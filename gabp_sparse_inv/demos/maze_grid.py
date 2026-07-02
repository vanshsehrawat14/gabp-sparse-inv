"""Maze-on-grids: a differentiable junction-tree inverse layer as the only long-range op.

The grid analogue of ``maze_tree.py``. A 2-D lattice is **loopy** (treewidth ~ side
length), so routing a source across it genuinely needs the junction-tree (filled-pattern)
selected inverse rather than a tree -- this is the point that ``maze_tree`` could only
proxy. One differentiable :func:`~gabp_sparse_inv.junction_solve` layer routes the source
exactly; a range-limited local message-passing model provably cannot.

Task (routing a source through the grid). The operator is a **grid-graph Laplacian plus a
small diagonal** ``A = L(w) + eps I`` with positive edge weights ``w`` -- a diffusion /
"maze" operator. A small ``eps`` lifts the constant-mode null space, pinning
``kappa ~ deg/eps`` and giving ``A^{-1}`` (the grid Green's function) a long correlation
length: the long-range regime. Each example draws a random maze geometry and a random
**source** node ``s``; the label is the induced field ``x = A^{-1} e_s`` (the potential
routed out from the source across the lattice), computed with ``junction_solve``.

Edge weights are factored through per-node potentials, ``w_ij = phi_i * phi_j`` (positive),
so -- exactly as a tree hands each node its single parent-edge weight -- the operator is a
function of strictly **local** node features that the per-node encoder can reconstruct.
This keeps the only obstacle for the local model the *global solve*, not missing inputs.

Models (identical inputs: each node's local features ``[phi_v, weighted_degree_v, b_v]``).
  * **gabp** -- a per-node MLP maps the local geometry features (not the source) to positive
    node potentials, hence to edge weights of a precision ``hatA = L(hat w) + eps I`` (SPD by
    construction: the flagged maze-conditioning risk is handled by the Laplacian + ``eps``
    parameterization), then one ``junction_solve`` returns ``hatA^{-1} b`` as the prediction.
  * **local** -- the same features, then ``K`` rounds of nearest-neighbour averaging
    (``K`` much smaller than the grid diameter) and a readout MLP. No global operator, so it
    cannot route a source farther than ``K`` hops.

Run ``python -m gabp_sparse_inv.demos.maze_grid`` for the size-sweep table.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..generators import grid_edges
from ..junction import junction_solve
from ..layout import BlockSparseSym

__all__ = [
    "grid_laplacian_from_potentials",
    "gen_dataset",
    "GaBPMazeGrid",
    "LocalMazeGrid",
    "train_eval",
    "run_demo",
    "causal_solve_sweep",
    "run_causal",
]


def _jacobi_solve(diag, i_idx, j_idx, edge_val, b, steps, omega):
    """``steps`` damped-Jacobi sweeps for ``A x = b`` on the grid pattern (``A = L(w)+eps I``).

    Each sweep is one matvec with ``A``, so information travels exactly **one hop per step**:
    ``K`` steps reach ``K`` hops, and ``K -> infinity`` recovers the exact solve. ``A`` is
    strictly diagonally dominant (the ``eps`` lift), so plain Jacobi (``omega=1``) converges;
    damping only slows it. Out-of-place (autograd-safe). Used by the matched-capacity *causal*
    control: the identical model with its exact solve replaced by a ``K``-hop one.
    """
    x = torch.zeros_like(b)
    for _ in range(steps):
        ax = diag * x
        ax = ax.index_add(-1, i_idx, edge_val * x.index_select(-1, j_idx))
        ax = ax.index_add(-1, j_idx, edge_val * x.index_select(-1, i_idx))
        x = x + omega * (b - ax) / diag
    return x


def grid_laplacian_from_potentials(phi, eps, i_idx, j_idx):
    """Build ``A = L(w) + eps I`` from per-node potentials, ``w_ij = phi_i * phi_j``.

    Returns ``(diag, edge_val, w)`` with ``diag[..., v] = eps + sum_{e ~ v} w_e`` (the
    weighted degree lifted by ``eps``), ``edge_val[..., k] = -w_k`` (the off-diagonal
    ``A_{i_k, j_k}``), and ``w`` the per-edge weights. SPD by strict diagonal dominance:
    the incident off-diagonal magnitudes sum to the weighted degree, the diagonal is that
    plus ``eps``. All ops are out-of-place (autograd-safe).
    """
    w = phi[..., i_idx] * phi[..., j_idx]                    # [..., m], positive
    deg = phi.new_zeros(phi.shape).index_add(-1, i_idx, w).index_add(-1, j_idx, w)
    diag = eps + deg
    edge_val = -w
    return diag, edge_val, w


def gen_dataset(
    rows: int,
    cols: int,
    num: int,
    *,
    seed: int,
    eps: float = 0.05,
    w_lo: float = 0.5,
    w_hi: float = 1.5,
    dtype: torch.dtype = torch.float64,
):
    """Random Laplacian precisions, random sources, and the exactly-routed fields.

    Returns ``(edge_index, feats, target)`` with ``edge_index`` ``[2, m]`` (the shared grid
    pattern), ``feats[..., v] = [phi_v, weighted_degree_v, b_v]`` ``[num, n, 3]`` and
    ``target[..., v] = (A^{-1} e_s)_v`` ``[num, n]``.
    """
    edge_index = grid_edges(rows, cols)
    i_idx, j_idx = edge_index[0], edge_index[1]
    n = rows * cols
    g = torch.Generator().manual_seed(seed)

    # Each example: its own random potentials (maze geometry) and a random source node s.
    phi = w_lo + (w_hi - w_lo) * torch.rand(num, n, generator=g, dtype=dtype)
    diag, edge_val, _w = grid_laplacian_from_potentials(phi, eps, i_idx, j_idx)
    deg = diag - eps                                          # weighted degree

    src = torch.randint(0, n, (num,), generator=g)
    b = torch.zeros(num, n, dtype=dtype)
    b[torch.arange(num), src] = 1.0

    # Label = the field the source induces, x = A^{-1} e_s (the grid Green's-function column
    # at s), computed exactly by junction_solve on the loopy pattern (scalar blocks b=1). A
    # node far from s still has a nonzero potential depending on the whole lattice between.
    x = junction_solve(diag[..., None, None], edge_index, edge_val[..., None, None], b[..., None])
    target = x[..., 0]

    # Inputs both models see, per node: its potential, weighted degree, and source indicator.
    # All LOCAL -- no node is handed anything about distant nodes; the only way to learn the
    # global field is a global operator.
    feats = torch.stack([phi, deg, b], dim=-1)
    return edge_index, feats, target


class GaBPMazeGrid(nn.Module):
    """Local encoder -> SPD grid Laplacian -> one ``junction_solve`` routing the source."""

    def __init__(self, rows: int, cols: int, eps: float = 0.05, hidden: int = 16,
                 solve_steps: int | None = None, jacobi_omega: float = 1.0):
        super().__init__()
        edge_index = grid_edges(rows, cols)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("i_idx", edge_index[0])
        self.register_buffer("j_idx", edge_index[1])
        self.eps = float(eps)
        # solve_steps=None -> the exact junction_solve (the model proper). An int K replaces it
        # with K Jacobi hops on the SAME learned A: the matched-capacity causal knob (identical
        # parameters, only the solve's reach changes). See causal_solve_sweep.
        self.solve_steps = solve_steps
        self.jacobi_omega = float(jacobi_omega)
        self.enc = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def build_precision(self, feats: Tensor):
        # Read the maze geometry from the first 2 features only (NOT the source b_v).
        # softplus keeps potentials positive, so w_ij = phi_i phi_j > 0 and the Laplacian
        # stays SPD no matter what the encoder produces -- the conditioning-risk story,
        # handled by construction (eps pins kappa).
        phi = torch.nn.functional.softplus(self.enc(feats[..., :2])[..., 0])
        diag, edge_val, _w = grid_laplacian_from_potentials(phi, self.eps, self.i_idx, self.j_idx)
        return diag, edge_val

    def forward(self, feats: Tensor) -> Tensor:
        diag, edge_val = self.build_precision(feats)
        b = feats[..., 2]
        if self.solve_steps is not None:
            # Causal ablation: the SAME model, but the solve is truncated to K hops of reach.
            return _jacobi_solve(diag, self.i_idx, self.j_idx, edge_val, b,
                                 self.solve_steps, self.jacobi_omega)
        # The only long-range operator in the model: junction_solve routes the injected
        # source across the whole loopy lattice in one structured solve. Everything else is
        # a pointwise MLP, so if this solves the task the credit is unambiguous.
        x = junction_solve(diag[..., None, None], self.edge_index, edge_val[..., None, None], b[..., None])
        return x[..., 0]


class LocalMazeGrid(nn.Module):
    """Local features + ``K`` rounds of neighbour-mean message passing + readout (no global op)."""

    def __init__(self, rows: int, cols: int, hidden: int = 16, rounds: int = 2):
        super().__init__()
        n = rows * cols
        edge_index = grid_edges(rows, cols)
        adj = torch.zeros(n, n)
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            adj[i, j] = adj[j, i] = 1.0
        self.register_buffer("adj", adj)
        self.register_buffer("deg", adj.sum(-1, keepdim=True).clamp_min(1.0))
        self.rounds = rounds
        self.enc = nn.Linear(3, hidden)
        self.msg = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh()) for _ in range(rounds)]
        )
        self.read = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats: Tensor) -> Tensor:
        # Same inputs as GaBPMazeGrid, but the only mixing is `rounds` steps of averaging
        # over grid neighbours. After K rounds a node has seen only K hops out, so it cannot
        # know about a source farther than K away. With K << diameter the model is blind to
        # most of the routing task no matter how long it trains.
        h = torch.tanh(self.enc(feats))
        for layer in self.msg:
            nb = torch.einsum("ij,bjh->bih", self.adj, h) / self.deg
            h = layer(torch.cat([h, nb], dim=-1))
        return self.read(h)[..., 0]


def _kappa_range(model: GaBPMazeGrid, feats: Tensor):
    """Condition-number summary of the learned precisions over a feature batch."""
    with torch.no_grad():
        diag, edge_val = model.build_precision(feats)
        dense = BlockSparseSym(
            diag[..., None, None], model.edge_index, edge_val[..., None, None]
        ).to_dense()
        kappa = torch.linalg.cond(dense)
    return float(kappa.median()), float(kappa.max())


def train_eval(
    rows: int,
    cols: int,
    *,
    eps: float = 0.05,
    n_train: int = 64,
    n_test: int = 64,
    steps: int = 250,
    lr: float = 0.02,
    rounds: int = 2,
    hidden: int = 16,
    seed: int = 0,
):
    """Train the gabp and local models on one grid size; return test MSEs and kappa range."""
    _ei_tr, feats_tr, y_tr = gen_dataset(rows, cols, n_train, seed=seed, eps=eps)
    _ei_te, feats_te, y_te = gen_dataset(rows, cols, n_test, seed=seed + 1000, eps=eps)

    out = {"baseline": float(torch.mean((y_te - y_tr.mean()) ** 2))}
    for name in ("gabp", "local"):
        torch.manual_seed(seed)
        model = (
            GaBPMazeGrid(rows, cols, eps, hidden)
            if name == "gabp"
            else LocalMazeGrid(rows, cols, hidden, rounds)
        ).double()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            torch.mean((model(feats_tr) - y_tr) ** 2).backward()
            opt.step()
        with torch.no_grad():
            out[name] = float(torch.mean((model(feats_te) - y_te) ** 2))
        if name == "gabp":
            out["kappa_median"], out["kappa_max"] = _kappa_range(model, feats_te)
    return out


def run_demo(sizes=((4, 4), (5, 5), (6, 6)), eps=0.05, **kw):
    """Print the grid-size-sweep table (gabp vs local vs predict-mean baseline)."""
    header = (
        f"{'grid':>7} {'n':>5} {'diam':>5} {'gabp':>11} {'local':>11} "
        f"{'baseline':>11} {'kappa(med/max)':>16}"
    )
    print(header)
    print("-" * len(header))
    rows_out = {}
    for (rr, cc) in sizes:
        r = train_eval(rr, cc, eps=eps, **kw)
        rows_out[(rr, cc)] = r
        diam = (rr - 1) + (cc - 1)
        print(
            f"{f'{rr}x{cc}':>7} {rr * cc:>5} {diam:>5} {r['gabp']:>11.2e} {r['local']:>11.2e} "
            f"{r['baseline']:>11.2e} {r['kappa_median']:>7.1f}/{r['kappa_max']:<8.1f}"
        )
    return rows_out


def causal_solve_sweep(
    rows: int,
    cols: int,
    *,
    steps_list=(1, 2, 4, 8, 16, None),
    eps: float = 0.05,
    n_train: int = 64,
    n_test: int = 64,
    steps: int = 250,
    lr: float = 0.02,
    hidden: int = 16,
    jacobi_omega: float = 1.0,
    seed: int = 0,
):
    """Matched-capacity **causal** control: the identical model, intervening on *only* reach.

    For each ``K`` in ``steps_list`` (``None`` = the exact ``junction_solve``), train and
    evaluate the **same** ``GaBPMazeGrid`` architecture and parameter count with its solve
    replaced by ``K`` Jacobi sweeps (= ``K`` hops). Architecture, capacity, inputs, data and
    training schedule are identical across ``K`` -- the single intervened variable is the
    solve's reach -- so a monotone dose-response in ``K``, with only the exact (global) solve
    reaching the predict-mean floor, isolates the inverse's **globality** as the causal factor
    rather than architecture or parameter count. Returns ``{K: test_mse}`` (+ ``"baseline"``,
    the predict-train-mean floor). Unlike the cross-architecture GNN/Transformer baselines
    (`maze_baselines.py`), this varies nothing but the one mechanism.
    """
    _ei_tr, feats_tr, y_tr = gen_dataset(rows, cols, n_train, seed=seed, eps=eps)
    _ei_te, feats_te, y_te = gen_dataset(rows, cols, n_test, seed=seed + 1000, eps=eps)
    out = {"baseline": float(torch.mean((y_te - y_tr.mean()) ** 2))}
    for K in steps_list:
        torch.manual_seed(seed)
        model = GaBPMazeGrid(rows, cols, eps, hidden, solve_steps=K, jacobi_omega=jacobi_omega).double()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            torch.mean((model(feats_tr) - y_tr) ** 2).backward()
            opt.step()
        with torch.no_grad():
            out[K] = float(torch.mean((model(feats_te) - y_te) ** 2))
    return out


def run_causal(sizes=((6, 6),), steps_list=(1, 2, 4, 8, 16, None), **kw):
    """Print the matched-capacity causal dose-response: test MSE vs solve depth K (None=exact)."""
    res = {}
    for (rr, cc) in sizes:
        r = causal_solve_sweep(rr, cc, steps_list=steps_list, **kw)
        res[(rr, cc)] = r
        diam = (rr - 1) + (cc - 1)
        print(f"grid {rr}x{cc} (diameter {diam}) -- same model + params, vary ONLY solve reach K:")
        print(f"  {'K hops':>7} {'test MSE':>11}   (None = exact junction_solve = unbounded reach)")
        for K in steps_list:
            label = "exact" if K is None else str(K)
            print(f"  {label:>7} {r[K]:>11.2e}")
        print(f"  {'mean':>7} {r['baseline']:>11.2e}   (predict-train-mean floor)")
    return res


if __name__ == "__main__":
    run_demo()
    print()
    run_causal()
