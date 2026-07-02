"""Matched-capacity long-range baselines for the grid maze (Track B / E4).

The grid maze (``maze_grid.py``) pits one global ``junction_solve`` (``GaBPMazeGrid``)
against a deliberately myopic ``K``-hop averaging model (``LocalMazeGrid``). That is an
*architectural ablation*: the local model is range-limited **by construction**, so its
failure is structural -- not evidence the inverse buys a *capability*. This module adds
**fair** long-range baselines that *can* route globally, to test the honest question:

    Does the exact selected inverse buy something a matched-or-over-matched universal
    long-range learner cannot get from data?

Baselines (both share the same strictly-local per-node inputs ``[phi_v, deg_v, b_v]`` as
``GaBPMazeGrid``, so the only difference is the long-range mechanism):
  * ``GNNMaze`` -- ``depth`` rounds of (self + mean-neighbour) message passing with
    residuals. A fair long-range operator (unlike the fixed ``K=2`` local model): with
    ``depth >= diameter`` it can route -- but that *is* an unrolled iterative solver, with
    ``O(diameter)`` depth and a ``depth``-hop receptive field.
  * ``TransformerMaze`` -- full self-attention over the ``n`` nodes with a non-parametric
    2-D sinusoidal positional encoding. Global reach at every layer; the strongest fair
    fixed-capacity baseline.

Both have **>> the (65-param) GaBP encoder's** capacity, so "matched capacity" is the easy
direction -- the test is whether the over-resourced learner still can't match the solve.

Two axes:
  * **Extrapolation (headline).** The GaBP encoder is size-independent and the solve is
    exact at any size, so weights trained on a small grid transfer and stay accurate on a
    larger one. A fixed-capacity GNN/Transformer trained small should degrade as the test
    diameter grows. ``evaluate_extrapolation`` trains at one size and evaluates every model
    on a range of (larger) sizes by transferring the size-independent weights.
  * **Capacity (control).** ``capacity_sweep`` varies a baseline's width at a fixed size to
    check the in-distribution gap does not close as capacity grows.

Numbers from ``run_benchmark`` are BLAS/seed-dependent diagnostics; the machine-independent
facts (harness runs, baselines over-parameterised vs GaBP, weights transfer across sizes)
are gated in ``tests/test_maze_baselines.py``.

Run ``python -m gabp_sparse_inv.demos.maze_baselines`` for the extrapolation table.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..generators import grid_edges
from ..junction import junction_solve, selinv_junction
from .maze_grid import GaBPMazeGrid, gen_dataset, grid_laplacian_from_potentials

__all__ = [
    "GNNMaze",
    "TransformerMaze",
    "GaBPResistance",
    "gen_resistance_dataset",
    "count_params",
    "build_models",
    "train_model",
    "evaluate_extrapolation",
    "extrapolation_curve",
    "capacity_sweep",
    "run_benchmark",
    "run_curve",
]


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters (the capacity-matching currency)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _grid_adjacency(rows: int, cols: int) -> Tensor:
    n = rows * cols
    ei = grid_edges(rows, cols)
    adj = torch.zeros(n, n)
    adj[ei[0], ei[1]] = 1.0
    adj[ei[1], ei[0]] = 1.0
    return adj


def _row_normalize(adj: Tensor) -> Tensor:
    return adj / adj.sum(-1, keepdim=True).clamp_min(1.0)


def _sinusoidal(pos: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal features of a 1-D coordinate; ``pos`` ``[n]`` -> ``[n, dim]``."""
    div = torch.exp(torch.arange(0, dim, 2).double() * (-math.log(10000.0) / max(dim, 1)))
    out = torch.zeros(pos.shape[0], dim, dtype=torch.float64)
    out[:, 0::2] = torch.sin(pos[:, None] * div[: out[:, 0::2].shape[1]])
    out[:, 1::2] = torch.cos(pos[:, None] * div[: out[:, 1::2].shape[1]])
    return out


def _grid_pe(rows: int, cols: int, hidden: int) -> Tensor:
    """Non-parametric 2-D sinusoidal positional encoding over normalised grid coords.

    Normalised to ``[0, 1]`` per axis so the encoding is defined at any grid size -- the
    most baseline-favourable choice for the extrapolation axis (a learned per-position
    table could not transfer at all).
    """
    idx = torch.arange(rows * cols)
    r = (idx // cols).double() / max(rows - 1, 1)
    c = (idx % cols).double() / max(cols - 1, 1)
    half = hidden // 2
    return torch.cat([_sinusoidal(r, half), _sinusoidal(c, hidden - half)], dim=-1)


class GNNMaze(nn.Module):
    """Deep residual message-passing baseline (``depth`` rounds of self + mean-neighbour).

    Size-dependent state (the adjacency) is a non-persistent buffer, so the trainable
    weights are size-independent and transfer across grid sizes; the receptive field is
    ``depth`` hops, so on a grid of diameter ``> depth`` the model is structurally blind to
    the far field -- the honest "fixed-capacity learner cannot extrapolate" mechanism.
    """

    def __init__(self, rows: int, cols: int, hidden: int = 32, depth: int = 10):
        super().__init__()
        self.register_buffer("adj_norm", _row_normalize(_grid_adjacency(rows, cols)), persistent=False)
        self.enc = nn.Linear(3, hidden)
        self.self_w = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth))
        self.nbr_w = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth))
        self.read = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats: Tensor) -> Tensor:
        h = torch.tanh(self.enc(feats))
        for sw, nw in zip(self.self_w, self.nbr_w):
            nbr = torch.einsum("ij,bjh->bih", self.adj_norm, h)
            h = h + torch.tanh(sw(h) + nw(nbr))  # residual keeps deep stacks trainable
        return self.read(h)[..., 0]


class TransformerMaze(nn.Module):
    """Full self-attention baseline: global reach at every layer, any grid size.

    The positional encoding is non-parametric (``_grid_pe``), so all trainable weights are
    size-independent and transfer across sizes -- giving the Transformer its best shot at
    the extrapolation axis.
    """

    def __init__(self, rows: int, cols: int, hidden: int = 32, heads: int = 2, layers: int = 2):
        super().__init__()
        self.register_buffer("pe", _grid_pe(rows, cols, hidden), persistent=False)
        self.enc = nn.Linear(3, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=0.0, batch_first=True
        )
        self.tf = nn.TransformerEncoder(layer, layers)
        self.read = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats: Tensor) -> Tensor:
        h = self.enc(feats) + self.pe
        h = self.tf(h)
        return self.read(h)[..., 0]


# --------------------------------------------------------------------------- #
# Held-out long-range task: effective resistance (a non-linear functional of A^{-1},
# not a single solve column -- it broadens the claim past reconstructing A^{-1} e_s).
# --------------------------------------------------------------------------- #
def gen_resistance_dataset(rows, cols, num, *, seed, eps=0.05, w_lo=0.5, w_hi=1.5,
                           dtype=torch.float64):
    """Effective-resistance field on the same maze precision as :func:`gen_dataset`.

    Same SPD grid Laplacian ``A = L(phi) + eps I`` and random source ``s``, but the label is
    the **effective resistance** ``R(v, s) = (A^{-1})_vv - 2 (A^{-1})_vs + (A^{-1})_ss`` -- the
    canonical global graph metric. It is a non-linear functional of the inverse, using its
    **diagonal** (``selinv_junction``) as well as a solve column (``junction_solve``), so it is
    *not* the exact function any single solve computes. Returns ``(edge_index, feats, target)``
    with ``feats[..., v] = [phi_v, weighted_degree_v, b_v]`` and ``target[..., v] = R(v, s)``.
    """
    edge_index = grid_edges(rows, cols)
    i_idx, j_idx = edge_index[0], edge_index[1]
    n = rows * cols
    g = torch.Generator().manual_seed(seed)

    phi = w_lo + (w_hi - w_lo) * torch.rand(num, n, generator=g, dtype=dtype)
    diag, edge_val, _w = grid_laplacian_from_potentials(phi, eps, i_idx, j_idx)
    deg = diag - eps

    src = torch.randint(0, n, (num,), generator=g)
    b = torch.zeros(num, n, dtype=dtype)
    b[torch.arange(num), src] = 1.0

    x = junction_solve(diag[..., None, None], edge_index, edge_val[..., None, None], b[..., None])[..., 0]
    G_diag, _Si, _Gl = selinv_junction(diag[..., None, None], edge_index, edge_val[..., None, None])
    d = G_diag[..., 0, 0]                                    # diag(A^{-1}) -- the selected inverse
    x_s = (x * b).sum(-1, keepdim=True)                      # (A^{-1})_ss
    target = d - 2 * x + x_s                                 # R(v, s)

    feats = torch.stack([phi, deg, b], dim=-1)
    return edge_index, feats, target


class GaBPResistance(GaBPMazeGrid):
    """GaBP encoder -> precision -> (``selinv_junction`` diagonal + ``junction_solve``) -> R(v, s).

    Same size-independent encoder as :class:`GaBPMazeGrid` (so weights transfer across sizes),
    but the readout is the deterministic effective-resistance formula. It *contains* the two
    selected-inverse ops the task needs, so it computes ``R`` near-exactly once the encoder
    recovers the geometry -- the inductive-bias point, now for a non-linear inverse functional.
    """

    def forward(self, feats: Tensor) -> Tensor:
        diag, edge_val = self.build_precision(feats)
        b = feats[..., 2]
        x = junction_solve(diag[..., None, None], self.edge_index, edge_val[..., None, None], b[..., None])[..., 0]
        G_diag, _Si, _Gl = selinv_junction(diag[..., None, None], self.edge_index, edge_val[..., None, None])
        d = G_diag[..., 0, 0]
        x_s = (x * b).sum(-1, keepdim=True)
        return d - 2 * x + x_s


_GEN = {"route": gen_dataset, "resistance": gen_resistance_dataset}


def build_models(rows, cols, *, eps=0.05, hidden=32, gnn_depth=10, tf_layers=2, task="route"):
    """The three models at one grid size (GaBP encoder kept at its native width 16).

    ``task`` selects the GaBP readout: ``"route"`` (source field ``A^{-1} e_s``, the default)
    or ``"resistance"`` (the effective-resistance functional). The GNN / Transformer baselines
    are identical across tasks -- only their regression target changes.
    """
    gabp = (GaBPResistance if task == "resistance" else GaBPMazeGrid)(rows, cols, eps, hidden=16)
    return {
        "gabp": gabp,
        "gnn": GNNMaze(rows, cols, hidden=hidden, depth=gnn_depth),
        "transformer": TransformerMaze(rows, cols, hidden=hidden, layers=tf_layers),
    }


def _transfer_params(dst: nn.Module, src: nn.Module) -> None:
    """Copy size-independent trainable parameters src -> dst (buffers rebuilt per size)."""
    src_params = dict(src.named_parameters())
    with torch.no_grad():
        for name, p in dst.named_parameters():
            p.copy_(src_params[name])


def train_model(model: nn.Module, feats: Tensor, y: Tensor, *, steps=300, lr=0.01) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        torch.mean((model(feats) - y) ** 2).backward()
        opt.step()
    return model


def evaluate_extrapolation(
    train_size=(6, 6),
    test_sizes=((6, 6), (8, 8), (10, 10)),
    *,
    eps=0.05,
    n_train=48,
    n_test=48,
    steps=300,
    seed=0,
    hidden=32,
    gnn_depth=10,
    tf_layers=2,
    task="route",
):
    """Train every model at ``train_size``; evaluate each on every ``test_sizes`` grid.

    Returns ``(mse, params)`` where ``mse[(name, size)]`` is the held-out MSE and
    ``params[name]`` the trainable-parameter count. ``name`` ranges over
    ``gabp / gnn / transformer`` plus ``baseline`` (predict-the-training-mean). ``task`` is
    ``"route"`` (default, label ``A^{-1} e_s``) or ``"resistance"`` (the held-out effective-
    resistance target); the GaBP readout and the dataset change, the baselines do not.
    """
    gen = _GEN[task]
    rr, cc = train_size
    _ei, feats_tr, y_tr = gen(rr, cc, n_train, seed=seed, eps=eps)

    trained, params = {}, {}
    for name, model in build_models(rr, cc, eps=eps, hidden=hidden, gnn_depth=gnn_depth, tf_layers=tf_layers, task=task).items():
        torch.manual_seed(seed)
        model = model.double()
        params[name] = count_params(model)
        train_model(model, feats_tr, y_tr, steps=steps)
        trained[name] = model

    mse = {}
    for (tr, tc) in test_sizes:
        _ei2, feats_te, y_te = gen(tr, tc, n_test, seed=seed + 1000, eps=eps)
        mse[("baseline", (tr, tc))] = float(torch.mean((y_te - y_tr.mean()) ** 2))
        for name, fresh in build_models(tr, tc, eps=eps, hidden=hidden, gnn_depth=gnn_depth, tf_layers=tf_layers, task=task).items():
            fresh = fresh.double()
            _transfer_params(fresh, trained[name])  # size-independent weights transfer
            with torch.no_grad():
                mse[(name, (tr, tc))] = float(torch.mean((fresh(feats_te) - y_te) ** 2))
    return mse, params


def extrapolation_curve(
    train_size=(6, 6),
    test_sizes=((6, 6), (8, 8), (10, 10)),
    *,
    seeds=(0, 1, 2),
    task="route",
    **kw,
):
    """Multi-seed extrapolation curve: median + per-seed (min, max) over ``seeds``.

    Runs :func:`evaluate_extrapolation` for each seed and aggregates. Returns ``(curve,
    params)`` where ``curve[(name, size)] = (median, lo, hi)`` over seeds. The headline read is
    that the GaBP median stays flat and below every baseline at every size, and that the
    *spreads do not overlap* at the largest grid -- a seed-robust separation, not a lucky seed.
    """
    runs = [evaluate_extrapolation(train_size, test_sizes, seed=s, task=task, **kw) for s in seeds]
    mses = [r[0] for r in runs]
    params = runs[0][1]
    names = ("gabp", "gnn", "transformer", "baseline")
    curve = {}
    for size in test_sizes:
        for name in names:
            vals = sorted(m[(name, size)] for m in mses)
            curve[(name, size)] = (vals[len(vals) // 2], vals[0], vals[-1])
    return curve, params


def capacity_sweep(size=(6, 6), hiddens=(16, 32, 64), *, model="transformer", eps=0.05,
                   n_train=48, n_test=48, steps=300, seed=0):
    """In-distribution control: vary a baseline's width; return ``{hidden: (params, mse)}``.

    Checks the gap does not close as the baseline is given more capacity at a fixed size.
    """
    rr, cc = size
    _ei, feats_tr, y_tr = gen_dataset(rr, cc, n_train, seed=seed, eps=eps)
    _ei2, feats_te, y_te = gen_dataset(rr, cc, n_test, seed=seed + 1000, eps=eps)
    out = {}
    for h in hiddens:
        torch.manual_seed(seed)
        m = (GNNMaze(rr, cc, hidden=h) if model == "gnn" else TransformerMaze(rr, cc, hidden=h)).double()
        train_model(m, feats_tr, y_tr, steps=steps)
        with torch.no_grad():
            out[h] = (count_params(m), float(torch.mean((m(feats_te) - y_te) ** 2)))
    return out


def run_benchmark(train_size=(6, 6), test_sizes=((6, 6), (8, 8), (10, 10)), *, task="route", **kw):
    """Print the extrapolation table (MSE per model per test grid) + parameter counts."""
    mse, params = evaluate_extrapolation(train_size, test_sizes, task=task, **kw)
    names = ("gabp", "gnn", "transformer", "baseline")
    print(f"task={task};  train grid {train_size[0]}x{train_size[1]};  params: " +
          ", ".join(f"{k}={params[k]}" for k in ("gabp", "gnn", "transformer")))
    header = f"{'test grid':>10} {'diam':>5} " + " ".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))
    for (tr, tc) in test_sizes:
        diam = (tr - 1) + (tc - 1)
        cells = " ".join(f"{mse[(n, (tr, tc))]:>12.2e}" for n in names)
        print(f"{f'{tr}x{tc}':>10} {diam:>5} {cells}")
    return mse, params


def run_curve(train_size=(6, 6), test_sizes=((6, 6), (8, 8), (10, 10)), *, seeds=(0, 1, 2),
              task="route", **kw):
    """Print the multi-seed extrapolation curve: ``median [min, max]`` per model per grid."""
    curve, params = extrapolation_curve(train_size, test_sizes, seeds=seeds, task=task, **kw)
    names = ("gabp", "gnn", "transformer", "baseline")
    print(f"task={task};  train {train_size[0]}x{train_size[1]};  {len(seeds)} seeds  "
          f"(median [min, max]);  params: " + ", ".join(f"{k}={params[k]}" for k in ("gabp", "gnn", "transformer")))
    header = f"{'test grid':>9} {'diam':>5} " + " ".join(f"{n:>22}" for n in names)
    print(header)
    print("-" * len(header))
    for (tr, tc) in test_sizes:
        diam = (tr - 1) + (tc - 1)
        cells = " ".join(f"{curve[(n, (tr, tc))][0]:>8.1e} [{curve[(n, (tr, tc))][1]:.0e},{curve[(n, (tr, tc))][2]:.0e}]"
                         for n in names)
        print(f"{f'{tr}x{tc}':>9} {diam:>5} {cells}")
    return curve, params


if __name__ == "__main__":
    print("=== source routing (A^-1 e_s) ===")
    run_curve(task="route")
    print("\n=== held-out: effective resistance R(v,s) ===")
    run_curve(task="resistance")
