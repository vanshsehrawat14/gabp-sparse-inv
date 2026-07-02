"""Maze-on-trees: a differentiable tree-inverse layer as the only long-range operator.

The grid maze of the roadmap is loopy (treewidth ~sqrt(n)) and needs the junction-tree
selected inverse. This is the **tree proxy** that de-risks it on the shipped kernels: a
source-routing task where the answer at a node depends on a source that may be anywhere in
the tree, so a range-limited local model cannot solve it, but one differentiable tree
GaBP layer (`tree_solve` -- the linear-system form of the same collect/distribute
selected-inversion machinery as `selinv_tree`) solves it exactly.

Task (routing a source through the tree). The operator is a **tree-graph Laplacian plus a
small diagonal** ``A = L(w) + eps I`` with positive edge weights ``w`` -- a diffusion /
"maze" operator. Off-diagonals ``A_{p(v),v} = -w[v]``; diagonal ``A_vv = eps + (weighted
degree)``; SPD by diagonal dominance with margin ``eps``. A *small* ``eps`` makes ``A``
nearly singular along the constant mode, so ``kappa ~ deg/eps`` is large and ``A^{-1}``
(the tree Green's function) has a **long correlation length** -- the long-range regime.
Each example draws random ``w`` and a random **source** node ``s``; the label is the
induced field ``x = A^{-1} e_s`` (the potential routed out from the source along the unique
tree paths). A node far from the source has a small but nonzero potential that depends on
the whole path back to ``s``.

Models (identical inputs: each node's local row of ``A`` and the source indicator,
``[A_vv, edge[v], sum_children w, b_v]``).
  * **gabp** -- a per-node MLP maps the local features to positive edge weights of a
    precision ``hatA = L(hat w) + eps I`` (SPD by construction: the flagged "maze
    conditioning" risk is handled by the diagonal-dominant parameterization, with ``eps``
    fixing ``kappa``), then one `tree_solve` layer returns ``hatA^{-1} b`` as the
    prediction. With the inputs encoding ``A``, the layer represents the exact field.
  * **local** -- the same features, then ``K`` rounds of nearest-neighbour message passing
    (``K`` much smaller than the tree diameter) and a readout MLP. No global operator, so
    it cannot route a source farther than ``K`` hops.

Run ``python -m gabp_sparse_inv.demos.maze_tree`` for the depth-sweep table.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..layout import BlockTree, _as_parent_tensor, tree_orders
from ..gmrf import tree_solve

__all__ = [
    "balanced_parent",
    "tree_laplacian",
    "gen_dataset",
    "GaBPMaze",
    "LocalMaze",
    "train_eval",
    "run_demo",
]


def balanced_parent(depth: int) -> list[int]:
    """Parent array of a complete binary tree of the given depth (``n = 2^{depth+1}-1``)."""
    n = 2 ** (depth + 1) - 1
    return [-1] + [(v - 1) // 2 for v in range(1, n)]


def _scatter_index(parent: Tensor) -> tuple[Tensor, int]:
    """Index mapping each node to its parent (root -> itself), for child-edge scatters."""
    root, _, _ = tree_orders(parent)
    plist = parent.tolist()
    idx = torch.tensor([pp if pp != -1 else root for pp in plist], dtype=torch.long)
    return idx, root


def tree_laplacian(w, eps, par_idx, root):
    """Build ``A = L(w) + eps I`` for per-node parent-edge weights ``w`` (``w[root]`` unused).

    Returns ``(diag, edge)`` with ``edge[v] = -w[v]`` (the off-diagonal ``A_{p(v),v}``) and
    ``A_vv = eps + w[v] + sum_{c in ch(v)} w[c]`` (eps + weighted degree). SPD by strict
    diagonal dominance (the off-diagonal magnitudes sum to ``deg``, the diagonal is
    ``eps + deg``), with condition number set by ``eps``.
    """
    # w[v] is the weight of the edge from v to its parent. The root has no parent edge,
    # so force w[root]=0 (and edge[root]=0, which the tree kernel requires anyway).
    w = w.index_fill(-1, torch.tensor([root], device=w.device), 0.0)
    # Weighted degree splits into "up" (w[v], the parent edge) + "down" (sum of child
    # edge weights). Scatter each child's w into its parent to get the down part.
    child_deg = torch.zeros_like(w).index_add(-1, par_idx, w)        # sum_{c in ch(v)} w[c]
    # Graph Laplacian: A_vv = (weighted degree), A_{p,v} = -w[v]. Adding eps*I lifts the
    # constant-mode null space so A is strictly diagonally dominant => SPD, and pins the
    # condition number at ~deg/eps. Small eps = long correlation length = the long-range
    # target regime (and the knob that controls the flagged maze-conditioning risk).
    diag = eps + w + child_deg
    edge = -w
    return diag, edge, child_deg


def gen_dataset(
    parent,
    num: int,
    *,
    seed: int,
    eps: float = 0.03,
    w_lo: float = 0.5,
    w_hi: float = 1.5,
    dtype: torch.dtype = torch.float64,
):
    """Random Laplacian precisions, random sources, and the exactly-routed fields.

    Returns ``(feats, target)`` with ``feats[..., v] = [A_vv, edge[v], sum_children w, b_v]``
    ``[num, n, 4]`` and ``target[..., v] = (A^{-1} e_s)_v`` ``[num, n]``.
    """
    parent = _as_parent_tensor(parent)
    par_idx, root = _scatter_index(parent)
    n = int(parent.numel())
    g = torch.Generator().manual_seed(seed)

    # Each example gets its own random maze geometry (edge weights) ...
    w = w_lo + (w_hi - w_lo) * torch.rand(num, n, generator=g, dtype=dtype)
    diag, edge, child_deg = tree_laplacian(w, eps, par_idx, root)

    # ... and a random source node s. b = e_s is a one-hot "inject here" signal.
    src = torch.randint(0, n, (num,), generator=g)
    b = torch.zeros(num, n, dtype=dtype)
    b[torch.arange(num), src] = 1.0

    # The label is the field the source induces, x = A^{-1} e_s (the tree Green's function
    # column at s). tree_solve does this exactly in O(n); blocks are scalar, so add the
    # trailing [1,1]/[1] dims the kernel expects. This is the long-range target: a node far
    # from s still has a nonzero potential that depends on the whole path back to s.
    x = tree_solve(diag[..., None, None], edge[..., None, None], parent, b[..., None])
    target = x[..., 0]

    # Features both models see, per node: its own row of A (diag, parent-edge, down-degree)
    # plus the source indicator b_v. Everything here is local; no node is handed anything
    # about distant nodes; the only way to learn the global field is a global operator.
    feats = torch.stack([diag, edge, child_deg, b], dim=-1)
    return feats, target


class GaBPMaze(nn.Module):
    """Local encoder -> SPD tree Laplacian -> one ``tree_solve`` layer routing the source."""

    def __init__(self, parent, eps: float = 0.03, hidden: int = 16):
        super().__init__()
        parent = _as_parent_tensor(parent)
        par_idx, root = _scatter_index(parent)
        self.register_buffer("parent", parent)
        self.register_buffer("par_idx", par_idx)
        self.root = int(root)
        self.eps = float(eps)
        self.enc = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def build_precision(self, feats: Tensor):
        # Read the maze geometry from the first 3 features only (NOT the source b_v): the
        # operator A is a property of the maze, not of where you inject. softplus keeps the
        # edge weights positive, so tree_laplacian stays SPD no matter what the encoder does
        # That is the conditioning-risk story, handled by construction.
        w = torch.nn.functional.softplus(self.enc(feats[..., :3])[..., 0])
        return tree_laplacian(w, self.eps, self.par_idx, self.root)

    def forward(self, feats: Tensor) -> Tensor:
        diag, edge, _ = self.build_precision(feats)
        b = feats[..., 3]
        # This one line is the only long-range operator in the whole model: tree_solve
        # propagates the injected source across the entire tree in O(n). Everything else is
        # a pointwise MLP. If this solves the task, the credit is unambiguous.
        x = tree_solve(diag[..., None, None], edge[..., None, None], self.parent, b[..., None])
        return x[..., 0]


class LocalMaze(nn.Module):
    """Local features + ``K`` rounds of neighbour-mean message passing + readout (no global op)."""

    def __init__(self, parent, hidden: int = 16, rounds: int = 2):
        super().__init__()
        parent = _as_parent_tensor(parent)
        n = int(parent.numel())
        adj = torch.zeros(n, n)
        for v, pp in enumerate(parent.tolist()):
            if pp != -1:
                adj[v, pp] = adj[pp, v] = 1.0
        self.register_buffer("adj", adj)
        self.register_buffer("deg", adj.sum(-1, keepdim=True).clamp_min(1.0))
        self.rounds = rounds
        self.enc = nn.Linear(4, hidden)
        self.msg = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh()) for _ in range(rounds)]
        )
        self.read = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats: Tensor) -> Tensor:
        # Same inputs as GaBPMaze, but the only mixing is `rounds` steps of averaging over
        # tree neighbours. After K rounds a node has only seen K hops out, so it physically
        # cannot know about a source farther than K away. With K << diameter this model is
        # blind to most of the routing task regardless of training duration.
        h = torch.tanh(self.enc(feats))
        for layer in self.msg:
            nb = torch.einsum("ij,bjh->bih", self.adj, h) / self.deg   # mean of neighbours
            h = layer(torch.cat([h, nb], dim=-1))
        return self.read(h)[..., 0]


def _kappa_range(model: GaBPMaze, feats: Tensor):
    """Condition-number summary of the learned precisions over a feature batch."""
    with torch.no_grad():
        diag, edge, _ = model.build_precision(feats)
        dense = BlockTree(diag[..., None, None], edge[..., None, None], model.parent).to_dense()
        kappa = torch.linalg.cond(dense)
    return float(kappa.median()), float(kappa.max())


def train_eval(
    parent,
    *,
    eps: float = 0.03,
    n_train: int = 96,
    n_test: int = 128,
    steps: int = 400,
    lr: float = 0.02,
    rounds: int = 2,
    hidden: int = 16,
    seed: int = 0,
):
    """Train the gabp and local models on one topology; return test MSEs and kappa range."""
    feats_tr, y_tr = gen_dataset(parent, n_train, seed=seed, eps=eps)
    feats_te, y_te = gen_dataset(parent, n_test, seed=seed + 1000, eps=eps)

    out = {"baseline": float(torch.mean((y_te - y_tr.mean()) ** 2))}
    for name in ("gabp", "local"):
        torch.manual_seed(seed)
        model = (GaBPMaze(parent, eps, hidden) if name == "gabp" else LocalMaze(parent, hidden, rounds)).double()
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


def run_demo(depths=(3, 4, 5), eps=0.03, **kw):
    """Print the depth-sweep table (gabp vs local vs predict-mean baseline)."""
    header = f"{'depth':>5} {'n':>5} {'diam':>5} {'gabp':>11} {'local':>11} {'baseline':>11} {'kappa(med/max)':>16}"
    print(header)
    print("-" * len(header))
    rows = {}
    for d in depths:
        parent = balanced_parent(d)
        r = train_eval(parent, eps=eps, **kw)
        rows[d] = r
        print(
            f"{d:>5} {len(parent):>5} {2 * d:>5} {r['gabp']:>11.2e} {r['local']:>11.2e} "
            f"{r['baseline']:>11.2e} {r['kappa_median']:>7.1f}/{r['kappa_max']:<8.1f}"
        )
    return rows


if __name__ == "__main__":
    run_demo()
