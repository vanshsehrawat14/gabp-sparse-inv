"""Gates for the maze-on-trees demo (`gabp_sparse_inv/demos/maze_tree.py`).

The claim under test: a single differentiable tree-solve layer is the *only* long-range
operator that solves the source-routing task; a range-limited local message-passing model
provably cannot. The tests check three things: the task labels really are the exact tree solve
`A^{-1} e_s`; the learned precision stays SPD and well-conditioned (the flagged maze-
conditioning risk); and the tree-solve model beats the local ablation by a wide margin.
"""

from __future__ import annotations

import torch

from gabp_sparse_inv.demos.maze_tree import (
    GaBPMaze,
    balanced_parent,
    gen_dataset,
    train_eval,
)
from gabp_sparse_inv.layout import BlockTree, _as_parent_tensor

torch.manual_seed(0)


def test_task_labels_are_exact_solve():
    # The label at each node IS the routed field x = A^{-1} e_s; cross-check the kernel's
    # tree_solve against a plain dense solve of the reconstructed operator.
    parent = balanced_parent(3)
    feats, target = gen_dataset(parent, 6, seed=0, eps=0.03)
    p = _as_parent_tensor(parent)
    A = BlockTree(feats[..., 0][..., None, None], feats[..., 1][..., None, None], p).to_dense()
    x_dense = torch.linalg.solve(A, feats[..., 3].unsqueeze(-1)).squeeze(-1)
    assert (x_dense - target).abs().max() < 1e-10


def test_learned_precision_spd_and_conditioned():
    # The flagged maze-conditioning risk: the diagonally-dominant Laplacian + eps*I
    # parameterization must keep the learned precision SPD and well-conditioned for any
    # weights the encoder produces. Check a fresh model over a feature batch.
    parent = balanced_parent(4)
    feats, _ = gen_dataset(parent, 8, seed=1, eps=0.03)
    model = GaBPMaze(parent, eps=0.03).double()
    with torch.no_grad():
        diag, edge, _ = model.build_precision(feats)
    A = BlockTree(diag[..., None, None], edge[..., None, None], _as_parent_tensor(parent)).to_dense()
    assert float(torch.linalg.eigvalsh(A).min()) > 0       # SPD
    assert float(torch.linalg.cond(A).max()) < 1e3         # bounded kappa (eps fixes it)


def test_gabp_beats_local_ablation():
    # Headline: one tree_solve layer routes the source near-exactly (diameter 8 here),
    # while 2-hop local message passing cannot reach past its receptive field.
    r = train_eval(balanced_parent(4), eps=0.03, steps=150, n_train=48, n_test=48, seed=0)
    assert r["gabp"] < 5e-2                 # near-exact global routing
    assert r["gabp"] < 0.1 * r["local"]     # wide architectural margin over the ablation
    assert r["kappa_max"] < 1e3             # conditioning stayed bounded through training
