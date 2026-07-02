"""Gates for the maze-on-grids demo (`gabp_sparse_inv/demos/maze_grid.py`).

The grid analogue of `test_maze_tree.py`. A 2-D lattice is loopy, so routing a source
across it genuinely exercises the junction-tree (filled-pattern) selected inverse via
`junction_solve` -- the point the tree proxy could only stand in for. The tests check three
things: the task labels really are the exact solve `A^{-1} e_s`; the learned precision
stays SPD and well-conditioned (the flagged maze-conditioning risk); and the
junction-solve model beats the range-limited local ablation by a wide margin.
"""

from __future__ import annotations

import torch

from gabp_sparse_inv.demos.maze_grid import (
    GaBPMazeGrid,
    causal_solve_sweep,
    gen_dataset,
    train_eval,
)
from gabp_sparse_inv.layout import BlockSparseSym

torch.manual_seed(0)


def test_task_labels_are_exact_solve():
    # The label at each node IS the routed field x = A^{-1} e_s; cross-check the kernel's
    # junction_solve (used to build the labels) against a plain dense solve of the
    # reconstructed loopy operator.
    edge_index, feats, target = gen_dataset(4, 4, 6, seed=0, eps=0.05)
    phi, deg, b = feats[..., 0], feats[..., 1], feats[..., 2]
    diag = (0.05 + deg)[..., None, None]
    i_idx, j_idx = edge_index[0], edge_index[1]
    edge_val = (-(phi[..., i_idx] * phi[..., j_idx]))[..., None, None]
    A = BlockSparseSym(diag, edge_index, edge_val).to_dense()
    x_dense = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    assert (x_dense - target).abs().max() < 1e-10


def test_learned_precision_spd_and_conditioned():
    # The flagged maze-conditioning risk: the Laplacian + eps*I parameterization must keep
    # the learned precision SPD and well-conditioned for any potentials the encoder emits.
    edge_index, feats, _ = gen_dataset(5, 5, 8, seed=1, eps=0.05)
    model = GaBPMazeGrid(5, 5, eps=0.05).double()
    with torch.no_grad():
        diag, edge_val = model.build_precision(feats)
    A = BlockSparseSym(diag[..., None, None], edge_index, edge_val[..., None, None]).to_dense()
    assert float(torch.linalg.eigvalsh(A).min()) > 0       # SPD
    assert float(torch.linalg.cond(A).max()) < 1e3         # bounded kappa (eps fixes it)


def test_gabp_beats_local_ablation():
    # Headline: one junction_solve layer routes the source near-exactly on the loopy grid,
    # while 2-hop local message passing cannot reach past its receptive field.
    r = train_eval(4, 4, eps=0.05, steps=100, n_train=24, n_test=24, seed=0)
    assert r["gabp"] < 5e-2                 # near-exact global routing
    assert r["gabp"] < 0.1 * r["local"]     # wide architectural margin over the ablation
    assert r["kappa_max"] < 1e3             # conditioning stayed bounded through training


def test_matched_capacity_causal_dose_response():
    # Matched-capacity CAUSAL control: the identical GaBPMazeGrid (same architecture and
    # parameter count), intervening on ONLY the solve's reach -- the exact junction_solve vs
    # K Jacobi hops. The exact (global) solve beats the predict-mean floor while a few-hop
    # truncation cannot, and more reach helps monotonically. Nothing but the one mechanism
    # varies, so the gap is causally the inverse's globality -- not architecture or capacity.
    res = causal_solve_sweep(5, 5, steps_list=(1, 4, None), steps=150,
                             n_train=32, n_test=32, seed=0)
    assert res[None] < res["baseline"]      # only the exact global solve beats predict-mean
    assert res[1] > res["baseline"]         # a 1-hop truncation is worse than trivial
    assert res[None] < 0.05 * res[1]        # exact is orders below the 1-hop truncation
    assert res[4] < res[1]                  # more reach (4 hops vs 1) helps -- the dose-response
