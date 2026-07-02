"""Gates for the matched-capacity maze baselines (`demos/maze_baselines.py`, Track B/E4).

These assert the **machine-independent** facts of the benchmark harness -- not the
hardware/seed-dependent MSE numbers (those are diagnostics from `run_benchmark`). The tests check
that the fair baselines are over-parameterised relative to the tiny GaBP encoder (so a
surviving gap cannot be a capacity artifact), that the size-independent weights transfer
across grid sizes (the extrapolation mechanism works mechanically), and that the exact
solve still beats the predict-mean baseline on an *extrapolated* (larger) grid.
"""

from __future__ import annotations

import math

import torch

from gabp_sparse_inv import grid_edges
from gabp_sparse_inv.demos.maze_baselines import (
    build_models,
    count_params,
    evaluate_extrapolation,
    extrapolation_curve,
    gen_resistance_dataset,
)
from gabp_sparse_inv.demos.maze_grid import grid_laplacian_from_potentials

torch.manual_seed(0)


def test_baselines_over_parameterised_vs_gabp():
    # Capacity matching is the easy direction: the GaBP "model" is just an 81-param encoder
    # plus the parameter-free solve, so any fair baseline dwarfs it. Gate that so a measured
    # gap is never explainable as "the baseline was starved of capacity".
    models = build_models(6, 6)
    p = {k: count_params(m) for k, m in models.items()}
    assert p["gnn"] >= p["gabp"]
    assert p["transformer"] >= p["gabp"]


def test_extrapolation_harness_runs_and_transfers():
    # Tiny config: train at 4x4, evaluate at 4x4 and the larger 5x5 (weights must transfer).
    mse, params = evaluate_extrapolation(
        train_size=(4, 4),
        test_sizes=((4, 4), (5, 5)),
        n_train=16,
        n_test=16,
        steps=80,
        seed=0,
        hidden=16,
        gnn_depth=6,
    )
    names = ("gabp", "gnn", "transformer", "baseline")
    # Every (model, size) cell exists and is finite -- in particular the baselines produced
    # finite outputs at 5x5 from 4x4-trained weights, so the cross-size transfer is sound.
    for size in ((4, 4), (5, 5)):
        for name in names:
            assert math.isfinite(mse[(name, size)])

    # The exact solve routes, and -- the whole point of the extrapolation axis -- still beats
    # predict-the-mean on the *larger* grid it was never trained on (size-independent encoder
    # + exact solve), a robust seed-independent inequality.
    assert mse[("gabp", (4, 4))] < mse[("baseline", (4, 4))]
    assert mse[("gabp", (5, 5))] < mse[("baseline", (5, 5))]


# --------------------------------------------------------------------------- #
# Held-out task: effective resistance R(v,s) -- a non-linear functional of A^{-1}.
# --------------------------------------------------------------------------- #
def test_resistance_label_matches_dense_oracle():
    # The held-out label must be the exact effective resistance, so the demonstration rests on a
    # correct target. Compare gen_resistance_dataset to R(v,s) = inv_vv - 2 inv_vs + inv_ss from a
    # dense fp64 inverse, on each example's own precision.
    rows, cols, num = 3, 3, 4
    ei, feats, R = gen_resistance_dataset(rows, cols, num, seed=1, eps=0.05)
    n = rows * cols
    phi, b = feats[..., 0], feats[..., 2]
    diag, edge_val, _w = grid_laplacian_from_potentials(phi, 0.05, ei[0], ei[1])
    for ex in range(num):
        A = torch.zeros(n, n, dtype=torch.float64)
        A[torch.arange(n), torch.arange(n)] = diag[ex]
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            A[i, j] = A[j, i] = edge_val[ex, k]
        inv = torch.linalg.inv(A)
        s = int(b[ex].argmax())
        ref = torch.tensor([inv[v, v] - 2 * inv[v, s] + inv[s, s] for v in range(n)], dtype=torch.float64)
        assert torch.allclose(R[ex], ref, atol=1e-10)
        assert float(R[ex, s].abs()) < 1e-10                      # resistance to self is 0
        assert float(R[ex].min()) > -1e-10                        # effective resistance >= 0


def test_resistance_gabp_beats_baselines_and_extrapolates():
    # The GaBP model (which *contains* selinv_junction + junction_solve) computes the resistance
    # functional far better than the over-parameterised universal learners, and the gap holds on
    # a larger grid it was never trained on -- broadening the claim past reconstructing A^{-1} e_s.
    mse, params = evaluate_extrapolation(
        train_size=(4, 4), test_sizes=((4, 4), (5, 5)),
        n_train=16, n_test=16, steps=80, seed=0, hidden=16, gnn_depth=6, task="resistance",
    )
    assert params["gnn"] >= params["gabp"] and params["transformer"] >= params["gabp"]
    for size in ((4, 4), (5, 5)):
        assert mse[("gabp", size)] < mse[("baseline", size)]
        assert mse[("gabp", size)] < mse[("gnn", size)]
        assert mse[("gabp", size)] < mse[("transformer", size)]


def test_extrapolation_curve_multiseed_structure_and_separation():
    # The multi-seed curve aggregates median + (min, max) per (model, size); gate the structure
    # and the seed-robust separation (GaBP median below the predict-mean baseline at every size).
    curve, _params = extrapolation_curve(
        train_size=(4, 4), test_sizes=((4, 4), (5, 5)), seeds=(0, 1),
        n_train=12, n_test=12, steps=50, hidden=16, gnn_depth=5, task="route",
    )
    for size in ((4, 4), (5, 5)):
        for name in ("gabp", "gnn", "transformer", "baseline"):
            med, lo, hi = curve[(name, size)]
            assert lo <= med <= hi
        assert curve[("gabp", size)][0] < curve[("baseline", size)][0]
