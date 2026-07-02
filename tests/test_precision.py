"""Correctness gates for the precision-study harness (bench/precision.py).

These check the *harness* is wired correctly -- the fp64 oracle, the on-pattern
projection of the dense baselines, and the error scoring -- not the competitive
inequality (selinv vs dense at low precision), which is a hardware-dependent diagnostic
the study reports rather than asserts.
"""

from __future__ import annotations

import json

import pytest
import torch

from gabp_sparse_inv.bench import metrics, precision
from gabp_sparse_inv.generators import grid_edges, random_spd_laplacian
from gabp_sparse_inv.junction import selected_inverse_junction


@pytest.mark.parametrize("problem", ["chain", "star", "tree"])
def test_fp64_all_methods_match_oracle(problem):
    # In fp64 the kernel and both dense baselines must reproduce the oracle on the pattern
    # to ~machine precision: confirms oracle + projection + scoring agree across methods.
    rec = precision.precision_one(problem, size=16, b=3, seed=0, diag_load=2.0, precision="fp64")
    assert rec.kappa > 1.0
    assert rec.selinv_normwise < 1e-9
    assert rec.dense_chol_normwise < 1e-9
    assert rec.dense_inv_normwise < 1e-8


@pytest.mark.parametrize("problem", ["chain", "star", "tree"])
def test_low_precision_records_are_finite(problem):
    rec = precision.precision_one(problem, size=16, b=3, seed=1, diag_load=1.0, precision="fp32")
    for v in (rec.selinv_normwise, rec.dense_chol_normwise, rec.dense_inv_normwise,
              rec.advantage_chol, rec.advantage_inv):
        assert v == v and v != float("inf")          # finite, not nan/inf
    assert rec.selinv_normwise < 1e-2                 # fp32, well-conditioned: loose sanity


def test_bf16_uses_store_low_compute_fp32_note():
    rec = precision.precision_one("chain", size=8, b=4, seed=2, diag_load=4.0, precision="bf16")
    assert "store-low" in rec.note or "native" in rec.note
    assert rec.selinv_normwise < 5e-1                 # bf16 storage is coarse; sanity only


def test_advantage_matches_error_ratio():
    rec = precision.precision_one("tree", size=24, b=2, seed=3, diag_load=1e-2, precision="fp32")
    assert rec.advantage_chol == pytest.approx(
        rec.dense_chol_normwise / max(rec.selinv_normwise, 1e-300), rel=1e-6
    )


def test_main_writes_outputs(tmp_path):
    out = tmp_path / "prec"
    rc = precision.main([
        "--problem", "chain", "--size", "12", "--b", "2",
        "--precisions", "fp64", "fp32", "--diag-loads", "1.0", "0.01",
        "--seeds", "0", "1", "--out", str(out),
    ])
    assert rc == 0
    assert out.with_suffix(".json").exists() and out.with_suffix(".csv").exists()
    data = json.loads(out.with_suffix(".json").read_text())
    assert len(data) == 2 * 2 * 2                     # precisions x diag_loads x seeds
    assert {r["precision"] for r in data} == {"fp64", "fp32"}


# --------------------------------------------------------------------------- #
# Junction / grid problem + the output-dynamic-range measure (T3 additions).
# --------------------------------------------------------------------------- #
def test_fp64_grid_matches_oracle_on_filled_pattern():
    # The junction kernel scored on its own filled pattern S must reproduce the fp64 oracle
    # to ~machine precision: confirms forward_error_junction + the projection are wired right.
    rec = precision.precision_one("grid", size=4, b=2, seed=0, diag_load=1.0, precision="fp64")
    assert rec.kappa > 1.0
    assert rec.selinv_normwise < 1e-9
    assert rec.dense_chol_normwise < 1e-8
    assert rec.dyn_range == rec.dyn_range and rec.dyn_range >= 0.0     # finite, non-negative


def test_grid_low_precision_records_finite():
    rec = precision.precision_one("grid", size=4, b=2, seed=1, diag_load=1e-2, precision="fp32")
    for v in (rec.selinv_normwise, rec.dense_chol_normwise, rec.dyn_range,
              rec.advantage_chol):
        assert v == v and v != float("inf")


def test_forward_error_junction_zero_on_oracle_itself():
    # Projecting the oracle onto S and scoring it against the same oracle must give ~0 error.
    sp = random_spd_laplacian(9, grid_edges(3, 3), 2, eps=1e-1, seed=2)
    A = sp.to_dense().to(torch.float64)
    oracle = torch.linalg.inv(A)
    oracle = 0.5 * (oracle + oracle.mT)
    G_diag, S_index, G_lower = selected_inverse_junction(sp.diag, sp.edge_index, sp.edge_val)
    ref_d, ref_l = metrics._selected_from_dense_junction(oracle, 9, S_index, 2)
    fe = metrics.forward_error_junction(ref_d, S_index, ref_l, oracle)
    assert fe.normwise < 1e-12
    # And the kernel's own fp64 blocks match the oracle to machine precision.
    fe_kernel = metrics.forward_error_junction(G_diag, S_index, G_lower, oracle)
    assert fe_kernel.normwise < 1e-9


def test_output_dynamic_range_matches_hand_computation():
    # Build a tiny 3-block (b=1) inverse with a known off/on structure.
    inv = torch.tensor([[1.0, 0.0, 5.0],
                        [0.0, 2.0, 0.0],
                        [5.0, 0.0, 1.0]], dtype=torch.float64)
    # On-pattern: diagonal only. max_on = 2 (block (1,1)); max_off = 5 (block (0,2)).
    dr = metrics.output_dynamic_range(inv, [], 3, 1)
    assert dr == pytest.approx(5.0 / 2.0, rel=1e-9)
    # Mark (0,2) on-pattern -> only block (something) off; remaining off-blocks are zero.
    dr2 = metrics.output_dynamic_range(inv, [(0, 2)], 3, 1)
    assert dr2 == pytest.approx(0.0, abs=1e-12)


def test_grid_dyn_range_grows_as_eps_shrinks():
    # A graph-Laplacian inverse is a Green's function whose correlation length grows as the
    # lift eps -> 0, so off-pattern mass rises relative to on-pattern: the measured output
    # dynamic range increases. (Measurement sanity, not an asserted precision claim.)
    lo = precision.precision_one("grid", size=5, b=2, seed=0, diag_load=1.0, precision="fp64")
    hi = precision.precision_one("grid", size=5, b=2, seed=0, diag_load=1e-3, precision="fp64")
    assert hi.dyn_range > lo.dyn_range


# --------------------------------------------------------------------------- #
# Elimination-order sensitivity study (the committed "is the edge an ordering
# artifact?" experiment). Wiring only -- the ratios are heavy-tailed diagnostics.
# --------------------------------------------------------------------------- #
def test_order_sensitivity_fp64_ratios_bounded():
    # In fp64 every method is ~exact (the natural-order dense baseline IS the oracle, so its
    # error is ~0 and nat/ker -> 0). The wiring check is that the LEAF-first permutation
    # round-trip does not corrupt anything: a broken unpermute would leave an O(1) error and
    # blow the ratio up to ~1e14. Bounded ratios confirm permute -> factor -> unpermute is right.
    recs = precision.tree_order_sensitivity(
        size=16, b=2, diag_load=1.0, precision="fp64", seeds=[0, 1])
    assert len(recs) == 2
    for _kappa, ratios in recs:
        for m in precision._ORDER_METRICS:
            nat, leaf = ratios[m]
            assert 0.0 <= nat < 10.0 and 0.0 <= leaf < 10.0


def test_order_sensitivity_fp32_records_finite():
    # fp32 records must be finite and positive; the test does NOT assert the inequality (it is a
    # heavy-tailed diagnostic -- read the median+spread the CLI prints, never one ratio).
    recs = precision.tree_order_sensitivity(
        size=32, b=2, diag_load=1e-3, precision="fp32", seeds=[0, 1, 2])
    for kappa, ratios in recs:
        assert kappa > 1.0
        for m in precision._ORDER_METRICS:
            nat, leaf = ratios[m]
            assert nat == nat and nat != float("inf") and nat > 0.0
            assert leaf == leaf and leaf != float("inf") and leaf > 0.0


def test_compare_orders_cli_runs():
    rc = precision.main([
        "--compare-orders", "--size", "16", "--b", "2",
        "--precisions", "fp32", "--diag-loads", "1e-2", "--seeds", "0", "1",
    ])
    assert rc == 0
