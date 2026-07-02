"""Correctness gates for the backward-stability harness (bench/stability.py).

These check the *harness* is wired correctly -- the fp64 reference, the LDL^T reassembly
from the kernel's own factors, and the error scoring -- not the height/kappa-independence
of the backward constant, which is the diagnostic the study reports rather than asserts
(derivations.md Sec 6, eq. 9a).
"""

from __future__ import annotations

import json

import torch

from gabp_sparse_inv.bench import stability


def test_fp64_factors_reproduce_A_to_machine_precision():
    # In fp64 the LDL^T reassembled from the kernel factors must reproduce A on the tree to
    # ~machine precision: confirms the reassembly + spectral-norm scoring are correct.
    rec = stability.stability_one(depth=4, b=3, seed=0, diag_load=1.0, storage="fp64")
    assert rec.backward_norm < 1e-12
    assert rec.forward_norm < 1e-10
    assert rec.kappa > 1.0


def test_fp32_records_are_finite_and_modest():
    rec = stability.stability_one(depth=5, b=2, seed=1, diag_load=1e-2, storage="fp32")
    for v in (rec.backward_norm, rec.backward_over_u, rec.forward_norm,
              rec.forward_over_kappa_u):
        assert v == v and v != float("inf")            # finite, not nan/inf
    # The backward constant ||E||/(u||A||) is a small low-degree poly in (b, Delta); a loose
    # ceiling (not the tuned constant) guards against an accidental growth-factor regression.
    assert rec.backward_over_u < 1e3


def test_backward_constant_under_loose_ceiling_across_height_and_kappa():
    # NOT the eq. (9a) height/kappa-independence assertion (that stays a reported diagnostic,
    # per this module's docstring) -- a regression guard that extends the single-point loose
    # ceiling above across a small height x kappa sweep, so an accidental growth-with-height
    # or growth-with-kappa factor in the backward constant would trip it somewhere.
    for depth in (3, 5, 6):                                 # n = 15, 63, 127
        for diag_load in (1.0, 1e-2, 1e-4):
            rec = stability.stability_one(depth=depth, b=2, seed=0, diag_load=diag_load,
                                          storage="fp32")
            assert rec.backward_over_u == rec.backward_over_u            # finite (not nan)
            assert rec.backward_over_u != float("inf")
            assert rec.backward_over_u < 1e3, (depth, diag_load, rec.backward_over_u)


def test_forward_error_tracks_kappa_times_u():
    # eq. (9): forward_norm ~ c * kappa * u, so forward_norm / (kappa u) is O(1)-ish and the
    # recorded ratio must equal the definition.
    rec = stability.stability_one(depth=4, b=2, seed=2, diag_load=1e-3, storage="fp32")
    u = float(torch.finfo(torch.float32).eps)
    assert rec.forward_over_kappa_u == _approx(rec.forward_norm / (rec.kappa * u))
    assert rec.forward_over_kappa_u < 1e2


def test_main_writes_outputs(tmp_path):
    out = tmp_path / "stab"
    rc = stability.main([
        "--depths", "3", "4", "--b", "2",
        "--diag-loads", "1.0", "0.01", "--seeds", "0", "1", "--out", str(out),
    ])
    assert rc == 0
    assert out.with_suffix(".json").exists() and out.with_suffix(".csv").exists()
    data = json.loads(out.with_suffix(".json").read_text())
    assert len(data) == 2 * 2 * 2                       # depths x diag_loads x seeds
    assert {r["depth"] for r in data} == {3, 4}


def _approx(x: float):
    import pytest
    return pytest.approx(x, rel=1e-6)
