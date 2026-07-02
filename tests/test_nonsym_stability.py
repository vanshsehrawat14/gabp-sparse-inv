"""Gates for the no-pivot stability-boundary study (``bench/nonsym_stability.py``).

The non-symmetric kernel eliminates with **no pivoting**, so the static-pattern factorization
is only safe while block-diagonally dominant (``derivations.md`` §10.4). These gates check the
harness wiring and the **robust, monotone** qualitative boundary: parity with pivoted LU when
dominant, a collapsing Schur-pivot floor and a growing no-pivot error when dominance is lost;
not the noisy per-seed magnitudes (those are diagnostics).
"""

from __future__ import annotations

import math

import torch

from gabp_sparse_inv.bench.nonsym_stability import (
    nonsym_stability_one,
    random_nonsym_dominant,
)

torch.manual_seed(0)
SEEDS = [0, 1, 2, 3, 4]


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def _sweep(alpha):
    return [nonsym_stability_one(alpha=alpha, rows=4, cols=4, b=2, seed=s) for s in SEEDS]


def test_record_fields_well_formed():
    rec = nonsym_stability_one(alpha=2.0, rows=3, cols=3, b=2, seed=0)
    assert rec.n == 9 and rec.b == 2
    assert rec.kappa > 1.0 and rec.pivot_floor > 0.0
    assert math.isfinite(rec.err_nopivot) and math.isfinite(rec.err_pivoted)
    assert rec.dom_ratio > 0.0


def test_dominant_regime_is_parity_with_pivoted_lu():
    # Strongly block-diagonally dominant: the no-pivot kernel matches pivoted dense LU (penalty
    # ~ 1) and is accurate in fp32 (tracks kappa*u, here ~1e-7).
    recs = _sweep(3.0)
    assert _median([r.penalty for r in recs]) < 5.0
    assert _median([r.err_nopivot for r in recs]) < 1e-5


def test_losing_dominance_collapses_pivot_floor_and_grows_error():
    # The structural, monotone facts of the boundary (robust to seed noise):
    dom = _sweep(3.0)            # strongly dominant
    weak = _sweep(0.2)          # not dominant
    # 1) the Schur-pivot floor collapses as dominance is lost.
    assert _median([r.pivot_floor for r in dom]) > 3.0 * _median([r.pivot_floor for r in weak])
    # 2) the no-pivot selected-inverse error grows.
    assert _median([r.err_nopivot for r in weak]) > _median([r.err_nopivot for r in dom])
    # 3) and it does so *beyond* what pivoting would suffer (the no-pivot penalty rises).
    assert _median([r.penalty for r in weak]) > _median([r.penalty for r in dom])


def test_measured_dominance_tracks_alpha():
    # The dominance knob is monotone: a larger alpha gives a larger measured dominance ratio.
    assert _median([r.dom_ratio for r in _sweep(3.0)]) > _median([r.dom_ratio for r in _sweep(0.5)])


def test_generator_dominance_structure():
    # At large alpha each diagonal block dominates its off-diagonal row sum; at small alpha it
    # need not -- the generator's core knob.
    diag, ei, lower, upper = random_nonsym_dominant(9, torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 0, 3, 4, 5, 6, 7]], dtype=torch.long), 2, alpha=4.0, seed=1)
    n = diag.shape[0]
    R = torch.zeros(n, dtype=torch.float64)
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        R[i] += torch.linalg.matrix_norm(lower[k], ord=2)
        R[j] += torch.linalg.matrix_norm(upper[k], ord=2)
    for v in range(n):
        if R[v] > 0:
            assert float(torch.linalg.svdvals(diag[v])[-1]) > float(R[v])   # dominant at alpha=4
