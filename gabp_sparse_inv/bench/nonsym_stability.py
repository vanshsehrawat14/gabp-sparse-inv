"""No-pivot stability boundary for the general non-symmetric (LU) selected inverse.

The non-symmetric junction kernel (``selected_inverse_nonsym_junction``, ``derivations.md`` §10)
eliminates in a **fixed symbolic order with no pivoting** - that is what keeps the pattern
static, but its work still follows the squared front sizes. It requires every
Schur-complement pivot ``D_v`` to stay
non-singular. §10.4 names the safe regime: block diagonal dominance / the well-scaled input the
project targets. This harness measures **where that regime ends**.

This harness builds a non-symmetric block matrix on a graph with a tunable block-diagonal-**dominance**
``alpha`` (``sigma_min(A_vv) ≈ alpha * (off-diagonal row sum)``; the Feingold-Varga boundary is
``alpha = 1``), and sweeps it from strongly dominant down through ``alpha < 1``. Per setting it
record, against a dense fp64 oracle:

  * ``pivot_floor`` ``= min_v sigma_min(D_v) / ||A||`` from the kernel's own factorization - the
    static-pattern Schur floor. It collapses toward 0 as the order hits a near-singular pivot.
  * ``err_nopivot`` - the fp32 no-pivot kernel's selected-block error.
  * ``err_pivoted`` - the fp32 **dense LU with partial pivoting** (``torch.linalg.inv``) on the
    same blocks, the stability baseline pivoting buys.
  * ``penalty = err_nopivot / err_pivoted`` - the price of *not* pivoting, at equal precision.

The honest reading (§10.4): while block-diagonally dominant (``alpha >~ 1``) the no-pivot kernel
is at **parity** with pivoted LU and tracks ``kappa * u``; as dominance is lost the Schur floor
collapses and ``err_nopivot`` departs from ``kappa * u`` while pivoted LU does not - the boundary
of the static-pattern regime. Pivoting would fix it but makes the symbolic pattern
data-dependent, so it is deliberately out of scope (``docs/ROADMAP.md``). Like
the rest of ``bench/`` these are dtype/BLAS-dependent **diagnostics** - recorded, not asserted.

    python -m gabp_sparse_inv.bench.nonsym_stability
    python -m gabp_sparse_inv.bench.nonsym_stability --alphas 2 1.5 1 0.7 0.5 0.3 --rows 4 --cols 4
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..generators import grid_edges
from ..nonsym_junction import _factor_nonsym, selected_inverse_nonsym_junction

__all__ = ["NonsymStabilityRecord", "random_nonsym_dominant", "nonsym_stability_one", "main"]


@dataclass
class NonsymStabilityRecord:
    alpha: float                  # the dominance knob (target sigma_min(A_vv) / off-row-sum)
    dom_ratio: float              # measured min_v sigma_min(A_vv) / off-row-sum_v
    n: int
    b: int
    seed: int
    kappa: float
    pivot_floor: float            # min_v sigma_min(D_v) / ||A||  (static-pattern Schur floor)
    err_nopivot: float            # fp32 no-pivot kernel selected-block rel error vs fp64 oracle
    err_pivoted: float            # fp32 dense LU (partial pivoting) rel error vs fp64 oracle
    penalty: float                # err_nopivot / err_pivoted


def random_nonsym_dominant(n, edges, b, *, alpha, seed, dtype=torch.float64):
    """Non-symmetric block matrix on ``edges`` with block-diagonal-dominance ``alpha``.

    Off-diagonal blocks (both orientations, independent) are ``N(0, 1)``. For node ``v`` with
    off-diagonal row sum ``R_v = sum_j ||A_vj||_2`` the diagonal block is
    ``A_vv = alpha * R_v * I + 0.2 * R_v * N`` (``N`` random) - so ``sigma_min(A_vv) ≈ alpha R_v``
    with a non-normal perturbation. ``alpha > 1`` is strictly block-diagonally dominant (the
    no-pivot-safe regime); ``alpha < 1`` is not. Returns ``(diag, edge_index, lower, upper)``.
    """
    g = torch.Generator().manual_seed(seed)
    pairs = sorted({(max(int(edges[0, k]), int(edges[1, k])), min(int(edges[0, k]), int(edges[1, k])))
                    for k in range(edges.shape[1])})
    ei = torch.tensor(list(zip(*pairs)), dtype=torch.long)
    m = ei.shape[1]
    eye = torch.eye(b, dtype=dtype)
    lower = torch.randn(m, b, b, generator=g, dtype=dtype)
    upper = torch.randn(m, b, b, generator=g, dtype=dtype)
    # Row sum R_v of off-diagonal block 2-norms (node v sees A_{v,j} for every incident edge).
    R = torch.zeros(n, dtype=dtype)
    for k in range(m):
        i, j = int(ei[0, k]), int(ei[1, k])
        R[i] = R[i] + torch.linalg.matrix_norm(lower[k], ord=2)        # A_{i,j}
        R[j] = R[j] + torch.linalg.matrix_norm(upper[k], ord=2)        # A_{j,i}
    R = R.clamp_min(1e-6)
    diag = (alpha * R)[:, None, None] * eye + 0.2 * R[:, None, None] * torch.randn(n, b, b, generator=g, dtype=dtype)
    return diag, ei, lower, upper


def _assemble(diag, ei, lower, upper):
    n, b = diag.shape[-3], diag.shape[-1]
    N = n * b
    A = diag.new_zeros((N, N))
    for v in range(n):
        A[v * b:(v + 1) * b, v * b:(v + 1) * b] = diag[v]
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        A[i * b:(i + 1) * b, j * b:(j + 1) * b] = lower[k]
        A[j * b:(j + 1) * b, i * b:(i + 1) * b] = upper[k]
    return A


def _selected_err(Gd, Si, Gl, Gu, inv64, b):
    """Relative (normwise) error of selected blocks vs a dense fp64 inverse."""
    num_sq = torch.zeros((), dtype=torch.float64)
    den_sq = torch.zeros((), dtype=torch.float64)
    Gd = Gd.to(torch.float64)
    for v in range(Gd.shape[-3]):
        ref = inv64[v * b:(v + 1) * b, v * b:(v + 1) * b]
        num_sq = num_sq + (Gd[v] - ref).pow(2).sum()
        den_sq = den_sq + ref.pow(2).sum()
    Gl = Gl.to(torch.float64); Gu = Gu.to(torch.float64)
    for k in range(Si.shape[1]):
        i, j = int(Si[0, k]), int(Si[1, k])
        rl = inv64[i * b:(i + 1) * b, j * b:(j + 1) * b]
        ru = inv64[j * b:(j + 1) * b, i * b:(i + 1) * b]
        num_sq = num_sq + (Gl[k] - rl).pow(2).sum() + (Gu[k] - ru).pow(2).sum()
        den_sq = den_sq + rl.pow(2).sum() + ru.pow(2).sum()
    return float((num_sq.sqrt() / den_sq.sqrt().clamp_min(1e-30)))


def _dense_pivoted_err(A32, inv64, Si, b):
    """fp32 dense LU (partial pivoting) selected-block error vs the fp64 oracle."""
    try:
        inv32 = torch.linalg.inv(A32).to(torch.float64)
    except RuntimeError:
        return float("inf")
    num_sq = torch.zeros((), dtype=torch.float64)
    den_sq = torch.zeros((), dtype=torch.float64)
    n = inv64.shape[-1] // b
    for v in range(n):
        r = slice(v * b, (v + 1) * b)
        num_sq = num_sq + (inv32[r, r] - inv64[r, r]).pow(2).sum()
        den_sq = den_sq + inv64[r, r].pow(2).sum()
    for k in range(Si.shape[1]):
        i, j = int(Si[0, k]), int(Si[1, k])
        ri, rj = slice(i * b, (i + 1) * b), slice(j * b, (j + 1) * b)
        for (a, c) in ((ri, rj), (rj, ri)):
            num_sq = num_sq + (inv32[a, c] - inv64[a, c]).pow(2).sum()
            den_sq = den_sq + inv64[a, c].pow(2).sum()
    return float((num_sq.sqrt() / den_sq.sqrt().clamp_min(1e-30)))


def nonsym_stability_one(*, alpha, rows, cols, b, seed, storage="fp32"):
    """One no-pivot stability record on a loopy grid at a given dominance ``alpha``."""
    n = rows * cols
    dtype = torch.float32 if storage == "fp32" else torch.float64
    u = float(torch.finfo(dtype).eps)
    diag, ei, lower, upper = random_nonsym_dominant(n, grid_edges(rows, cols), b, alpha=alpha, seed=seed)

    A = _assemble(diag, ei, lower, upper)
    normA = float(torch.linalg.matrix_norm(A, ord=2))
    sv = torch.linalg.svdvals(A)
    kappa = float(sv[0] / sv[-1].clamp_min(1e-30))
    inv64 = torch.linalg.inv(A)

    # Measured dominance ratio: min_v sigma_min(A_vv) / off-row-sum_v.
    Rrow = torch.zeros(n, dtype=torch.float64)
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        Rrow[i] += float(torch.linalg.matrix_norm(lower[k], ord=2))
        Rrow[j] += float(torch.linalg.matrix_norm(upper[k], ord=2))
    dom = min(float(torch.linalg.svdvals(diag[v])[-1]) / max(float(Rrow[v]), 1e-30) for v in range(n))

    # Static-pattern Schur floor: min_v sigma_min(D_v) = 1 / max_v ||D_v^{-1}||_2  (fp64 factor).
    try:
        _o, _p, _U, _S, Dinv, _Lf, _Uf = _factor_nonsym(diag, ei, lower, upper, None)
        max_dinv = max(float(torch.linalg.matrix_norm(Dinv[v], ord=2)) for v in range(n))
        pivot_floor = (1.0 / max_dinv) / max(normA, 1e-30)
    except RuntimeError:
        pivot_floor = 0.0

    # fp32 no-pivot kernel.
    try:
        Gd, Si, Gl, Gu = selected_inverse_nonsym_junction(diag.to(dtype), ei, lower.to(dtype), upper.to(dtype))
        err_nopivot = _selected_err(Gd, Si, Gl, Gu, inv64, b)
        if not (err_nopivot == err_nopivot and err_nopivot < float("inf")):
            err_nopivot = float("inf")
    except RuntimeError:
        # exact-singular Schur pivot in the static order: the no-pivot factorization fails.
        Si = selected_inverse_nonsym_junction(diag, ei, lower, upper)[1]
        err_nopivot = float("inf")

    err_pivoted = _dense_pivoted_err(A.to(dtype), inv64, Si, b)
    penalty = err_nopivot / err_pivoted if err_pivoted > 0 else float("inf")

    return NonsymStabilityRecord(
        alpha=alpha, dom_ratio=dom, n=n, b=b, seed=seed, kappa=kappa, pivot_floor=pivot_floor,
        err_nopivot=err_nopivot, err_pivoted=err_pivoted, penalty=penalty,
    )


DEFAULT_ALPHAS = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1]


def _median(xs):
    finite = sorted(x for x in xs if x == x and x < float("inf"))
    if not finite:
        return float("inf")
    return finite[len(finite) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="No-pivot stability boundary for the non-symmetric selected inverse "
                    "(derivations.md §10.4): where block diagonal dominance stops protecting the "
                    "static-pattern factorization.")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                    help="block-diagonal-dominance knob (sigma_min(A_vv)/off-row-sum); 1 = boundary")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--storage", default="fp32", choices=["fp32", "fp64"])
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--out", default="nonsym_stability_results", help="output path prefix")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.num_threads)

    records = []
    for alpha in args.alphas:
        for seed in args.seeds:
            records.append(asdict(nonsym_stability_one(
                alpha=alpha, rows=args.rows, cols=args.cols, b=args.b, seed=seed, storage=args.storage)))

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(records[0].keys())
        for r in records:
            w.writerow(r.values())
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.csv')}")
    print(f"grid {args.rows}x{args.cols}, b={args.b}, storage={args.storage}  "
          f"(median over {len(args.seeds)} seeds; diagnostics, not asserted)")
    print("Sec 10.4: no-pivot is at parity with pivoted LU while block-diagonally dominant "
          "(alpha >~ 1); below the boundary the Schur floor collapses and err_nopivot diverges.\n")
    print(f"{'alpha':>6} {'dom_ratio':>10} {'kappa':>10} {'pivot_floor':>12} "
          f"{'err_nopivot':>12} {'err_pivoted':>12} {'penalty':>9}")
    by_alpha = {}
    for r in records:
        by_alpha.setdefault(r["alpha"], []).append(r)
    for alpha in sorted(by_alpha, reverse=True):
        rs = by_alpha[alpha]
        print(f"{alpha:>6.2f} {_median([r['dom_ratio'] for r in rs]):>10.2f} "
              f"{_median([r['kappa'] for r in rs]):>10.1e} {_median([r['pivot_floor'] for r in rs]):>12.1e} "
              f"{_median([r['err_nopivot'] for r in rs]):>12.1e} {_median([r['err_pivoted'] for r in rs]):>12.1e} "
              f"{_median([r['penalty'] for r in rs]):>9.1f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
