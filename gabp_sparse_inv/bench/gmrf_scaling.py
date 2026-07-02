"""Scaling study for the differentiable tree-GMRF (M2 substance).

Times one marginal-likelihood forward+backward step for the ``O(n)`` tree path
(:func:`gabp_sparse_inv.gmrf.marginal_log_likelihood`) against a dense-autograd
baseline (assemble ``Q(theta)``, ``inv``, MVN log-prob, backprop -- ``O(N^3)`` time,
``O(N^2)`` memory) across tree sizes ``n``. Reports the log-log time slope of each,
the gradient agreement where both run, and the size beyond which the dense baseline is
skipped as infeasible. Also exercises the posterior-variance calibration objective
(the general selinv adjoint) at a size the dense baseline cannot reach.

    python -m gabp_sparse_inv.bench.gmrf_scaling --values 31 63 127 255 511 1023

The table in ``docs/APPLICATIONS.md`` used ``--samples 16``; the default here is 64.
Times are BLAS/thread/device dependent diagnostics -- recorded, not asserted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from ..gmrf import (
    marginal_log_likelihood,
    posterior_marginal_variances,
    sample_tree_gmrf,
    tree_gmrf_precision,
)
from ..layout import _as_parent_tensor

THETA = dict(a=0.6, kappa=1.5, root_prec=2.0, sigma2=0.3)


def balanced_binary_parent(n: int) -> list[int]:
    """Parent array of a balanced binary tree on ``n`` nodes (root 0)."""
    return [-1] + [(i - 1) // 2 for i in range(1, n)]


def _time_call(fn, *, repeats: int = 5) -> float:
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def _tree_fwdbwd(parent, y, *, batched=False):
    a = torch.tensor(THETA["a"], dtype=torch.float64, requires_grad=True)
    k = torch.tensor(THETA["kappa"], dtype=torch.float64, requires_grad=True)
    r = torch.tensor(THETA["root_prec"], dtype=torch.float64, requires_grad=True)
    s = torch.tensor(THETA["sigma2"], dtype=torch.float64, requires_grad=True)

    def step():
        for g in (a, k, r, s):
            g.grad = None
        (-marginal_log_likelihood(parent, y, a, k, r, s, batched=batched).sum()).backward()
        return torch.tensor([a.grad, k.grad, r.grad, s.grad])

    return step


def _dense_fwdbwd(parent, y):
    p = _as_parent_tensor(parent)
    n = len(parent)
    eye = torch.eye(n, dtype=torch.float64)
    const = 0.5 * n * math.log(2 * math.pi)

    def step():
        a = torch.tensor(THETA["a"], dtype=torch.float64, requires_grad=True)
        k = torch.tensor(THETA["kappa"], dtype=torch.float64, requires_grad=True)
        r = torch.tensor(THETA["root_prec"], dtype=torch.float64, requires_grad=True)
        s = torch.tensor(THETA["sigma2"], dtype=torch.float64, requires_grad=True)
        diag, edge = tree_gmrf_precision(p, a, k, r)
        N = n
        Q = diag.new_zeros((N, N))
        plist = p.tolist()
        for v in range(n):
            Q[v, v] = diag[v, 0, 0]
            pv = plist[v]
            if pv != -1:
                Q[pv, v] = edge[v, 0, 0]
                Q[v, pv] = edge[v, 0, 0]
        Sigma = torch.linalg.inv(Q) + s * eye
        L = torch.linalg.cholesky(Sigma)
        yb = y.reshape(-1, n, 1)
        sol = torch.cholesky_solve(yb, L)
        quad = (yb * sol).sum(dim=(-1, -2))
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        ll = (-0.5 * quad - 0.5 * logdet - const).sum()
        (-ll).backward()
        return torch.tensor([a.grad, k.grad, r.grad, s.grad])

    return step


def _loglog_slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    m = len(lx)
    mx, my = sum(lx) / m, sum(ly) / m
    num = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    den = sum((x - mx) ** 2 for x in lx)
    return num / den if den else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tree-GMRF scaling: tree O(n) vs dense O(N^3).")
    ap.add_argument("--values", type=int, nargs="+",
                    default=[31, 63, 127, 255, 511, 1023, 2047])
    ap.add_argument("--samples", type=int, default=64, help="iid fields per problem")
    ap.add_argument("--dense-max-n", type=int, default=511,
                    help="skip the dense O(N^3) baseline beyond this n (infeasible)")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--out", default="gmrf_scaling")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(0)

    records = []
    for n in args.values:
        parent = balanced_binary_parent(n)
        fields = torch.cat(
            [sample_tree_gmrf(parent, THETA["a"], THETA["kappa"], THETA["root_prec"], seed=s)
             for s in range(args.samples)], dim=1)
        y = (fields + math.sqrt(THETA["sigma2"]) * torch.randn(n, args.samples, dtype=torch.float64))
        y = y.T.reshape(args.samples, n, 1)

        t_tree = _time_call(_tree_fwdbwd(parent, y))
        g_tree = _tree_fwdbwd(parent, y)()
        t_tree_b = _time_call(_tree_fwdbwd(parent, y, batched=True))
        g_tree_b = _tree_fwdbwd(parent, y, batched=True)()
        batch_grad_err = float((g_tree - g_tree_b).abs().max())

        t_dense = None
        grad_err = None
        if n <= args.dense_max_n:
            t_dense = _time_call(_dense_fwdbwd(parent, y), repeats=3)
            g_dense = _dense_fwdbwd(parent, y)()
            grad_err = float((g_tree - g_dense).abs().max())

        records.append({
            "n": n, "samples": args.samples,
            "tree_fwdbwd_s": t_tree, "tree_batched_fwdbwd_s": t_tree_b,
            "dense_fwdbwd_s": t_dense,
            "grad_err_vs_dense": grad_err, "grad_err_batched_vs_loop": batch_grad_err,
            "speedup": (t_dense / t_tree) if t_dense else None,
            "batched_speedup_vs_loop": t_tree / t_tree_b,
        })
        line = (f"n={n:5d}  tree={t_tree*1e3:8.2f} ms  batched={t_tree_b*1e3:8.2f} ms"
                f"  (loop/batched={t_tree/t_tree_b:4.1f}x)")
        if t_dense:
            line += f"  dense={t_dense*1e3:9.2f} ms  speedup={t_dense/t_tree:6.1f}x  grad_err={grad_err:.1e}"
        else:
            line += "  dense=  (skipped: infeasible)"
        print(line)

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(records[0].keys())
        for r in records:
            w.writerow(r.values())

    ns = [r["n"] for r in records]
    tree_slope = _loglog_slope(ns, [r["tree_fwdbwd_s"] for r in records])
    batched_slope = _loglog_slope(ns, [r["tree_batched_fwdbwd_s"] for r in records])
    dns = [(r["n"], r["dense_fwdbwd_s"]) for r in records if r["dense_fwdbwd_s"]]
    print(f"\ntree fwd+bwd log-log slope vs n = {tree_slope:.2f} (expect ~1, linear)")
    print(f"tree (batched) fwd+bwd log-log slope vs n = {batched_slope:.2f} (expect ~1, linear)")
    if len(dns) >= 2:
        dslope = _loglog_slope([x for x, _ in dns], [t for _, t in dns])
        print(f"dense fwd+bwd log-log slope vs n = {dslope:.2f} (expect ~3, cubic)")

    # General-adjoint objective at a size the dense baseline cannot reach.
    big = max(args.values)
    parent = balanced_binary_parent(big)
    a = torch.tensor(THETA["a"], dtype=torch.float64, requires_grad=True)
    t0 = time.perf_counter()
    var = posterior_marginal_variances(parent, a, THETA["kappa"], THETA["root_prec"], THETA["sigma2"])
    ((var - 0.2) ** 2).sum().backward()
    dt = time.perf_counter() - t0
    print(f"\nposterior-variance calibration (general selinv adjoint) at n={big}: "
          f"fwd+bwd {dt*1e3:.1f} ms, d/da = {float(a.grad):.4e} (nonzero => uses G_edge)")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
