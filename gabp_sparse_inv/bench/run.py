"""CLI: sweep chain length, star leaves, or block size; emit CSV/JSON diagnostics.

Examples
--------
    python -m gabp_sparse_inv.bench.run --problem chain --sweep L --b 8
    python -m gabp_sparse_inv.bench.run --problem star --sweep K --b 8
    python -m gabp_sparse_inv.bench.run --problem chain --sweep b --L 64
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .harness import (
    DEFAULT_PRECISIONS,
    bench_grad_tree,
    bench_one,
    bench_one_star,
    bench_one_tree,
)

DEFAULT_L_SWEEP = [4, 8, 16, 32, 64, 128, 256]
DEFAULT_B_SWEEP = [1, 2, 4, 8, 16, 32]


def _loglog_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log(y) vs log(x)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys if y > 0]
    if len(ly) != len(lx) or len(lx) < 2:
        return float("nan")
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


def _crossover(records: list[dict], baseline_key: str, size_label: str = "L") -> str:
    """Smallest primary size at which the structured time beats the dense baseline."""
    pts = [(r["L"], r["time_median_s"], r[baseline_key]) for r in records
           if r.get(baseline_key) is not None]
    pts.sort()
    for size, t_struct, t_dense in pts:
        if t_struct < t_dense:
            return f"{size_label}={size}"
    return "crossover not observed within tested range"


def _run_grad(args) -> int:
    """Differentiable forward+backward sweep over tree size ``n`` (diagnostics)."""
    sweep_vals = args.values if args.values is not None else DEFAULT_L_SWEEP
    records: list[dict] = []
    for n in sweep_vals:
        for seed in args.seeds:
            rec = bench_grad_tree(
                n=n, b=args.b, seed=seed, diag_load=args.diag_load, kind=args.tree_kind,
                device=args.device, num_threads=args.num_threads,
                dense_baseline_max_n=args.dense_max_L, time_min_run_s=args.min_run_s,
            )
            records.append(asdict(rec))

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(records, indent=2))
    keys = [k for k in records[0] if not isinstance(records[0][k], dict)]
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in records:
            w.writerow([r[k] for k in keys])

    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.csv')}")
    print(f"problem=tree-grad device={args.device} threads={args.num_threads} "
          f"kind={args.tree_kind} b={args.b}\n")
    by_n: dict[int, list[dict]] = {}
    for r in records:
        by_n.setdefault(r["n"], []).append(r)
    xs = sorted(by_n)

    def _median(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    ys_loop = [_median([x["grad_time_loop_s"] for x in by_n[v]]) for v in xs]
    ys_batch = [_median([x["grad_time_batched_s"] for x in by_n[v]]) for v in xs]
    print(f"fwd+bwd log-log time slope vs n: loop={_loglog_slope([float(x) for x in xs], ys_loop):.3f} "
          f"batched={_loglog_slope([float(x) for x in xs], ys_batch):.3f} (expect ~1 = linear)")
    worst_adj = max((r["adjoint_err_vs_dense"] for r in records
                     if r["adjoint_err_vs_dense"] is not None), default=float("nan"))
    print(f"worst gradient error vs dense autograd = {worst_adj:.3e}")
    big = max(records, key=lambda r: r["n"])
    mem = big["grad_mem"]
    print(f"backward memory @ n={big['n']}: structured {mem['structured_grad_elems']} elems "
          f"vs dense-autograd {mem['dense_autograd_elems']} elems "
          f"({mem['ratio_dense_over_structured']:.1f}x)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Selected-inverse benchmark (chain / star / tree).")
    p.add_argument("--problem", choices=["chain", "star", "tree"], default="chain",
                   help="chain: sweep chain length L; star: sweep leaves K; tree: sweep nodes n")
    p.add_argument("--sweep", choices=["L", "K", "n", "b"], default="L",
                   help="size axis to sweep: primary size (L/K/n) or b")
    p.add_argument("--L", type=int, default=64, help="fixed primary size (L/K/n) when sweeping b")
    p.add_argument("--b", type=int, default=8, help="fixed b when sweeping the primary size")
    p.add_argument("--values", type=int, nargs="+", default=None, help="override sweep values")
    p.add_argument("--precisions", nargs="+", default=list(DEFAULT_PRECISIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--diag-load", type=float, default=1.0)
    p.add_argument("--tree-kind", default="random",
                   help="tree topology family (random/path/star/balanced) for --problem tree")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--dense-max-L", type=int, default=64)
    p.add_argument("--min-run-s", type=float, default=0.1)
    p.add_argument("--out", default="bench_results", help="output path prefix")
    p.add_argument("--grad", action="store_true",
                   help="benchmark differentiable forward+backward (tree only): grad-time "
                        "scaling, loop-vs-batched, gradient correctness, memory diagnostic")
    args = p.parse_args(argv)
    if args.sweep in ("K", "n"):   # star/tree-friendly aliases for the primary-size axis
        args.sweep = "L"

    if args.grad:
        return _run_grad(args)

    if args.values is not None:
        sweep_vals = args.values
    else:
        sweep_vals = DEFAULT_L_SWEEP if args.sweep == "L" else DEFAULT_B_SWEEP

    records: list[dict] = []
    for prec in args.precisions:
        for val in sweep_vals:
            size = val if args.sweep == "L" else args.L   # primary size: L (chain) or K (star)
            b = val if args.sweep == "b" else args.b
            for seed in args.seeds:
                if args.problem == "chain":
                    rec = bench_one(
                        L=size, b=b, seed=seed, precision=prec, diag_load=args.diag_load,
                        device=args.device, num_threads=args.num_threads,
                        dense_baseline_max_L=args.dense_max_L, time_min_run_s=args.min_run_s,
                    )
                elif args.problem == "star":
                    rec = bench_one_star(
                        K=size, b=b, seed=seed, precision=prec, diag_load=args.diag_load,
                        device=args.device, num_threads=args.num_threads,
                        dense_baseline_max_K=args.dense_max_L, time_min_run_s=args.min_run_s,
                    )
                else:
                    rec = bench_one_tree(
                        n=size, b=b, seed=seed, precision=prec, diag_load=args.diag_load,
                        kind=args.tree_kind, device=args.device, num_threads=args.num_threads,
                        dense_baseline_max_n=args.dense_max_L, time_min_run_s=args.min_run_s,
                    )
                records.append(asdict(rec))

    out = Path(args.out)
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    json_path.write_text(json.dumps(records, indent=2))

    if records:
        keys = [k for k in records[0] if not isinstance(records[0][k], dict)]
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["precision_name"])
            for r in records:
                w.writerow([r[k] for k in keys] + [r["precision"]["name"]])

    # Report log-log slope + crossover per precision (median over seeds at each value).
    # Diagnostics only -- the slope/crossover depend on BLAS, device, threads, b, K.
    _primary = {"chain": "L", "star": "K", "tree": "n"}[args.problem]
    size_label = _primary if args.sweep == "L" else "b"
    record_key = "L" if args.sweep == "L" else "b"   # star stores K in the L field
    print(f"\nwrote {json_path} and {csv_path}")
    print(f"problem={args.problem} device={args.device} threads={args.num_threads} "
          f"sweep={size_label}\n")
    for prec in args.precisions:
        sub = [r for r in records if r["precision"]["name"] == prec]
        by_val: dict[int, list[dict]] = {}
        for r in sub:
            by_val.setdefault(r[record_key], []).append(r)
        xs = sorted(by_val)
        ys = [sorted(x["time_median_s"] for x in by_val[v])[len(by_val[v]) // 2] for v in xs]
        slope = _loglog_slope([float(x) for x in xs], ys)
        expected = 1.0 if args.sweep == "L" else 3.0
        print(f"[{prec}] log-log time slope vs {size_label} = {slope:.3f} (expect ~{expected:.0f})")
        if args.sweep == "L":
            print(f"        crossover vs dense inv : {_crossover(sub, 'dense_inv_time_s', size_label)}")
            print(f"        crossover vs dense chol: {_crossover(sub, 'dense_chol_time_s', size_label)}")
        worst = max(r["forward_normwise"] for r in sub)
        print(f"        worst forward normwise error = {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
