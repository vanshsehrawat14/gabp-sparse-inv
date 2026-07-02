#!/usr/bin/env python
"""Reproducible figure-data pipeline for the paper.

Each paper figure/table is backed either by one of the project's benchmark CLIs
(``gabp_sparse_inv.bench.*``) or by a demonstration's data function
(``gabp_sparse_inv.demos.*``). This script centralizes the exact invocations the paper
needs and writes their CSV/JSON into ``paper/figures/data/`` so every figure regenerates
from a single command. We emit *data*, not rendered PDFs: the LaTeX consumes the CSVs via
``pgfplotstable`` (see paper/README.md). There is no plotting dependency to verify here.

Usage:
    python paper/make_figures.py                 # all figures, modest configs
    python paper/make_figures.py --quick         # smaller/faster configs (smoke)
    python paper/make_figures.py --only precision_tree,deq_robustness
    python paper/make_figures.py --list          # list figure ids

Everything is a CPU diagnostic (BLAS/device-dependent), matching the benches' own stance.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import time
from pathlib import Path

from gabp_sparse_inv.bench import (
    gmrf_scaling,
    nonsym_stability,
    precision,
    run as run_bench,
    stability,
)

# --- bench-CLI figures: id -> (bench main, argv-builder(out_prefix, quick), description). The
# builder returns the argv (incl. --out) passed to the bench's main(); the bench writes the CSV.
def _timing_tree(out: str, quick: bool) -> list[str]:
    vals = ["31", "63", "127"] if quick else ["31", "63", "127", "255", "511"]
    return ["--problem", "tree", "--sweep", "n", "--values", *vals,
            "--b", "2", "--seeds", "0", "1", "--out", out]


def _gmrf_scaling(out: str, quick: bool) -> list[str]:
    vals = ["127", "255"] if quick else ["127", "255", "511"]
    samples = "8" if quick else "16"
    return ["--values", *vals, "--samples", samples, "--out", out]


def _precision_tree(out: str, quick: bool) -> list[str]:
    seeds = ["0", "1"] if quick else ["0", "1", "2", "3"]
    return ["--problem", "tree", "--size", "127", "--b", "2", "--precisions", "fp32",
            "--diag-loads", "1.0", "0.01", "0.0001", "--seeds", *seeds, "--out", out]


def _precision_grid(out: str, quick: bool) -> list[str]:
    side = "6" if quick else "8"
    seeds = ["0", "1"] if quick else ["0", "1", "2", "3"]
    return ["--problem", "grid", "--size", side, "--b", "2", "--precisions", "fp32",
            "--diag-loads", "1.0", "0.01", "0.0001", "--seeds", *seeds, "--out", out]


def _precision_compare_orders(out: str, quick: bool) -> list[str]:
    # The honest ordering experiment: most of the apparent tree "edge" is elimination order.
    seeds = ["0", "1", "2", "3"] if quick else [str(s) for s in range(16)]
    return ["--problem", "tree", "--size", "127", "--b", "2", "--precisions", "fp32",
            "--compare-orders", "--seeds", *seeds, "--out", out]


def _stability(out: str, quick: bool) -> list[str]:
    depths = ["3", "4", "5"] if quick else ["3", "4", "5", "6"]
    return ["--depths", *depths, "--b", "2", "--diag-loads", "1.0", "0.01", "0.0001",
            "--seeds", "0", "1", "--out", out]


def _nonsym_stability(out: str, quick: bool) -> list[str]:
    seeds = ["0", "1"] if quick else ["0", "1", "2", "3", "4"]
    return ["--rows", "4", "--cols", "4", "--b", "2", "--seeds", *seeds, "--out", out]


WIRED = {
    "timing_tree": (run_bench.main, _timing_tree,
                    "Fwd+bwd time vs n on a tree; dense crossover (bench/run.py)."),
    "gmrf_scaling": (gmrf_scaling.main, _gmrf_scaling,
                     "Differentiable GMRF: tree O(n) vs dense O(N^3) (bench/gmrf_scaling.py)."),
    "precision_tree": (precision.main, _precision_tree,
                       "Selinv vs dense at equal precision, tree, across kappa."),
    "precision_grid": (precision.main, _precision_grid,
                       "Selinv vs dense at equal precision, loopy grid (junction), across kappa."),
    "precision_compare_orders": (precision.main, _precision_compare_orders,
                                 "Ordering controls the apparent tree precision 'edge'."),
    "stability": (stability.main, _stability,
                  "Backward constant flat across height/kappa; forward ~ kappa*u."),
    "nonsym_stability": (nonsym_stability.main, _nonsym_stability,
                         "No-pivot selinv vs pivoted LU across block-diagonal dominance."),
}

# These bench modes print a console diagnostic and return 0 without writing a CSV; we capture
# their stdout to a .txt artifact instead of expecting <out>.csv.
CONSOLE_ONLY = {"precision_compare_orders"}


# --- demo-callable figures: id -> (producer(out_prefix, quick) -> rc, description). The
# producer calls a demos.* data function and writes the CSV itself.
def _demo_maze_extrapolation(out: str, quick: bool) -> int:
    from gabp_sparse_inv.demos import maze_baselines
    test_sizes = ((6, 6), (8, 8)) if quick else ((6, 6), (8, 8), (10, 10))
    seeds = (0,) if quick else (0, 1, 2)
    kw = dict(n_train=16, n_test=16, steps=60) if quick else {}
    curve, params = maze_baselines.extrapolation_curve(
        train_size=(6, 6), test_sizes=test_sizes, seeds=seeds, task="route", **kw)
    names = ("gabp", "gnn", "transformer", "baseline")
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_rows", "test_cols", "diameter",
                    "mse_median", "mse_lo", "mse_hi", "n_params"])
        for (tr, tc) in test_sizes:
            diam = (tr - 1) + (tc - 1)
            for name in names:
                med, lo, hi = curve[(name, (tr, tc))]
                w.writerow([name, tr, tc, diam, med, lo, hi, params.get(name, "")])
    return 0


def _demo_maze_causal(out: str, quick: bool) -> int:
    from gabp_sparse_inv.demos.maze_grid import causal_solve_sweep
    steps_list = (1, 4, 16, None) if quick else (1, 2, 4, 8, 16, 32, None)
    sw = dict(steps=120, n_train=32, n_test=32) if quick else dict(steps=250, n_train=64, n_test=64)
    res = causal_solve_sweep(6, 6, steps_list=steps_list, **sw)
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solve_reach_K", "test_mse"])
        for K in steps_list:
            w.writerow(["exact" if K is None else K, res[K]])
        w.writerow(["predict_mean", res["baseline"]])
    return 0


def _demo_deq_robustness(out: str, quick: bool) -> int:
    from gabp_sparse_inv.demos import deq_fixedpoint
    from gabp_sparse_inv.generators import grid_edges
    rows, cols, b = 3, 3, 2
    rhos = (0.5, 0.9, 0.99) if quick else (0.5, 0.9, 0.99, 0.999)
    Ks = (8, 16, 32)
    edge_index = grid_edges(rows, cols)
    res = deq_fixedpoint.backward_accuracy_sweep(edge_index, rows * cols, b, rhos=rhos, Ks=Ks)
    kcols = [f"neumann{k}" for k in Ks]
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "exact", *kcols])
        for rho, row in res.items():
            w.writerow([rho, row["exact"], *[row[k] for k in kcols]])
    return 0


DEMO = {
    "maze_extrapolation": (_demo_maze_extrapolation,
                           "Maze extrapolation: gabp vs matched-capacity GNN/Transformer/baseline "
                           "vs grid size/diameter (demos/maze_baselines.py)."),
    "maze_causal": (_demo_maze_causal,
                    "Maze matched-capacity causal dose-response: identical model, test MSE vs "
                    "solve reach K (exact vs K Jacobi hops) (demos/maze_grid.py)."),
    "deq_robustness": (_demo_deq_robustness,
                       "Fixed-point backward: exact vs Neumann-K gradient error as rho(J)->1 "
                       "(demos/deq_fixedpoint.py)."),
}


def _run_one(name: str, out_prefix: str, quick: bool) -> tuple[int, bool, str]:
    """Run one figure; return (rc, output_exists, ext)."""
    if name in WIRED:
        bench_main, build_argv, _ = WIRED[name]
        argv_i = build_argv(out_prefix, quick)
        if name in CONSOLE_ONLY:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = bench_main(argv_i)
            Path(out_prefix + ".txt").write_text(buf.getvalue())
            return rc, Path(out_prefix + ".txt").exists(), ".txt"
        rc = bench_main(argv_i)
        return rc, Path(out_prefix + ".csv").exists(), ".csv"
    producer, _ = DEMO[name]
    rc = producer(out_prefix, quick)
    return rc, Path(out_prefix + ".csv").exists(), ".csv"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "figures" / "data"),
                    help="where the CSV/JSON land")
    ap.add_argument("--only", default=None,
                    help="comma-separated figure ids (default: all)")
    ap.add_argument("--quick", action="store_true", help="smaller/faster configs (smoke)")
    ap.add_argument("--list", action="store_true", help="list figure ids and exit")
    args = ap.parse_args(argv)

    all_ids = list(WIRED) + list(DEMO)
    if args.list:
        print("Bench-CLI figures:")
        for name, (_, _, desc) in WIRED.items():
            print(f"  {name:28s} {desc}")
        print("Demo figures:")
        for name, (_, desc) in DEMO.items():
            print(f"  {name:28s} {desc}")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = all_ids if args.only is None else [s.strip() for s in args.only.split(",")]
    unknown = [s for s in selected if s not in WIRED and s not in DEMO]
    if unknown:
        print(f"unknown figure id(s): {unknown}; valid: {all_ids}", file=sys.stderr)
        return 2

    failures = []
    for name in selected:
        out_prefix = str(out_dir / name)
        t0 = time.perf_counter()
        try:
            rc, ok, ext = _run_one(name, out_prefix, args.quick)
        except Exception as exc:  # keep going; report at the end
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(name)
            continue
        dt = time.perf_counter() - t0
        if rc == 0 and ok:
            print(f"[ok]   {name:28s} {dt:6.1f}s -> {name}{ext}")
        else:
            print(f"[FAIL] {name}: rc={rc} {ext}_exists={ok}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} figure(s) failed: {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(selected)} figure(s) ok -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
