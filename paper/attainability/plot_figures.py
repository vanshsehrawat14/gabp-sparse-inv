#!/usr/bin/env python
"""Render the paper's figures (PDF) from the CSVs in figures/data/.

Data is produced by make_figures.py (which drives the tested bench/demo CLIs);
this script only draws. Style: Computer Modern mathtext to match the paper,
Okabe-Ito colorblind-safe palette, distinct markers/linestyles so every series
survives grayscale print.

    python paper/plot_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "figures" / "data"
OUT = HERE / "figures"

# Okabe-Ito (colorblind-safe). Ours/exact is always black.
BLACK = "#000000"
ORANGE = "#E69F00"
BLUE = "#56B4E9"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GRAY = "#7F7F7F"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.4,
        # Opaque legend box so data lines never strike through legend text.
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def rows(name: str) -> list[dict]:
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def maze_extrapolation() -> None:
    data = rows("maze_extrapolation.csv")
    sizes = sorted({int(r["test_rows"]) for r in data})
    series = [
        ("gabp", "gabp (exact solve)", BLACK, "o", "-"),
        ("gnn", "GNN", ORANGE, "s", "--"),
        ("transformer", "Transformer", BLUE, "^", ":"),
    ]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for key, label, color, marker, ls in series:
        med = [float(r["mse_median"]) for r in data if r["model"] == key]
        lo = [float(r["mse_lo"]) for r in data if r["model"] == key]
        hi = [float(r["mse_hi"]) for r in data if r["model"] == key]
        ax.plot(sizes, med, marker=marker, ls=ls, color=color, label=label)
        ax.fill_between(sizes, lo, hi, color=color, alpha=0.15, lw=0)
    floor = [float(r["mse_median"]) for r in data if r["model"] == "baseline"]
    ax.plot(sizes, floor, ls="--", color=GRAY, label="predict-the-mean floor")
    ax.set_yscale("log")
    ax.set_xticks(sizes, [f"${s}\\times{s}$" for s in sizes])
    ax.set_xlabel("test grid size (trained at $6\\times6$)")
    ax.set_ylabel("test MSE")
    ax.legend(loc="center right")
    fig.savefig(OUT / "maze_extrapolation.pdf")
    plt.close(fig)


def maze_causal() -> None:
    data = rows("maze_causal.csv")
    ks = [int(r["solve_reach_K"]) for r in data if r["solve_reach_K"].isdigit()]
    mses = [float(r["test_mse"]) for r in data if r["solve_reach_K"].isdigit()]
    exact = next(float(r["test_mse"]) for r in data if r["solve_reach_K"] == "exact")
    floor = next(
        (float(r["test_mse"]) for r in data if r["solve_reach_K"] == "predict_mean"),
        None,
    )
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(ks, mses, marker="s", ls="-", color=VERMILLION, label="$K$ Jacobi sweeps")
    xr = max(ks) * 2
    ax.plot([xr], [exact], marker="*", ms=11, ls="none", color=BLACK, label="exact solve")
    if floor is not None:
        ax.axhline(floor, ls="--", color=GRAY, lw=1.0, label="predict-the-mean floor")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks + [xr], [str(k) for k in ks] + ["exact"])
    ax.set_xlabel("solve reach $K$ (sweeps; one hop per sweep)")
    ax.set_ylabel("test MSE")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "maze_causal.pdf")
    plt.close(fig)


def deq_robustness() -> None:
    data = rows("deq_robustness.csv")
    rho = [float(r["rho"]) for r in data]
    x = [1.0 - v for v in rho]
    series = [
        ("exact", "sparse direct", BLACK, "o", "-"),
        ("neumann8", "Neumann-8", ORANGE, "s", "--"),
        ("neumann16", "Neumann-16", BLUE, "^", ":"),
        ("neumann32", "Neumann-32", GREEN, "D", "-."),
    ]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for key, label, color, marker, ls in series:
        ax.plot(x, [float(r[key]) for r in data], marker=marker, ls=ls, color=color, label=label)
    ax.set_xscale("log")
    ax.invert_xaxis()  # stiffness (rho -> 1) increases to the right
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{v:g}" for v in rho])
    ax.set_xlabel("spectral radius $\\rho(J)$")
    ax.set_ylabel("parameter-gradient relative error")
    ax.legend(loc="center right")
    fig.savefig(OUT / "deq_robustness.pdf")
    plt.close(fig)


def scaling() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.7))

    with open(HERE.parent.parent / "bench_results.csv", newline="") as f:
        t = list(csv.DictReader(f))
    ns: dict[int, dict[str, list[float]]] = {}
    for r in t:
        n = int(r["L"])
        d = ns.setdefault(n, {"selinv": [], "dense": []})
        d["selinv"].append(float(r["time_median_s"]))
        if r["dense_chol_time_s"]:
            d["dense"].append(float(r["dense_chol_time_s"]))
    xs = sorted(ns)
    med = lambda v: sorted(v)[len(v) // 2]
    ax1.plot(xs, [med(ns[n]["selinv"]) for n in xs], marker="o", color=BLACK,
             label="selected inverse (fwd+bwd)")
    xd = [n for n in xs if ns[n]["dense"]]
    ax1.plot(xd, [med(ns[n]["dense"]) for n in xd], marker="s", ls="--",
             color=ORANGE, label="dense Cholesky")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("chain length $n$")
    ax1.set_ylabel("time (s)")
    ax1.set_title("chain selected inverse ($b=8$)")
    ax1.set_ylim(top=0.4)  # headroom so the legend sits clear of the data
    ax1.legend(loc="upper left")

    g = rows("gmrf_scaling.csv")
    n2 = [int(r["n"]) for r in g]
    ax2.plot(n2, [float(r["tree_fwdbwd_s"]) for r in g], marker="o", color=BLACK,
             label="selinv (per-node loop)")
    ax2.plot(n2, [float(r["tree_batched_fwdbwd_s"]) for r in g], marker="^", ls=":",
             color=GREEN, label="selinv (batched)")
    ax2.plot(n2, [float(r["dense_fwdbwd_s"]) for r in g], marker="s", ls="--",
             color=ORANGE, label="dense autograd")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("GMRF nodes $n$")
    ax2.set_ylabel("time per step (s)")
    ax2.set_title("differentiable GMRF learning")
    ax2.set_ylim(top=4.0)  # headroom so the legend sits clear of the data
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "scaling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    maze_extrapolation()
    maze_causal()
    deq_robustness()
    scaling()
    print("wrote", *[p.name for p in sorted(OUT.glob("*.pdf"))])
