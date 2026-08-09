#!/usr/bin/env python
"""Recompute the quantitative claims shared by the Paper 1 manuscripts.

This is a read-only audit over committed CSV/JSON records. It does not rerun
training or consume new seeds.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAPER = ROOT / "paper" / "attainability"
DATA = PAPER / "figures" / "data"
FROZEN = ROOT / "archive" / "results" / "CONFIRMATORY" / "matched_fit_taskA"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, *, rel_tol: float = 1e-12) -> None:
    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=0.0), (
        f"{actual!r} != {expected!r}"
    )


def truncated(value: float, digits: int = 0) -> float:
    scale = 10**digits
    return math.floor(value * scale) / scale


def verify_diagnostics() -> None:
    deq = {float(row["rho"]): row for row in csv_rows(DATA / "deq_robustness.csv")}
    assert set(deq) == {0.5, 0.9, 0.99, 0.999}
    exact_max = max(float(row["exact"]) for row in deq.values())
    close(exact_max, 1.073346053470098e-12)
    close(float(deq[0.99]["neumann32"]), 0.771341187894034)
    close(float(deq[0.999]["neumann32"]), 0.9746359129567539)

    maze = csv_rows(DATA / "maze_extrapolation.csv")
    assert len(maze) == 12
    assert {
        row["model"] for row in maze
    } == {"gabp", "gnn", "transformer", "baseline"}
    expected_params = {"gabp": "65", "gnn": "22337", "transformer": "18305"}
    for row in maze:
        size = int(row["test_rows"])
        assert size == int(row["test_cols"])
        assert int(row["diameter"]) == 2 * (size - 1)
        if row["model"] in expected_params:
            assert row["n_params"] == expected_params[row["model"]]
    assert 22337 // 65 == 343
    assert 18305 // 65 == 281
    by_key = {
        (row["model"], int(row["test_rows"])): float(row["mse_median"])
        for row in maze
    }
    ratios = {
        (model, size): by_key[(model, size)] / by_key[("gabp", size)]
        for model in ("gnn", "transformer")
        for size in (6, 10)
    }
    assert [math.floor(ratios[key]) for key in (
        ("gnn", 6), ("transformer", 6), ("gnn", 10), ("transformer", 10)
    )] == [212, 903, 9448, 11242]
    for size in (6, 8, 10):
        gabp_hi = float(next(
            row["mse_hi"] for row in maze
            if row["model"] == "gabp" and int(row["test_rows"]) == size
        ))
        learned_lo = min(
            float(row["mse_lo"]) for row in maze
            if row["model"] in {"gnn", "transformer"}
            and int(row["test_rows"]) == size
        )
        assert gabp_hi < learned_lo
        assert by_key[("gabp", size)] < 1e-4

    causal_rows = csv_rows(DATA / "maze_causal.csv")
    causal = {row["jacobi_sweeps_K"]: float(row["test_mse"]) for row in causal_rows}
    assert set(causal) == {"1", "2", "4", "8", "16", "32", "exact", "predict_mean"}
    finite = [causal[str(k)] for k in (1, 2, 4, 8, 16, 32)]
    assert all(left > right for left, right in zip(finite, finite[1:]))
    assert all(value > causal["predict_mean"] for value in finite)
    assert causal["exact"] < causal["predict_mean"]
    close(causal["32"] / causal["exact"], 4514.04455088631)


def verify_precision_and_scaling() -> None:
    ratios: list[float] = []
    for name in ("precision_tree.csv", "precision_grid.csv"):
        rows = csv_rows(DATA / name)
        for diag_load in (1.0, 0.01, 0.0001):
            cell = [row for row in rows if float(row["diag_load"]) == diag_load]
            assert len(cell) == 4
            ratios.append(statistics.median(float(row["advantage_chol"]) for row in cell))
    close(min(ratios), 0.34609193566595825)
    close(max(ratios), 1.954387872784074)
    assert truncated(min(ratios), 2) == 0.34
    assert truncated(max(ratios), 2) == 1.95

    timing = [
        row for row in csv_rows(ROOT / "bench_results.csv")
        if row["precision_name"] == "fp64"
    ]
    assert {int(row["b"]) for row in timing} == {8}
    cells: dict[int, list[dict[str, str]]] = {}
    for row in timing:
        cells.setdefault(int(row["L"]), []).append(row)
    crossover = []
    for size, rows in sorted(cells.items()):
        dense = [float(row["dense_chol_time_s"]) for row in rows if row["dense_chol_time_s"]]
        if dense:
            selected = statistics.median(float(row["time_median_s"]) for row in rows)
            crossover.append((size, selected < statistics.median(dense)))
    assert crossover == [(4, False), (8, False), (16, False), (32, False), (64, True)]


def verify_frozen_first_read() -> None:
    summary = load_json(FROZEN / "summary.json")
    assert summary["run_tag"] == "CONFIRMATORY"
    assert summary["seeds"] == list(range(1000, 1030))
    close(summary["match_tolerance"], 0.05)
    close(summary["m_min"], 0.60)
    close(summary["floor_abs"], 2e-6)

    deq = load_json(FROZEN / "deq_matched_fit.json")
    # These two nested tags are the disclosed provenance-label defect.
    assert deq["run_tag"] == "EXPLORATORY"
    assert deq["seeds"] == list(range(1000, 1030))
    assert deq["rhos"] == [0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99]
    assert deq["k_grid"] == [1, 2, 4, 8, 16, 32]
    rows = {(float(row["rho"]), int(row["K"])): row for row in deq["rows"]}
    expected = {
        (0.98, 1): (0, 13.234417511390973),
        (0.98, 2): (1, 5.4263659347508675),
        (0.98, 4): (7, 1.4579376558490833),
        (0.99, 1): (3, 10.86077896931651),
        (0.99, 2): (4, 7.151350591249369),
        (0.99, 4): (7, 3.881743520172767),
    }
    for key, (count, gap) in expected.items():
        assert rows[key]["n_matchable_A"] == count
        close(rows[key]["median_rel_train_gap"], gap)
    assert rows[(0.98, 32)]["n_matchable_A"] == 22
    assert rows[(0.99, 32)]["n_matchable_A"] == 19
    close(rows[(0.98, 32)]["armA_matched_fdist"]["mean"], 9.113603548447139e-4)
    close(rows[(0.99, 32)]["armA_matched_fdist"]["mean"], 1.8385704884063436e-4)
    for key in ((0.98, 32), (0.99, 32)):
        close(rows[key]["floor_F"], 2 * rows[key]["armA_matched_fdist"]["mean"])
        assert rows[key]["armA_verdict"].startswith("COLLAPSE")
    assert rows[(0.98, 32)]["armA_matched_fdist"]["ci_low"] > 2e-6

    # The absolute-scale sensitivity is direction-dependent: on the mean both
    # near-exact cells sit above 2e-6, on the median both sit below it. The
    # appendix reports both, so both are checked here.
    close(rows[(0.98, 32)]["armA_matched_fdist"]["median"], 5.465129580149062e-8)
    close(rows[(0.99, 32)]["armA_matched_fdist"]["median"], 3.2554474168122184e-8)
    for key in ((0.98, 32), (0.99, 32)):
        stats = rows[key]["armA_matched_fdist"]
        assert stats["mean"] > 2e-6 > stats["median"]
        assert stats["mean"] / stats["median"] > 5e3
    # The rho=0.99 interval straddles the absolute scale, so that cell decides
    # nothing in either direction.
    straddle = rows[(0.99, 32)]["armA_matched_fdist"]
    assert straddle["ci_low"] < 2e-6 < straddle["ci_high"]
    # Low-difficulty control cells are not skewed the same way; the tail is a
    # property of the two adjudicable near-exact cells, not of the metric.
    for k in (1, 2):
        stats = rows[(0.5, k)]["armA_matched_fdist"]
        assert stats["mean"] / stats["median"] < 5
    low = [rows[(0.5, k)]["median_rel_train_gap"] for k in (1, 2, 4)]
    assert all(value > 0.1 for value in low)
    high_fdist = max(
        rows[(rho, k)]["as_trained_fdist_mean"]
        for rho in (0.98, 0.99)
        for k in (1, 2)
    )
    low_fdist = max(
        rows[(0.5, k)]["as_trained_fdist_mean"] for k in (1, 2, 4)
    )
    assert truncated(math.log10(high_fdist / low_fdist), 1) == 4.1

    maze = load_json(FROZEN / "maze_matched_fit.json")
    assert maze["run_tag"] == "EXPLORATORY"
    assert maze["seeds"] == list(range(1000, 1030))
    assert maze["K_match"] == 7
    assert maze["k_floor"] == 256
    assert maze["sizes"] == [6, 12, 24, 48]
    assert maze["n_matchable_A"] == 0
    close(maze["proving"]["median_L_exact_train"], 1.4514363739124565e-5)
    close(maze["proving"]["median_b2_best_exact_train"], 2.9611736040451635e-2)
    close(maze["proving"]["median_rel_gap"], 2039.1676968195968)
    close(maze["floor_indistribution_gap"]["mean"], 2.7042480100640097e-5)
    assert maze["floor_sound"] is False
    assert maze["gap_shrinks_with_size"] is True
    close(maze["as_trained_gap_by_size"]["6"]["median"], 8.1760453048587e-2)
    close(maze["as_trained_gap_by_size"]["48"]["median"], 1.3026685553230526e-3)
    orders = [
        math.log10(
            abs(maze["as_trained_gap_by_size"][str(size)]["mean"])
            / abs(maze["floor_gap_by_size"][str(size)]["mean"])
        )
        for size in (12, 24, 48)
    ]
    assert truncated(min(orders), 1) == 3.8
    assert truncated(max(orders), 1) == 4.6


def verify_budget_window() -> None:
    """Recompute the reach and saturation figures the budget-window text quotes.

    One convention throughout: budgets are labelled by sweeps or Neumann terms,
    and K sweeps from a zero start spend K-1 products with A. The reach bound at
    K sweeps is therefore K-1 hops, and saturation arrives once the span holds
    A^-1 b, which happens at the grade of b and so no later than n. A fixed
    budget is informative only below both thresholds, and the sizes are
    recomputed from the same generators and FLOP accounting the probes used.
    """
    from gabp_sparse_inv.bench.matched_compute import n_match
    from gabp_sparse_inv.generators import grid_edges

    # Routed field: 6x6 grid, scalar blocks, FLOP-matched Jacobi budget.
    maze_nodes = 6 * 6
    k_match = n_match(
        maze_nodes, grid_edges(6, 6), b=1, k=1,
        method="jacobi", target="solve_adjoint",
    )
    assert k_match == 7, k_match
    assert k_match == load_json(FROZEN / "maze_matched_fit.json")["K_match"]
    assert k_match < maze_nodes                      # below saturation
    grid_diameters = {size: 2 * (size - 1) for size in (6, 12, 24, 48)}
    assert list(grid_diameters.values()) == [10, 22, 46, 94]
    products = k_match - 1                           # K sweeps spend K-1 products
    assert products == 6
    assert products < min(grid_diameters.values())   # below reach

    # DEQ training probe: 16-node chain, scalar blocks, K up to 32. The runner
    # lives under archive/, which is deliberately not importable, so the two
    # sizes are read off its frozen source the way manuscript literals are.
    runner = (ROOT / "archive" / "bench" / "matched_fit_taskA.py").read_text(
        encoding="utf-8"
    )
    assert "def run_deq(seeds, *, rhos, k_grid, n=16," in runner
    assert "edge_index = chain_edges(n)" in runner
    assert "random_coupling(n, edge_index, 1, seed=seed, rho=1.0)" in runner
    deq = load_json(FROZEN / "deq_matched_fit.json")
    deq_n = 16
    chain_diameter = deq_n - 1
    hard_ks = [1, 2, 4]
    assert set(hard_ks) <= set(deq["k_grid"])
    for k in hard_ks:
        assert k < deq_n and k - 1 < chain_diameter  # K terms spend K-1 products
        assert k <= deq_n // 4
    for k in (16, 32):
        assert k in deq["k_grid"] and k >= deq_n     # past saturation
    # On a path graph the reach and dimension thresholds coincide.
    assert chain_diameter + 1 == deq_n

    # Affine gradient sweep: 3x3 grid at block size 2.
    sweep_src = (PAPER / "make_figures.py").read_text(encoding="utf-8")
    assert "rows, cols, b = 3, 3, 2" in sweep_src
    assert "Ks = (8, 16, 32)" in sweep_src
    assert "backward_accuracy_sweep(edge_index, rows * cols, b," in sweep_src
    sweep_n = 3 * 3 * 2
    assert sweep_n == 18
    assert 32 >= sweep_n > 16 > 8                    # only K=32 is past saturation
    assert 8 - 1 > 2 * (3 - 1)                       # every K is past reach

    # The Neumann column is the geometric remainder, which is what makes it a
    # statement about the rule rather than about the budget.
    row = {float(r["rho"]): r for r in csv_rows(DATA / "deq_robustness.csv")}
    for rho in (0.99, 0.999):
        measured = float(row[rho]["neumann32"])
        assert rho**32 < measured < 1.1 * rho**32
    assert truncated(0.99**32, 3) == 0.724
    assert truncated(0.999**32, 3) == 0.968


def verify_reach_floor() -> None:
    """Recompute the reach floor and the ratios Appendix app:floor quotes.

    The floor is the exact field's energy outside the reach ball, so it lower-bounds
    the test MSE of every rule the reach proposition covers without fitting
    anything. It is recomputed here from the generator the probe itself uses, then
    checked against the committed table and against the measured dose-response.
    Two of the assertions run against the paper: the floor clears predict-the-mean
    only at the three smallest budgets, and it is vacuous at the two largest.
    """
    from gabp_sparse_inv.demos.maze_grid import reach_floor_sweep

    budgets = (1, 2, 4, 7, 8, 16, 32)
    live = reach_floor_sweep(6, 6, budgets=budgets)
    assert live["diameter"] == 10

    table = {
        int(r["jacobi_sweeps_K"]): r
        for r in csv_rows(DATA / "maze_reach_floor.csv")
        if int(r["rows"]) == 6
    }
    for k in budgets:
        assert int(table[k]["products_P"]) == k - 1   # K sweeps spend K-1 products
        close(float(table[k]["floor_mse"]), live[k])

    causal = {
        r["jacobi_sweeps_K"]: float(r["test_mse"])
        for r in csv_rows(DATA / "maze_causal.csv")
    }
    exact, mean = causal["exact"], causal["predict_mean"]

    # No covered rule attains the exact arm at the FLOP-matched budget.
    assert round(live[7] / exact) == 1460

    # At the tightest budget the trained arm sits on its own floor, so the
    # observed failure there is reach and not optimization.
    assert 1.0 < causal["1"] / live[1] < 1.002

    # The floor is weaker than the measurement past K=4, and says nothing at all
    # once the reach covers the diameter.
    for k in (1, 2, 4):
        assert live[k] > mean
    assert live[7] < mean and live[8] > 0.0
    assert live[16] == 0.0 and live[32] == 0.0


def verify_manuscript_literals() -> None:
    paths = [
        PAPER / "main.tex",
        PAPER / "tmlr" / "main.tex",
        PAPER / "dynafront" / "main.tex",
        *sorted((PAPER / "sections").glob("*.tex")),
    ]
    existing = [path for path in paths if path.is_file()]
    assert existing, "no manuscript sources found"
    text = "\n".join(path.read_text(encoding="utf-8") for path in existing)
    required = [
        "0$--$23.3\\%",
        "$343\\times$",
        "$281\\times$",
        "212\\times$--$903\\times",
        "9{,}448\\times$--$11{,}242\\times",
        "97.4\\%",
        "3.8--4.6 orders",
        "0.34$ to $1.95",
    ]
    # The low-vs-hard order separation is reported only in the shortened
    # workshop manuscript, which is intentionally absent from the anonymous
    # TMLR supplement.
    if (PAPER / "dynafront" / "main.tex").is_file():
        required.append("4.1\\) orders below")
    for fragment in required:
        assert fragment in text, f"missing manuscript fragment: {fragment}"
    for stale in (
        "97.5\\%",
        "4.2\\) orders below",
        "$344\\times$",
        "$282\\times$",
        "28{,}000\\times",
        "3.5--4.7 orders",
    ):
        assert stale not in text, f"stale manuscript fragment: {stale}"


def main() -> None:
    verify_diagnostics()
    verify_precision_and_scaling()
    verify_frozen_first_read()
    verify_budget_window()
    verify_reach_floor()
    verify_manuscript_literals()
    print("Paper 1 quantitative claims verified against committed records.")


if __name__ == "__main__":
    main()
