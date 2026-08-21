#!/usr/bin/env python
"""Recompute the quantitative claims shared by the Paper 1 manuscripts.

This is a read-only audit over committed CSV/JSON records. It does not rerun
training or consume new seeds.
"""

from __future__ import annotations

import csv
import hashlib
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
GMRES_RAW = (ROOT / "archive" / "results" / "EXPLORATORY"
             / "deq_gmres_falsifier")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    One convention throughout: K zero-start Jacobi sweeps charge K sparse
    products, but the first is applied to zero and the returned polynomial has
    degree K-1. The source-Krylov reach bound is informative only
    for a localized source; saturation arrives no later than dimension n. The
    sizes are recomputed from the same generators and FLOP accounting the probes
    used.
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
    charged_products = k_match
    propagating_products = k_match - 1
    assert charged_products == 7 and propagating_products == 6
    assert charged_products < min(grid_diameters.values())

    compute_rows = {int(row["side"]): row for row in csv_rows(DATA / "maze_compute_budget.csv")}
    assert [int(compute_rows[size]["forward_match_K"]) for size in (6, 12, 24, 48)] == [6, 14, 35, 88]
    assert [int(compute_rows[size]["training_match_K"]) for size in (6, 12, 24, 48)] == [7, 17, 47, 126]

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
    hard_ks = [1, 2, 4]
    assert set(hard_ks) <= set(deq["k_grid"])
    for k in hard_ks:
        assert k < deq_n                              # below dimensional saturation
        assert k <= deq_n // 4
    # K denotes Neumann terms, hence K-1 products. Span saturation occurs after
    # d-1 powers, while a black-box residual-optimal solve is guaranteed by d
    # products. K=16 is therefore below, and only K=32 exceeds, the n=16 ceiling.
    assert 16 in deq["k_grid"] and 16 - 1 < deq_n
    assert 32 in deq["k_grid"] and 32 - 1 >= deq_n
    # Dense training inputs make the MSE adjoint source generally dense, so the
    # routed-field reach argument is not used for this probe.
    assert "b_tr = input_scale * torch.randn(n_train, n, 1" in runner

    # Affine gradient sweep: 3x3 grid at block size 2.
    sweep_src = (PAPER / "make_figures.py").read_text(encoding="utf-8")
    assert "rows, cols, b = 3, 3, 2" in sweep_src
    assert "Ks = (8, 16, 32)" in sweep_src
    assert "backward_accuracy_sweep(edge_index, rows * cols, b," in sweep_src
    sweep_n = 3 * 3 * 2
    assert sweep_n == 18
    assert 32 - 1 >= sweep_n > 16 - 1 > 8 - 1       # only K=32 is past query saturation

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
        assert int(table[k]["propagating_products_P"]) == k - 1
        close(float(table[k]["floor_mse"]), live[k])

    causal = {
        r["jacobi_sweeps_K"]: float(r["test_mse"])
        for r in csv_rows(DATA / "maze_causal.csv")
    }
    exact, mean = causal["exact"], causal["predict_mean"]

    # Literal K=7 Jacobi has six propagating powers. A generic source-generated
    # competitor granted all seven charged products receives the more generous
    # P=7 floor, which still rules out the exact test error.
    assert round(live[7] / exact) == 1460
    assert round(live[8] / exact) == 578

    training_rows = list(csv_rows(DATA / "maze_training_support_floor.csv"))
    assert len(training_rows) == 30
    ratios_p6 = []
    ratios_p7 = []
    for row in training_rows:
        exact_train = float(row["exact_train_mse"])
        assert float(row["support_floor_P6"]) > exact_train
        assert float(row["support_floor_P7"]) > exact_train
        ratios_p6.append(float(row["ratio_P6_to_exact"]))
        ratios_p7.append(float(row["ratio_P7_to_exact"]))
    assert truncated(min(ratios_p6), 1) == 63.3
    assert truncated(min(ratios_p7), 1) == 24.6

    # At the tightest budget the trained arm sits on its own floor, so the
    # observed failure there is reach and not optimization.
    assert 1.0 < causal["1"] / live[1] < 1.002

    # The floor is weaker than the measurement past K=4, and says nothing at all
    # once the reach covers the diameter.
    for k in (1, 2, 4):
        assert live[k] > mean
    assert live[7] < mean and live[8] > 0.0
    assert live[16] == 0.0 and live[32] == 0.0


def verify_deq_gmres() -> None:
    """Audit the post-hoc GMRES result without upgrading it to confirmation."""
    rows = {
        (float(row["rho"]), int(row["products"])): row
        for row in csv_rows(DATA / "deq_gmres_falsifier.csv")
    }
    assert len(rows) == 18
    assert set(rows) == {
        (rho, products)
        for rho in (0.5, 0.98, 0.99)
        for products in (1, 2, 4, 8, 16, 32)
    }

    audit = load_json(DATA / "deq_gmres_falsifier_audit.json")
    assert audit["campaign_uuid"] == "d3d1654b-3f65-4272-8f16-a58e027668d3"
    assert audit["source_commit"] == "70536f06c63cbf293e99747dbf5186086abeaa2b"
    assert audit["protocol_sha256"] == (
        "63e626e179bb8e3650f70dc7c6b5033ced6597fe450446401dd22a8221c63aaf"
    )
    assert audit["record_count"] == 20 and audit["cell_count"] == 60
    assert audit["config"]["n"] == 256
    assert audit["config"]["products"] == [1, 2, 4, 8, 16, 32]
    assert audit["independent_summary_matches_runner"] is True
    assert audit["frozen_predictions"] == {
        "mechanism": False,
        "training": False,
        "control": True,
    }
    assert audit["initial_diagnostic_is_hard_spectrum"] is False
    close(audit["initial_diagnostic_gain_min"], 0.2)
    close(audit["initial_diagnostic_gain_max"], 0.2)
    assert audit["structured_exact_vs_dense_relative_error_max"] < 2.68e-16
    assert audit["minimum_relative_arnoldi_norm"] > 1.0
    assert set(audit["raw_sha256"]) == {
        *(f"seed{seed}.json" for seed in range(4000, 4020)),
        "summary.json",
    }
    for name, digest in audit["raw_sha256"].items():
        assert sha256(GMRES_RAW / name) == digest
    assert audit["raw_sha256"]["summary.json"] == (
        "1b1355f57b8973f08df12f8ceb072a640211e9e4a66676cf7413f84750e6ce77"
    )

    expected_p8 = {
        0.98: (0.25, 0.90, 1.3720221287815375, 1.0011094393206976,
               3.775037051012564e-06),
        0.99: (0.20, 0.75, 1.8310668389541163, 0.9963976427661598,
               3.973460470541035e-06),
    }
    for rho, expected in expected_p8.items():
        row = rows[(rho, 8)]
        actual = (
            float(row["neumann_matchable_fraction"]),
            float(row["gmres_matchable_fraction"]),
            float(row["median_neumann_test_loss_ratio_to_exact"]),
            float(row["median_gmres_test_loss_ratio_to_exact"]),
            float(row["median_abs_gmres_gain_minus_exact"]),
        )
        for got, want in zip(actual, expected):
            close(got, want)
        assert row["classification"] == "GMRES_OPENS_COMPARABLE_FIT"
        runtime_ratio = (
            float(row["median_gmres_training_seconds"])
            / float(row["median_neumann_training_seconds"])
        )
        assert 1.10 < runtime_ratio < 1.16

    for rho in (0.98, 0.99):
        for products in (1, 2, 4):
            assert rows[(rho, products)]["classification"] == (
                "JOINT_NONATTAINMENT_UNRESOLVED"
            )
        for products in (8, 16, 32):
            assert rows[(rho, products)]["classification"] == (
                "GMRES_OPENS_COMPARABLE_FIT"
            )
    close(float(rows[(0.98, 32)]["median_neumann_test_loss_ratio_to_exact"]),
          0.2617243493995922)
    designated = [
        float(rows[(rho, products)]["median_log10_train_loss_gmres_over_neumann"])
        for rho in (0.98, 0.99)
        for products in (16, 32)
    ]
    close(min(designated), -0.7393039483566021)
    assert min(designated) > -1.0


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
        "$578$ times",
        "$24.6$",
        "$18/20$ and $15/20$",
        "$5/20$ and $4/20$",
        "$1.001$ and $0.996$",
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
        "spends $d-1$ products",
        "residual-optimal rules remain unevaluated",
        "they were not evaluated here",
    ):
        assert stale not in text, f"stale manuscript fragment: {stale}"


def main() -> None:
    verify_diagnostics()
    verify_precision_and_scaling()
    verify_frozen_first_read()
    verify_budget_window()
    verify_reach_floor()
    verify_deq_gmres()
    verify_manuscript_literals()
    print("Paper 1 quantitative claims verified against committed records.")


if __name__ == "__main__":
    main()
