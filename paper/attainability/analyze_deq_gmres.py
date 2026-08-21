#!/usr/bin/env python
"""Independently audit and tabulate the frozen DEQ GMRES falsifier records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


SEEDS = tuple(range(4000, 4020))
RHOS = (0.5, 0.98, 0.99)
PRODUCTS = (1, 2, 4, 8, 16, 32)
CAMPAIGN_UUID = "d3d1654b-3f65-4272-8f16-a58e027668d3"
MATCHABILITY_BAR = 0.60
MATCH_TOL = 0.05


def median(values) -> float:
    return float(statistics.median(float(value) for value in values))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(neumann_fraction: float, gmres_fraction: float) -> str:
    if gmres_fraction >= MATCHABILITY_BAR and neumann_fraction < MATCHABILITY_BAR:
        return "GMRES_OPENS_COMPARABLE_FIT"
    if gmres_fraction < MATCHABILITY_BAR and neumann_fraction < MATCHABILITY_BAR:
        return "JOINT_NONATTAINMENT_UNRESOLVED"
    if gmres_fraction >= MATCHABILITY_BAR and neumann_fraction >= MATCHABILITY_BAR:
        return "BOTH_ATTAIN"
    return "GMRES_WORSE_OR_UNSTABLE"


def close(actual: float, expected: float) -> None:
    assert math.isclose(float(actual), float(expected), rel_tol=1e-13, abs_tol=1e-15)


def recompute_matchability(losses: list[float], exact_final: float) -> dict:
    denominator = max(float(exact_final), sys.float_info.min)
    gaps = [abs(float(value) - exact_final) / denominator for value in losses]
    in_band = [index for index, gap in enumerate(gaps) if gap <= MATCH_TOL]
    return {
        "matched": bool(in_band),
        "first_step": in_band[0] if in_band else None,
        "best_relative_gap": min(gaps),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", default="archive/results/EXPLORATORY/deq_gmres_falsifier"
    )
    parser.add_argument(
        "--out-csv",
        default="paper/attainability/figures/data/deq_gmres_falsifier.csv",
    )
    parser.add_argument(
        "--out-audit",
        default="paper/attainability/figures/data/deq_gmres_falsifier_audit.json",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    seed_paths = [raw_dir / f"seed{seed}.json" for seed in SEEDS]
    missing = [str(path) for path in seed_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"campaign incomplete; missing {missing}")
    summary_path = raw_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("campaign incomplete; summary.json is absent")

    records = [json.loads(path.read_text(encoding="utf-8")) for path in seed_paths]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    envelope = records[0]
    expected_config = {
        "seeds": list(SEEDS),
        "rhos": list(RHOS),
        "products": list(PRODUCTS),
        "n": 256,
        "steps": 120,
        "lr": 0.05,
        "n_train": 32,
        "n_test": 32,
        "input_scale": 0.2,
    }
    for record, seed in zip(records, SEEDS):
        assert record["run_tag"] == "EXPLORATORY"
        assert record["campaign_uuid"] == CAMPAIGN_UUID
        assert record["seed"] == seed
        assert record["config"] == expected_config
        assert record["source_commit"] == envelope["source_commit"]
        assert record["protocol_sha256"] == envelope["protocol_sha256"]
        assert [cell["rho"] for cell in record["cells"]] == list(RHOS)
        assert math.isfinite(float(record["seed"]))

    assert summary["run_tag"] == "EXPLORATORY"
    assert summary["status"] == "post-hoc adversarial study; not confirmatory"
    assert summary["campaign_uuid"] == CAMPAIGN_UUID
    assert summary["source_commit"] == envelope["source_commit"]
    assert summary["protocol_sha256"] == envelope["protocol_sha256"]
    assert summary["config"] == expected_config
    assert set(summary["predictions_frozen_before_run"]) == {
        "mechanism", "training", "control"
    }
    assert float(summary["seconds"]) > 0

    cells = {
        (record["seed"], float(cell["rho"])): cell
        for record in records
        for cell in record["cells"]
    }
    rows = []
    dense_errors = []
    arnoldi_norms = []
    initial_gains = []
    for rho in RHOS:
        for products in PRODUCTS:
            key = str(products)
            selected = [cells[(seed, rho)] for seed in SEEDS]
            for cell in selected:
                initial_gain = 0.999 / (1.0 + math.exp(-float(cell["raw_init"])))
                close(initial_gain, 0.2)
                initial_gains.append(initial_gain)
                assert set(map(int, cell["arms"])) == set(PRODUCTS)
                exact = cell["exact"]
                exact_training = exact["training"]
                assert exact_training["mode"] == "exact"
                assert len(exact_training["trajectory_raw"]) == 121
                assert len(exact_training["trajectory_exact_forward_loss"]) == 121
                close(
                    exact_training["trajectory_exact_forward_loss"][-1],
                    exact["final_train_loss"],
                )
                close(exact_training["trajectory_raw"][-1], exact_training["raw"])
                arm = cell["arms"][key]
                diagnostic = arm["adjoint_at_initialization"]
                assert diagnostic["products"] == products
                assert diagnostic["neumann_internal"] == {
                    "products": products,
                    "terms": products + 1,
                }
                assert diagnostic["gmres_internal"]["products"] == products
                for mode in ("neumann", "gmres"):
                    assert math.isfinite(diagnostic[mode]["relative_adjoint_error"])
                    assert math.isfinite(diagnostic[mode]["relative_residual"])
                    assert diagnostic[mode]["relative_adjoint_error"] >= 0
                    assert diagnostic[mode]["relative_residual"] >= 0
                arnoldi_norms.append(
                    diagnostic["gmres_internal"]["min_relative_arnoldi_norm"]
                )
                if products == max(PRODUCTS):
                    dense_errors.append(
                        diagnostic[
                            "structured_exact_vs_dense_relative_error_first_rhs"
                        ]
                    )
                for mode in ("neumann", "gmres"):
                    finite = arm[mode]
                    trained = finite["training"]
                    assert trained["mode"] == mode
                    assert trained["products"] == products
                    assert len(trained["trajectory_raw"]) == 121
                    assert len(trained["trajectory_exact_forward_loss"]) == 121
                    assert trained["seconds"] > 0
                    assert math.isfinite(finite["final_train_loss_exact_forward"])
                    assert math.isfinite(finite["final_test_loss_exact_forward"])
                    assert finite["final_train_loss_exact_forward"] >= 0
                    assert finite["final_test_loss_exact_forward"] >= 0
                    close(trained["trajectory_raw"][-1], trained["raw"])
                    close(
                        trained["trajectory_exact_forward_loss"][-1],
                        finite["final_train_loss_exact_forward"],
                    )
                    expected_match = recompute_matchability(
                        trained["trajectory_exact_forward_loss"],
                        exact["final_train_loss"],
                    )
                    assert finite["matchability"]["matched"] == expected_match["matched"]
                    assert finite["matchability"]["first_step"] == expected_match["first_step"]
                    close(
                        finite["matchability"]["best_relative_gap"],
                        expected_match["best_relative_gap"],
                    )
                    close(
                        finite["train_loss_ratio_to_exact"],
                        finite["final_train_loss_exact_forward"]
                        / exact["final_train_loss"],
                    )
                close(
                    arm["paired_log10_train_loss_gmres_over_neumann"],
                    math.log10(
                        arm["gmres"]["final_train_loss_exact_forward"]
                        / arm["neumann"]["final_train_loss_exact_forward"]
                    ),
                )

            neumann = [cell["arms"][key]["neumann"] for cell in selected]
            gmres = [cell["arms"][key]["gmres"] for cell in selected]
            exact_train = [cell["exact"]["final_train_loss"] for cell in selected]
            exact_test = [cell["exact"]["final_test_loss"] for cell in selected]
            exact_gain = [cell["exact"]["training"]["gain"] for cell in selected]
            neumann_fraction = sum(
                arm["matchability"]["matched"] for arm in neumann
            ) / len(neumann)
            gmres_fraction = sum(
                arm["matchability"]["matched"] for arm in gmres
            ) / len(gmres)
            residual_half_count = sum(
                cell["arms"][key]["adjoint_at_initialization"]["gmres"]
                ["relative_residual"]
                <= 0.5
                * cell["arms"][key]["adjoint_at_initialization"]["neumann"]
                ["relative_residual"]
                for cell in selected
            )
            rows.append(
                {
                    "rho": rho,
                    "products": products,
                    "n": len(selected),
                    "neumann_matchable_fraction": neumann_fraction,
                    "gmres_matchable_fraction": gmres_fraction,
                    "median_log10_train_loss_gmres_over_neumann": median(
                        cell["arms"][key][
                            "paired_log10_train_loss_gmres_over_neumann"
                        ]
                        for cell in selected
                    ),
                    "median_neumann_train_loss_ratio_to_exact": median(
                        arm["train_loss_ratio_to_exact"] for arm in neumann
                    ),
                    "median_gmres_train_loss_ratio_to_exact": median(
                        arm["train_loss_ratio_to_exact"] for arm in gmres
                    ),
                    "median_neumann_test_loss_ratio_to_exact": median(
                        arm["final_test_loss_exact_forward"] / reference
                        for arm, reference in zip(neumann, exact_test)
                    ),
                    "median_gmres_test_loss_ratio_to_exact": median(
                        arm["final_test_loss_exact_forward"] / reference
                        for arm, reference in zip(gmres, exact_test)
                    ),
                    "median_abs_neumann_gain_minus_exact": median(
                        abs(arm["training"]["gain"] - reference)
                        for arm, reference in zip(neumann, exact_gain)
                    ),
                    "median_abs_gmres_gain_minus_exact": median(
                        abs(arm["training"]["gain"] - reference)
                        for arm, reference in zip(gmres, exact_gain)
                    ),
                    "n_neumann_final_train_at_most_exact": sum(
                        arm["final_train_loss_exact_forward"] <= reference
                        for arm, reference in zip(neumann, exact_train)
                    ),
                    "n_gmres_final_train_at_most_exact": sum(
                        arm["final_train_loss_exact_forward"] <= reference
                        for arm, reference in zip(gmres, exact_train)
                    ),
                    "median_neumann_initial_relative_adjoint_error": median(
                        cell["arms"][key]["adjoint_at_initialization"]["neumann"]
                        ["relative_adjoint_error"]
                        for cell in selected
                    ),
                    "median_gmres_initial_relative_adjoint_error": median(
                        cell["arms"][key]["adjoint_at_initialization"]["gmres"]
                        ["relative_adjoint_error"]
                        for cell in selected
                    ),
                    "median_neumann_initial_relative_residual": median(
                        cell["arms"][key]["adjoint_at_initialization"]["neumann"]
                        ["relative_residual"]
                        for cell in selected
                    ),
                    "median_gmres_initial_relative_residual": median(
                        cell["arms"][key]["adjoint_at_initialization"]["gmres"]
                        ["relative_residual"]
                        for cell in selected
                    ),
                    "n_gmres_residual_at_most_half_neumann": residual_half_count,
                    "median_neumann_training_seconds": median(
                        arm["training"]["seconds"] for arm in neumann
                    ),
                    "median_gmres_training_seconds": median(
                        arm["training"]["seconds"] for arm in gmres
                    ),
                    "classification": classify(neumann_fraction, gmres_fraction),
                }
            )

    produced = {(row["rho"], row["products"]): row for row in rows}
    frozen = {
        (float(row["rho"]), int(row["products"])): row for row in summary["rows"]
    }
    assert produced.keys() == frozen.keys()
    shared_fields = (
        "neumann_matchable_fraction",
        "gmres_matchable_fraction",
        "median_log10_train_loss_gmres_over_neumann",
        "median_neumann_train_loss_ratio_to_exact",
        "median_gmres_train_loss_ratio_to_exact",
        "median_neumann_initial_relative_adjoint_error",
        "median_gmres_initial_relative_adjoint_error",
        "median_neumann_initial_relative_residual",
        "median_gmres_initial_relative_residual",
    )
    for key, row in produced.items():
        assert row["classification"] == frozen[key]["classification"]
        for field in shared_fields:
            assert math.isclose(
                row[field], frozen[key][field], rel_tol=1e-13, abs_tol=1e-15
            )

    mechanism = all(
        produced[(rho, 32)]["n_gmres_residual_at_most_half_neumann"] >= 15
        for rho in (0.98, 0.99)
    )
    training = any(
        produced[(rho, products)][
            "median_log10_train_loss_gmres_over_neumann"
        ]
        <= -1.0
        for rho in (0.98, 0.99)
        for products in (16, 32)
    )
    control = all(
        produced[(0.5, 32)][field] < 1e-4
        for field in (
            "median_neumann_initial_relative_adjoint_error",
            "median_gmres_initial_relative_adjoint_error",
        )
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "campaign_uuid": CAMPAIGN_UUID,
        "source_commit": envelope["source_commit"],
        "protocol_sha256": envelope["protocol_sha256"],
        "record_count": len(records),
        "cell_count": len(cells),
        "config": expected_config,
        "independent_summary_matches_runner": True,
        "structured_exact_vs_dense_relative_error_max": max(dense_errors),
        "minimum_relative_arnoldi_norm": min(arnoldi_norms),
        "initial_diagnostic_gain_min": min(initial_gains),
        "initial_diagnostic_gain_max": max(initial_gains),
        "initial_diagnostic_is_hard_spectrum": False,
        "frozen_predictions": {
            "mechanism": mechanism,
            "training": training,
            "control": control,
        },
        "raw_sha256": {
            path.name: sha256(path) for path in [*seed_paths, summary_path]
        },
    }
    out_audit = Path(args.out_audit)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"audited {len(records)} records / {len(cells)} cells -> {out_csv}")
    print(json.dumps(audit["frozen_predictions"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
