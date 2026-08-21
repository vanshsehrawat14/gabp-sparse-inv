"""Mutation check for the Paper 1 quantitative verifier."""

from __future__ import annotations

import csv
import importlib.util
import shutil
from pathlib import Path

import pytest


VERIFY_PATH = Path(__file__).resolve().parents[1] / "paper" / "attainability" / "verify_claims.py"
SPEC = importlib.util.spec_from_file_location("paper_verify_claims", VERIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
verify_claims = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_claims)


def test_reach_floor_verifier_rejects_a_mutated_record(tmp_path, monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(verify_claims.DATA, data)
    record = data / "maze_reach_floor.csv"
    with record.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["floor_mse"] = "999"
    with record.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(verify_claims, "DATA", data)
    with pytest.raises(AssertionError):
        verify_claims.verify_reach_floor()
