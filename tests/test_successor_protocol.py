"""Successor and legacy artifacts must not be confusable.

The successor judges the whole output; the legacy protocol scored the first
assertion. Both write files named `*_validation_*.json` with a field called
`function_kill_rate`, so a directory holding both would contain two different
metrics under one name, and a reader could not tell which they had.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_successor_protocol_receipt import (
    DEFINING_SOURCES, LEGACY_PROTOCOL_NAME, PROTOCOL_NAME, build,
    results_directory_conflict,
)

RECEIPT = ROOT / "results" / "v4_2_successor_protocol_receipt.json"


def _receipt(**overrides):
    kwargs = dict(candidates=8, seeds=[42], prompt_budget=1024,
                  completion_budget=128, sequence_budget=2048,
                  raw_output_dir="results/{run}/raw_generations")
    kwargs.update(overrides)
    return build(**kwargs)


def test_a_directory_holding_legacy_artifacts_is_refused(tmp_path):
    run = tmp_path / "local_sft_relearn_v2_seed42"
    run.mkdir()
    (run / "sft_validation_standard_seed_42.json").write_text("{}", encoding="utf-8")

    conflict = results_directory_conflict(run)

    assert conflict is not None
    assert "own run directory" in conflict


def test_a_fresh_directory_is_allowed(tmp_path):
    assert results_directory_conflict(tmp_path / "successor_run") is None


def test_a_directory_of_successor_artifacts_is_not_a_conflict(tmp_path):
    run = tmp_path / "successor_run"
    run.mkdir()
    (run / "sft_validation_successor_seed_42.json").write_text("{}", encoding="utf-8")

    assert results_directory_conflict(run) is None


def test_the_receipt_names_the_protocol_and_what_it_supersedes():
    receipt = _receipt()
    assert receipt["protocol_name"] == PROTOCOL_NAME
    assert receipt["supersedes"]["protocol_name"] == LEGACY_PROTOCOL_NAME
    assert "HISTORICAL, NOT INVALID" in receipt["supersedes"]["status"]


def test_the_receipt_hashes_every_source_that_defines_behaviour():
    """A silent edit to the evaluator would change what a number means."""
    receipt = _receipt()
    for name in DEFINING_SOURCES:
        entry = receipt["defining_sources"][name]
        assert len(entry["sha256"]) == 64
        assert (ROOT / entry["path"]).exists()


def test_a_budget_that_does_not_fit_is_reported_as_not_fitting():
    """2048 prompt + 1024 completion cannot fit a 2048 sequence."""
    receipt = _receipt(prompt_budget=2048, completion_budget=1024,
                       sequence_budget=2048)
    assert receipt["budgets"]["fits"] is False


def test_a_budget_that_fits_is_reported_as_fitting():
    assert _receipt()["budgets"]["fits"] is True


def test_the_receipt_records_generation_and_scoring_rules():
    receipt = _receipt()
    generation = receipt["generation"]
    assert generation["candidate_parse_mode"] == "whole_output"
    assert generation["retain_raw_output"] is True
    assert generation["no_reranking"] is True
    assert "SAME retained raw output" in generation["generated_once"]


def test_the_receipt_states_the_locked_validation_policy():
    receipt = _receipt()
    policy = receipt["locked_validation_policy"]
    assert "ablation_dev only" in policy
    assert "frozen" in policy


def test_the_committed_receipt_matches_the_current_sources():
    """A source edited after the receipt was written invalidates it."""
    if not RECEIPT.exists():
        return
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    rebuilt = _receipt()
    assert committed["defining_sources"] == rebuilt["defining_sources"], (
        "a file defining the protocol changed after the receipt was written; "
        "rebuild the receipt before running anything under it"
    )
