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


# --- the successor must not overwrite the legacy artifact ------------------
#
# The evaluation profile slug encodes split, smoke size, feedback rounds,
# diversity, holdout and prompt/instruction variants - but not the parse mode.
# A whole_output run on ablation_dev therefore wrote
# sft_validation_ablation-dev_seed_42.json: byte-for-byte the legacy
# artifact's name, silently replacing a different metric under the same field
# names.

def test_the_parse_mode_appears_in_the_results_filename(monkeypatch):
    import scripts.train_on_dataset as driver

    monkeypatch.setattr(driver, "EVALUATION_SPLIT", "ablation_dev")
    monkeypatch.setattr(driver, "CANDIDATE_PARSE_MODE", "whole_output")
    successor = driver.evaluation_results_filename("sft", 42)

    monkeypatch.setattr(driver, "CANDIDATE_PARSE_MODE", "first_assertion")
    legacy = driver.evaluation_results_filename("sft", 42)

    assert successor != legacy, (
        "a successor run would overwrite the legacy artifact it is meant to "
        "be compared against"
    )
    assert "parse-whole-output" in successor
    assert "parse-" not in legacy, (
        "the frozen protocol's filenames must not change, or every historical "
        "artifact stops being found by the name it was written under"
    )


def test_the_frozen_filename_is_unchanged(monkeypatch):
    import scripts.train_on_dataset as driver

    monkeypatch.setattr(driver, "EVALUATION_SPLIT", "val")
    monkeypatch.setattr(driver, "CANDIDATE_PARSE_MODE", "first_assertion")
    assert driver.evaluation_results_filename("base", 42) == \
        "base_validation_standard_seed_42.json"


# --- the scoring protocol is part of the evaluation's identity -------------
#
# _adapter_evaluation_context returned identical output whether the run used
# first_assertion or whole_output, retained raw output or not, and accepted
# test functions or not. Progress recorded under one protocol could therefore
# be adopted by a resume under the other, producing one artifact whose
# candidates were scored by two different rules. The filename fix made this
# latent rather than active; relying on a filename to carry an identity the
# identity record omits is not a guarantee.

def test_the_resume_identity_separates_the_two_protocols(monkeypatch):
    import scripts.train_on_dataset as driver

    arguments = ("dataset", "adapter", "sha", "ablation_dev", None, "scope", 1)

    monkeypatch.setattr(driver, "CANDIDATE_PARSE_MODE", "first_assertion")
    monkeypatch.setattr(driver, "RETAIN_RAW_OUTPUT", False)
    monkeypatch.setattr(driver, "ALLOW_TEST_FUNCTION_CANDIDATES", False)
    legacy = driver._adapter_evaluation_context(*arguments)

    monkeypatch.setattr(driver, "CANDIDATE_PARSE_MODE", "whole_output")
    monkeypatch.setattr(driver, "RETAIN_RAW_OUTPUT", True)
    monkeypatch.setattr(driver, "ALLOW_TEST_FUNCTION_CANDIDATES", True)
    successor = driver._adapter_evaluation_context(*arguments)

    assert legacy != successor, (
        "a successor resume could adopt legacy progress and score one "
        "artifact's candidates by two different rules"
    )
    assert legacy["evaluation_profile_sha256"] != \
        successor["evaluation_profile_sha256"]


def test_each_protocol_setting_changes_the_identity_on_its_own(monkeypatch):
    """One combined flag would hide a change to either of the others."""
    import scripts.train_on_dataset as driver

    arguments = ("dataset", "adapter", "sha", "ablation_dev", None, "scope", 1)
    monkeypatch.setattr(driver, "CANDIDATE_PARSE_MODE", "first_assertion")
    monkeypatch.setattr(driver, "RETAIN_RAW_OUTPUT", False)
    monkeypatch.setattr(driver, "ALLOW_TEST_FUNCTION_CANDIDATES", False)
    baseline = driver._adapter_evaluation_context(*arguments)["evaluation_profile_sha256"]

    for name, value in (("CANDIDATE_PARSE_MODE", "whole_output"),
                        ("RETAIN_RAW_OUTPUT", True),
                        ("ALLOW_TEST_FUNCTION_CANDIDATES", True)):
        monkeypatch.setattr(driver, name, value)
        changed = driver._adapter_evaluation_context(*arguments)["evaluation_profile_sha256"]
        assert changed != baseline, f"{name} does not affect the resume identity"
        monkeypatch.setattr(
            driver, name,
            "first_assertion" if name == "CANDIDATE_PARSE_MODE" else False)


def test_the_completion_budget_appears_in_the_results_filename(monkeypatch):
    """Two budgets are two measurements, not a collision.

    The completion budget is already part of the evaluation scope hash, so a
    256-token run was correctly REFUSED as a mismatch against the 128-token
    artifact - but it shared that artifact's name, so a legitimate second
    measurement read as an immutability violation.
    """
    import scripts.train_on_dataset as driver

    monkeypatch.setattr(driver, "EVALUATION_SPLIT", "ablation_dev")
    monkeypatch.setattr(driver, "CANDIDATE_PARSE_MODE", "whole_output")

    monkeypatch.setattr(driver, "MAX_NEW_TOKENS_OVERRIDE", 128)
    default_budget = driver.evaluation_results_filename("sft", 42)
    monkeypatch.setattr(driver, "MAX_NEW_TOKENS_OVERRIDE", 256)
    raised_budget = driver.evaluation_results_filename("sft", 42)

    assert default_budget != raised_budget
    assert "completion256" in raised_budget
    assert "completion" not in default_budget, (
        "the default budget must not rename every historical artifact"
    )


def test_a_prompt_and_completion_that_cannot_share_a_sequence_are_refused():
    """2048 prompt + 1024 completion does not fit a 2048-token sequence."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    driver = _Path(__file__).resolve().parent.parent / "scripts" / "train_on_dataset.py"
    result = subprocess.run(
        [_sys.executable, str(driver), "--phase", "base_eval", "--run-name", "x",
         "--sft-prompt-token-limit", "1024",
         "--generation-completion-token-limit", "1500"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "does not fit" in (result.stdout + result.stderr)
