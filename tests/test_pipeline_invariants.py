"""End-to-end invariants across the seams between components.

Every defect this project has hit lived at a seam, and every test written
before this file checked one component in isolation:

* the trainer keyed verified completions one way and the relearning builder
  another, and both were individually correct;
* the trainer emitted multi-assertion test functions while the evaluator
  accepted only a lone assertion, and both were individually correct;
* relearning corrections were mined from a split the trainer never draws from,
  and both the miner and the trainer were individually correct;
* the generator collapses a model output to its first assertion while the
  candidate policy is happy to validate the whole thing, and both are
  individually correct.

Component tests cannot see any of that. These tests deliberately cross the
seams: they take what one stage produces and assert the next stage can consume
it, or that the loss is recorded rather than silent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import (
    executable_candidate, validate_generated_test,
)
from harness.multi_mutant_examples import verified_completions_by_record
from harness.safe_execution import execute_code

MULTI_MUTANT = ROOT / "data" / "training_views" / "multi_mutant_v1"
VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)


def _examples() -> list[dict]:
    path = MULTI_MUTANT / "train.examples.json"
    if not path.exists():
        pytest.skip("multi-mutant training view not built")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- seam 1
# what the trainer learns  ->  what the evaluator will accept

def test_every_training_completion_is_a_shape_the_evaluator_can_accept():
    """The model is trained to produce these. The scorer must admit them.

    This is the train/eval shape mismatch, asserted over the real corpus
    instead of a fixture. A completion the evaluator would reject is a target
    the model can never be credited for hitting.
    """
    rejected = []
    for item in _examples()[:400]:
        completion = str(item["completion"])
        entry = str(item.get("entry_point") or "")
        if not entry:
            continue
        result = validate_generated_test(completion, entry, allow_test_function=True)
        if not result.valid:
            rejected.append((item.get("displayed_record_id"), result.reason))
    assert not rejected, (
        f"{len(rejected)} training completions would be rejected by the "
        f"evaluator's own policy; first: {rejected[:3]}"
    )


def test_a_training_completion_survives_the_widened_generation_parser():
    """A completion routed through generation must reach execution intact."""
    from engine.generator import Phi3Generator

    example = next(
        item for item in _examples()
        if str(item["completion"]).lstrip().startswith("def test")
    )
    completion = str(example["completion"])
    entry = str(example["entry_point"])

    generator = Phi3Generator.__new__(Phi3Generator)
    generator.stats = {"total_generated": 0, "valid_generated": 0,
                       "invalid_generated": 0}
    generator.parse_mode = "whole_output"

    parsed = generator._parse_output(completion, entry, entry)
    assert parsed.is_valid, parsed.parse_error
    assert parsed.input_code.count("assert ") == completion.count("assert "), (
        "the parser dropped assertions the model was trained to produce"
    )


def test_the_frozen_parser_loss_is_known_and_asserted():
    """The frozen parser DOES drop assertions. That is recorded, not fixed.

    Changing the default would break comparability with every reported result,
    and the loss was measured to be a wash. What must never happen is the loss
    being present and unknown, so it is pinned here with its magnitude.
    """
    from engine.generator import Phi3Generator

    example = next(
        item for item in _examples()
        if str(item["completion"]).count("assert ") >= 3
    )
    completion = str(example["completion"])
    entry = str(example["entry_point"])

    generator = Phi3Generator.__new__(Phi3Generator)
    generator.stats = {"total_generated": 0, "valid_generated": 0,
                       "invalid_generated": 0}
    parsed = generator._parse_output(completion, entry, entry)
    assert parsed.input_code.count("assert ") == 1, (
        "frozen mode is expected to keep exactly one assertion"
    )


# ---------------------------------------------------------------- seam 2
# what the policy validates  ->  what the worker executes

def test_a_validated_test_function_is_actually_invoked_when_executed():
    """Validation passing is not enough; the worker must run the assertions.

    A defined-but-uncalled test function passes on the reference AND on the
    mutant, so it looks like a survivor rather than an error. This silently
    zeroed a rescore until the number contradicted the pipeline's own.
    """
    reference = "def f(x):\n    return x + 1\n"
    mutant = "def f(x):\n    return x + 2\n"
    test = "def test_f():\n    assert f(1) == 2\n"

    result = validate_generated_test(test, "f", allow_test_function=True)
    assert result.valid and result.shape == "test_function"

    runnable = executable_candidate(test, result.shape)
    reference_ok, _, _ = execute_code(reference, runnable, 5.0)
    mutant_ok, _, _ = execute_code(mutant, runnable, 5.0)
    assert reference_ok, "a valid test must pass on the reference"
    assert not mutant_ok, (
        "the mutant survived a test that should kill it - the test function "
        "was defined but never called"
    )


# ---------------------------------------------------------------- seam 3
# what supervision covers  ->  what the trainer can consume

def test_verified_completions_cover_the_records_they_were_verified_against():
    """Keying by the displayed record alone discards most of the supervision.

    Each completion is executed against every sibling mutant and the kills are
    recorded, so those records are supervised. Coverage collapsing back toward
    the number of lineages means the loader regressed.
    """
    examples = _examples()
    completions = verified_completions_by_record(examples)
    lineages = len(examples)
    assert len(completions) > lineages * 3, (
        f"coverage {len(completions)} is near the lineage count {lineages}; "
        "the loader is keying by displayed record again"
    )


# ---------------------------------------------------------------- seam 4
# what the evaluation scope holds  ->  what actually gets scored

def test_targets_dropped_before_scoring_are_repository_only_and_counted():
    """The evaluator scores function pairs and silently drops repository ones.

    781 val records become a 757-target panel. That is a deliberate design
    choice - repository execution needs a native project environment - but a
    reader who sees "757 held-out functions" cannot tell the balanced corpus
    was never measured. The drop is asserted here so it can never grow, move
    to another category, or be forgotten.
    """
    records_path = VIEW / "val.records.json"
    if not records_path.exists():
        pytest.skip("development view not built")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    repository = [r for r in records if r.get("task_mode") == "repository"]
    function = [r for r in records if r.get("task_mode") != "repository"]

    assert len(function) == 757
    assert len(repository) == 24, (
        "the repository count changed; every reported Kill@8 covers only the "
        "function records, so this number is part of what the results mean"
    )

    import scripts.train_on_dataset as driver

    assert driver.REPOSITORY_EVALUATION_STATUS == (
        "not_implemented_requires_native_project_environment"
    ), "repository evaluation status changed; the panel composition claim in "
    "the reports must be re-derived"


def test_no_reported_evaluation_artifact_contains_a_repository_target():
    """Derived from artifacts, so it cannot drift from what was measured."""
    import glob

    records_path = VIEW / "val.records.json"
    if not records_path.exists():
        pytest.skip("development view not built")
    repository_ids = {
        str(r["id"]) for r in json.loads(records_path.read_text(encoding="utf-8"))
        if r.get("task_mode") == "repository"
    }

    checked = 0
    for path in glob.glob(str(ROOT / "results" / "*" / "*validation_standard*.json")):
        if ".progress." in path:
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scored = {
            str(row.get("record_id")) for row in payload.get("function_results") or []
        }
        assert not (scored & repository_ids), (
            f"{path} scored repository targets; the synthetic-only claim in "
            "the reports is now wrong and must be updated"
        )
        checked += 1
    assert checked, "no validation artifacts found to check"


# ---------------------------------------------------------------- seam 5
# what was measured  ->  what the report says was measured

def test_a_comparison_report_states_its_panel_composition():
    """A result must carry what it measured, in the same artifact.

    The evaluator recorded 781 records, 757 scored and 24 repository held, all
    along. No report surfaced any of it, so "757 held-out functions" read as
    though the balanced corpus had been measured. Correct data that no report
    carries is not evidence a reader can use.
    """
    from scripts.compare_base_vs_sft import panel_composition

    payloads = [{
        "evaluation_split_records": 781,
        "function_validation_records": 757,
        "repository_validation_records_held": 24,
    }]
    row = panel_composition(payloads)
    assert row["records_in_split"] == [781]
    assert row["targets_scored"] == [757]
    assert row["repository_targets_held_unscored"] == [24]
    assert row["all_arms_share_one_panel"] is True
    assert "SYNTHETIC" in row["reporting_consequence"]


def test_panel_composition_flags_arms_that_did_not_share_one_panel():
    """Comparing arms scored on different panels is not a paired comparison."""
    from scripts.compare_base_vs_sft import panel_composition

    row = panel_composition([
        {"evaluation_split_records": 781, "function_validation_records": 757,
         "repository_validation_records_held": 24},
        {"evaluation_split_records": 781, "function_validation_records": 700,
         "repository_validation_records_held": 24},
    ])
    assert row["all_arms_share_one_panel"] is False
