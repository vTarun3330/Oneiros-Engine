"""Tests for the hard-example relearning round.

The properties under test are safety properties, not conveniences: a validation
or sealed-test case must never enter the queue, and a failed model output must
never become a training label.
"""
from __future__ import annotations

import pytest

from harness.relearning import (
    Correction,
    LoserCase,
    assert_split_is_eligible,
    attach_corrections,
    balanced_replay,
    classify_loser,
    relearning_dataset_sha256,
)


def _result(**overrides):
    payload = {
        "record_id": "r1",
        "requested_candidates": 8,
        "parsed_candidates": 8,
        "reference_valid_candidates": 8,
        "killed_candidates": 0,
        "function_killed": False,
        "candidates": [],
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# split isolation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["val", "validation", "test", "VAL", "Test"])
def test_validation_and_sealed_test_are_refused(split):
    with pytest.raises(ValueError, match="never enter relearning"):
        assert_split_is_eligible(split)


@pytest.mark.parametrize("split", ["train", "ablation_dev"])
def test_development_splits_are_eligible(split):
    assert_split_is_eligible(split) is None


def test_classify_refuses_a_validation_case_before_doing_anything():
    with pytest.raises(ValueError):
        classify_loser(_result(), "val")


def test_ablation_dev_hyphen_spelling_is_still_refused_if_forbidden():
    """'ablation-dev' normalizes to the eligible split, 'sealed-test' does not."""
    assert assert_split_is_eligible("ablation-dev") is None


# --------------------------------------------------------------------------
# what counts as a loser
# --------------------------------------------------------------------------

def test_a_killed_function_is_not_a_loser():
    assert classify_loser(_result(function_killed=True, killed_candidates=3),
                          "ablation_dev") is None


def test_a_function_the_model_missed_is_a_loser():
    loser = classify_loser(_result(), "ablation_dev")
    assert loser is not None
    assert loser.record_id == "r1"
    assert loser.dominant_category in {"no_kill", "syntax_invalid"}


def test_killed_but_worse_than_base_is_still_recorded():
    """A regression against the untrained base is a loser worth keeping."""
    loser = classify_loser(
        _result(function_killed=False),
        "ablation_dev",
        base_result=_result(function_killed=True),
    )
    assert loser is not None
    assert loser.worse_than_base is True
    assert loser.base_model_killed is True
    assert loser.categories.get("worse_than_base") == 1


def test_candidate_failure_modes_drive_the_category():
    loser = classify_loser(
        _result(candidates=[
            {"failure_mode": "wrong_expected_value"},
            {"failure_mode": "wrong_expected_value"},
            {"failure_mode": "syntax_invalid"},
        ]),
        "ablation_dev",
    )
    assert loser.dominant_category == "wrong_oracle"
    assert loser.categories["wrong_oracle"] == 2
    assert loser.categories["syntax_invalid"] == 1


def test_reference_failures_map_to_a_wrong_oracle():
    loser = classify_loser(
        _result(candidates=[{"failure_mode": "reference_assertion_error"}]),
        "ablation_dev",
    )
    assert loser.categories["wrong_oracle"] == 1


def test_passes_both_is_covering_without_detecting():
    loser = classify_loser(
        _result(candidates=[{"failure_mode": "passes_both"}]), "ablation_dev",
    )
    assert loser.dominant_category == "covers_without_detecting"


# --------------------------------------------------------------------------
# corrections are verified supervision, never model output
# --------------------------------------------------------------------------

def _loser(record_id="r1", project="synthetic", family="arithmetic",
           category="no_kill"):
    return LoserCase(
        record_id=record_id, split="ablation_dev", origin_group="synthetic_function",
        bug_family=family, complexity_tier="moderate", project=project,
        dominant_category=category, categories={category: 1},
        diversity_key=project,
        requested_candidates=8, parsed_candidates=8, reference_valid_candidates=8,
        killed_candidates=0, base_model_killed=None, worse_than_base=False,
        model_run="run", checkpoint_step=50, seed=42, prompt_version="v2",
    )


def test_a_loser_without_verified_supervision_yields_no_correction():
    """Fabricating a label here would train on an unverified completion."""
    corrections, summary = attach_corrections([_loser()], {})
    assert corrections == []
    assert summary["corrections"] == 0
    assert summary["skipped_without_verified_supervision"] == 1


def test_corrections_come_from_verified_completions_and_say_so():
    corrections, summary = attach_corrections(
        [_loser()], {"r1": "def test_r1():\n    assert f(1) == 2\n"},
    )
    assert summary["corrections"] == 1
    correction = corrections[0]
    assert correction.verified is True
    assert correction.supervision_source == "multi_mutant_verified_completion"
    assert correction.verification_evidence["executed_against_every_sibling_mutant"]


def test_correction_carries_the_category_it_is_meant_to_repair():
    corrections, _ = attach_corrections(
        [_loser(category="wrong_oracle")], {"r1": "def test_r1():\n    assert f(1) == 2\n"},
    )
    assert corrections[0].loser_category == "wrong_oracle"


# --------------------------------------------------------------------------
# balanced replay
# --------------------------------------------------------------------------

def _correction(record_id):
    return Correction(
        record_id=record_id, loser_category="no_kill",
        completion=f"def test_{record_id}():\n    assert f(1) == 2\n",
        completion_shape="test_function",
        supervision_source="multi_mutant_verified_completion", verified=True,
    )


def test_one_project_cannot_dominate_the_round():
    corrections = [_correction(f"r{index}") for index in range(10)]
    losers = {
        f"r{index}": _loser(record_id=f"r{index}", project="django")
        for index in range(10)
    }
    kept, summary = balanced_replay(corrections, losers, max_per_group=3)
    assert len(kept) == 3
    assert summary["dropped_by_cap"]["diversity_group_cap"] == 7


def test_caps_are_independent_across_projects():
    corrections = [_correction(f"r{index}") for index in range(6)]
    losers = {
        f"r{index}": _loser(
            record_id=f"r{index}", project="django" if index < 3 else "sympy",
        )
        for index in range(6)
    }
    kept, _ = balanced_replay(corrections, losers, max_per_group=3)
    assert len(kept) == 6


def test_replay_selection_is_deterministic():
    corrections = [_correction(f"r{index}") for index in range(8)]
    losers = {f"r{index}": _loser(record_id=f"r{index}") for index in range(8)}
    first, _ = balanced_replay(corrections, losers, max_per_group=4)
    second, _ = balanced_replay(list(reversed(corrections)), losers, max_per_group=4)
    assert [item.record_id for item in first] == [item.record_id for item in second]


def test_dataset_hash_is_order_independent_and_content_sensitive():
    left = [_correction("r1"), _correction("r2")]
    right = [_correction("r2"), _correction("r1")]
    assert relearning_dataset_sha256(left) == relearning_dataset_sha256(right)
    changed = [_correction("r1"), Correction(
        record_id="r2", loser_category="no_kill", completion="different",
        completion_shape="test_function", supervision_source="x", verified=True,
    )]
    assert relearning_dataset_sha256(left) != relearning_dataset_sha256(changed)
