from __future__ import annotations

from collections import Counter

import pytest

from harness.execution_supervision_sidecar import (
    ARM_SIZE,
    REPLACEMENT_COUNT,
    assemble_controlled_arms,
    freeze_lineage_split,
)


def _row(index: int, *, source: str = "mbpp") -> dict:
    return {
        "record_id": f"record-{index}",
        "function_lineage": f"lineage-{index}",
        "source_dataset": source,
        "bug_family": "boundary",
        "complexity_tier": "moderate",
        "completion": f"assert f({index}) == {index + 1}",
        "canonical_prompt": f"canonical {index}",
        "focused_prompt": f"focused {index}",
    }


def test_lineage_split_is_exact_disjoint_deterministic_and_stratified() -> None:
    rows = [
        _row(index, source="mbpp" if index < 531 else
             "humaneval" if index < 581 else "manual_curated_examples")
        for index in range(585)
    ]
    first = freeze_lineage_split(rows)
    second = freeze_lineage_split(reversed(rows))
    assert first == second
    assert len(first["train_lineages"]) == 385
    assert len(first["pilot_development_lineages"]) == 100
    assert len(first["unopened_confirmation_lineages"]) == 100
    assert first["no_lineage_overlap"] is True
    assert first["source_counts"]["eligible"] == {
        "humaneval": 50, "manual_curated_examples": 4, "mbpp": 531,
    }
    assert sum(first["source_counts"]["train"].values()) == 385


def test_split_refuses_attrition_hidden_by_a_declared_count() -> None:
    with pytest.raises(ValueError, match="eligible lineage universe"):
        freeze_lineage_split([_row(index) for index in range(584)])


def test_controlled_arms_differ_only_in_128_prompts() -> None:
    shared = [_row(index) for index in range(ARM_SIZE - REPLACEMENT_COUNT)]
    replacement = [
        _row(10_000 + index, source="humaneval" if index < 40 else "mbpp")
        for index in range(REPLACEMENT_COUNT)
    ]
    arm_a, arm_b, report = assemble_controlled_arms(shared, replacement)
    assert len(arm_a) == len(arm_b) == 1024
    assert report["replacement_examples"] == 128
    assert report["completion_bytes_equal_at_every_position"] is True
    changed = [index for index, pair in enumerate(zip(arm_a, arm_b))
               if pair[0] != pair[1]]
    assert changed == report["replacement_positions"]
    assert Counter(row["task_kind"] for row in arm_a) == {
        "test_generation": 1024,
    }
    assert Counter(row["task_kind"] for row in arm_b) == {
        "test_generation": 896,
        "execution_output_prediction": 128,
    }


def test_controlled_arm_counts_are_not_silently_padded_or_trimmed() -> None:
    shared = [_row(index) for index in range(895)]
    replacement = [_row(10_000 + index) for index in range(128)]
    with pytest.raises(ValueError, match="shared row count"):
        assemble_controlled_arms(shared, replacement)
