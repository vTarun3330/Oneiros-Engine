"""The relearning round must emphasise failures, and refuse what it cannot use.

A round that trains on the ordinary mixture is a second epoch wearing the name
of relearning. These tests pin the two properties that make it neither: the
failed records are guaranteed a place and repeated, and corrections the trainer
could not consume are refused loudly rather than silently ignored.
"""
from __future__ import annotations

import json
from collections import Counter

import pytest

from scripts.train_on_dataset import apply_relearning_round


def _view(tmp_path, record_ids, split="train", verified=True):
    directory = tmp_path / "relearning"
    directory.mkdir(exist_ok=True)
    (directory / "corrections.json").write_text(
        json.dumps([
            {"record_id": record_id, "verified": verified,
             "completion": "def test_x():\n    assert f(1) == 2\n"}
            for record_id in record_ids
        ]),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps({
            "source_evaluation": {"split": split},
            "dataset_sha256": "d" * 64,
        }),
        encoding="utf-8",
    )
    return directory


def _pairs(count):
    return [{"id": f"r{index}"} for index in range(count)]


def test_a_failed_record_the_selection_missed_is_injected(tmp_path):
    everything = _pairs(50)
    selected = [{"id": "r0"}]
    result, stats = apply_relearning_round(
        selected, everything, _view(tmp_path, ["r7"]), 3,
    )
    assert stats["injected_records"] == 1
    assert Counter(pair["id"] for pair in result)["r7"] == 3


def test_a_failed_record_already_selected_is_repeated_not_duplicated_twice(tmp_path):
    everything = _pairs(50)
    selected = [{"id": "r0"}, {"id": "r1"}]
    result, stats = apply_relearning_round(
        selected, everything, _view(tmp_path, ["r1"]), 3,
    )
    assert stats["injected_records"] == 0
    assert Counter(pair["id"] for pair in result)["r1"] == 3


def test_records_that_did_not_fail_are_left_alone(tmp_path):
    everything = _pairs(50)
    selected = [{"id": "r0"}, {"id": "r2"}]
    result, _ = apply_relearning_round(
        selected, everything, _view(tmp_path, ["r9"]), 3,
    )
    counts = Counter(pair["id"] for pair in result)
    assert counts["r0"] == 1 and counts["r2"] == 1


def test_repeats_of_one_still_guarantees_inclusion(tmp_path):
    result, stats = apply_relearning_round(
        [{"id": "r0"}], _pairs(50), _view(tmp_path, ["r4"]), 1,
    )
    assert Counter(pair["id"] for pair in result)["r4"] == 1
    assert stats["injected_records"] == 1


def test_ablation_dev_corrections_are_refused(tmp_path):
    """They can never reach an optimizer step, and it is the selection panel."""
    with pytest.raises(ValueError, match="checkpoint-selection panel"):
        apply_relearning_round(
            [{"id": "r0"}], _pairs(50),
            _view(tmp_path, ["r1"], split="ablation_dev"), 3,
        )


def test_unverified_corrections_are_refused(tmp_path):
    with pytest.raises(ValueError, match="never a label"):
        apply_relearning_round(
            [{"id": "r0"}], _pairs(50),
            _view(tmp_path, ["r1"], verified=False), 3,
        )


def test_a_missing_view_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        apply_relearning_round([{"id": "r0"}], _pairs(50), tmp_path / "absent", 3)


def test_a_correction_for_an_unknown_record_is_skipped_not_fatal(tmp_path):
    """The corpus is the authority on what exists; a stale id is not a crash."""
    result, stats = apply_relearning_round(
        [{"id": "r0"}], _pairs(5), _view(tmp_path, ["not_in_corpus"]), 3,
    )
    assert stats["injected_records"] == 0
    assert len(result) == 1


def test_the_round_records_that_labels_are_verified_supervision(tmp_path):
    _, stats = apply_relearning_round(
        [{"id": "r0"}], _pairs(50), _view(tmp_path, ["r3"]), 3,
    )
    assert stats["labels_are_verified_supervision"] is True
    assert stats["model_output_used_as_label"] is False
