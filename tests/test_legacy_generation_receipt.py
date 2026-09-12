"""A legacy-protocol artifact must be labelled unusable where it sits.

An evaluation artifact carries no visible sign of which protocol produced it -
legacy runs predate the field entirely. This receipt is what stops the full
train-split legacy run being picked up months later by someone who sees only a
promising filename and a large record count.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_legacy_generation_receipt import (
    PERMITTED_USE, PROHIBITED_USES, RECEIPT_NAME, build,
)

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = (ROOT / "results" / "local_base_qwen_train_full_s42" / RECEIPT_NAME)


def _artifact(tmp_path, profile, split="train"):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "evaluation_split": split,
        "final_test_measurement": False,
        "seed": 42,
        "adapter": "base_model",
        "evaluation_profile": profile,
        "function_validation_records": 5595,
    }), encoding="utf-8")
    return path


def test_a_legacy_run_is_marked_unsuitable(tmp_path):
    receipt = build(_artifact(tmp_path, {
        "candidate_parse_mode": "first_assertion",
        "retain_raw_output": False,
        "allow_test_function_candidates": False,
    }))
    assert receipt["suitable_for_oracle_structured_dataset"] is False
    assert receipt["raw_outputs_available"] is False
    assert receipt["evaluation_protocol"] == "first_assertion"
    assert receipt["permitted_use"] == PERMITTED_USE


def test_all_three_disqualifiers_are_named_individually(tmp_path):
    """A single lumped reason would not survive someone fixing one of them."""
    receipt = build(_artifact(tmp_path, {
        "candidate_parse_mode": "first_assertion",
        "retain_raw_output": False,
        "allow_test_function_candidates": False,
    }))
    assert len(receipt["disqualifiers"]) == 3
    joined = " ".join(receipt["disqualifiers"])
    assert "first_assertion" in joined
    assert "retain_raw_output" in joined
    assert "allow_test_function_candidates" in joined


def test_oracle_dataset_construction_is_explicitly_prohibited(tmp_path):
    receipt = build(_artifact(tmp_path, {
        "candidate_parse_mode": "first_assertion", "retain_raw_output": False,
    }))
    assert "oracle_structured_dataset_construction" in receipt["prohibited_uses"]
    assert "wrong_oracle_correction_labels" in receipt["prohibited_uses"]
    assert set(PROHIBITED_USES) <= set(receipt["prohibited_uses"])


def test_a_successor_run_with_raw_output_has_no_disqualifiers(tmp_path):
    """The receipt must not condemn the artifact it is meant to license."""
    receipt = build(_artifact(tmp_path, {
        "candidate_parse_mode": "whole_output",
        "retain_raw_output": True,
        "allow_test_function_candidates": True,
    }))
    assert receipt["disqualifiers"] == []
    assert receipt["raw_outputs_available"] is True
    assert receipt["evaluation_protocol"] == "whole_output"


def test_the_receipt_records_the_permitted_use_rather_than_only_refusing(tmp_path):
    receipt = build(_artifact(tmp_path, {
        "candidate_parse_mode": "first_assertion", "retain_raw_output": False,
    }))
    assert "composition" in receipt["permitted_use"]
    assert "mbpp" in receipt["why_it_is_kept"]


def test_sealed_test_is_never_marked_accessed(tmp_path):
    receipt = build(_artifact(tmp_path, {"retain_raw_output": False}))
    assert receipt["sealed_final_test_accessed"] is False
    assert receipt["final_test_measurement"] is False


@pytest.mark.skipif(not RECEIPT.exists(), reason="receipt not written here")
def test_the_committed_receipt_says_what_the_report_says():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["evaluation_protocol"] == "first_assertion"
    assert receipt["raw_outputs_available"] is False
    assert receipt["suitable_for_oracle_structured_dataset"] is False
    assert receipt["evaluation_split"] == "train"
    assert receipt["sealed_final_test_accessed"] is False
