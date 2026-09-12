"""An absent protocol field means legacy, and mixing protocols is refused.

The trap this guards is specific. Legacy artifacts predate
``candidate_parse_mode`` entirely, so they record nothing at all. Treating that
silence as "unknown, probably fine" is what lets a legacy kill@8 of 0.7159 be
subtracted from a successor 0.5480 to produce a seventeen-point delta that
measures the parser instead of the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.evaluation_protocol import (
    LEGACY, PROTOCOL_OF_COMMITTED_BASELINE, PROTOCOL_OF_RECORD_FOR_NEW_WORK,
    SUCCESSOR, assert_comparable, describe, protocol_of,
)

ROOT = Path(__file__).resolve().parent.parent

LEGACY_ARTIFACT = {"evaluation_profile": {"candidate_budget": 8}}
SUCCESSOR_ARTIFACT = {
    "evaluation_profile": {"candidate_parse_mode": "whole_output"}}


def test_an_absent_field_is_legacy_not_unknown():
    assert protocol_of(LEGACY_ARTIFACT) == LEGACY
    assert protocol_of({}) == LEGACY


def test_an_explicit_field_is_honoured():
    assert protocol_of(SUCCESSOR_ARTIFACT) == SUCCESSOR
    assert protocol_of(
        {"evaluation_profile": {"candidate_parse_mode": "first_assertion"}}
    ) == LEGACY


def test_an_unrecognised_protocol_raises_rather_than_defaulting():
    """Silently treating a new mode as legacy is how this defect recurs."""
    with pytest.raises(ValueError, match="unknown candidate_parse_mode"):
        protocol_of({"evaluation_profile": {"candidate_parse_mode": "per_assertion"}})


def test_comparing_across_protocols_is_refused():
    with pytest.raises(SystemExit, match="different evaluation"):
        assert_comparable([LEGACY_ARTIFACT, SUCCESSOR_ARTIFACT], ["base", "arm"])


def test_the_refusal_names_which_side_is_which():
    with pytest.raises(SystemExit) as raised:
        assert_comparable([LEGACY_ARTIFACT, SUCCESSOR_ARTIFACT], ["base", "arm"])
    message = str(raised.value)
    assert "base = first_assertion" in message
    assert "arm = whole_output" in message


def test_same_protocol_is_allowed_and_returned():
    assert assert_comparable([LEGACY_ARTIFACT, dict(LEGACY_ARTIFACT)]) == LEGACY
    assert assert_comparable(
        [SUCCESSOR_ARTIFACT, dict(SUCCESSOR_ARTIFACT)]) == SUCCESSOR


def test_describe_distinguishes_recorded_from_inferred():
    assert describe(LEGACY_ARTIFACT)["recorded_explicitly"] is False
    assert describe(SUCCESSOR_ARTIFACT)["recorded_explicitly"] is True


def test_the_declared_protocols_are_what_the_report_says():
    """The committed baseline is legacy; new work uses the successor."""
    assert PROTOCOL_OF_COMMITTED_BASELINE == LEGACY
    assert PROTOCOL_OF_RECORD_FOR_NEW_WORK == SUCCESSOR


def test_the_committed_baseline_artifacts_really_are_legacy():
    """If this fails, the arm table in the report is mixing protocols."""
    for relative in (
        "local_base_qwen_val_seed42/base_validation_standard_seed_42.json",
        "local_sft_relearn_v2_seed42/sft_validation_standard_seed_42.json",
    ):
        path = ROOT / "results" / relative
        if not path.exists():
            pytest.skip(f"{relative} not present in this checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert protocol_of(payload) == PROTOCOL_OF_COMMITTED_BASELINE


def test_slice_kill_rate_refuses_a_mixed_comparison(tmp_path):
    """The guard has to be wired in, not merely available."""
    from scripts.slice_kill_rate import build

    def write(name, protocol_profile, split="val"):
        path = tmp_path / name
        path.write_text(json.dumps({
            "evaluation_split": split,
            "evaluation_profile": protocol_profile,
            "function_results": [
                {"record_id": "r1", "dataset_name": "mbpp", "killed": True},
            ],
        }), encoding="utf-8")
        return path

    arm = write("arm.json", {"candidate_parse_mode": "whole_output"})
    base = write("base.json", {"candidate_budget": 8})
    with pytest.raises(SystemExit, match="different evaluation"):
        build(arm, base)
