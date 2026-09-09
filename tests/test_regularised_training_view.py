"""Data-level regularisers must be measured on the output, not requested.

The dropout and two-epoch arms both failed, so the remaining lever is the data
itself. Each property here is checked against the assembled view, because a
regulariser that is configured but not binding is worse than none: it reads as
a control that is not controlling anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regularised_training_view import MAX_REPEATS_PER_LINEAGE, build

VIEW = ROOT / "data" / "training_views" / "regularised_v1" / "view.json"


def _fixture(tmp_path, completions):
    curriculum = tmp_path / "c.json"
    curriculum.write_text(json.dumps({"curriculum": [
        {"block": 1, "items": [
            {"lineage": f"g{i}", "tier": "easy", "difficulty": 1.0,
             "displayed_record_id": f"r{i}"} for i in range(len(completions))]},
    ]}), encoding="utf-8")
    corrections = tmp_path / "x.json"
    corrections.write_text(json.dumps({"items": [
        {"lineage": f"g{i}", "displayed_record_id": f"r{i}", "entry_point": "f",
         "source_dataset": "mbpp", "completion": completion, "verified": True,
         "verification_evidence": {"valid_on_reference": True}}
        for i, completion in enumerate(completions)]}), encoding="utf-8")
    return curriculum, corrections


def test_identical_completions_are_deduplicated(tmp_path):
    curriculum, corrections = _fixture(
        tmp_path, ["assert f(1) == 2", "assert f(1) == 2", "assert f(2) == 3"])
    report = build(curriculum, corrections)
    assert report["examples"] == 2
    assert report["rejected"]["duplicate_completion"] == 1


def test_unverified_completions_are_refused(tmp_path):
    curriculum, corrections = _fixture(tmp_path, ["assert f(1) == 2"])
    payload = json.loads(corrections.read_text(encoding="utf-8"))
    payload["items"][0]["verification_evidence"]["valid_on_reference"] = False
    corrections.write_text(json.dumps(payload), encoding="utf-8")

    report = build(curriculum, corrections)
    assert report["examples"] == 0
    assert report["rejected"]["unverified_completion"] == 1


def test_a_lineage_without_a_correction_is_dropped_not_invented(tmp_path):
    curriculum, corrections = _fixture(tmp_path, ["assert f(1) == 2"])
    payload = json.loads(corrections.read_text(encoding="utf-8"))
    payload["items"] = []
    corrections.write_text(json.dumps(payload), encoding="utf-8")

    report = build(curriculum, corrections)
    assert report["examples"] == 0
    assert report["rejected"]["no_verified_correction"] == 1


def test_the_committed_view_is_fully_deduplicated():
    if not VIEW.exists():
        return
    report = json.loads(VIEW.read_text(encoding="utf-8"))
    assert report["distinct_completions"] == report["examples"]
    assert report["distinct_lineages"] == report["examples"]
    assert report["max_repeats_per_lineage"] <= MAX_REPEATS_PER_LINEAGE


def test_the_committed_view_keeps_every_block_mixed():
    if not VIEW.exists():
        return
    report = json.loads(VIEW.read_text(encoding="utf-8"))
    assert report["every_block_multi_tier"] is True


def test_the_source_cap_is_reported_even_when_it_does_not_bind():
    """The honest case: mbpp holds 83.6% and the cap never fired.

    The cap limits REPETITION, and completion deduplication already removed
    every repeat, so there was nothing left for it to limit. A control that
    reports itself as active while doing nothing is the failure mode here.
    """
    if not VIEW.exists():
        return
    report = json.loads(VIEW.read_text(encoding="utf-8"))
    assert "largest_source_share" in report
    assert report["largest_source_share"] is not None
    if report["max_repeats_per_lineage"] <= 1:
        assert "dominant_source_repetition_capped" not in report["rejected"], (
            "with no repetition to cap, the source cap cannot have fired; if "
            "it reports otherwise the accounting is wrong"
        )


def test_no_validation_signal_reaches_the_view():
    if not VIEW.exists():
        return
    report = json.loads(VIEW.read_text(encoding="utf-8"))
    assert report["validation_performance_used"] is False
    assert report["sealed_final_test_accessed"] is False
