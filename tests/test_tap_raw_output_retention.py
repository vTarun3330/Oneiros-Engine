"""The TAP comparison must retain complete generations, not prefixes.

The first capacity-gate run clipped every stored ``raw`` to 300 characters. Most
of the adapter's outputs are longer than that, so 309 of 600 TAP-ref rows and
376 of 600 TAP-mut rows were stored truncated, and the recorded hashes covered a
prefix rather than what the model produced. A result whose evidence cannot be
re-derived from the artifact is not reproducible, so this is enforced by test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_tap_adapter_compare import (
    atomic_json_write,
    checkpoint_payload,
    load_checkpoint,
    score,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "results" / "tap_adapter_compare.json"

#: A generation comfortably longer than the old 300-character clip.
LONG_OUTPUT = (
    "# STEP 1 - the intended behaviour is to return the rounded average of the\n"
    "# integers from n to m inclusive, converted to binary, and -1 when n > m.\n"
    "# " + ("x" * 400) + "\n"
    "assert rounded_avg(1, 5) == '0b11'\n"
)


def _item(expected="'0b11'"):
    return {"id": "t::0", "benchmark": "humaneval", "expected_repr": expected}


def test_score_retains_the_complete_generation():
    _, detail = score([_item()], [LONG_OUTPUT], [False])
    row = detail[0]
    assert row["raw"] == LONG_OUTPUT
    assert row["raw_chars"] == len(LONG_OUTPUT)
    assert len(LONG_OUTPUT) > 300, "the fixture must exceed the old clip"


def test_recorded_hash_covers_the_whole_generation():
    _, detail = score([_item()], [LONG_OUTPUT], [False])
    row = detail[0]
    assert row["raw_sha256"] == hashlib.sha256(LONG_OUTPUT.encode("utf-8")).hexdigest()
    prefix = hashlib.sha256(LONG_OUTPUT[:300].encode("utf-8")).hexdigest()
    assert row["raw_sha256"] != prefix, "hash must not cover only a prefix"


def test_expected_and_predicted_are_not_clipped():
    long_expected = "[" + ", ".join(str(n) for n in range(60)) + "]"
    output = f"assert f(x) == {long_expected}\n"
    _, detail = score([_item(long_expected)], [output], [False])
    row = detail[0]
    assert row["expected"] == long_expected
    assert row["predicted"] == long_expected
    assert row["verdict"] == "correct"
    assert len(long_expected) > 80, "the fixture must exceed the old clip"


def test_every_row_carries_a_hash_matching_its_raw_text():
    outputs = ["assert f(1) == 2", "", LONG_OUTPUT]
    items = [_item("2"), _item("2"), _item("'0b11'")]
    _, detail = score(items, outputs, [False, False, True])
    assert len(detail) == 3
    for row, text in zip(detail, outputs):
        assert row["raw"] == text
        assert row["raw_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_checkpoint_round_trip_preserves_complete_raw_outputs(tmp_path):
    items = [_item()]
    _, detail = score(items, [LONG_OUTPUT], [False])
    results = {"base::TAP-ref": {
        "per_benchmark": {"ALL": {"correct": 1}},
        "detail": detail,
        "truncated": 0,
    }}
    contract = {"schema_version": "test-contract", "items": 1}
    checkpoint = tmp_path / "tap.partial.json"
    atomic_json_write(
        checkpoint,
        checkpoint_payload(contract, results, "2026-09-19T00:00:00Z",
                           status="in_progress"),
    )

    loaded, created = load_checkpoint(checkpoint, contract, ["t::0"])
    assert created == "2026-09-19T00:00:00Z"
    assert loaded["base::TAP-ref"]["detail"][0]["raw"] == LONG_OUTPUT


def test_resume_refuses_a_different_contract(tmp_path):
    checkpoint = tmp_path / "tap.partial.json"
    atomic_json_write(
        checkpoint,
        checkpoint_payload({"items": 1}, {}, "2026-09-19T00:00:00Z",
                           status="in_progress"),
    )
    with pytest.raises(RuntimeError, match="contract mismatch"):
        load_checkpoint(checkpoint, {"items": 2}, ["t::0"])


def test_resume_refuses_raw_hash_mismatch(tmp_path):
    result = {
        "base::TAP-ref": {
            "per_benchmark": {},
            "truncated": 0,
            "detail": [{"id": "t::0", "raw": "assert f(1) == 2",
                        "raw_sha256": "wrong"}],
        }
    }
    checkpoint = tmp_path / "tap.partial.json"
    contract = {"items": 1}
    atomic_json_write(
        checkpoint,
        checkpoint_payload(contract, result, "2026-09-19T00:00:00Z",
                           status="in_progress"),
    )
    with pytest.raises(RuntimeError, match="invalid raw evidence"):
        load_checkpoint(checkpoint, contract, ["t::0"])


@pytest.mark.skipif(not ARTIFACT.exists(), reason="capacity-gate artifact not present")
def test_published_artifact_holds_unclipped_rows():
    """Guard the artifact itself, not only the function that writes it."""
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for arm, rows in data["detail"].items():
        assert rows, f"{arm} has no detail rows"
        for row in rows:
            assert "raw_sha256" in row, f"{arm} row {row['id']} has no raw hash"
            assert row["raw_sha256"] == hashlib.sha256(
                row["raw"].encode("utf-8")).hexdigest(), (
                f"{arm} row {row['id']} hash does not match its retained text")
        clipped = sum(1 for row in rows if row.get("raw_chars", 0) == 300)
        assert clipped == 0, f"{arm} still has {clipped} rows clipped at 300 chars"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="capacity-gate artifact not present")
def test_both_conditions_present_for_every_arm():
    """Gate 1 needs TAP-mut for each arm, or code sensitivity is unmeasurable."""
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    arms = {key.split("::")[0] for key in data["detail"]}
    for arm in arms:
        for condition in ("TAP-ref", "TAP-mut"):
            assert f"{arm}::{condition}" in data["detail"], (
                f"{arm} is missing {condition}; code sensitivity cannot be measured")
