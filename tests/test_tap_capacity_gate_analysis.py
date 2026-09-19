"""Statistical and integrity guards for the TAP capacity gate."""
from __future__ import annotations

import hashlib

import pytest

from scripts.analyse_tap_capacity_gate import (
    Z90,
    Z95,
    mcnemar_exact,
    paired_difference,
    tost,
    validate_artifact,
)


def _row(item_id: str, raw: str = "assert f(1) == 2") -> dict:
    return {
        "id": item_id,
        "raw": raw,
        "raw_chars": len(raw),
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "verdict": "correct",
        "predicted": "2",
    }


def _artifact() -> dict:
    rows = [_row("pilot"), _row("primary")]
    return {
        "detail": {
            f"{arm}::{condition}": [dict(row) for row in rows]
            for arm in ("base", "a431", "a431_sysprompt")
            for condition in ("TAP-ref", "TAP-mut")
        }
    }


def test_equivalence_requires_the_90_percent_interval_inside_margin():
    assert tost(-4.9, 4.9).startswith("EQUIVALENT")
    assert tost(-5.0, 4.0).startswith("NOT EQUIVALENT")
    assert tost(-4.0, 5.0).startswith("NOT EQUIVALENT")


def test_null_difference_does_not_automatically_imply_equivalence():
    # Only two discordant observations: the point estimate is zero, but the
    # interval is far too wide to establish +/-5pp equivalence.
    pairs = [(True, False), (False, True)]
    diff, low, high, _, _ = paired_difference(pairs, z=Z90)
    assert diff == 0
    assert low < -5 and high > 5
    assert tost(low, high).startswith("NOT EQUIVALENT")


def test_95_percent_interval_is_wider_than_tost_interval():
    pairs = [(True, True)] * 90 + [(True, False)] * 5 + [(False, True)] * 5
    _, lo95, hi95, _, _ = paired_difference(pairs, z=Z95)
    _, lo90, hi90, _, _ = paired_difference(pairs, z=Z90)
    assert lo95 < lo90 < hi90 < hi95


def test_exact_mcnemar_is_one_for_no_or_balanced_discordance():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0


def test_integrity_validation_accepts_all_six_complete_conditions():
    indexed = validate_artifact(
        _artifact(), {"pilot_ids": ["pilot"], "pilot_n": 1}
    )
    assert len(indexed) == 6


def test_integrity_validation_refuses_missing_condition():
    artifact = _artifact()
    del artifact["detail"]["a431_sysprompt::TAP-mut"]
    with pytest.raises(ValueError, match="conditions mismatch"):
        validate_artifact(artifact, {"pilot_ids": ["pilot"], "pilot_n": 1})


def test_integrity_validation_refuses_clipped_or_modified_raw_evidence():
    artifact = _artifact()
    artifact["detail"]["base::TAP-ref"][0]["raw"] += " modified"
    with pytest.raises(ValueError, match="raw hash mismatch"):
        validate_artifact(artifact, {"pilot_ids": ["pilot"], "pilot_n": 1})
