"""Unit tests for the post-gate trace-pilot diagnosis helpers."""
from __future__ import annotations

import pytest

from scripts.diagnose_execution_trace_pilot import (
    answer_class, exact_gain_upper, newcombe_paired, same_value, track,
)


def _pairs(both, only_a, only_b, neither):
    return ([(True, True)] * both + [(True, False)] * only_a
            + [(False, True)] * only_b + [(False, False)] * neither)


def test_newcombe_matches_independent_computation():
    # 23 correct in both, 2 switched to correct, none lost, 72 wrong in both.
    # Reference values computed step by step at full precision (Wilson 90%
    # bounds [0.173824, 0.314670] and [0.191957, 0.336655], phi 0.946116).
    # The formula subtracts two near-equal terms, so rounded hand arithmetic
    # is not a usable reference: a four-decimal attempt produced -0.30.
    low, high = newcombe_paired(_pairs(23, 0, 2, 72))
    assert low == pytest.approx(-0.5622, abs=1e-3)
    assert high == pytest.approx(4.8595, abs=1e-3)


def test_newcombe_is_not_degenerate_without_discordant_pairs():
    """The frozen Wald interval collapses to [0, 0] here; Newcombe must not."""
    low, high = newcombe_paired(_pairs(26, 0, 0, 71))
    assert low < 0 < high


def test_newcombe_is_symmetric_under_arm_swap():
    low, high = newcombe_paired(_pairs(20, 5, 1, 71))
    swapped_low, swapped_high = newcombe_paired(_pairs(20, 1, 5, 71))
    assert low == pytest.approx(-swapped_high, abs=1e-9)
    assert high == pytest.approx(-swapped_low, abs=1e-9)


def test_exact_gain_bound_rule_of_three_and_monotone():
    # zero gains out of 97: 1 - 0.05 ** (1/97) = 3.04%
    assert exact_gain_upper(0, 97) == pytest.approx(3.04, abs=0.02)
    assert exact_gain_upper(2, 97) > exact_gain_upper(1, 97) > exact_gain_upper(0, 97)
    assert exact_gain_upper(97, 97) == 100.0


def test_same_value_requires_matching_type():
    assert same_value("1", {"type": "int", "literal": "1"})
    assert not same_value("1.0", {"type": "int", "literal": "1"})
    assert not same_value("True", {"type": "int", "literal": "1"})
    assert same_value("'a b'", {"type": "str", "literal": "'a b'"})
    assert not same_value(None, {"type": "str", "literal": "'a'"})


def test_track_separates_intended_shown_and_neither():
    intended = {"type": "int", "literal": "3"}
    actual = {"type": "int", "literal": "4"}
    assert track("3", intended, actual) == "intended"
    assert track("4", intended, actual) == "shown_code_actual"
    assert track("5", intended, actual) == "neither"
    assert track(None, intended, actual) == "no_usable_answer"


def test_answer_class_collapses_format_failures():
    assert answer_class("correct") == "correct"
    assert answer_class("wrong_value") == "wrong_answer"
    assert answer_class("wrong_type") == "wrong_answer"
    for verdict in ("invalid_syntax", "multiple_or_missing_statements",
                    "not_single_equality", "prediction_not_literal", "wrong_call"):
        assert answer_class(verdict) == "no_usable_answer"


def test_tracked_trace_evidence_chain_verifies_on_any_checkout():
    """Analysis -> diagnosis -> decision receipt must verify from tracked bytes.

    These three files are committed and cross-reference each other by hash.
    They are written as LF bytes because .gitattributes stores *.json as LF;
    a CRLF write on Windows would make the recorded hashes fail on a fresh
    clone. Only tracked files are read, so this runs anywhere.
    """
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "results"
    analysis = root / "v4_3_execution_trace_pilot_analysis.json"
    diagnosis = root / "v4_3_execution_trace_pilot_diagnosis.json"
    receipt = root / "v4_3_execution_trace_pilot_decision_receipt.json"
    if not receipt.exists():
        pytest.skip("trace-pilot decision receipt not present")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for path in (analysis, diagnosis, receipt):
        assert b"\r" not in path.read_bytes(), f"{path.name} contains CR bytes"
    diag = json.loads(diagnosis.read_text(encoding="utf-8"))
    rec = json.loads(receipt.read_text(encoding="utf-8"))
    assert diag["inputs"]["analysis_sha256"] == digest(analysis)
    assert rec["analysis"]["sha256"] == digest(analysis)
    assert rec["diagnosis"]["sha256"] == digest(diagnosis)
    assert rec["analysis"]["mechanism_gate_passed"] is False
    assert rec["decision"]["advance"] is False
    assert all(value is False for value in rec["leakage"].values()
               if isinstance(value, bool))
