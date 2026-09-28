"""Phase 1 causal ledger: exclusive/exhaustive categories and faithful instrumentation."""
from __future__ import annotations

import itertools

import pytest

from harness.causal_ledger import (
    CATEGORY_DEFINITIONS, TERMINAL_CATEGORIES, classify_terminal, instrument, measure_function,
    static_facts,
)
from harness.safe_execution import classify_assertions

FLAGS = ("parse_valid", "duplicate", "policy_valid", "reference_valid", "killed",
         "executed", "oracle_holds", "discriminating")


def test_every_flag_combination_gets_exactly_one_known_category():
    seen = set()
    for values in itertools.product((False, True), repeat=len(FLAGS)):
        category = classify_terminal(**dict(zip(FLAGS, values)))
        assert category in TERMINAL_CATEGORIES
        seen.add(category)
    assert seen == set(TERMINAL_CATEGORIES)           # every category is reachable
    assert set(CATEGORY_DEFINITIONS) == set(TERMINAL_CATEGORIES)


@pytest.mark.parametrize("flags, expected", [
    (dict(parse_valid=False), "parse_failure"),
    (dict(duplicate=True, killed=True, reference_valid=True), "duplicate"),
    (dict(reference_valid=True, killed=True, discriminating=False), "valid_kill"),
    (dict(executed=False), "execution_failure"),
    (dict(discriminating=False, reference_valid=True), "non_discriminating_input"),
    (dict(oracle_holds=False), "discriminating_wrong_oracle"),
    (dict(policy_valid=False), "correct_content_invalid_contract"),
    (dict(reference_valid=True), "valid_non_kill"),
])
def test_precedence(flags, expected):
    base = dict(parse_valid=True, duplicate=False, policy_valid=True, reference_valid=False,
                killed=False, executed=True, oracle_holds=True, discriminating=True)
    assert classify_terminal(**{**base, **flags}) == expected


REFERENCE = "def f(x):\n    return x + 1 if x > 2 else x\n"
MUTANT = "def f(x):\n    return x + 1 if x >= 2 else x\n"          # differs only at x == 2
RECURSIVE_REF = "def f(n):\n    return 0 if n <= 0 else 1 + f(n - 1)\n"


def _run(code, golden=REFERENCE, mutant=MUTANT):
    row = classify_assertions([instrument(code, "f")], golden, mutant)[0]
    return row["golden"]["result"], (row["mutant"] or {}).get("result")


def test_instrumentation_records_calls_and_does_not_abort_on_failure():
    ref, mut = _run("assert f(2) == 99")
    assert ref["asserts"] == [[1, False]] and ref["calls"][0][:2] == ["ok", "2"]
    assert mut["calls"][0][:2] == ["ok", "3"]


def test_recursion_depth_is_unchanged_by_the_wrapper():
    # 150 levels fit under the worker's recursion limit only if the wrapper
    # does not add a frame to every recursive call.
    ref, _ = _run("assert f(150) == 150", RECURSIVE_REF, RECURSIVE_REF)
    assert ref["asserts"] == [[1, True]] and len(ref["calls"]) == 1


def _outcome(code, **kw):
    base = {"rank": 1, "parse_valid": True, "code": code, "policy_valid": True,
            "candidate_shape": "assertion", "execution_valid": True}
    return {**base, **kw}


def _ledger(outcomes):
    record = {"entry_point": "f", "reference_code": REFERENCE, "code_under_test": MUTANT}
    return measure_function({"candidate_outcomes": outcomes}, record)


def test_wrong_oracle_on_a_discriminating_input_is_separated_from_a_benign_input():
    rows = _ledger([
        _outcome("assert f(2) == 99", reference_status="assertion_error"),      # disc, wrong
        _outcome("assert f(0) == 99", reference_status="assertion_error"),      # agree, wrong
        _outcome("assert f(2) == 2", reference_status="pass", reference_valid=True,
                 killed=True),
        _outcome("assert f(2) == 2", reference_status="pass", reference_valid=True,
                 killed=True),                                                  # duplicate
        _outcome("assert f(5) > 0", reference_status="pass", reference_valid=True),
        _outcome("assert f(2) != 7", reference_status="pass", reference_valid=True),
        {"rank": 7, "parse_valid": False, "code": None},
        _outcome("x = f(2)\nassert x == 2", policy_valid=False, execution_valid=False),
    ])
    assert [r["terminal_category"] for r in rows] == [
        "discriminating_wrong_oracle", "non_discriminating_input", "valid_kill", "duplicate",
        "non_discriminating_input", "valid_non_kill", "parse_failure",
        "correct_content_invalid_contract"]
    assert rows[0]["predicted_expected_value"] == "99"
    assert rows[0]["reference_output"] == "2" and rows[0]["buggy_output"] == "3"
    assert all(r["reproduction"]["agrees"] for r in rows if r["reproduction"])


def test_discrimination_after_the_first_failing_assertion_does_not_count():
    code = "def test_a():\n    assert f(0) == 5\n    assert f(2) == 2\n"
    rows = _ledger([_outcome(code, candidate_shape="test_function",
                             reference_status="assertion_error")])
    assert rows[0]["discrimination"]["horizon_calls"] == 1
    assert rows[0]["terminal_category"] == "non_discriminating_input"


def test_an_unsafe_policy_invalid_candidate_is_never_executed():
    rows = _ledger([_outcome("import os\nassert f(2) == 2", policy_valid=False,
                             execution_valid=False)])
    assert rows[0]["execution_detail"].startswith("unsafe:")
    assert rows[0]["terminal_category"] == "execution_failure"


def test_static_facts_name_the_expected_value_side():
    facts = static_facts("assert 7 == f(3)", "f")
    assert facts["assertions"][0]["expected_source"] == "7"
    assert facts["assertion_forms"] == ["compare_Eq"]
