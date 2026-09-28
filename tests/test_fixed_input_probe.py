"""Phase 3A fixed-input probe: arm-blind selection, leakage guards, extraction, scoring."""
from __future__ import annotations

import pytest

from harness.fixed_input_probe import (
    CONTROL_LEVEL, build_prompt, extract_answer, perturbed_calls, score_answers,
    select_inputs, upstream_calls, verify_calls,
)

REFERENCE = "def f(x):\n    return x + 1 if x > 2 else x\n"
MUTANT = "def f(x):\n    return x + 1 if x >= 2 else x\n"
RECORD = {"entry_point": "f", "reference_code": REFERENCE, "code_under_test": MUTANT,
          "specification": "Return x plus one when x exceeds two.",
          "tests": [{"code": "assert f(5) == 6"}, {"code": "assert f(1) == 1"}]}


def test_perturbations_are_deterministic_and_exclude_the_original():
    first = perturbed_calls("f(1)")
    assert first == perturbed_calls("f(1)") and "f(1)" not in first
    assert first[:2] == ["f(2)", "f(0)"]


def test_discrimination_is_judged_under_equality_not_repr():
    rows = verify_calls(["f(2)", "f(5)"], REFERENCE, MUTANT)
    assert [r["usable"] for r in rows] == [True, False]
    assert rows[0]["expected_repr"] == "2" and rows[0]["buggy_repr"] == "3"
    equal_values = verify_calls(["g()"], "def g():\n    return 1\n", "def g():\n    return 1.0\n")
    assert equal_values[0]["usable"] is False           # 1 == 1.0: an == oracle cannot tell


def test_novel_inputs_avoid_every_group_call():
    picked = select_inputs(RECORD, set(upstream_calls(RECORD["tests"], "f")))
    assert picked["upstream"] is None                   # neither upstream call discriminates
    assert picked["novel"]["call"] == "f(2)"            # f(1) -> f(2) is the first that does


def test_prompts_never_leak_the_reference_or_a_readable_call():
    prompt = build_prompt(RECORD, "f(2)", "2", "A0")
    assert REFERENCE.strip() not in prompt["user"] and prompt["prefill"].endswith("f(2) == ")
    assert "type int" in build_prompt(RECORD, "f(2)", "2", "A3")["user"]
    assert REFERENCE.strip() in build_prompt(RECORD, "f(2)", "2", "A4")["user"]
    leaky = {**RECORD, "specification": "For example f(2) returns 2."}
    with pytest.raises(ValueError):
        build_prompt(leaky, "f(2)", "2", "A0")


def test_extraction_and_scoring_use_evaluator_semantics():
    assert extract_answer("2\n```", "A0", "f(2)")["answer"] == "2"
    assert extract_answer("ANSWER: 2", CONTROL_LEVEL, "f(2)")["method"] == "strict"
    assert extract_answer("assert f(2) == 3", CONTROL_LEVEL, "f(2)")["method"] == \
        "lenient_assertion"
    assert extract_answer("I think it is two", CONTROL_LEVEL, "f(2)")["answer"] is None
    verdicts = score_answers([{"call": "f(2)", "answer": "2"}, {"call": "f(2)", "answer": "3"},
                              {"call": "f(2)", "answer": "1 + 1"},
                              {"call": "f(2)", "answer": None}], REFERENCE)
    assert [v["correct"] for v in verdicts] == [True, False, True, False]
    assert verdicts[3]["status"] == "nonanswer"
