"""A five-assertion completion must not be collapsed to its first assertion.

The dataset produces one broad test per lineage, killing a mean of 8.43
sibling mutants. The frozen evaluation parser scans a model's output for the
first line starting with ``assert `` and discards the rest, so that capability
has never been measurable in the MODEL - only in the dataset.

These tests pin the successor path end to end. Retaining the text is not
enough: the assertions have to REACH the executor, and a mutant that only the
fifth assertion catches has to actually die. A parser that kept all five lines
and an executor that ran only the first would pass a text-level test and still
measure nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.generator import Phi3Generator
from harness.candidate_policy import (
    count_assertions, executable_candidate, validate_generated_test,
)
from metrics.research_evaluation import evaluate_candidate_slots

CLAMP = (
    "def clamp(value, low, high):\n"
    "    if value < low:\n"
    "        return low\n"
    "    if value > high:\n"
    "        return high\n"
    "    return value\n"
)

#: Differs from the reference ONLY at the upper bound, so the first four
#: assertions below pass on it and only the fifth catches it. If anything in
#: the path stops after the first assertion, this mutant survives.
UPPER_BOUND_MUTANT = (
    "def clamp(value, low, high):\n"
    "    if value < low:\n"
    "        return low\n"
    "    if value > high:\n"
    "        return value\n"
    "    return value\n"
)

FIVE = (
    "def test_clamp_boundaries():\n"
    "    assert clamp(5, 0, 10) == 5\n"
    "    assert clamp(0, 0, 10) == 0\n"
    "    assert clamp(-1, 0, 10) == 0\n"
    "    assert clamp(10, 0, 10) == 10\n"
    "    assert clamp(11, 0, 10) == 10\n"
)


def _parse(mode: str, output: str):
    generator = Phi3Generator.__new__(Phi3Generator)
    generator.stats = {
        "total_generated": 0, "valid_generated": 0, "invalid_generated": 0,
    }
    generator.parse_mode = mode
    return generator._parse_output(output, "clamp", "clamp")


def test_the_frozen_parser_collapses_five_assertions_to_one():
    """Documents the defect the successor exists to fix."""
    parsed = _parse("first_assertion", FIVE)
    assert parsed.input_code == "assert clamp(5, 0, 10) == 5"
    assert count_assertions(parsed.input_code) == 1


def test_the_successor_parser_keeps_all_five():
    parsed = _parse("whole_output", FIVE)
    assert count_assertions(parsed.input_code) == 5
    assert parsed.is_valid, parsed.parse_error


def test_all_five_assertions_reach_the_executor():
    """Text retention is not evaluation; the call has to be appended."""
    policy = validate_generated_test(FIVE, "clamp", allow_test_function=True)
    executable = executable_candidate(FIVE, policy.shape)
    assert count_assertions(executable) == 5
    assert executable.rstrip().endswith("test_clamp_boundaries()"), (
        "a defined-but-never-called test function asserts nothing and passes "
        "on the reference and the mutant alike"
    )


def test_a_mutant_only_the_fifth_assertion_catches_is_killed():
    """The end-to-end proof. Under the frozen parser this mutant survives."""
    outcomes = evaluate_candidate_slots(
        [{"code": FIVE, "parse_valid": True}], CLAMP, UPPER_BOUND_MUTANT,
        "clamp", allow_test_function=True,
    )
    outcome = outcomes[0]
    assert outcome["policy_valid"], outcome.get("policy_error")
    assert outcome["reference_valid"], "the test must pass on correct code"
    assert outcome["killed"], (
        "only the fifth assertion distinguishes this mutant; if it is not "
        "executed the multi-mutant capability is unmeasurable"
    )
    assert outcome["assertion_count"] == 5
    assert outcome["candidate_shape"] == "test_function"


def test_the_same_mutant_survives_a_single_first_assertion():
    """Confirms the mutant is only catchable by the later assertions."""
    outcomes = evaluate_candidate_slots(
        [{"code": "assert clamp(5, 0, 10) == 5", "parse_valid": True}],
        CLAMP, UPPER_BOUND_MUTANT, "clamp", allow_test_function=True,
    )
    assert outcomes[0]["reference_valid"]
    assert not outcomes[0]["killed"]
    assert outcomes[0]["assertion_count"] == 1


def test_assertion_count_is_recorded_for_every_candidate():
    """Without it, a collapsed candidate is indistinguishable in the artifact."""
    outcomes = evaluate_candidate_slots([
        {"code": FIVE, "parse_valid": True},
        {"code": "assert clamp(1, 0, 10) == 1", "parse_valid": True},
        {"code": "def broken(", "parse_valid": True},
    ], CLAMP, UPPER_BOUND_MUTANT, "clamp", allow_test_function=True)
    assert [o["assertion_count"] for o in outcomes] == [5, 1, 0]


def test_a_helper_function_beside_the_test_is_refused():
    """P0's capability rule: one self-contained test, nothing else."""
    with_helper = (
        "def helper(x):\n    return x\n\n"
        "def test_clamp():\n    assert clamp(helper(5), 0, 10) == 5\n"
    )
    policy = validate_generated_test(with_helper, "clamp", allow_test_function=True)
    assert not policy.valid


def test_an_import_inside_the_test_is_refused():
    with_import = (
        "def test_clamp():\n"
        "    import os\n"
        "    assert clamp(5, 0, 10) == 5\n"
    )
    policy = validate_generated_test(with_import, "clamp", allow_test_function=True)
    assert not policy.valid
