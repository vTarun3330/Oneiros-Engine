"""Value-free assertions must be value-free, legal, and actually discriminating.

Three properties matter and each is checked against a reference/mutant pair
whose relationship is known by construction:

* a proposal never names an expected value - that is the entire point;
* a proposal passes the FROZEN candidate policy, or the measured ceiling is one
  the pipeline could not reach;
* a proposal built from a relation the mutant breaks actually kills it.
"""
from __future__ import annotations

import ast

from harness.candidate_policy import validate_generated_test
from harness.metamorphic_relations import (
    RELATION_NAMES, extract_argument_tuples, propose,
)
from harness.safe_execution import classify_assertions

#: Sorting is idempotent and order-insensitive. The mutant drops the last
#: element, which breaks length preservation and idempotence but leaves plenty
#: of single-value assertions still passing - so the pair separates a relation
#: that works from one that merely looks reasonable.
REFERENCE = "def tidy(xs):\n    return sorted(xs)\n"
MUTANT = "def tidy(xs):\n    return sorted(xs)[:-1]\n"


def test_no_proposal_contains_a_literal_expected_value():
    """A proposal that names a value has failed at its only job."""
    for arguments in (["[3, 1, 2]"], ["1", "2"], ["'abc'"]):
        for proposal in propose("f", arguments):
            tree = ast.parse(proposal["code"])
            # Every constant in the assertion must have come from the caller's
            # own arguments; none may be an answer the proposer invented.
            source = " ".join(arguments)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and not isinstance(
                        node.value, bool):
                    assert str(node.value) in source or node.value in (-1, 1), (
                        f"{proposal['code']} names a value not present in "
                        f"its own arguments"
                    )


def test_every_proposal_passes_the_frozen_candidate_policy():
    for arguments in (["[3, 1, 2]"], ["1", "2"], ["'abc'"], ["5"]):
        for proposal in propose("tidy", arguments):
            result = validate_generated_test(proposal["code"], "tidy", True)
            assert result.valid, f"{proposal['code']}: {result.reason}"


def test_a_broken_relation_kills_the_mutant():
    """Length preservation holds on sorted() and fails on the truncating mutant."""
    proposals = propose("tidy", ["[3, 1, 2]"])
    by_relation = {p["relation"]: p["code"] for p in proposals}
    assertion = by_relation["length_preservation"]
    row = classify_assertions([assertion], REFERENCE, MUTANT, 5.0)[0]
    assert row["valid"], "the relation must hold on the reference"
    assert row["killed"], "the relation must fail on the mutant"


def test_a_relation_the_mutant_preserves_does_not_kill():
    """Otherwise every relation would 'work' and the ceiling would be noise."""
    # Sorting the argument first changes nothing for either implementation.
    proposals = propose("tidy", ["[3, 1, 2]"])
    by_relation = {p["relation"]: p["code"] for p in proposals}
    row = classify_assertions([by_relation["list_order_invariance"]],
                              REFERENCE, MUTANT, 5.0)[0]
    assert row["valid"]
    assert not row["killed"]


def test_proposals_cover_the_declared_relation_names():
    produced = {p["relation"] for p in propose("f", ["[1, 2]"])}
    produced |= {p["relation"] for p in propose("f", ["1", "2"])}
    unknown = produced - set(RELATION_NAMES)
    assert not unknown, f"undeclared relations: {unknown}"
    assert "idempotence" in produced
    assert "argument_permutation" in produced


def test_permutation_is_only_proposed_when_it_changes_the_call():
    """f(1, 1) swapped is f(1, 1); proposing it would pad the ceiling."""
    codes = [p["code"] for p in propose("f", ["1", "1"])
             if p["relation"] == "argument_permutation"]
    assert codes == []


def test_argument_tuples_come_from_the_candidate_not_from_us():
    candidate = "assert tidy([3, 1, 2]) == [1, 2, 3]"
    assert extract_argument_tuples(candidate, "tidy") == [["[3, 1, 2]"]]


def test_argument_tuples_are_deduplicated_and_skip_keyword_calls():
    candidate = ("assert tidy([1]) == tidy([1])\n"
                 "assert tidy(xs=[2]) == [2]\n")
    assert extract_argument_tuples(candidate, "tidy") == [["[1]"]]


def test_unparseable_candidates_yield_nothing_rather_than_raising():
    assert extract_argument_tuples("assert tidy([1,", "tidy") == []


def test_no_arguments_means_no_proposals():
    assert propose("f", []) == []
    assert propose("", ["1"]) == []
