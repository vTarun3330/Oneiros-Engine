"""The wrong-input / wrong-oracle split must be decided by execution.

The whole value of this refinement is that it separates two failures that look
identical in the recorded artifact. A test that only checked the labels exist
would pass while the classifier decided them backwards, so every case here
builds a reference and a mutant whose relationship is known by construction.
"""
from __future__ import annotations

import json

import pytest

from harness.oracle_diagnosis import (
    BENIGN_INPUT, BOTH_ERROR, NO_CALL, ORACLE_ERROR, call_expression,
    diagnose, refine,
)

REFERENCE = "def f(n):\n    return n * 2\n"
#: Differs from the reference only for n > 10, so the boundary is known and a
#: probe on either side of it has a predictable verdict.
MUTANT = "def f(n):\n    return n * 2 if n <= 10 else n * 3\n"


def test_a_probe_where_reference_and_mutant_differ_is_an_oracle_error():
    # f(20) is 40 on the reference and 60 on the mutant, so this candidate is
    # one correct value away from a kill.
    candidate = "assert f(20) == 999"
    result = diagnose(candidate, "f", REFERENCE, MUTANT)
    assert result["refined_label"] == ORACLE_ERROR
    assert result["probes_distinguishing_input"] is True
    assert result["call"] == "f(20)"
    assert result["reference_behaviour"] != result["mutant_behaviour"]


def test_a_probe_where_they_agree_is_an_input_error():
    # f(2) is 4 on both, so no expected value could ever have killed this.
    candidate = "assert f(2) == 999"
    result = diagnose(candidate, "f", REFERENCE, MUTANT)
    assert result["refined_label"] == BENIGN_INPUT
    assert result["probes_distinguishing_input"] is False
    assert result["reference_behaviour"] == result["mutant_behaviour"]


def test_the_two_cases_are_not_collapsed():
    """The defect this module exists to prevent: one label for both."""
    oracle = diagnose("assert f(20) == 0", "f", REFERENCE, MUTANT)
    benign = diagnose("assert f(2) == 0", "f", REFERENCE, MUTANT)
    assert oracle["refined_label"] != benign["refined_label"]


def test_identical_exceptions_on_both_are_not_an_oracle_error():
    candidate = "assert f('x', 'y') == 1"
    result = diagnose(candidate, "f", REFERENCE, MUTANT)
    assert result["refined_label"] == BOTH_ERROR


def test_a_candidate_that_never_calls_the_entry_point_is_labelled_as_such():
    result = diagnose("assert 1 == 1", "f", REFERENCE, MUTANT)
    assert result["refined_label"] == NO_CALL
    assert result["call"] is None


def test_repr_comparison_separates_one_from_true():
    """1 and True must be two behaviours, not one.

    A mutant returning the truthy int where the reference returns the bool is a
    real difference; comparing values with == would call it agreement and hide
    a killable mutant inside the benign bucket.
    """
    reference = "def g(n):\n    return True\n"
    mutant = "def g(n):\n    return 1\n"
    result = diagnose("assert g(0) == 'nope'", "g", reference, mutant)
    assert result["refined_label"] == ORACLE_ERROR


def test_only_wrong_expected_value_is_refined():
    passthrough = refine("syntax_invalid", {"code": "assert f(20) == 1"}, "f",
                         REFERENCE, MUTANT)
    assert passthrough["refined"] is False
    assert passthrough["refined_label"] == "syntax_invalid"

    refined = refine("wrong_expected_value", {"code": "assert f(20) == 1"}, "f",
                     REFERENCE, MUTANT)
    assert refined["refined"] is True
    assert refined["refined_label"] == ORACLE_ERROR


def test_call_expression_finds_the_entry_point_not_the_first_call():
    assert call_expression("assert len(f(3)) == 1", "f") == "f(3)"


@pytest.mark.parametrize("path,expected", [
    ("results/v4_2_oracle_vs_input_base_val_s42.json", {
        "mbpp_oracle_share": 0.707424,
        "mbpp_oracle_recoverable_functions": 207,
    }),
    ("results/v4_2_oracle_vs_input_relearn_val_s42.json", {
        "mbpp_oracle_share": 0.731003,
        "mbpp_oracle_recoverable_functions": 216,
    }),
])
def test_committed_locked_validation_figures_still_match(path, expected):
    """These are the numbers quoted in the report, on the LOCKED panel.

    The earlier 76% headroom figure came from ablation_dev, which overstates
    every arm on this project by about 3.2x. Pinning the locked-panel figures
    here is what stops that substitution happening again.
    """
    from pathlib import Path
    artifact = Path(__file__).resolve().parent.parent / path
    if not artifact.exists():
        pytest.skip(f"{path} not built in this checkout")
    report = json.loads(artifact.read_text(encoding="utf-8"))
    mbpp = report["per_benchmark"]["mbpp"]
    assert mbpp["oracle_share_of_decided"] == expected["mbpp_oracle_share"]
    assert (mbpp["functions_with_an_oracle_recoverable_candidate"]
            == expected["mbpp_oracle_recoverable_functions"])
    assert report["evaluation_split"] == "val"
    assert report["sealed_final_test_accessed"] is False
