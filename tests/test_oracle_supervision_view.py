"""Structured supervision must stay legal, stay verified, and invent nothing.

The risk in restating the expected value is that it becomes a place to write
something that was never checked. So every test here is about provenance: the
expected line comes from executing the reference, the completion still passes
the frozen policy, and anything that stops killing is dropped rather than
taught.
"""
from __future__ import annotations

import pytest

from harness.candidate_policy import count_assertions, validate_generated_test
from scripts.build_oracle_supervision_view import (
    LABELS, MAX_REPEATS, MIN_CLEAN_SHARE, _verify, structure,
)

REFERENCE = "def twice(n):\n    return n * 2\n"
MUTANT = "def twice(n):\n    return n * 3\n"


def _job(completion):
    return {"completion": completion, "entry_point": "twice",
            "reference": REFERENCE, "mutant": MUTANT, "timeout": 5.0}


def test_the_expected_line_is_executed_not_copied_from_the_assertion():
    """A wrong assertion must not be able to state a matching expectation.

    If the expected line were parsed out of the assertion, a completion
    asserting the wrong value would cheerfully document the wrong value. It is
    obtained by running the reference, so it disagrees instead.
    """
    built = structure("assert twice(5) == 999", "twice", REFERENCE, 5.0)
    assert built["expected"] == "10"
    assert "999" not in built["structured"].split("assert")[0]


def test_the_structured_completion_still_passes_the_frozen_policy():
    built = structure("assert twice(5) == 10", "twice", REFERENCE, 5.0)
    result = validate_generated_test(built["structured"], "twice", True)
    assert result.valid
    assert result.shape == "assertion"


def test_comments_do_not_add_assertions():
    built = structure("assert twice(5) == 10", "twice", REFERENCE, 5.0)
    assert count_assertions(built["structured"]) == 1


def test_the_three_fields_appear_in_the_order_the_model_must_think_in():
    built = structure("assert twice(5) == 10", "twice", REFERENCE, 5.0)
    lines = built["structured"].splitlines()
    assert lines[0].startswith("# input:")
    assert lines[1].startswith("# expected:")
    assert lines[2].startswith("assert ")


def test_a_verified_killing_completion_is_usable():
    result = _verify(_job("assert twice(5) == 10"))
    assert result["usable"] is True


def test_a_completion_that_stops_killing_is_dropped_not_taught():
    # twice(0) is 0 on both implementations, so this distinguishes nothing.
    result = _verify(_job("assert twice(0) == 0"))
    assert result["usable"] is False
    assert result["reason"] == "does_not_kill_displayed_target"


def test_a_completion_invalid_on_the_reference_is_dropped():
    result = _verify(_job("assert twice(5) == 999"))
    assert result["usable"] is False
    assert result["reason"] == "not_reference_valid"


def test_a_multi_assertion_completion_is_not_structured():
    """The format promises one bounded assertion; two would break that."""
    assert structure("assert twice(1) == 2\nassert twice(2) == 4",
                     "twice", REFERENCE, 5.0) is None


def test_a_completion_that_never_calls_the_entry_point_is_not_structured():
    assert structure("assert 1 == 1", "twice", REFERENCE, 5.0) is None


def test_an_unreachable_expected_value_is_not_invented():
    """If the reference cannot be executed on that call, emit nothing."""
    assert structure("assert twice('a', 'b') == 1", "twice", REFERENCE, 5.0) is None


def test_the_targeted_failure_is_oversampled_but_capped():
    assert MAX_REPEATS["wrong_oracle"] > MAX_REPEATS.get("wrong_input", 1)
    assert max(MAX_REPEATS.values()) <= 3, "oversampling must not become a monoculture"


def test_clean_examples_are_required():
    assert 0 < MIN_CLEAN_SHARE < 1


def test_the_label_set_covers_what_the_review_asked_for():
    for required in ("wrong_oracle", "wrong_input", "syntax", "api_misuse",
                     "reference_failure", "non_kill"):
        assert required in LABELS


def test_wrong_oracle_and_wrong_input_are_distinct_labels():
    """Collapsing them is the defect this whole line of work corrects."""
    assert "wrong_oracle" in LABELS and "wrong_input" in LABELS
    assert LABELS.index("wrong_oracle") != LABELS.index("wrong_input")
