"""The repair loop must be useful AND unable to cheat.

Half of these tests check that the feedback is correct. The other half check
that it is impossible for the feedback to carry anything derived from the
reference implementation - because a repair loop that quietly leaks the oracle
would produce a spectacular kill rate and mean nothing at all.
"""
from __future__ import annotations

import pytest

from harness.verifier_guided_repair import (
    FORBIDDEN_IN_FEEDBACK, OBSERVATIONS, assert_feedback_is_clean,
    build_repair_prompt, feedback_sentence, observe, plan_repairs,
)

#: What the model is shown. There is no reference anywhere in this module, on
#: purpose: the loop must work with only this.
CODE_UNDER_TEST = "def add(a, b):\n    return a - b\n"
PROMPT = "Write a test for add.\n\ndef add(a, b): ..."


def test_a_passing_assertion_is_reported_as_unable_to_reveal_a_defect():
    result = observe("assert add(5, 3) == 2", "add", CODE_UNDER_TEST)
    assert result["observation"] == "passes_on_code_under_test"
    assert "cannot reveal any defect" in feedback_sentence(result)


def test_a_failing_assertion_is_reported_as_defect_revealing():
    result = observe("assert add(5, 3) == 8", "add", CODE_UNDER_TEST)
    assert result["observation"] == "fails_on_code_under_test"


def test_an_unparseable_candidate_is_reported_as_such():
    result = observe("assert add(5, ", "add", CODE_UNDER_TEST)
    assert result["observation"] in {"parse_error", "policy_rejected"}
    assert result["executable"] is False


def test_a_raising_candidate_reports_the_exception():
    result = observe("assert add(5) == 1", "add", CODE_UNDER_TEST)
    assert result["observation"] == "raises_on_code_under_test"
    assert "TypeError" in result["detail"]
    assert "TypeError" in feedback_sentence(result)


def test_every_observation_has_a_sentence():
    for observation in OBSERVATIONS:
        sentence = feedback_sentence({"observation": observation, "detail": ""})
        assert sentence and sentence != "Your previous test could not be assessed."


# --------------------------------------------------------------------------
# The boundary.
# --------------------------------------------------------------------------

def test_no_feedback_sentence_mentions_hidden_information():
    for observation in OBSERVATIONS:
        sentence = feedback_sentence({"observation": observation, "detail": ""})
        assert_feedback_is_clean(sentence)


def test_the_guard_actually_rejects_a_leak():
    """If this passes trivially the other boundary tests prove nothing."""
    with pytest.raises(ValueError):
        assert_feedback_is_clean(
            "Your test does not pass on the reference implementation.")
    with pytest.raises(ValueError):
        assert_feedback_is_clean("Your test killed the mutant.")


def test_kill_status_is_never_a_feedback_category():
    """Kill status comes from the hidden reference, so it may not be observed."""
    joined = " ".join(OBSERVATIONS)
    assert "kill" not in joined
    assert "reference" not in joined


def test_observe_has_no_reference_parameter():
    """Structural, not textual: nothing hidden is in scope to leak."""
    import inspect
    parameters = set(inspect.signature(observe).parameters)
    assert parameters == {"candidate", "entry_point", "code_under_test",
                          "allow_test_function", "timeout"}
    assert not parameters & {"reference", "golden_code", "reference_code",
                             "mutant_code"}


def test_an_exception_detail_is_not_scanned_for_forbidden_words():
    """A user function whose error text says "reference" is not a leak.

    The detail comes from running against the code the model was already
    shown, so it cannot tell the model anything new - but scanning it would
    abort a legitimate repair round.
    """
    sentence = ("Running your previous test against the code under test raised "
                "KeyError: reference not found. Call the function the way its "
                "signature allows.")
    assert_feedback_is_clean(sentence, detail="KeyError: reference not found")


def test_repair_prompt_reuses_the_original_prompt_verbatim():
    observation = observe("assert add(5, 3) == 2", "add", CODE_UNDER_TEST)
    prompt = build_repair_prompt(PROMPT, "assert add(5, 3) == 2", observation)
    assert prompt.startswith(PROMPT.rstrip())
    assert "assert add(5, 3) == 2" in prompt
    assert_feedback_is_clean(prompt)


def test_plan_repairs_does_not_skip_already_failing_candidates():
    """Skipping them would bias the round toward unchecked oracles."""
    planned = plan_repairs(
        ["assert add(5, 3) == 8", "assert add(5, 3) == 2"],
        "add", CODE_UNDER_TEST, PROMPT)
    assert len(planned) == 2
    assert {row["observation"] for row in planned} == {
        "fails_on_code_under_test", "passes_on_code_under_test"}
    for row in planned:
        assert_feedback_is_clean(row["repair_prompt"], row["detail"])


def test_forbidden_list_covers_the_things_that_matter():
    for token in ("reference", "mutant", "killed", "gold"):
        assert token in FORBIDDEN_IN_FEEDBACK
