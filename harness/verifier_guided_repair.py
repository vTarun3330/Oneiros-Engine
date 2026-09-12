"""Execute the model's own candidate, tell it what happened, let it try again.

Every intervention in this project so far has changed the WEIGHTS. This one
changes nothing: it runs the candidate the model already produced, hands back
what a developer would have seen, and asks for one repair. If it works it costs
inference time instead of GPU training, and it applies to the untrained base
model as readily as to any adapter.

THE INFORMATION BOUNDARY, which is the whole design.

At inference the model is shown one function - the code under test - plus a
specification. It is legitimate for it to run its own test against the code it
was just shown, because that is what any developer does and it requires nothing
hidden. Everything else is off limits:

* the reference implementation is never executed here and never quoted;
* whether the candidate KILLS is never revealed - kill status is computed from
  the hidden reference, so it is evaluator information;
* the mutation diff, sibling mutants and gold tests never appear.

So the loop can say "your assertion passes on the code under test, so it cannot
reveal a defect in it" and "your test raises TypeError". It can never say "your
expected value is wrong", because knowing that requires the reference.

That asymmetry is the honest limitation and it is worth stating plainly: this
loop can fix INPUT selection and executability, and it cannot directly fix
ORACLE errors, which measurement puts at roughly 73% of the dominant mbpp
failure. It is expected to help, not to close the gap.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from harness.candidate_policy import executable_candidate, validate_generated_test
from harness.safe_execution import execute_assertions

#: What the loop is allowed to observe. Anything not on this list is a defect.
OBSERVATIONS = (
    "parse_error",
    "policy_rejected",
    "raises_on_code_under_test",
    "passes_on_code_under_test",
    "fails_on_code_under_test",
)

#: Tokens that must never appear in model-visible feedback. A test asserts this.
FORBIDDEN_IN_FEEDBACK = (
    "reference", "golden", "mutant", "mutation", "sibling", "gold",
    "killed", "kill@", "expected value is", "correct implementation",
)


def observe(candidate: str, entry_point: str, code_under_test: str,
            allow_test_function: bool = True,
            timeout: float = 5.0) -> dict[str, Any]:
    """What running this candidate against the SHOWN code produces.

    The reference is not a parameter of this function, which is the structural
    reason it cannot leak: there is nothing hidden in scope to leak.

    ``allow_test_function`` defaults to True so the loop scores candidates
    under the successor (whole-output) protocol rather than the frozen
    first-assertion parser, which acted as a repair step of its own.
    """
    validation = validate_generated_test(candidate, entry_point,
                                         allow_test_function)
    if not validation.valid:
        reason = str(getattr(validation, "reason", "") or "rejected")
        observation = ("parse_error" if "syntax" in reason.lower()
                       or "parse" in reason.lower() else "policy_rejected")
        return {"observation": observation, "detail": reason,
                "executable": False}

    executable = executable_candidate(candidate, validation.shape)
    rows = execute_assertions([executable], code_under_test, timeout)
    row = rows[0] if rows else {}
    if row.get("ok"):
        return {"observation": "passes_on_code_under_test", "detail": "",
                "executable": True}

    error = str(row.get("error") or "")
    exception = error.split(":", 1)[0].strip()
    if exception in {"AssertionError", ""} or row.get("status") == "assertion_error":
        return {"observation": "fails_on_code_under_test", "detail": "",
                "executable": True}
    return {"observation": "raises_on_code_under_test", "detail": error,
            "executable": True}


def feedback_sentence(observation: Mapping[str, Any]) -> str:
    """The model-visible sentence. Nothing here is derived from the reference."""
    kind = str(observation.get("observation") or "")
    detail = str(observation.get("detail") or "").strip()

    if kind == "parse_error":
        return ("Your previous test did not parse as Python"
                + (": " + detail if detail else "") + ".")
    if kind == "policy_rejected":
        return ("Your previous test was rejected by the candidate policy"
                + (": " + detail if detail else "")
                + ". Write a single self-contained assertion that calls only "
                  "the function under test, with no imports and no helpers.")
    if kind == "raises_on_code_under_test":
        return ("Running your previous test against the code under test raised "
                + (detail if detail else "an error")
                + ". Call the function the way its signature allows.")
    if kind == "passes_on_code_under_test":
        return ("Your previous test PASSES on the code under test, so it "
                "cannot reveal any defect in it. Choose a different input - one "
                "where you expect the specification and this code to disagree.")
    if kind == "fails_on_code_under_test":
        return ("Your previous test fails on the code under test, which is what "
                "a defect-revealing test should do. Keep that input if the "
                "value you assert is what the specification requires; correct "
                "the asserted value if it is not.")
    return "Your previous test could not be assessed."


def build_repair_prompt(original_prompt: str, candidate: str,
                        observation: Mapping[str, Any]) -> str:
    """The second-round prompt: the task, the attempt, and what happened.

    The original prompt is reused verbatim rather than rebuilt, so a repair
    round cannot silently widen the information the model was given.
    """
    return "\n\n".join([
        original_prompt.rstrip(),
        "You previously answered:",
        candidate.strip(),
        feedback_sentence(observation),
        "Write one corrected test. Output only the test.",
    ])


def assert_feedback_is_clean(text: str, detail: str = "") -> None:
    """Refuse feedback that mentions anything the model may not be told.

    ``detail`` is excluded from the scan rather than scanned and passed. It is
    an exception message produced by running the candidate against the code the
    model was already shown, so it cannot carry information the model lacks -
    while a user function whose own error message contains the word
    "reference" would otherwise abort a legitimate repair round.
    """
    lowered = str(text).lower()
    if detail:
        lowered = lowered.replace(str(detail).lower(), "")
    for token in FORBIDDEN_IN_FEEDBACK:
        if token in lowered:
            raise ValueError(
                f"feedback would leak hidden information (matched {token!r}): "
                f"{text!r}"
            )


def plan_repairs(candidates: Iterable[str], entry_point: str,
                 code_under_test: str, original_prompt: str,
                 allow_test_function: bool = True,
                 timeout: float = 5.0) -> list[dict[str, Any]]:
    """Observe every candidate and build the repair prompt for each.

    Candidates that already fail on the code under test are still repaired:
    failing is necessary for a kill but not sufficient, since the asserted
    value may be wrong on the reference too. Skipping them would bias the round
    towards exactly the candidates whose oracle is least checked.
    """
    planned: list[dict[str, Any]] = []
    for candidate in candidates:
        observation = observe(candidate, entry_point, code_under_test,
                              allow_test_function, timeout)
        sentence = feedback_sentence(observation)
        assert_feedback_is_clean(sentence, observation["detail"])
        planned.append({
            "candidate": candidate,
            "observation": observation["observation"],
            "detail": observation["detail"],
            "repair_prompt": build_repair_prompt(
                original_prompt, candidate, observation),
        })
    return planned
