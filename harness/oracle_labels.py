"""One candidate, one label, with a stated reason and an ordered precedence.

The builder this replaces assigned a MAJORITY label per function, which threw
away the thing the dataset is for: a function with three fabricated-API
candidates, four wrong oracles and one kill has seven different lessons in it,
and "wrong_oracle" describes none of them.

Precedence matters because several conditions genuinely co-occur. A candidate
whose call does not distinguish reference from mutant AND asserts the wrong
value is BOTH wrong_input and wrong_oracle - but only the input failure is
worth acting on, because no correction to the expected value could ever make
that candidate kill. So wrong_input is primary and wrong_oracle is recorded
beside it, rather than one of them being silently dropped.

Nothing here guesses. A candidate that cannot be resolved becomes `uncertain`,
which is excluded from positive supervision by construction.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

VALID_KILLING = "valid_killing_test"
VALID_NON_KILLING = "valid_non_killing_test"
WRONG_INPUT = "wrong_input"
WRONG_ORACLE = "wrong_oracle"
SYNTAX_OR_POLICY_INVALID = "syntax_or_policy_invalid"
FABRICATED_API = "fabricated_api"
SEMANTIC_EXECUTION_ERROR = "semantic_execution_error"
HARNESS_ENVIRONMENT_FAILURE = "harness_environment_failure"
UNCERTAIN = "uncertain"

LABELS = (
    VALID_KILLING, VALID_NON_KILLING, WRONG_INPUT, WRONG_ORACLE,
    SYNTAX_OR_POLICY_INVALID, FABRICATED_API, SEMANTIC_EXECUTION_ERROR,
    HARNESS_ENVIRONMENT_FAILURE, UNCERTAIN,
)

#: Labels that may never become a positive SFT target, whatever else is true.
NEVER_POSITIVE = frozenset({
    WRONG_INPUT, WRONG_ORACLE, SYNTAX_OR_POLICY_INVALID, FABRICATED_API,
    SEMANTIC_EXECUTION_ERROR, HARNESS_ENVIRONMENT_FAILURE, UNCERTAIN,
})

HARNESS_STATUSES = ("worker_error", "system_exit", "keyboard_interrupt")

#: A TypeError can mean either "you called an interface that does not exist
#: that way" or "you passed a value of the wrong type to an interface that
#: does". Only the first is fabrication, and the message is what separates
#: them - so the patterns are explicit rather than the exception type alone.
_ARITY_OR_KEYWORD = re.compile(
    r"takes \d+ positional argument|"
    r"missing \d+ required (positional|keyword-only) argument|"
    r"unexpected keyword argument|"
    r"takes no arguments|"
    r"got multiple values for argument|"
    r"argument after \*\* must be a mapping",
    re.IGNORECASE,
)

_FABRICATED_EXCEPTIONS = frozenset({
    "NameError", "AttributeError", "ImportError", "ModuleNotFoundError",
    "NotImplementedError",
})


def _exception_of(error: str) -> str:
    return str(error or "").split(":", 1)[0].strip()


def classify_candidate(
    outcome: Mapping[str, Any],
    *,
    distinguishing_call: bool | None = None,
) -> dict[str, Any]:
    """Label one candidate.

    ``distinguishing_call`` is the execution-backed answer to "does this
    candidate's own call separate the reference from the mutant". It is
    supplied only for candidates that failed their assertion on the reference,
    because that is the only case where the question decides the label. None
    means the question could not be answered, which yields ``uncertain``
    rather than a guess.
    """
    status = str(outcome.get("reference_status") or "")
    error = str(outcome.get("reference_error") or "")
    exception = _exception_of(error)

    # 1. The harness, not the candidate. Highest precedence: a candidate that
    #    was never scored has no other property worth reading.
    if status in HARNESS_STATUSES:
        return {"label": HARNESS_ENVIRONMENT_FAILURE, "secondary_label": None,
                "reason": f"execution harness returned {status}; never scored"}

    # 2. It never became a candidate at all.
    if not outcome.get("parse_valid"):
        return {"label": SYNTAX_OR_POLICY_INVALID, "secondary_label": None,
                "reason": "raw output did not parse into a candidate"}
    if not outcome.get("policy_valid"):
        return {"label": SYNTAX_OR_POLICY_INVALID, "secondary_label": None,
                "reason": "rejected by the candidate policy: "
                          + str(outcome.get("policy_error") or "unstated")}

    # 3. It ran on the reference and was valid there.
    if outcome.get("reference_valid"):
        if outcome.get("killed"):
            return {"label": VALID_KILLING, "secondary_label": None,
                    "reason": "valid on the reference and distinguishes the mutant"}
        return {"label": VALID_NON_KILLING, "secondary_label": None,
                "reason": "valid on the reference but the mutant survives it"}

    # 4. It failed its assertion on the reference: the expected value is wrong.
    #    Whether that is worth correcting depends on the CALL, which only
    #    execution can answer.
    if status == "assertion_error":
        if distinguishing_call is True:
            return {"label": WRONG_ORACLE, "secondary_label": None,
                    "reason": "the call distinguishes reference from mutant; "
                              "only the asserted value is wrong"}
        if distinguishing_call is False:
            # Both conditions hold. Input is primary: no correction to the
            # expected value could make this candidate kill.
            return {"label": WRONG_INPUT, "secondary_label": WRONG_ORACLE,
                    "reason": "reference and mutant agree on this call, so no "
                              "expected value could make it kill; the asserted "
                              "value is also wrong"}
        return {"label": UNCERTAIN, "secondary_label": None,
                "reason": "assertion failed on the reference but the call could "
                          "not be replayed to decide input versus oracle"}

    # 5. It raised. Fabricated interface, or a real value/type failure?
    if exception in _FABRICATED_EXCEPTIONS:
        return {"label": FABRICATED_API, "secondary_label": None,
                "reason": f"{exception}: the referenced interface does not exist"}
    if exception == "TypeError":
        if _ARITY_OR_KEYWORD.search(error):
            return {"label": FABRICATED_API, "secondary_label": None,
                    "reason": "TypeError on arity or keyword: the call does not "
                              "match the function's signature"}
        return {"label": SEMANTIC_EXECUTION_ERROR, "secondary_label": None,
                "reason": "TypeError from a value, not from the signature"}
    if status == "timeout":
        return {"label": SEMANTIC_EXECUTION_ERROR, "secondary_label": None,
                "reason": "the candidate did not terminate within the budget"}
    if status == "source_policy_error":
        return {"label": SYNTAX_OR_POLICY_INVALID, "secondary_label": None,
                "reason": "source policy refused the candidate at execution time"}
    if exception:
        return {"label": SEMANTIC_EXECUTION_ERROR, "secondary_label": None,
                "reason": f"{exception} raised on the reference"}

    return {"label": UNCERTAIN, "secondary_label": None,
            "reason": f"unresolved reference_status {status!r} with no exception"}


def needs_call_replay(outcome: Mapping[str, Any]) -> bool:
    """Only assertion failures need the expensive input/oracle distinction."""
    return (str(outcome.get("reference_status") or "") == "assertion_error"
            and bool(outcome.get("parse_valid"))
            and bool(outcome.get("policy_valid"))
            and not outcome.get("reference_valid"))
