"""Closed, versioned taxonomy of execution feedback for the repair arm.

Every candidate is assigned exactly one category from a closed set.  A category
is either REPAIRABLE (a demonstrably invalid artifact: the model may be told
what went wrong and asked for one repair) or RETAINED (the candidate is kept
exactly as generated and the model is told nothing about it).

The rules this module enforces:

* Assertion outcomes are never feedback.  An ``AssertionError`` on the code
  under test may be a legitimate kill; a clean pass may not be.  Saying either
  would reveal kill-relevant information, so both are RETAINED silently.
* Exceptions raised inside the code under test are RETAINED: the defect itself
  may be what raised.  Only exceptions whose innermost frame is the
  candidate's own code (a name it invented, a call with the wrong arity) are
  repairable.
* A timeout inside the code under test is RETAINED (the defect may loop); a
  timeout in the candidate's own code is repairable.
* Infrastructure failures are recorded separately and never blamed on, or fed
  back to, the model.

Messages are fixed templates.  The only variable content is drawn from the
candidate itself or from the closed policy/exception vocabularies, and every
message is scanned against the record's hidden material before it is used.
Every feedback object is canonical JSON with a SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

FEEDBACK_SCHEMA_VERSION = "oneiros_execution_feedback_v1"

REPAIRABLE = (
    "unparsable_output",
    "invalid_python_syntax",
    "prohibited_construct",
    "missing_test_invocation",
    "invalid_candidate_shape",
    "fabricated_or_unavailable_api",
    "unavailable_name",
    "malformed_invocation",
    "candidate_timeout",
    "duplicate_candidate",
)
RETAINED = (
    "executes_without_error",
    "observed_assertion_failure",
    "target_raised_exception",
    "target_timeout",
    "infrastructure_failure",
    "identical_to_parent",
)
CATEGORIES = REPAIRABLE + RETAINED

#: Categories a final-slot policy may treat as well-formed.  They are chosen
#: without reference to any assertion outcome: passing, failing and raising in
#: the target are all equally well-formed.
WELL_FORMED = ("executes_without_error", "observed_assertion_failure",
               "target_raised_exception", "target_timeout")

_TEMPLATES = {
    "unparsable_output": "Your previous answer contained no Python test that could be "
                         "extracted.",
    "invalid_python_syntax": "Your previous test is not valid Python ({detail}).",
    "prohibited_construct": "Your previous test uses a construct the test policy does "
                            "not allow ({detail}). Use only the function under test and "
                            "literal values, with no imports.",
    "missing_test_invocation": "Your previous test never calls the function under test.",
    "invalid_candidate_shape": "Your previous test does not have an accepted shape "
                               "({detail}). Write one assert statement, or one "
                               "argument-free def test_...() containing assert "
                               "statements.",
    "fabricated_or_unavailable_api": "Your previous test used an attribute that does "
                                     "not exist ({detail}).",
    "unavailable_name": "Your previous test referred to a name that is not available "
                        "({detail}). Use only the function under test.",
    "malformed_invocation": "Your previous test called the function in a way its "
                            "signature does not accept ({detail}).",
    "candidate_timeout": "Your previous test did not finish within the time limit "
                         "because of a loop in the test itself.",
    "duplicate_candidate": "Your previous test duplicates an earlier test. Write a "
                           "different one.",
}
_CLOSING = "Write one corrected test. Output only the test."

#: Model-visible text must never contain these.  Assertion outcomes and
#: evaluator vocabulary are included so a template edit cannot reintroduce them.
FORBIDDEN_TOKENS = (
    "reference", "golden", "gold", "fixed version", "fixed implementation",
    "mutant", "mutation", "killed", "kill", "correct implementation",
    "expected value", "passes", "passed", "fails on", "assertion failed",
    "assertionerror",
)

_POLICY_MAP = (
    ("empty_candidate", "unparsable_output"),
    ("syntax_error:", "invalid_python_syntax"),
    ("imports_not_allowed", "prohibited_construct"),
    ("disallowed_", "prohibited_construct"),
    ("target_entry_point_not_called", "missing_test_invocation"),
)
_SAFE_DETAIL = re.compile(r"[^A-Za-z0-9_ .:()'\-]")


def _clean_detail(text: str) -> str:
    return _SAFE_DETAIL.sub("", str(text))[:120].strip()


def classify_policy_failure(reason: str) -> tuple[str, str]:
    """Map a frozen candidate-policy reason code to (category, detail)."""
    reason = str(reason or "")
    for prefix, category in _POLICY_MAP:
        if reason.startswith(prefix):
            detail = reason.split(":", 1)[1] if ":" in reason else reason
            return category, _clean_detail(detail)
    return "invalid_candidate_shape", _clean_detail(reason)


def classify_execution(result: Mapping[str, Any]) -> tuple[str, str]:
    """Map a buggy-side execution result to (category, detail)."""
    status = str(result.get("status"))
    origin = str(result.get("origin"))
    exception = str(result.get("exception_type") or "")
    if status == "pass":
        return "executes_without_error", ""
    if status == "assertion_error":
        return "observed_assertion_failure", ""
    if status == "infrastructure_error" or origin == "harness":
        return "infrastructure_failure", ""
    if status == "timeout":
        return ("candidate_timeout", "") if origin == "candidate" else ("target_timeout", "")
    if status == "exception" and origin == "candidate":
        message = str(result.get("message") or "")
        if exception == "NameError":
            name = re.search(r"name '([A-Za-z_][A-Za-z0-9_]*)'", message)
            return "unavailable_name", name.group(1) if name else "NameError"
        if exception == "AttributeError":
            return "fabricated_or_unavailable_api", "AttributeError"
        if exception == "TypeError":
            return "malformed_invocation", _clean_detail(message)
        # Any other candidate-frame exception is a malformed artifact too, but
        # its message is not echoed: only the exception type is named.
        return "malformed_invocation", _clean_detail(exception)
    return "target_raised_exception", ""


def normalised_code(code: str) -> str:
    return re.sub(r"\s+", " ", str(code or "")).strip()


def assert_no_hidden_content(text: str, hidden: Iterable[str]) -> None:
    """Refuse model-visible text that echoes hidden material or evaluator words.

    ``hidden`` is the record's fixed implementation, gold tests and any other
    evaluator-only text.  Any of its non-trivial lines appearing verbatim in
    the feedback is a leak.
    """
    lowered = str(text).lower()
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            raise ValueError(f"feedback contains forbidden token {token!r}")
    for block in hidden:
        for line in str(block or "").splitlines():
            line = normalised_code(line)
            if len(line) >= 12 and line in normalised_code(text):
                raise ValueError("feedback echoes hidden material")


def build_feedback(category: str, detail: str, hidden: Iterable[str]) -> dict[str, Any]:
    """The machine-readable, hashable feedback object for one repairable candidate."""
    if category not in REPAIRABLE:
        raise ValueError(f"category {category!r} is not repairable; no feedback is given")
    hidden = list(hidden)
    detail = _clean_detail(detail)
    message = _TEMPLATES[category].format(detail=detail or "unspecified")
    try:
        assert_no_hidden_content(message, hidden)
    except ValueError:
        # The detail comes from the candidate; if it happens to contain a
        # forbidden word or a hidden line, drop it. The bare template must pass.
        detail = ""
        message = _TEMPLATES[category].format(detail="unspecified")
        assert_no_hidden_content(message, hidden)
    body = {"schema_version": FEEDBACK_SCHEMA_VERSION, "category": category,
            "detail": detail, "message": message}
    body["sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return body


def build_repair_prompt(original_prompt: str, candidate_text: str,
                        feedback: Mapping[str, Any], hidden: Iterable[str]) -> str:
    """Original prompt verbatim, the model's own attempt, one feedback sentence.

    Only the added instruction text is scanned: the prompt is the one the model
    already received, and the attempt is the model's own output.
    """
    assert_no_hidden_content(str(feedback["message"]) + " " + _CLOSING, hidden)
    return "\n\n".join([original_prompt.rstrip(), "You previously answered:",
                        str(candidate_text).strip(), str(feedback["message"]), _CLOSING])
