"""Separate a WRONG INPUT from a WRONG ORACLE, by execution rather than by text.

The evaluator's own classifier (``classify_candidate_failure``) is deliberately
post-hoc: it never re-executes anything, so it can label a candidate
``wrong_expected_value`` but cannot say *why* that value was wrong. Two
completely different failures land in that one bucket:

* the model probed an input where the reference and the mutant already DIFFER,
  and only mispredicted the value. That candidate is one correct value away
  from a kill, and better oracle supervision recovers it.
* the model probed an input where reference and mutant AGREE. No oracle fixes
  this one - the test could never have killed anything, and the lever is input
  selection, not value prediction.

Collapsing those two into a single ``wrong_oracle`` label is what made the
relearning queue aim at the wrong thing a second time: the first defect was
reading a key no artifact wrote, and the second is treating every recovered
case as an oracle problem when a measured share of them are not.

This module re-runs the candidate's OWN call - never an invented one, because
the question is what the model was already probing - against the reference and
against the mutant, and reports which case it is.

INFORMATION BOUNDARY. This is an analysis and training-side tool. It reads the
mutant, so nothing it returns may ever reach a model prompt. Callers that build
prompts must not import it.
"""
from __future__ import annotations

import ast
from typing import Any, Mapping

from harness.safe_execution import execute_code

#: Refined labels. The first two are the split that matters.
ORACLE_ERROR = "wrong_oracle_on_distinguishing_input"
BENIGN_INPUT = "wrong_input_reference_and_mutant_agree"
BOTH_ERROR = "call_errors_on_both"
NO_CALL = "no_parseable_call"

REFINED_LABELS = (ORACLE_ERROR, BENIGN_INPUT, BOTH_ERROR, NO_CALL)

#: Coarse taxonomy labels this refinement applies to. Every other coarse label
#: already names its own mechanism and is passed through untouched.
REFINABLE = frozenset({"wrong_expected_value"})


def call_expression(candidate: str, entry_point: str) -> str | None:
    """The candidate's own call to the function under test, as source."""
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.id if isinstance(function, ast.Name)
            else function.attr if isinstance(function, ast.Attribute)
            else None
        )
        if name == entry_point:
            try:
                return ast.unparse(node)
            except Exception:
                return None
    return None


def behaviour(code: str, call: str, timeout: float) -> tuple[str, str]:
    """``(status, repr of the value)`` of one call against one implementation.

    ``repr`` is used so that ``1``, ``True`` and ``"1"`` are three different
    behaviours rather than one - a mutant that returns the truthy int where the
    reference returns the bool is a real difference, and comparing values
    directly would call it agreement.
    """
    probe = "result = repr(" + call + ")"
    ok, result, error = execute_code(code, probe, timeout)
    if not ok:
        # Two implementations that raise the same exception type are agreeing,
        # so the type is the behaviour; the message may carry a value and is
        # deliberately discarded.
        return ("error", str(error).split(":", 1)[0].strip())
    return ("ok", str(result if result is not None else "").strip())


def diagnose(candidate_code: str, entry_point: str, reference: str,
             mutant: str, timeout: float = 5.0) -> dict[str, Any]:
    """Why one ``wrong_expected_value`` candidate failed.

    Returns the refined label plus the evidence it was derived from, so a
    report can show the call and the two behaviours rather than asking the
    reader to trust the label.
    """
    call = call_expression(candidate_code, entry_point)
    if call is None:
        return {"refined_label": NO_CALL, "call": None,
                "reference_behaviour": None, "mutant_behaviour": None}

    reference_behaviour = behaviour(reference, call, timeout)
    mutant_behaviour = behaviour(mutant, call, timeout)

    if reference_behaviour[0] == "error" and mutant_behaviour[0] == "error" \
            and reference_behaviour == mutant_behaviour:
        label = BOTH_ERROR
    elif reference_behaviour != mutant_behaviour:
        label = ORACLE_ERROR
    else:
        label = BENIGN_INPUT

    return {
        "refined_label": label,
        "call": call,
        "reference_behaviour": list(reference_behaviour),
        "mutant_behaviour": list(mutant_behaviour),
        "probes_distinguishing_input": label == ORACLE_ERROR,
    }


def refine(coarse_label: str, outcome: Mapping[str, Any], entry_point: str,
           reference: str, mutant: str, timeout: float = 5.0) -> dict[str, Any]:
    """Refine one coarse taxonomy label, or pass it through unchanged.

    Only ``wrong_expected_value`` is refinable. Everything else keeps its
    coarse label so that a refined report stays comparable with every taxonomy
    figure already committed.
    """
    if coarse_label not in REFINABLE:
        return {"coarse_label": coarse_label, "refined_label": coarse_label,
                "refined": False}
    diagnosis = diagnose(str(outcome.get("code") or ""), entry_point,
                         reference, mutant, timeout)
    return {"coarse_label": coarse_label, "refined": True, **diagnosis}
