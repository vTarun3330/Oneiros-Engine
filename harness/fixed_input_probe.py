"""Phase 3A fixed-input probe: can the model predict the fixed function's output
on an input that already distinguishes the buggy and fixed functions?

This removes the two things that confounded Phase 2's oracle mediator: which
input the model chooses, and which assertion form it uses.  Every model and
specification level receives the SAME frozen call and writes only the value.

Input selection is deterministic and arm-blind.  It uses the record's own
upstream test calls and fixed-order literal perturbations of them, executes each
on the reference and the mutant in the restricted worker, and keeps a call only
if the two differ under ``==`` (or the mutant raises) and the reference result
round-trips as a short Python literal.  No model output is ever consulted.

INFORMATION BOUNDARY: selection and scoring read the reference.  Prompts at
levels A0-A3 are built without it, and ``build_prompt`` refuses to emit a
non-A4 prompt containing the reference source or the expected value.
"""
from __future__ import annotations

import ast
import hashlib
from typing import Any, Iterable, Mapping, Sequence

from engine.test_generation_prompt import build_unified_user_prompt
from harness.safe_execution import classify_assertions

PROBE_SCHEMA_VERSION = "oneiros_fixed_input_probe_v1"
SELECTION_SEED = "20260928"
MAX_REPR_CHARS = 120
LEVELS = ("A0", "A3", "A4")
CONTROL_LEVEL = "A0_answer_schema"
LEVEL_DEFINITIONS = {
    "A0": "production information: buggy code under test and the record specification",
    "A1": ("NOT RUN: the records' specification already is the full public upstream "
           "statement; no additional public docstring or type information exists"),
    "A2": "NOT APPLICABLE: synthetic function records have no repository context",
    # FROZEN v1 wording.  It is written into the Phase 3A cohort under
    # PROBE_SCHEMA_VERSION "oneiros_fixed_input_probe_v1" and must stay byte-identical so
    # the freezer reproduces that cohort.  Its A3/A4 descriptions are scientifically
    # superseded: use CORRECTED_LEVEL_LABELS for any interpretation.
    "A3": ("A0 plus a semantic invariant that does not reveal the value: the result's Python "
           "type, and its length for sized results"),
    "A4": ("DIAGNOSTIC, NON-DEPLOYABLE upper bound: the fixed implementation is shown instead "
           "of the buggy one (pure execution reasoning); never used for training or "
           "confirmation"),
    CONTROL_LEVEL: ("schema/extraction control: A0 information, no prefill, answer requested as "
                    "one line 'ANSWER: <Python literal>'"),
}

#: Corrected scientific labels (Phase 3 corrections, 2026-09-28).  Analysis
#: metadata only; never written into v1 cohort metadata.
CORRECTED_LEVEL_LABELS = {
    "A0": "production information (buggy code under test and record specification)",
    "A3": ("DIAGNOSTIC, oracle-derived partial behavioural hint: the correct result's type "
           "and, where sized, its correct length (computed from the reference output); does "
           "not expose the complete value; non-deployable; prohibited from training and "
           "confirmation"),
    "A4": ("DIAGNOSTIC, NON-DEPLOYABLE: access to the complete fixed implementation instead of "
           "the buggy one; changes far more than specification detail; not an identifying "
           "manipulation of public specification quality; never used for training or "
           "confirmation"),
    CONTROL_LEVEL: "A0 information with the one-line 'ANSWER: <literal>' schema (no prefill)",
}


def stable_key(*parts: str) -> str:
    return hashlib.sha256(":".join((SELECTION_SEED, *parts)).encode("utf-8")).hexdigest()


# --- candidate calls ------------------------------------------------------------------------

def upstream_calls(tests: Iterable[Mapping[str, Any]], entry_point: str) -> list[str]:
    calls: list[str] = []
    for test in tests:
        try:
            tree = ast.parse(str(test.get("code") or ""))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == entry_point:
                source = ast.unparse(node)
                if source not in calls:
                    calls.append(source)
    return calls


def _variants(value: Any) -> list[Any]:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, value - 1, value + 2, 0, value * 2]
    if isinstance(value, float):
        return [value + 1.0, value / 2, 0.0]
    if isinstance(value, str):
        return [value + "a", value[1:], value[::-1], value.upper(), ""]
    if isinstance(value, (list, tuple)):
        items = list(value)
        out = [items[::-1], items[:-1], items + items[:1], items[1:]]
        return [type(value)(v) for v in out]
    return []


def perturbed_calls(call: str) -> list[str]:
    """Fixed-order single-argument literal perturbations of one call."""
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError:
        return []
    if not isinstance(node, ast.Call):
        return []
    out: list[str] = []
    for index, argument in enumerate(node.args):
        try:
            value = ast.literal_eval(argument)
        except (ValueError, SyntaxError, TypeError):
            continue
        for variant in _variants(value):
            copy = ast.parse(call, mode="eval").body
            copy.args[index] = ast.parse(repr(variant), mode="eval").body
            source = ast.unparse(copy)
            if source != call and source not in out:
                out.append(source)
    return out


# --- verification ----------------------------------------------------------------------------

def _literal(text: str) -> tuple[bool, Any]:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None
    return repr(value) == text, value


def verify_calls(calls: Sequence[str], reference: str, mutant: str,
                 timeout: float = 0.5) -> list[dict[str, Any]]:
    """Execute every call on reference and mutant; report discrimination under ==."""
    if not calls:
        return []
    rows = classify_assertions([f"result = repr({call})" for call in calls],
                               reference, mutant, timeout)
    out = []
    for call, row in zip(calls, rows):
        golden, mut = row.get("golden") or {}, row.get("mutant") or {}
        if not golden.get("ok"):
            out.append({"call": call, "usable": False, "reason": "reference_failed"})
            continue
        ref_repr = str(golden.get("result"))
        ok, ref_value = _literal(ref_repr)
        if not ok or len(ref_repr) > MAX_REPR_CHARS:
            out.append({"call": call, "usable": False, "reason": "reference_not_short_literal"})
            continue
        if not mut.get("ok"):
            discriminating, mutant_repr = True, f"<{mut.get('status')}>"
        else:
            mutant_repr = str(mut.get("result"))
            mok, mvalue = _literal(mutant_repr)
            discriminating = (not (mvalue == ref_value)) if mok else mutant_repr != ref_repr
        out.append({"call": call, "usable": discriminating,
                    "reason": "discriminating" if discriminating else "agrees_under_eq",
                    "expected_repr": ref_repr, "buggy_repr": mutant_repr[:MAX_REPR_CHARS]})
    return out


def select_inputs(record: Mapping[str, Any], group_upstream: set[str]) -> dict[str, Any]:
    """First discriminating upstream call and first discriminating novel perturbation."""
    entry = str(record["entry_point"])
    reference, mutant = str(record["reference_code"]), str(record["code_under_test"])
    own = upstream_calls(record.get("tests") or [], entry)
    novel_pool: list[str] = []
    for call in own:
        for variant in perturbed_calls(call):
            if variant not in group_upstream and variant not in novel_pool:
                novel_pool.append(variant)
    checked = verify_calls(own + novel_pool, reference, mutant)
    upstream = next((c for c in checked[:len(own)] if c["usable"]), None)
    novel = next((c for c in checked[len(own):] if c["usable"]), None)
    return {"upstream": upstream, "novel": novel,
            "upstream_candidates": len(own), "novel_candidates": len(novel_pool)}


# --- prompts -------------------------------------------------------------------------------

def _type_hint(expected_repr: str) -> str:
    _, value = _literal(expected_repr)
    text = f"The correct result has Python type {type(value).__name__}"
    if isinstance(value, (str, list, tuple, dict, set)):
        text += f" and length {len(value)}"
    return text + "."


def build_prompt(record: Mapping[str, Any], call: str, expected_repr: str,
                 level: str) -> dict[str, str]:
    """User prompt and assistant prefill for one item at one level."""
    entry = str(record["entry_point"])
    code = str(record["reference_code"]) if level == "A4" else \
        str(record.get("prompt_code_under_test") or record["code_under_test"])
    user = build_unified_user_prompt(
        code_under_test=code, execution_mode="function_assertion",
        specification=str(record.get("specification") or ""),
        support_context="", target_symbols=[entry], entry_point=entry)
    if call in user:
        # e.g. a docstring example: the output could be read, not predicted.
        raise ValueError("the frozen call already appears in the prompt")
    if level == "A3":
        user += "\n\nSemantic invariant: " + _type_hint(expected_repr)
    if level == "A4":
        user += ("\n\nThe code above is the CORRECT implementation. Predict exactly what it "
                 "returns.")
    if level == CONTROL_LEVEL:
        user += (f"\n\nWhat does the correct, intended function return for {call} ? Reply "
                 "with exactly one line of the form\nANSWER: <Python literal>")
        prefill = ""
    else:
        user += f"\n\nWrite the test as one assertion using exactly this call: {call}"
        # Bare, unfenced: the production instruction forbids fences and arm A's
        # SFT targets are bare assertions.
        prefill = f"assert {call} == "
    if level != "A4" and str(record["reference_code"]).strip() in user:
        raise ValueError("reference source leaked into a non-A4 prompt")
    return {"user": user, "prefill": prefill}


# --- answers and scoring --------------------------------------------------------------------

def extract_answer(text: str, level: str, call: str) -> dict[str, Any]:
    """The value expression the model wrote, and how it was found."""
    if level != CONTROL_LEVEL:
        value = text.split("\n", 1)[0].strip()
        if value.endswith("```"):
            value = value[:-3].strip()
        return {"answer": value or None, "method": "prefill_line"}
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if stripped.upper().startswith("ANSWER:"):
            return {"answer": stripped.split(":", 1)[1].strip() or None, "method": "strict"}
    marker = f"assert {call} =="
    for line in text.splitlines():
        if line.strip().startswith(marker):
            return {"answer": line.strip()[len(marker):].strip() or None,
                    "method": "lenient_assertion"}
    return {"answer": None, "method": "nonanswer"}


def score_answers(items: Sequence[Mapping[str, Any]], reference: str,
                  timeout: float = 0.5) -> list[dict[str, Any]]:
    """Run ``assert CALL == ANSWER`` on the reference (the evaluator's semantics)."""
    runnable = [(i, item) for i, item in enumerate(items) if item.get("answer")]
    tests = [f"assert {item['call']} == ({item['answer']})" for _, item in runnable]
    rows = classify_assertions(tests, reference, None, timeout) if tests else []
    verdicts: dict[int, dict[str, Any]] = {}
    for (index, _), row in zip(runnable, rows):
        golden = row.get("golden") or {}
        verdicts[index] = {"correct": bool(golden.get("ok")),
                           "status": str(golden.get("status"))}
    return [verdicts.get(i, {"correct": False, "status": "nonanswer"})
            for i in range(len(items))]
