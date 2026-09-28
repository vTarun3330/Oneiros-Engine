"""Candidate-level causal ledger for the SFT root-cause loop (Phase 1).

The evaluator records, for every generated candidate, whether it parsed, passed
the candidate policy, passed on the reference and killed the mutant.  It does
not record the two things the root-cause question turns on:

* did the candidate's INPUT make the buggy and fixed functions behave
  differently (a discriminating input), and
* given such an input, was the candidate's ORACLE right about the fixed
  behaviour?

This module measures both by re-executing each candidate's OWN calls - never an
invented one - in the same restricted worker, with the same timeout, against
the reference and against the mutant.  The candidate is instrumented, not
rewritten:

* the function under test is wrapped at call depth zero, so every top-level
  call records ``(status, repr, sha256(repr))``; recursive calls inside the
  implementation see the original function (the wrapper rebinds the global
  name while the call runs), so the recursion depth is unchanged;
* every ``assert T`` becomes a recorded check of ``bool(T)`` that does not
  abort, together with the number of calls made so far, so the first failing
  assertion and the calls that preceded it are known.

A real test stops at its first failing assertion, so for a candidate that fails
on the reference, "discriminating" is measured on the calls made up to and
including that assertion.  For a candidate that passes, it is measured on every
call.

The original candidates are executed again in the same worker call, and the
evaluator's stored verdicts (reference valid, killed) are checked against that
re-execution; a disagreement is recorded, never silently overwritten.

INFORMATION BOUNDARY: this module reads the reference and the mutant.  It is an
analysis tool; nothing it returns may reach a model prompt.
"""
from __future__ import annotations

import ast
import hashlib
from typing import Any, Mapping, Sequence

from harness.candidate_policy import executable_candidate, test_function_name
from harness.safe_execution import _source_policy_error, classify_assertions

LEDGER_SCHEMA_VERSION = "oneiros_sft_causal_ledger_v1"

#: Terminal categories in PRECEDENCE order.  Every candidate receives exactly
#: the first category whose condition holds (see ``classify_terminal``).
TERMINAL_CATEGORIES = (
    "parse_failure",
    "duplicate",
    "valid_kill",
    "execution_failure",
    "non_discriminating_input",
    "discriminating_wrong_oracle",
    "correct_content_invalid_contract",
    "valid_non_kill",
)

CATEGORY_DEFINITIONS = {
    "parse_failure": "no candidate could be parsed from the raw output",
    "duplicate": ("parsed, and its whitespace-normalised code equals an earlier-ranked "
                  "parsed candidate for the same function"),
    "valid_kill": "passes on the reference and fails on the mutant (evaluator verdict)",
    "execution_failure": ("did not run to its assertions on the reference: policy-valid "
                          "candidates whose reference status is not pass/assertion_error, "
                          "or policy-invalid candidates that are unsafe, raise outside an "
                          "assertion, time out, or reach no assertion"),
    "non_discriminating_input": ("ran on the reference, but no call to the function under "
                                 "test (up to the first failing assertion) behaved "
                                 "differently on the mutant; includes candidates that "
                                 "make no call"),
    "discriminating_wrong_oracle": ("a call behaved differently on the mutant, and an "
                                    "assertion failed on the reference"),
    "correct_content_invalid_contract": ("a discriminating call and assertions that hold on "
                                         "the reference, but the candidate violates the "
                                         "accepted test contract (policy-invalid)"),
    "valid_non_kill": ("policy-valid, passes on the reference, discriminating call, but "
                       "the assertion does not fail on the mutant (a weak oracle)"),
}


def classify_terminal(*, parse_valid: bool, duplicate: bool, policy_valid: bool,
                      reference_valid: bool, killed: bool, executed: bool,
                      oracle_holds: bool, discriminating: bool) -> str:
    """Exactly one terminal category, by fixed precedence (total over all inputs)."""
    if not parse_valid:
        return "parse_failure"
    if duplicate:
        return "duplicate"
    if policy_valid and reference_valid and killed:
        return "valid_kill"
    if not executed:
        return "execution_failure"
    if not discriminating:
        return "non_discriminating_input"
    if not oracle_holds:
        return "discriminating_wrong_oracle"
    if not policy_valid:
        return "correct_content_invalid_contract"
    return "valid_non_kill"


# --- instrumentation ------------------------------------------------------------------------

_PREFIX = '''
import hashlib as _oneiros_hashlib
_oneiros_calls = []
_oneiros_asserts = []
_oneiros_abort = None
_oneiros_original = {entry}

def _oneiros_record(status, text):
    _oneiros_calls.append([status, text[:300],
                           _oneiros_hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]])

def _oneiros_wrapper(*args, **kwargs):
    global {entry}
    {entry} = _oneiros_original
    try:
        value = _oneiros_original(*args, **kwargs)
    except Exception as error:
        if type(error).__name__ != "_ExecutionDeadline":
            _oneiros_record("error", type(error).__name__)
        raise
    finally:
        {entry} = _oneiros_wrapper
    _oneiros_record("ok", repr(value))
    return value

{entry} = _oneiros_wrapper
'''

_SUFFIX = '''
result = {"calls": _oneiros_calls, "asserts": _oneiros_asserts, "abort": _oneiros_abort}
'''


class _AssertRecorder(ast.NodeTransformer):
    """``assert T`` -> record bool(T) (or the exception type) without aborting."""

    def visit_Assert(self, node: ast.Assert) -> list[ast.stmt]:  # noqa: N802
        template = ast.parse(
            "try:\n"
            "    _oneiros_ok = bool(__TEST__)\n"
            "except Exception as _oneiros_error:\n"
            "    if type(_oneiros_error).__name__ == '_ExecutionDeadline':\n"
            "        raise\n"
            "    _oneiros_ok = 'error:' + type(_oneiros_error).__name__\n"
            "_oneiros_asserts.append([len(_oneiros_calls), _oneiros_ok])\n").body
        call = template[0].body[0].value
        call.args[0] = node.test
        return [ast.copy_location(statement, node) for statement in template]


def instrument(code: str, entry_point: str) -> str | None:
    """The instrumented executable form of one candidate, or None if impossible."""
    if not entry_point.isidentifier():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if any(isinstance(node, (ast.ImportFrom)) and node.module == "__future__"
           for node in ast.walk(tree)):
        return None
    tree = _AssertRecorder().visit(tree)
    body = list(tree.body)
    name = test_function_name(code)
    if name:
        body += ast.parse(f"{name}()").body
    guard = ast.parse(
        "try:\n"
        "    pass\n"
        "except Exception as _oneiros_error:\n"
        "    if type(_oneiros_error).__name__ == '_ExecutionDeadline':\n"
        "        raise\n"
        "    _oneiros_abort = type(_oneiros_error).__name__\n").body[0]
    guard.body = body + [ast.Pass()]
    module = ast.Module(body=[guard], type_ignores=[])
    ast.fix_missing_locations(module)
    return _PREFIX.format(entry=entry_point) + ast.unparse(module) + "\n" + _SUFFIX


# --- static helpers -----------------------------------------------------------------------------

def normalise(code: str) -> str:
    return " ".join(str(code).strip().split())


def _calls_to(tree: ast.AST, entry_point: str) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == entry_point]


def static_facts(code: str, entry_point: str) -> dict[str, Any]:
    """Assertions in source order, the call signatures and the expected-value side."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"assertions": [], "call_signatures": [], "assertion_forms": []}
    asserts = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Assert)),
                     key=lambda node: (node.lineno, node.col_offset))
    facts = []
    for node in asserts:
        test = node.test
        form, expected = "other", None
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left, right = test.left, test.comparators[0]
            op = type(test.ops[0]).__name__
            left_calls = bool(_calls_to(left, entry_point))
            right_calls = bool(_calls_to(right, entry_point))
            form = f"compare_{op}"
            if left_calls != right_calls:
                expected = ast.unparse(right if left_calls else left)[:300]
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            form = "not"
        elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name) \
                and test.func.id == "isinstance":
            form = "isinstance"
        elif _calls_to(test, entry_point) and isinstance(test, ast.Call):
            form = "truthiness"
        facts.append({"source": ast.unparse(test)[:300], "form": form,
                      "expected_source": expected})
    signatures = sorted({ast.unparse(call)[:300] for call in _calls_to(tree, entry_point)})
    return {"assertions": facts, "call_signatures": signatures,
            "assertion_forms": [fact["form"] for fact in facts]}


# --- measurement --------------------------------------------------------------------------------

def _first_failure(asserts: Sequence[Sequence[Any]]) -> int | None:
    for index, (_, value) in enumerate(asserts):
        if value is not True:
            return index
    return None


def _run_view(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"status": "not_run", "calls": None, "asserts": None, "abort": None}
    result = row.get("result") if row.get("ok") else None
    if not isinstance(result, dict):
        return {"status": str(row.get("status")), "calls": None, "asserts": None,
                "abort": None, "error": str(row.get("error", ""))[:200]}
    return {"status": "pass", "calls": result.get("calls") or [],
            "asserts": result.get("asserts") or [], "abort": result.get("abort")}


def discrimination(reference: Mapping[str, Any], mutant: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the candidate's own calls behaved differently, up to the first failure."""
    if reference["calls"] is None:
        return {"measured": False, "discriminating": False, "reason": "reference_run_failed"}
    failure = _first_failure(reference["asserts"])
    horizon = (reference["asserts"][failure][0] if failure is not None
               else len(reference["calls"]))
    ref_calls = [call[0:1] + call[2:3] for call in reference["calls"][:horizon]]
    if not ref_calls:
        return {"measured": True, "discriminating": False, "reason": "no_call_before_horizon",
                "horizon_calls": 0}
    if mutant["calls"] is None:
        return {"measured": True, "discriminating": True,
                "reason": f"mutant_run_{mutant['status']}", "horizon_calls": horizon}
    mut_calls = [call[0:1] + call[2:3] for call in mutant["calls"][:horizon]]
    differing = next((i for i, (a, b) in enumerate(zip(ref_calls, mut_calls)) if a != b), None)
    if differing is None and len(mut_calls) < len(ref_calls):
        differing = len(mut_calls)
    return {"measured": True, "discriminating": differing is not None,
            "reason": "call_behaviour_differs" if differing is not None else "calls_agree",
            "horizon_calls": horizon, "first_differing_call": differing}


def measure_function(result: Mapping[str, Any], record: Mapping[str, Any],
                     timeout: float = 0.5, allow_test_function: bool = True
                     ) -> list[dict[str, Any]]:
    """One ledger row per requested candidate of one evaluated function."""
    entry = str(record["entry_point"])
    golden, mutant = str(record["reference_code"]), str(record["code_under_test"])
    outcomes = list(result.get("candidate_outcomes") or [])
    tests: list[str] = []
    originals: dict[int, int] = {}
    instrumented: dict[int, int] = {}
    unsafe: dict[int, str] = {}
    for index, outcome in enumerate(outcomes):
        code = outcome.get("code")
        if not (outcome.get("parse_valid") and isinstance(code, str)):
            continue
        if outcome.get("policy_valid"):
            originals[index] = len(tests)
            tests.append(executable_candidate(code, str(outcome.get("candidate_shape") or "")))
        policy = _source_policy_error(code)
        if policy and not outcome.get("policy_valid"):
            unsafe[index] = policy
            continue
        source = instrument(code, entry)
        if source is not None:
            instrumented[index] = len(tests)
            tests.append(source)
    rows_out = classify_assertions(tests, golden, mutant, timeout) if tests else []

    seen: set[str] = set()
    ledger = []
    for index, outcome in enumerate(outcomes):
        code = outcome.get("code") if isinstance(outcome.get("code"), str) else None
        parse_valid = bool(outcome.get("parse_valid") and code)
        duplicate = False
        if parse_valid:
            key = normalise(code)
            duplicate = key in seen
            seen.add(key)
        policy_valid = bool(outcome.get("policy_valid"))
        reference_valid = bool(outcome.get("reference_valid"))
        killed = bool(outcome.get("killed"))
        reproduction = None
        if index in originals:
            row = rows_out[originals[index]]
            reproduction = {"reference_valid": bool(row.get("valid")),
                            "killed": bool(row.get("killed")),
                            "agrees": bool(row.get("valid")) == reference_valid
                            and bool(row.get("killed")) == killed}
        reference_run = mutant_run = _run_view(None)
        if index in instrumented:
            row = rows_out[instrumented[index]]
            reference_run, mutant_run = _run_view(row.get("golden")), _run_view(row.get("mutant"))
        facts = static_facts(code, entry) if parse_valid else {
            "assertions": [], "call_signatures": [], "assertion_forms": []}
        disc = (discrimination(reference_run, mutant_run) if parse_valid
                else {"measured": False, "discriminating": False, "reason": "not_parsed"})
        asserts = reference_run["asserts"] or []
        instrumented_all_hold = bool(asserts) and all(value is True for _, value in asserts) \
            and reference_run["abort"] is None
        if policy_valid:
            executed = bool(outcome.get("execution_valid"))
            oracle_holds = reference_valid
            execution_detail = str(outcome.get("reference_status"))
        else:
            if index in unsafe:
                executed, execution_detail = False, f"unsafe:{unsafe[index]}"
            elif reference_run["calls"] is None:
                executed, execution_detail = False, f"instrumented_{reference_run['status']}"
            elif reference_run["abort"] is not None:
                executed, execution_detail = False, f"raised:{reference_run['abort']}"
            elif not asserts:
                executed, execution_detail = False, "no_assertion_reached"
            elif any(isinstance(value, str) for _, value in asserts):
                executed, execution_detail = False, "assertion_raised"
            else:
                executed, execution_detail = True, "instrumented_pass"
            oracle_holds = instrumented_all_hold
        category = classify_terminal(
            parse_valid=parse_valid, duplicate=duplicate, policy_valid=policy_valid,
            reference_valid=reference_valid, killed=killed, executed=executed,
            oracle_holds=oracle_holds, discriminating=bool(disc["discriminating"]))
        failure = _first_failure(asserts) if asserts else None
        focus = failure if failure is not None else (0 if asserts else None)
        predicted = None
        if focus is not None and focus < len(facts["assertions"]):
            predicted = facts["assertions"][focus]
        start = asserts[focus - 1][0] if focus else 0
        end = asserts[focus][0] if focus is not None else 0
        ref_slice = (reference_run["calls"] or [])[start:end]
        mut_slice = (mutant_run["calls"] or [])[start:end]
        ledger.append({
            "rank": int(outcome.get("rank", index + 1)),
            "parse_valid": parse_valid,
            "policy_valid": policy_valid,
            "policy_error": outcome.get("policy_error"),
            "execution_valid": bool(outcome.get("execution_valid")),
            "executed": executed,
            "execution_detail": execution_detail,
            "reference_valid": reference_valid,
            "killed": killed,
            "reference_status": outcome.get("reference_status"),
            "mutant_status": outcome.get("mutant_status"),
            "duplicate": duplicate,
            "discriminating_input": bool(disc["discriminating"]),
            "discrimination": disc,
            "oracle_holds_on_reference": oracle_holds,
            "instrumented_oracle_consistent": (None if not policy_valid or not asserts
                                               else instrumented_all_hold == reference_valid),
            "reproduction": reproduction,
            "candidate_shape": outcome.get("candidate_shape"),
            "assertion_count": outcome.get("assertion_count"),
            "assertion_forms": facts["assertion_forms"],
            "call_signatures": facts["call_signatures"],
            "focus_assertion": predicted["source"] if predicted else None,
            "predicted_expected_value": predicted["expected_source"] if predicted else None,
            "reference_output": ref_slice[0][1] if ref_slice else None,
            "buggy_output": mut_slice[0][1] if mut_slice else None,
            "normalised_code_sha256": (hashlib.sha256(normalise(code).encode("utf-8"))
                                       .hexdigest() if code else None),
            "raw_output_sha256": outcome.get("raw_output_sha256"),
            "evaluator_failure_mode": outcome.get("failure_mode"),
            "terminal_category": category,
        })
    return ledger
