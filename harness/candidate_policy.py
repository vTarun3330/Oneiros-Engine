"""Fail-closed policy for model-generated function-level tests.

The policy is deliberately narrower than Python: a candidate must be exactly
one ``assert`` statement, must call the requested entry point, and may not use
syntax or names that provide filesystem, process, network, or introspection
capabilities.  This is a validation layer; execution still happens in a
separate restricted process (see :mod:`harness.safe_execution`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


MAX_CANDIDATE_BYTES = 4_096
MAX_AST_NODES = 160
MAX_AST_DEPTH = 24
MAX_LITERAL_CONTAINER_ITEMS = 128
MAX_STRING_BYTES = 2_048
MAX_INTEGER_ABS = 10**12

_DANGEROUS_NAMES = {
    "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec",
    "exit", "getattr", "globals", "help", "input", "locals", "memoryview",
    "open", "quit", "setattr", "vars",
    "builtins", "ctypes", "importlib", "multiprocessing", "os", "pathlib",
    "resource", "shutil", "signal", "socket", "subprocess", "sys", "tempfile",
}

_DISALLOWED_NODES = (
    ast.Await,
    ast.Delete,
    ast.DictComp,
    ast.FormattedValue,
    ast.GeneratorExp,
    ast.Lambda,
    ast.ListComp,
    ast.NamedExpr,
    ast.SetComp,
    ast.Starred,
    ast.Yield,
    ast.YieldFrom,
)


@dataclass(frozen=True)
class CandidatePolicyResult:
    valid: bool
    reason: str = ""


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_depth(child) for child in children)


def validate_function_assertion(test_code: str, entry_point: str) -> CandidatePolicyResult:
    """Validate a generated assertion without executing it."""
    if not isinstance(test_code, str) or not test_code.strip():
        return CandidatePolicyResult(False, "empty_candidate")
    if len(test_code.encode("utf-8")) > MAX_CANDIDATE_BYTES:
        return CandidatePolicyResult(False, "candidate_too_large")
    if not entry_point or not entry_point.isidentifier():
        return CandidatePolicyResult(False, "invalid_entry_point")

    try:
        tree = ast.parse(test_code, mode="exec")
    except SyntaxError as exc:
        return CandidatePolicyResult(False, f"syntax_error:{exc.msg}")

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return CandidatePolicyResult(False, "candidate_must_be_one_assert")

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        return CandidatePolicyResult(False, "candidate_ast_too_large")
    if _depth(tree) > MAX_AST_DEPTH:
        return CandidatePolicyResult(False, "candidate_ast_too_deep")

    target_called = False
    for node in nodes:
        if isinstance(node, _DISALLOWED_NODES):
            return CandidatePolicyResult(False, f"disallowed_syntax:{type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return CandidatePolicyResult(False, "imports_not_allowed")
        if isinstance(node, ast.Name) and (
            node.id in _DANGEROUS_NAMES or node.id.startswith("__")
        ):
            return CandidatePolicyResult(False, f"disallowed_name:{node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr in _DANGEROUS_NAMES or node.attr.startswith("__")
        ):
            return CandidatePolicyResult(False, f"disallowed_attribute:{node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == entry_point:
                target_called = True
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_NAMES:
                return CandidatePolicyResult(False, f"disallowed_call:{node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _DANGEROUS_NAMES:
                return CandidatePolicyResult(False, f"disallowed_call:{node.func.attr}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value.encode("utf-8")) > MAX_STRING_BYTES:
                return CandidatePolicyResult(False, "string_literal_too_large")
            if isinstance(node.value, int) and abs(node.value) > MAX_INTEGER_ABS:
                return CandidatePolicyResult(False, "integer_literal_too_large")
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > MAX_LITERAL_CONTAINER_ITEMS:
            return CandidatePolicyResult(False, "container_literal_too_large")
        if isinstance(node, ast.Dict) and len(node.keys) > MAX_LITERAL_CONTAINER_ITEMS:
            return CandidatePolicyResult(False, "container_literal_too_large")

    if not target_called:
        return CandidatePolicyResult(False, "target_entry_point_not_called")
    return CandidatePolicyResult(True)


# ---------------------------------------------------------------------------
# Multi-assertion test functions
#
# ``validate_function_assertion`` above is the frozen single-assertion policy
# and is deliberately left unchanged: every historical result, baseline, and
# Kill@k number was produced under it.
#
# The multi-mutant builder needs a second, wider shape - one ``def test_*()``
# carrying several assertions that together distinguish more than one sibling
# mutant.  It is a SEPARATE validator with the same safety rules, not a
# relaxation of the existing one, so a caller must opt in explicitly.
# ---------------------------------------------------------------------------

MAX_TEST_FUNCTION_BYTES = 8_192
MAX_TEST_FUNCTION_AST_NODES = 900
MAX_TEST_FUNCTION_ASSERTS = 24

#: Statement kinds allowed inside a generated test body.  Setup is permitted
#: because a realistic test binds a value before asserting on it, but control
#: flow, loops, and definitions are not: they make a test's meaning depend on
#: state the evaluator cannot see.
_ALLOWED_TEST_BODY_NODES = (ast.Assert, ast.Assign, ast.AnnAssign, ast.Pass)


def validate_test_function(test_code: str, entry_point: str) -> CandidatePolicyResult:
    """Validate one ``def test_*()`` containing setup and several assertions.

    Same capability restrictions as :func:`validate_function_assertion`: no
    imports, no filesystem/process/network names, no dunder access, bounded
    literals.  The shape differs only in permitting multiple assertions and
    straight-line setup inside a single zero-argument test function.
    """
    if not isinstance(test_code, str) or not test_code.strip():
        return CandidatePolicyResult(False, "empty_candidate")
    if len(test_code.encode("utf-8")) > MAX_TEST_FUNCTION_BYTES:
        return CandidatePolicyResult(False, "candidate_too_large")
    if not entry_point or not entry_point.isidentifier():
        return CandidatePolicyResult(False, "invalid_entry_point")

    try:
        tree = ast.parse(test_code, mode="exec")
    except SyntaxError as exc:
        return CandidatePolicyResult(False, f"syntax_error:{exc.msg}")

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return CandidatePolicyResult(False, "candidate_must_be_one_test_function")
    function = tree.body[0]
    if not function.name.startswith("test"):
        return CandidatePolicyResult(False, "test_function_must_be_named_test")
    arguments = function.args
    if (
        arguments.args or arguments.posonlyargs or arguments.kwonlyargs
        or arguments.vararg or arguments.kwarg
    ):
        # A parameterized test needs a fixture the evaluator does not provide,
        # so it could never be executed as written.
        return CandidatePolicyResult(False, "test_function_must_take_no_arguments")
    if function.decorator_list:
        return CandidatePolicyResult(False, "test_function_must_not_be_decorated")

    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
        getattr(body[0], "value", None), ast.Constant
    ) and isinstance(body[0].value.value, str):
        body = body[1:]
    assert_count = 0
    for statement in body:
        if not isinstance(statement, _ALLOWED_TEST_BODY_NODES):
            return CandidatePolicyResult(
                False, f"disallowed_statement:{type(statement).__name__}"
            )
        assert_count += isinstance(statement, ast.Assert)
    if assert_count == 0:
        return CandidatePolicyResult(False, "test_function_has_no_assertion")
    if assert_count > MAX_TEST_FUNCTION_ASSERTS:
        return CandidatePolicyResult(False, "test_function_too_many_assertions")

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_TEST_FUNCTION_AST_NODES:
        return CandidatePolicyResult(False, "candidate_ast_too_large")
    if _depth(tree) > MAX_AST_DEPTH:
        return CandidatePolicyResult(False, "candidate_ast_too_deep")

    target_called = False
    for node in nodes:
        if isinstance(node, _DISALLOWED_NODES):
            return CandidatePolicyResult(False, f"disallowed_syntax:{type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return CandidatePolicyResult(False, "imports_not_allowed")
        if isinstance(node, ast.Name) and (
            node.id in _DANGEROUS_NAMES or node.id.startswith("__")
        ):
            return CandidatePolicyResult(False, f"disallowed_name:{node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr in _DANGEROUS_NAMES or node.attr.startswith("__")
        ):
            return CandidatePolicyResult(False, f"disallowed_attribute:{node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == entry_point:
                target_called = True
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_NAMES:
                return CandidatePolicyResult(False, f"disallowed_call:{node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _DANGEROUS_NAMES:
                return CandidatePolicyResult(False, f"disallowed_call:{node.func.attr}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value.encode("utf-8")) > MAX_STRING_BYTES:
                return CandidatePolicyResult(False, "string_literal_too_large")
            if isinstance(node.value, int) and abs(node.value) > MAX_INTEGER_ABS:
                return CandidatePolicyResult(False, "integer_literal_too_large")
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > MAX_LITERAL_CONTAINER_ITEMS:
            return CandidatePolicyResult(False, "container_literal_too_large")
        if isinstance(node, ast.Dict) and len(node.keys) > MAX_LITERAL_CONTAINER_ITEMS:
            return CandidatePolicyResult(False, "container_literal_too_large")

    if not target_called:
        return CandidatePolicyResult(False, "target_entry_point_not_called")
    return CandidatePolicyResult(True)


def test_function_name(test_code: str) -> str:
    """Return the defined test function's name, or an empty string."""
    try:
        tree = ast.parse(test_code, mode="exec")
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            return node.name
    return ""


def executable_test_function(test_code: str) -> str:
    """Append the invocation the restricted worker needs to run the test.

    The completion the model produces is the definition alone, matching how a
    pytest test is written.  Execution needs the call, so it is added here
    rather than being trained into the completion.
    """
    name = test_function_name(test_code)
    if not name:
        return test_code
    return f"{test_code.rstrip()}\n{name}()\n"


#: Candidate shapes the evaluator understands.
#: ``assertion``     - the frozen single-``assert`` contract (default).
#: ``test_function`` - one ``def test_*()`` carrying several assertions.
CANDIDATE_SHAPES = ("assertion", "test_function")


@dataclass(frozen=True)
class CandidateShapeResult:
    valid: bool
    shape: str = ""
    reason: str = ""


def validate_generated_test(
    test_code: str, entry_point: str, allow_test_function: bool = False,
) -> CandidateShapeResult:
    """Validate a generated candidate under one or both accepted shapes.

    With ``allow_test_function=False`` this is exactly the frozen
    single-assertion policy, so every historical result and every baseline is
    scored by the identical rule and stays comparable.

    With it enabled, a single assertion is STILL validated first and by the
    same policy; only candidates that are not a lone assertion are offered to
    the wider test-function policy. Enabling the wider shape can therefore
    never change the verdict on a single-assertion candidate.
    """
    assertion = validate_function_assertion(test_code, entry_point)
    if assertion.valid:
        return CandidateShapeResult(True, "assertion")
    if not allow_test_function:
        return CandidateShapeResult(False, "", assertion.reason)
    function = validate_test_function(test_code, entry_point)
    if function.valid:
        return CandidateShapeResult(True, "test_function")
    # Report the failure of the shape the candidate actually resembles.
    return CandidateShapeResult(
        False, "", function.reason
        if test_function_name(test_code) else assertion.reason,
    )


def executable_candidate(test_code: str, shape: str) -> str:
    """Return the source the execution worker should run for this candidate."""
    if shape == "test_function":
        return executable_test_function(test_code)
    return test_code
