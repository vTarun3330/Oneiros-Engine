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
