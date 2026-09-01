"""Leakage-safe static complexity metrics for localized Python functions.

The metrics in this module are derived exclusively from the buggy-side source
shown to the model.  They never inspect a reference implementation, patch,
gold test, mutation description, or execution result, so they are safe to use
for training-corpus composition and reporting.
"""
from __future__ import annotations

import ast
import io
import tokenize
import warnings
from dataclasses import asdict, dataclass
from typing import Any


COMPLEXITY_POLICY_VERSION = "oneiros_buggy_ast_complexity_v1"
COMPLEX_THRESHOLDS = {
    "cyclomatic_complexity": 6,
    "max_control_nesting": 3,
    "logical_lines": 24,
    "ast_node_count": 80,
}
MODERATE_THRESHOLDS = {
    "cyclomatic_complexity": 3,
    "max_control_nesting": 2,
    "logical_lines": 12,
    "ast_node_count": 40,
}


@dataclass(frozen=True)
class FunctionComplexity:
    policy_version: str
    tier: str
    cyclomatic_complexity: int
    max_control_nesting: int
    logical_lines: int
    ast_node_count: int
    call_count: int
    parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SIMPLE_DECISIONS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.IfExp,
)
_NESTING_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)


def _target_function(tree: ast.AST, entry_point: str) -> ast.AST:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
    ]
    if not matches:
        raise ValueError(f"entry point {entry_point!r} is not defined")
    return min(matches, key=lambda node: (node.lineno, node.col_offset))


def _cyclomatic_complexity(node: ast.AST) -> int:
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, _SIMPLE_DECISIONS):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(0, len(child.values) - 1)
        elif isinstance(child, ast.Try):
            complexity += len(child.handlers)
        elif isinstance(child, ast.comprehension):
            complexity += 1 + len(child.ifs)
        elif isinstance(child, ast.Match):
            complexity += max(0, len(child.cases) - 1)
    return complexity


def _max_control_nesting(node: ast.AST) -> int:
    maximum = 0

    def visit(current: ast.AST, depth: int) -> None:
        nonlocal maximum
        nested_depth = depth + 1 if isinstance(current, _NESTING_NODES) else depth
        maximum = max(maximum, nested_depth)
        for child in ast.iter_child_nodes(current):
            visit(child, nested_depth)

    visit(node, 0)
    return maximum


def _logical_lines(source: str, node: ast.AST) -> int:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    lines: set[int] = set()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    for token in tokens:
        if token.type in ignored or not token.string.strip():
            continue
        first = max(start, token.start[0])
        last = min(end, token.end[0])
        if first <= last:
            lines.update(range(first, last + 1))

    body = getattr(node, "body", [])
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(getattr(body[0].value, "value", None), str)
    ):
        lines.difference_update(
            range(body[0].lineno, getattr(body[0], "end_lineno", body[0].lineno) + 1)
        )
    return len(lines)


def _tier(metrics: dict[str, int]) -> str:
    if any(metrics[name] >= threshold for name, threshold in COMPLEX_THRESHOLDS.items()):
        return "complex"
    if any(metrics[name] >= threshold for name, threshold in MODERATE_THRESHOLDS.items()):
        return "moderate"
    return "simple"


def analyze_function_complexity(source: str, entry_point: str) -> FunctionComplexity:
    """Classify one localized function using buggy-side static source only."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be non-empty Python")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError("entry_point must be a Python identifier")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
        target = _target_function(tree, entry_point)
        logical_lines = _logical_lines(source, target)
    except (SyntaxError, tokenize.TokenError) as exc:
        raise ValueError("source must be parseable Python") from exc

    metrics = {
        "cyclomatic_complexity": _cyclomatic_complexity(target),
        "max_control_nesting": _max_control_nesting(target),
        "logical_lines": logical_lines,
        "ast_node_count": sum(1 for _ in ast.walk(target)),
    }
    arguments = target.args
    parameter_count = (
        len(arguments.posonlyargs)
        + len(arguments.args)
        + len(arguments.kwonlyargs)
        + int(arguments.vararg is not None)
        + int(arguments.kwarg is not None)
    )
    return FunctionComplexity(
        policy_version=COMPLEXITY_POLICY_VERSION,
        tier=_tier(metrics),
        call_count=sum(isinstance(child, ast.Call) for child in ast.walk(target)),
        parameter_count=parameter_count,
        **metrics,
    )
