"""Leakage-safe static complexity metrics for localized repository regions.

``harness.function_complexity`` classifies a single self-contained function by
its entry point.  Repository records have no entry point: the buggy-side prompt
carries a localized *region* of a real source file, and the defect may live in
a method, a module-level helper, or several cooperating callables.

Every metric here is computed from the buggy-side prompt payload only - the
same text the model is shown - plus the size of the support context.  No fixed
revision, gold patch, gold test body, or execution result is read, so these
values are safe for corpus composition and for reporting.
"""
from __future__ import annotations

import ast
import builtins
import io
import tokenize
import warnings
from dataclasses import asdict, dataclass
from typing import Any


REPOSITORY_COMPLEXITY_POLICY_VERSION = "oneiros_buggy_region_repository_complexity_v1"

#: Frozen tier thresholds.  A region is ``complex`` when it reaches ANY complex
#: threshold and ``moderate`` when it reaches any moderate threshold.  The
#: any-of rule matches ``function_complexity`` so the two tier vocabularies mean
#: comparable things, but the numbers are deliberately higher: a repository
#: region is a slice of a real file and is larger than a benchmark function of
#: equal difficulty.
REPOSITORY_COMPLEX_THRESHOLDS = {
    "cyclomatic_complexity": 12,
    "max_control_nesting": 4,
    "logical_lines": 60,
    "ast_node_count": 320,
    "branch_count": 10,
}
REPOSITORY_MODERATE_THRESHOLDS = {
    "cyclomatic_complexity": 5,
    "max_control_nesting": 2,
    "logical_lines": 22,
    "ast_node_count": 110,
    "branch_count": 3,
}

_TIER_FIELDS = tuple(REPOSITORY_COMPLEX_THRESHOLDS)
_BUILTIN_NAMES = frozenset(dir(builtins))

_BRANCH_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.Match)
_NESTING_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
    ast.With, ast.AsyncWith, ast.Match, ast.FunctionDef,
    ast.AsyncFunctionDef, ast.ClassDef,
)
_COLLECTION_NODES = (
    ast.List, ast.Dict, ast.Set, ast.Tuple, ast.ListComp,
    ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Subscript,
)
_ASYNC_NODES = (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)


@dataclass(frozen=True)
class RepositoryComplexity:
    """One buggy-side repository region, measured and tiered."""

    policy_version: str
    tier: str
    parse_status: str
    logical_lines: int
    cyclomatic_complexity: int
    branch_count: int
    max_control_nesting: int
    ast_node_count: int
    parameter_count: int
    call_count: int
    external_symbol_count: int
    state_mutation_count: int
    exception_path_count: int
    collection_operation_count: int
    class_interaction_count: int
    async_construct_count: int
    function_count: int
    class_count: int
    context_characters: int
    context_lines: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_leading_comments(source: str) -> str:
    """Drop the ``# File: ...`` banner the prompt builder prepends."""
    lines = source.splitlines()
    index = 0
    while index < len(lines) and lines[index].lstrip().startswith("#"):
        index += 1
    return "\n".join(lines[index:])


def _parse(source: str) -> ast.AST | None:
    """Parse a region, tolerating the fact that a slice may not stand alone."""
    for candidate in (source, _strip_leading_comments(source)):
        if not candidate.strip():
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return ast.parse(candidate)
        except (SyntaxError, ValueError):
            continue
    return None


def _logical_lines(source: str) -> int:
    lines: set[int] = set()
    ignored = {
        tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT,
        tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT,
    }
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in ignored or not token.string.strip():
                continue
            lines.update(range(token.start[0], token.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # A region that cannot be tokenized is still measurable.  Returning
        # zero here would misfile a large unparseable slice as ``simple``.
        return sum(
            1 for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return len(lines)


def _cyclomatic_complexity(tree: ast.AST) -> int:
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += max(0, len(node.values) - 1)
        elif isinstance(node, ast.Try):
            complexity += len(node.handlers)
        elif isinstance(node, ast.comprehension):
            complexity += 1 + len(node.ifs)
        elif isinstance(node, ast.Match):
            complexity += max(0, len(node.cases) - 1)
    return complexity


def _max_control_nesting(tree: ast.AST) -> int:
    maximum = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal maximum
        nested = depth + 1 if isinstance(node, _NESTING_NODES) else depth
        maximum = max(maximum, nested)
        for child in ast.iter_child_nodes(node):
            visit(child, nested)

    visit(tree, 0)
    return maximum


def _defined_names(tree: ast.AST) -> set[str]:
    """Names bound inside the region: definitions, imports, and assignments."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            arguments = getattr(node, "args", None)
            if arguments is not None:
                for argument in (
                    list(arguments.posonlyargs) + list(arguments.args)
                    + list(arguments.kwonlyargs)
                    + [arguments.vararg, arguments.kwarg]
                ):
                    if argument is not None:
                        names.add(argument.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _external_symbols(tree: ast.AST) -> int:
    """Loaded names the region never binds - its dependency surface."""
    defined = _defined_names(tree)
    external = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in defined
        and node.id not in _BUILTIN_NAMES
    }
    return len(external)


def _parameter_count(tree: ast.AST) -> int:
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            total += (
                len(arguments.posonlyargs) + len(arguments.args)
                + len(arguments.kwonlyargs)
                + int(arguments.vararg is not None)
                + int(arguments.kwarg is not None)
            )
    return total


def _state_mutations(tree: ast.AST) -> int:
    """Attribute/subscript writes, augmented assignment, deletes, and rebinding.

    These are the writes that make a defect stateful rather than pure, which is
    the property separating realistic repository code from a benchmark function.
    """
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign):
            total += 1
        elif isinstance(node, ast.Assign):
            total += sum(
                isinstance(target, (ast.Attribute, ast.Subscript))
                for target in node.targets
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, (ast.Attribute, ast.Subscript)
        ):
            total += 1
        elif isinstance(node, ast.Delete):
            total += len(node.targets)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            total += len(node.names)
    return total


def _exception_paths(tree: ast.AST) -> int:
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            total += len(node.handlers) + int(bool(node.finalbody))
        elif isinstance(node, ast.Raise):
            total += 1
    return total


def _count(tree: ast.AST, types: tuple[type, ...]) -> int:
    return sum(isinstance(node, types) for node in ast.walk(tree))


def _tier(metrics: dict[str, int]) -> str:
    if any(
        metrics[name] >= REPOSITORY_COMPLEX_THRESHOLDS[name] for name in _TIER_FIELDS
    ):
        return "complex"
    if any(
        metrics[name] >= REPOSITORY_MODERATE_THRESHOLDS[name] for name in _TIER_FIELDS
    ):
        return "moderate"
    return "simple"


def analyze_repository_complexity(
    prompt_code: str, support_context: str = "",
) -> RepositoryComplexity:
    """Measure one buggy-side repository region shown in the model prompt.

    ``prompt_code`` must be the prompt payload, never the fixed revision.  A
    region that cannot be parsed is still measured on its text metrics and
    reported with ``parse_status='unparsed'`` rather than dropped, so an
    unparseable slice can never masquerade as a simple one.
    """
    if not isinstance(prompt_code, str):
        raise TypeError("prompt_code must be a string")
    support_context = support_context or ""
    logical_lines = _logical_lines(prompt_code)
    context_characters = len(support_context)
    context_lines = len(support_context.splitlines())
    tree = _parse(prompt_code)

    if tree is None:
        metrics = {
            "cyclomatic_complexity": 0,
            "branch_count": 0,
            "max_control_nesting": 0,
            "ast_node_count": 0,
            "logical_lines": logical_lines,
        }
        return RepositoryComplexity(
            policy_version=REPOSITORY_COMPLEXITY_POLICY_VERSION,
            tier=_tier(metrics),
            parse_status="unparsed",
            parameter_count=0,
            call_count=0,
            external_symbol_count=0,
            state_mutation_count=0,
            exception_path_count=0,
            collection_operation_count=0,
            class_interaction_count=0,
            async_construct_count=0,
            function_count=0,
            class_count=0,
            context_characters=context_characters,
            context_lines=context_lines,
            **metrics,
        )

    metrics = {
        "cyclomatic_complexity": _cyclomatic_complexity(tree),
        "branch_count": _count(tree, _BRANCH_NODES),
        "max_control_nesting": _max_control_nesting(tree),
        "ast_node_count": sum(1 for _ in ast.walk(tree)),
        "logical_lines": logical_lines,
    }
    return RepositoryComplexity(
        policy_version=REPOSITORY_COMPLEXITY_POLICY_VERSION,
        tier=_tier(metrics),
        parse_status="parsed",
        parameter_count=_parameter_count(tree),
        call_count=_count(tree, (ast.Call,)),
        external_symbol_count=_external_symbols(tree),
        state_mutation_count=_state_mutations(tree),
        exception_path_count=_exception_paths(tree),
        collection_operation_count=_count(tree, _COLLECTION_NODES),
        class_interaction_count=_count(tree, (ast.Attribute,)),
        async_construct_count=_count(tree, _ASYNC_NODES),
        function_count=_count(tree, (ast.FunctionDef, ast.AsyncFunctionDef)),
        class_count=_count(tree, (ast.ClassDef,)),
        context_characters=context_characters,
        context_lines=context_lines,
        **metrics,
    )
