"""Leakage-safe model-visible field construction for Oneiros V4.1.

This module deliberately has no reference-code, gold-patch, oracle-output, or
official-test-body input.  Keeping the API narrow makes prohibited data flow a
construction error instead of a convention callers can accidentally violate.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PROHIBITED_LINEAGE_MARKERS = (
    "reference_code",
    "gold_patch",
    "gold_test",
    "official_test_body",
    "expected_completion",
    "oracle_result",
    "mutation_operator",
    "fixed_execution",
    "hidden_correction",
)


@dataclass(frozen=True)
class RepositoryPromptContext:
    target_symbols: tuple[str, ...]
    prompt_code_under_test: str
    support_context: str
    localization_source: str
    field_lineage: Mapping[str, tuple[str, ...]]


def _named_top_level_nodes(source: str) -> list[ast.AST]:
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return []
    return [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _identifier_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(text or ""))
        if len(token) >= 3
    }


def select_buggy_side_targets(
    buggy_localized_source: str,
    specification: str,
    declared_entry_point: str = "",
    max_targets: int = 3,
) -> tuple[list[str], str]:
    """Select target names without seeing a fixed implementation or gold test."""
    nodes = _named_top_level_nodes(buggy_localized_source)
    names = [str(getattr(node, "name")) for node in nodes]
    if declared_entry_point and declared_entry_point in names:
        return [declared_entry_point], "explicit_public_entry_point"

    spec_tokens = _identifier_tokens(specification)
    mentioned = [name for name in names if name.lower() in spec_tokens]
    if mentioned:
        return mentioned[:max_targets], "public_issue_metadata"

    public = [name for name in names if not name.startswith("_")]
    if public:
        # Source-order public definitions are a reproducible buggy-side-only
        # heuristic. Fault localization itself remains an explicit upstream
        # assumption; this does not claim to discover the defective line.
        return public[:max_targets], "buggy_side_static_analysis"
    if names:
        return names[:max_targets], "buggy_side_static_analysis"
    return [], "declared_localized_source_excerpt"


def partition_buggy_source(
    buggy_localized_source: str,
    target_symbols: Iterable[str],
) -> tuple[str, str]:
    """Partition complete top-level units using only the buggy source."""
    targets = set(target_symbols)
    if not targets:
        return "", str(buggy_localized_source or "").strip()
    try:
        tree = ast.parse(buggy_localized_source)
    except SyntaxError:
        return "", str(buggy_localized_source or "").strip()
    lines = buggy_localized_source.splitlines()
    target_ranges = [
        (node.lineno - 1, int(getattr(node, "end_lineno", node.lineno)))
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and getattr(node, "name", "") in targets
    ]
    if not target_ranges:
        return "", str(buggy_localized_source or "").strip()
    indexes = {index for start, end in target_ranges for index in range(start, end)}
    target = "\n\n".join(
        "\n".join(lines[start:end]).strip() for start, end in target_ranges
    ).strip()
    support = "\n".join(
        line for index, line in enumerate(lines) if index not in indexes
    ).strip()
    file_markers = [line for line in lines if line.lstrip().startswith("# File:")]
    if file_markers and target and not target.startswith("# File:"):
        target = f"{file_markers[0]}\n{target}"
    return support, target


def extract_non_gold_test_environment(
    buggy_test_module_source: str,
) -> tuple[str, list[str]]:
    """Extract imports/constants/fixtures/helpers, never any test function.

    The extraction is intentionally independent of the selected official test:
    every ``test_*`` method/function and every ``Test*`` class is omitted.  A
    change to a gold test body therefore cannot affect the visible context.
    """
    source = str(buggy_test_module_source or "")
    if not source.strip():
        return "", []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "", []
    lines = source.splitlines()
    units: list[str] = []
    lineage: list[str] = []
    for node in tree.body:
        keep = False
        source_kind = ""
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep, source_kind = True, "buggy_repo:test_module_import"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            keep, source_kind = True, "buggy_repo:test_module_constant"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = {
                ast.unparse(item) if hasattr(ast, "unparse") else ""
                for item in node.decorator_list
            }
            if not node.name.startswith("test_"):
                keep = True
                source_kind = (
                    "buggy_repo:test_module_fixture"
                    if any("fixture" in item for item in decorators)
                    else "buggy_repo:test_module_helper"
                )
        elif isinstance(node, ast.ClassDef):
            contains_test_method = any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
                for child in node.body
            )
            is_test_container = (
                node.name.startswith("Test")
                or node.name.endswith("Test")
                or node.name.endswith("TestCase")
                or contains_test_method
            )
            if not is_test_container:
                keep, source_kind = True, "buggy_repo:test_module_class"
        if not keep:
            continue
        start = node.lineno - 1
        end = int(getattr(node, "end_lineno", node.lineno))
        unit = "\n".join(lines[start:end]).strip()
        if unit:
            units.append(unit)
            lineage.append(source_kind)
    return "\n\n".join(units), list(dict.fromkeys(lineage))


def build_repository_prompt_context(
    *,
    buggy_localized_source: str,
    specification: str,
    execution_mode: str,
    declared_entry_point: str = "",
    public_test_module_path: str = "",
    buggy_test_environment_source: str = "",
) -> RepositoryPromptContext:
    """Construct all repository model-visible fields without gold inputs."""
    targets, localization_source = select_buggy_side_targets(
        buggy_localized_source,
        specification,
        declared_entry_point,
    )
    source_support, target_code = partition_buggy_source(
        buggy_localized_source, targets
    )
    test_support, test_lineage = extract_non_gold_test_environment(
        buggy_test_environment_source
    )
    framework = "pytest" if "pytest" in execution_mode else "unittest"
    header = [
        f"Test framework: {framework}.",
        "The test executes in the native buggy-repository test environment.",
    ]
    support_lineage = ["static_config:test_framework"]
    if public_test_module_path:
        header.append(f"Public test module path: {public_test_module_path}")
        support_lineage.append("public_metadata:test_module_path")
    bodies = [*header]
    if test_support:
        bodies.extend(["", "Non-gold test-module environment:", test_support])
        support_lineage.extend(test_lineage)
    if source_support:
        bodies.extend(["", "Buggy-source imports and non-target definitions:", source_support])
        support_lineage.append("buggy_repo:localized_source_non_target_units")
    lineage = {
        "task_mode": ("static_config:execution_mode",),
        "test_format": ("static_config:execution_mode",),
        "specification": ("upstream_public:behavioral_specification",),
        "prompt_code_under_test": ("buggy_revision:declared_localized_region",),
        "target_symbols": (f"{localization_source}:target_symbols",),
        "support_context": tuple(dict.fromkeys(support_lineage)),
    }
    return RepositoryPromptContext(
        target_symbols=tuple(targets),
        prompt_code_under_test=target_code or buggy_localized_source.strip(),
        support_context="\n".join(bodies).strip(),
        localization_source=localization_source,
        field_lineage=lineage,
    )


def prohibited_lineage_entries(field_lineage: Mapping[str, Sequence[str]]) -> list[str]:
    failures: list[str] = []
    for field, entries in field_lineage.items():
        for entry in entries:
            normalized = str(entry).lower().replace("-", "_").replace(" ", "_")
            if any(marker in normalized for marker in PROHIBITED_LINEAGE_MARKERS):
                failures.append(f"{field}:{entry}")
    return failures
