"""v2.5 deterministic conversion of verified train-only SFT completions into ``pytest_module_v1``
(protocol v2.5 A/B.1). Pure and standard-library only (Windows and WSL).

Synthetic (function-mode) examples keep their verified assertion as the oracle, unchanged, inside
one test function. The target and any helper it needs are imported from the stable train-only
interface module ``oneiros_target`` (whose bytes are the reference code on the fixed revision and
the mutant on the buggy revision; the test never contains either implementation). A standard-
library module is imported only when the assertion uses it AND the reference code imports it.
Anything else unresolved is REJECTED, never guessed.

Repository fragments keep their verified test unchanged and gain the public "Non-gold
test-module environment" import header recorded in their train-only support context.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import re
import sys
from typing import Dict, List, Optional

VERSION = "oneiros_v25_module_conversion_v1"
OUTPUT_TYPE = "pytest_module_v1"
INTERFACE_MODULE = "oneiros_target"
BUILTINS = frozenset(dir(builtins))
STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))
HEADER_MARK = "Non-gold test-module environment:"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _loads_and_binds(tree: ast.AST):
    loads, binds = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loads if isinstance(node.ctx, ast.Load) else binds).add(node.id)
        elif isinstance(node, ast.arg):
            binds.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            binds.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            binds.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            binds.add(node.name)
    return loads, binds


def _top_level_names(code: str) -> Optional[set]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


def _imported_modules(code: str) -> set:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    return {(a.asname or a.name).split(".")[0] for n in tree.body if isinstance(n, ast.Import)
            for a in n.names if a.asname is None or a.asname == a.name.split(".")[0]}


def _test_name(entry_point: str) -> str:
    return "test_" + re.sub(r"\W", "_", entry_point).strip("_") + "_defect"


def convert_synthetic(completion: str, entry_point: str, reference_code: str,
                      code_under_test: str) -> Dict[str, object]:
    """A verified bare assertion -> one pytest_module_v1 module, or a rejection reason."""
    text = completion.strip()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {"accepted": False, "reason": "completion_does_not_parse"}
    if not tree.body or not all(isinstance(n, ast.Assert) for n in tree.body):
        return {"accepted": False, "reason": "not_an_assertion_only_completion"}
    loads, binds = _loads_and_binds(tree)
    unresolved = sorted(loads - binds - BUILTINS)
    ref_names, cut_names = _top_level_names(reference_code), _top_level_names(code_under_test)
    if ref_names is None or cut_names is None:
        return {"accepted": False, "reason": "target_code_does_not_parse"}
    if entry_point not in loads:
        return {"accepted": False, "reason": "target_not_referenced"}
    from_target = sorted(n for n in unresolved if n in ref_names and n in cut_names)
    ref_modules = _imported_modules(reference_code)
    stdlib = sorted(n for n in unresolved if n not in from_target and n in STDLIB
                    and n in ref_modules)
    missing = [n for n in unresolved if n not in from_target and n not in stdlib]
    if missing:
        return {"accepted": False, "reason": f"unresolvable_name:{missing[0]}"}
    if entry_point not in from_target:
        return {"accepted": False, "reason": "target_not_defined_in_both_revisions"}
    lines = [f"import {m}" for m in stdlib]
    lines.append(f"from {INTERFACE_MODULE} import {', '.join(from_target)}")
    body = "\n".join("    " + line if line.strip() else "" for line in text.splitlines())
    module = "\n".join(lines) + f"\n\n\ndef {_test_name(entry_point)}():\n{body}\n"
    return {"accepted": True, "module": module, "target_import":
            f"from {INTERFACE_MODULE} import {entry_point}",
            "imports_added": lines, "module_sha256": sha256(module)}


def repository_header(support_context: str) -> Optional[str]:
    """The public import header recorded in the train-only support context, or None."""
    if HEADER_MARK not in support_context:
        return None
    block = support_context.split(HEADER_MARK, 1)[1]
    kept: List[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not re.match(r"(from\s+[\w.]+\s+import\s|import\s+[\w.])", stripped):
            if kept and line.startswith((" ", "\t", ")")):
                kept.append(line)
                continue
            break
        kept.append(line)
    header = "\n".join(kept).strip()
    try:
        ast.parse(header)
    except (SyntaxError, ValueError):
        return None
    return header or None


def convert_repository(fragment: str, support_context: str) -> Dict[str, object]:
    header = repository_header(support_context)
    if header is None:
        return {"accepted": False, "reason": "no_public_import_header"}
    module = header + "\n\n\n" + fragment.strip() + "\n"
    try:
        tree = ast.parse(module)
    except (SyntaxError, ValueError):
        return {"accepted": False, "reason": "converted_module_does_not_parse"}
    if "pytest." in fragment and not any(
            isinstance(n, ast.Import) and any(a.name == "pytest" for a in n.names)
            for n in ast.walk(tree)):
        module = "import pytest\n" + module
    return {"accepted": True, "module": module, "imports_added": header.splitlines(),
            "module_sha256": sha256(module), "verification": "pending_native_environment"}
