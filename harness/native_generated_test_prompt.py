"""Permitted-view prompt builder for native generated tests (protocol v2, sections 2, 3, 6).

Privilege boundary: ``build_prompt`` receives ONLY a sanitised target DTO (exactly the keys
in ``DTO_KEYS``) and the text of the buggy-side target file. It never receives - and this
module never opens - the rehearsal manifest, official tests or their names, patches,
fixed code, evidence sidecars, issue text or commit messages.

The prompt uses the production unified test-generation prompt in repository mode
(``repository_pytest_fragment``), the family arm A's SFT data used, so base and SFT are
compared on the prompt family SFT was trained on. The prompt is sealed (SHA-256) before
any leakage scan.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from engine.test_generation_prompt import PROMPT_SCHEMA_VERSION, build_unified_user_prompt

BUILDER_VERSION = "oneiros_native_generated_test_prompt_v1"
DTO_KEYS = ("target_key", "repository", "buggy_commit", "target_file", "qualname")
MAX_CONSTANTS = 20
MAX_CONSTANT_CHARS = 200


class PromptRefused(ValueError):
    """The target cannot be rendered from the permitted view."""


def module_name(target_file: str) -> str:
    parts = list(PurePosixPath(target_file).with_suffix("").parts)
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def sanitise_dto(dto: Mapping[str, Any]) -> Dict[str, str]:
    """Exactly the allowlisted fields, as strings; anything else refuses."""
    extra = sorted(set(dto) - set(DTO_KEYS))
    missing = sorted(set(DTO_KEYS) - set(dto))
    if extra or missing:
        raise PromptRefused(f"DTO fields not allowlisted: extra {extra}, missing {missing}")
    return {key: str(dto[key]) for key in DTO_KEYS}


def _segment(source: str, node: ast.AST) -> str:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    lines = source.splitlines()
    return "\n".join(lines[start - 1:node.end_lineno])


def _signature_stub(source: str, node: ast.FunctionDef) -> str:
    lines = source.splitlines()
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    header_end = node.body[0].lineno - 1
    header = lines[start - 1:header_end]
    indent = " " * (node.col_offset + 4)
    doc = ast.get_docstring(node)
    stub = header + ([f'{indent}"""{doc.splitlines()[0]}"""'] if doc else []) + [f"{indent}..."]
    return "\n".join(stub)


def _find(tree: ast.Module, qualname: str):
    parts = qualname.split(".")
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                return None, node
    elif len(parts) == 2:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == parts[0]:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                            child.name == parts[1]:
                        return node, child
    raise PromptRefused(f"target {qualname!r} not found at module or class level")


def permitted_view(dto: Mapping[str, Any], buggy_source: str) -> Dict[str, Any]:
    """The buggy-side context the prompt may contain, and nothing else."""
    clean = sanitise_dto(dto)
    try:
        tree = ast.parse(buggy_source)
    except SyntaxError as exc:
        raise PromptRefused(f"buggy source does not parse: {exc}") from None
    klass, target = _find(tree, clean["qualname"])
    imports = [_segment(buggy_source, n) for n in tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    constants = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and len(constants) < MAX_CONSTANTS:
            text = _segment(buggy_source, node)
            if len(text) <= MAX_CONSTANT_CHARS:
                constants.append(text)
    skeleton = None
    if klass is not None:
        lines = buggy_source.splitlines()
        header = lines[min([klass.lineno] + [d.lineno for d in klass.decorator_list]) - 1:
                       klass.body[0].lineno - 1]
        parts = list(header)
        doc = ast.get_docstring(klass)
        if doc:
            parts.append(f'    """{doc.splitlines()[0]}"""')
        for child in klass.body:
            if child is target:
                parts.append("    # target method (full source below)")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append(_signature_stub(buggy_source, child))
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                text = _segment(buggy_source, child)
                if len(text) <= MAX_CONSTANT_CHARS:
                    parts.append(text)
        skeleton = "\n".join(parts)
    module = module_name(clean["target_file"])
    return {"dto": clean, "module": module,
            "import_name": clean["qualname"].split(".")[0],
            "target_source": _segment(buggy_source, target),
            "docstring": ast.get_docstring(target) or "",
            "imports": imports, "constants": constants, "class_skeleton": skeleton}


def _context(view: Mapping[str, Any]) -> str:
    blocks = [f"Module: `{view['module']}`",
              f"Import the target with: `from {view['module']} import {view['import_name']}`"]
    if view["imports"]:
        blocks.append("Module-level imports in the code under test:\n" + "\n".join(view["imports"]))
    if view["constants"]:
        blocks.append("Module-level constants:\n" + "\n".join(view["constants"]))
    if view["class_skeleton"]:
        blocks.append("Enclosing class (other method bodies elided):\n" + view["class_skeleton"])
    return "\n\n".join(blocks)


def seal(prompt: str, view: Mapping[str, Any], buggy_source: str, condition: str,
         scaffold: Optional[str] = None) -> Dict[str, Any]:
    return {"builder_version": BUILDER_VERSION, "prompt_schema": PROMPT_SCHEMA_VERSION,
            "condition": condition, "target_key": view["dto"]["target_key"],
            "dto": view["dto"],
            "dto_sha256": hashlib.sha256(repr(sorted(view["dto"].items())).encode()).hexdigest(),
            "buggy_source_sha256": hashlib.sha256(buggy_source.encode("utf-8")).hexdigest(),
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "scaffold": scaffold}


def build_prompt(dto: Mapping[str, Any], buggy_source: str) -> Dict[str, Any]:
    """Primary condition: the model writes a complete pytest module."""
    view = permitted_view(dto, buggy_source)
    prompt = build_unified_user_prompt(
        code_under_test=view["target_source"], execution_mode="repository_pytest_fragment",
        specification=view["docstring"], support_context=_context(view),
        target_symbols=[view["dto"]["qualname"]],
        entry_point=view["dto"]["qualname"].split(".")[-1],
        information_variant="full", output_instruction_variant="self_contained")
    return seal(prompt, view, buggy_source, "whole_module")


def build_scaffold_prompt(dto: Mapping[str, Any], buggy_source: str) -> Dict[str, Any]:
    """Secondary diagnostic: a fixed safe scaffold; the model writes only the test body."""
    view = permitted_view(dto, buggy_source)
    scaffold = (f"import pytest\nfrom {view['module']} import {view['import_name']}\n\n\n"
                "def test_generated():\n")
    context = (_context(view) + "\n\nThe test module is already written up to this line:\n"
               + scaffold + "Write ONLY the indented body of `test_generated` (four-space "
               "indentation), with no imports and no other functions.")
    prompt = build_unified_user_prompt(
        code_under_test=view["target_source"], execution_mode="repository_pytest_fragment",
        specification=view["docstring"], support_context=context,
        target_symbols=[view["dto"]["qualname"]],
        entry_point=view["dto"]["qualname"].split(".")[-1],
        information_variant="full", output_instruction_variant="self_contained")
    return seal(prompt, view, buggy_source, "scaffolded_diagnostic", scaffold)


def compose_scaffolded(scaffold: str, body: str) -> str:
    lines = body.rstrip("\n").splitlines() or ["pass"]
    indented = [line if line.startswith("    ") or not line.strip() else "    " + line
                for line in lines]
    return scaffold + "\n".join(indented) + "\n"
