"""Permitted-view prompt builder for native generated tests (protocol v2, sections 2, 3, 6;
amendment v2.2 section C).

Privilege boundary: ``build_prompt`` receives ONLY a sanitised target DTO (exactly the keys
in ``DTO_KEYS``) and the text of the buggy-side target file. It never receives - and this
module never opens - the rehearsal manifest, official tests or their names, patches,
fixed code, evidence sidecars, issue text or commit messages.

Builder v2 selects context deterministically from the buggy source's AST, identically for
every target: the full target function (never truncated), the enclosing class chain
headers, constructor information, directly referenced helper signatures, and the module
imports, constants, class attributes and same-module definitions that the retained code
references (a fixed point over names). Unreferenced siblings and every helper body are
omitted. There are no target-specific rules.

The prompt uses the production unified test-generation prompt in repository mode
(``repository_pytest_fragment``), the family arm A's SFT data used, so base and SFT are
compared on the prompt family SFT was trained on. The prompt is sealed (SHA-256) before
any leakage scan.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import PurePosixPath
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.test_generation_prompt import PROMPT_SCHEMA_VERSION, build_unified_user_prompt

BUILDER_VERSION = "oneiros_native_generated_test_prompt_v2"
DTO_KEYS = ("target_key", "repository", "buggy_commit", "target_file", "qualname")
MAX_VALUE_CHARS = 400
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
BINDERS = (ast.Assign, ast.AnnAssign)


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


# --- source segments --------------------------------------------------------------------------

def _start(node: ast.AST) -> int:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _segment(lines: List[str], node: ast.AST) -> str:
    return "\n".join(lines[_start(node) - 1:node.end_lineno])


def _doc_line(node: ast.AST, indent: int) -> List[str]:
    doc = ast.get_docstring(node)
    return [f'{" " * indent}"""{doc.splitlines()[0]}"""'] if doc and doc.strip() else []


def _header(lines: List[str], node: ast.AST) -> List[str]:
    """Decorators and the ``def``/``class`` header exactly as in the source; a header that
    shares its line with the body is re-rendered from the AST (the body never appears)."""
    first = node.body[0]
    if first.lineno > node.lineno and first.col_offset > node.col_offset:
        header = lines[_start(node) - 1:first.lineno - 1]
        while header and (not header[-1].strip() or header[-1].lstrip().startswith("#")):
            header = header[:-1]          # comments between the header and the body
        if header:
            return header
    clone = copy.copy(node)
    clone.body = [ast.Pass()]
    text = ast.unparse(clone).splitlines()[:-1]
    return [" " * node.col_offset + line for line in text]


def _signature_stub(lines: List[str], node: ast.AST) -> str:
    indent = node.col_offset + 4
    return "\n".join(_header(lines, node) + _doc_line(node, indent) + [f"{' ' * indent}..."])


def _class_header(lines: List[str], node: ast.ClassDef) -> List[str]:
    return _header(lines, node) + _doc_line(node, node.col_offset + 4)


def _bound(node: ast.AST) -> List[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [sub.id for target in targets for sub in ast.walk(target) if isinstance(sub, ast.Name)]


def _value_text(lines: List[str], node: ast.AST) -> Tuple[str, bool]:
    """Full source of a binding, or ``NAME = ...`` when it exceeds MAX_VALUE_CHARS."""
    text = _segment(lines, node)
    if len(text) <= MAX_VALUE_CHARS:
        return text, False
    return f"{' ' * node.col_offset}{', '.join(_bound(node))} = ...", True


# --- names ------------------------------------------------------------------------------------

def _names(nodes) -> List[str]:
    out = []
    for node in nodes:
        if node is None:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                out.append(sub.id)
    return out


def _signature_parts(node: ast.AST) -> List[ast.AST]:
    a = node.args
    parts = list(node.decorator_list) + [node.returns]
    for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
        if arg is not None:
            parts.append(arg.annotation)
    return parts + list(a.defaults) + [d for d in a.kw_defaults if d is not None]


def _class_parts(node: ast.ClassDef) -> List[ast.AST]:
    return list(node.decorator_list) + list(node.bases) + [k.value for k in node.keywords]


def _binding_parts(node: ast.AST) -> List[ast.AST]:
    return [node.value] + ([node.annotation] if isinstance(node, ast.AnnAssign) else [])


def _module_statements(tree: ast.Module) -> List[ast.AST]:
    """Module-scope statements, descending into module-level if/try/with blocks only."""
    out, stack = [], list(tree.body)
    while stack:
        node = stack.pop(0)
        out.append(node)
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            inner = list(node.body) + list(getattr(node, "orelse", []))
            inner += list(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                inner += list(handler.body)
            stack = inner + stack
    return sorted(out, key=lambda n: (n.lineno, n.col_offset))


def _find(tree: ast.Module, qualname: str):
    """(enclosing class chain, target function); classes may be nested."""
    parts = qualname.split(".")
    scope, chain = tree.body, []
    for name in parts[:-1]:
        klass = next((n for n in scope if isinstance(n, ast.ClassDef) and n.name == name), None)
        if klass is None:
            raise PromptRefused(f"target {qualname!r}: class {name!r} not found")
        chain.append(klass)
        scope = klass.body
    for node in scope:
        if isinstance(node, FUNCS) and node.name == parts[-1]:
            return chain, node
    raise PromptRefused(f"target {qualname!r} not found at module or class level")


def _members(klass: ast.ClassDef) -> Dict[str, List[ast.AST]]:
    out: Dict[str, List[ast.AST]] = {}
    for child in klass.body:
        if isinstance(child, FUNCS + (ast.ClassDef,)):
            out.setdefault(child.name, []).append(child)
        elif isinstance(child, BINDERS):
            for name in _bound(child):
                out.setdefault(name, []).append(child)
    return out


def _receiver(target: ast.AST, in_class: bool) -> Optional[str]:
    if not in_class:
        return None
    if any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in target.decorator_list):
        return None
    params = target.args.posonlyargs + target.args.args
    return params[0].arg if params else None


# --- permitted view ---------------------------------------------------------------------------

def permitted_view(dto: Mapping[str, Any], buggy_source: str) -> Dict[str, Any]:
    """The buggy-side context the prompt may contain, and nothing else (builder v2)."""
    clean = sanitise_dto(dto)
    try:
        tree = ast.parse(buggy_source)
    except SyntaxError as exc:
        raise PromptRefused(f"buggy source does not parse: {exc}") from None
    lines = buggy_source.splitlines()
    chain, target = _find(tree, clean["qualname"])

    imports: Dict[str, List[Tuple[ast.AST, ast.alias]]] = {}
    future: List[ast.AST] = []
    functions: Dict[str, List[ast.AST]] = {}
    classes: Dict[str, List[ast.ClassDef]] = {}
    constants: Dict[str, List[ast.AST]] = {}
    import_total = 0
    for node in _module_statements(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                import_total += 1
                imports.setdefault(alias.asname or alias.name.split(".")[0], []) \
                    .append((node, alias))
        elif isinstance(node, FUNCS):
            functions.setdefault(node.name, []).append(node)
        elif isinstance(node, ast.ClassDef):
            classes.setdefault(node.name, []).append(node)
        elif isinstance(node, BINDERS):
            for name in _bound(node):
                constants.setdefault(name, []).append(node)

    retained: Dict[int, Dict[str, Any]] = {}
    oversized: List[str] = []
    queue: List[str] = []

    def keep(kind: str, node: ast.AST, name: str, text: str, parts, owner=None,
             order: Optional[Tuple[int, int]] = None) -> None:
        if id(node) not in retained:
            retained[id(node)] = {"kind": kind, "name": name, "node": node, "text": text,
                                  "owner": owner,
                                  "order": order or (_start(node), node.col_offset)}
            queue.extend(_names(parts))

    def keep_member(owner: ast.ClassDef, node: ast.AST, name: str, kind: str) -> None:
        if node is target:
            return
        label = f"{owner.name}.{name}"
        if isinstance(node, FUNCS):
            keep(kind, node, label, _signature_stub(lines, node), _signature_parts(node), owner)
        elif isinstance(node, ast.ClassDef):
            keep("nested_class_header", node, label, "\n".join(
                _class_header(lines, node) + [" " * (node.col_offset + 4) + "..."]),
                _class_parts(node), owner)
        else:
            text, big = _value_text(lines, node)
            if big:
                oversized.append(label)
            keep(kind, node, label, text, _binding_parts(node), owner)

    def mro(klass: ast.ClassDef) -> List[ast.ClassDef]:
        """The class, then same-module bases breadth-first (resolved by simple name)."""
        order, pending = [], [klass]
        while pending:
            current = pending.pop(0)
            if any(current is o for o in order):
                continue
            order.append(current)
            for base in current.bases:
                if isinstance(base, ast.Name) and base.id in classes:
                    pending.extend(classes[base.id])
        return order

    def lookup(klass: ast.ClassDef, attr: str, skip_self: bool = False):
        for owner in mro(klass)[1 if skip_self else 0:]:
            found = _members(owner).get(attr)
            if found:
                return owner, found
        return None, []

    # 1. enclosing class chain and constructor information
    for klass in chain:
        queue.extend(_names(_class_parts(klass)))
    if chain:
        inner = chain[-1]
        ctor_owner = next((c for c in mro(inner)
                           if {"__init__", "__new__"} & set(_members(c))), None)
        if ctor_owner is not None:
            for name in ("__init__", "__new__"):
                for node in _members(ctor_owner).get(name, []):
                    keep_member(ctor_owner, node, name, "constructor_signature")
        elif inner.decorator_list:
            for child in inner.body:
                if isinstance(child, ast.AnnAssign):
                    keep_member(inner, child, _bound(child)[0], "constructor_field")
    # 2. the target and the class members it references directly
    queue.extend(_names([target]))
    receiver = _receiver(target, bool(chain))
    chain_names = {k.name: k for k in chain}
    for sub in ast.walk(target):
        if not isinstance(sub, ast.Attribute):
            continue
        value, owner_class, skip = sub.value, None, False
        if isinstance(value, ast.Name) and receiver and value.id == receiver:
            owner_class = chain[-1]
        elif isinstance(value, ast.Name) and value.id in chain_names:
            owner_class = chain_names[value.id]
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and chain:
            if value.func.id == "super":
                owner_class, skip = chain[-1], True
            elif value.func.id == "type" and receiver and value.args and \
                    isinstance(value.args[0], ast.Name) and value.args[0].id == receiver:
                owner_class = chain[-1]
        if owner_class is None:
            continue
        owner, nodes = lookup(owner_class, sub.attr, skip_self=skip)
        for node in nodes:
            kind = "helper_signature" if isinstance(node, FUNCS) else "class_attribute"
            keep_member(owner, node, sub.attr, kind)
    # 3. module-level names used by the retained code, until nothing new is added
    seen: set = set()
    skip_names = {chain[0].name} if chain else {target.name}
    while queue:
        name = queue.pop(0)
        if name in seen or name in skip_names:
            continue
        seen.add(name)
        for stmt, alias in imports.get(name, []):
            keep("import", alias, name, "", [], owner=stmt, order=(stmt.lineno, stmt.col_offset))
        for node in constants.get(name, []):
            text, big = _value_text(lines, node)
            if big:
                oversized.append(name)
            keep("module_constant", node, name, text, _binding_parts(node))
        for node in functions.get(name, []):
            keep("module_function_signature", node, name, _signature_stub(lines, node),
                 _signature_parts(node))
        for node in classes.get(name, []):
            keep("module_class_header", node, name, "", _class_parts(node))

    entries = sorted(retained.values(), key=lambda e: e["order"])

    # imports: only the kept aliases; a whole top-level statement verbatim if none dropped
    rendered_imports = [_segment(lines, n) for n in future]
    by_statement: Dict[int, Tuple[ast.AST, List[ast.alias]]] = {}
    for e in entries:
        if e["kind"] == "import":
            by_statement.setdefault(id(e["owner"]), (e["owner"], []))[1].append(e["node"])
    for stmt, kept_aliases in by_statement.values():
        aliases = [a for a in stmt.names if any(a is k for k in kept_aliases)]
        if len(aliases) == len(stmt.names) and stmt.col_offset == 0:
            rendered_imports.append(_segment(lines, stmt))
        else:
            clone = (ast.Import(names=aliases) if isinstance(stmt, ast.Import) else
                     ast.ImportFrom(module=stmt.module, names=aliases, level=stmt.level))
            rendered_imports.append(ast.unparse(clone))
    constants_text = [e["text"] for e in entries if e["kind"] == "module_constant"]

    # class-owned members: enclosing chain -> skeleton; other same-module owners -> definitions
    members = [e for e in entries if e["owner"] is not None and e["kind"] != "import"]

    def in_chain(owner: ast.AST) -> bool:
        return any(owner is k for k in chain)

    def owner_block(owner: ast.ClassDef) -> str:
        own = [e["text"] for e in members if e["owner"] is owner]
        return "\n".join(_class_header(lines, owner)
                         + (own or [" " * (owner.col_offset + 4) + "..."]))
    definitions: List[str] = []
    rendered_owners: List[ast.AST] = []
    for e in entries:
        if e["kind"] == "module_function_signature":
            definitions.append(e["text"])
        elif e["kind"] == "module_class_header" and not in_chain(e["node"]):
            definitions.append(owner_block(e["node"]))
            rendered_owners.append(e["node"])
    for e in members:
        owner = e["owner"]
        if not in_chain(owner) and not any(owner is o for o in rendered_owners):
            definitions.append(owner_block(owner))
            rendered_owners.append(owner)
    skeleton = None
    if chain:
        def render(depth: int) -> List[str]:
            klass = chain[depth]
            items = [(e["order"], e["text"].split("\n")) for e in members if e["owner"] is klass]
            if depth + 1 < len(chain):
                items.append(((_start(chain[depth + 1]), 0), render(depth + 1)))
            else:
                items.append(((_start(target), 0),
                              [" " * target.col_offset + "# target method (full source below)"]))
            out = _class_header(lines, klass)
            for _, block in sorted(items, key=lambda x: x[0]):
                out.extend(block)
            return out
        skeleton = "\n".join(render(0))

    # omitted accounting (counts only)
    member_total = sum(1 for k in chain for c in k.body
                       if isinstance(c, FUNCS + BINDERS + (ast.ClassDef,)))
    chain_members = sum(1 for e in members if in_chain(e["owner"]))
    omitted = {
        "enclosing_class_members_not_referenced":
            member_total - chain_members - (len(chain) - 1) - (1 if chain else 0),
        "imports_not_referenced": import_total - sum(e["kind"] == "import" for e in entries),
        "module_constants_not_referenced": len(set(constants) - seen),
        "module_functions_not_referenced": len(set(functions) - seen - {target.name}),
        "module_classes_not_referenced": len(set(classes) - seen - skip_names),
        "helper_bodies_elided": sum(isinstance(e["node"], FUNCS) for e in entries),
        "oversized_value_elided": sorted(set(oversized)),
    }
    module = module_name(clean["target_file"])
    view = {"dto": clean, "module": module,
            "import_name": clean["qualname"].split(".")[0],
            "target_source": textwrap.dedent(_segment(lines, target)),
            "docstring": ast.get_docstring(target) or "",
            "imports": rendered_imports, "constants": constants_text,
            "definitions": definitions, "class_skeleton": skeleton,
            "retained_components": [{"kind": e["kind"], "name": e["name"]} for e in entries],
            "omitted": omitted}
    view["view_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in view.items() if k not in ("retained_components", "omitted")},
        sort_keys=True).encode("utf-8")).hexdigest()
    return view


def _context(view: Mapping[str, Any]) -> str:
    blocks = [f"Module: `{view['module']}`",
              f"Target: `{view['dto']['qualname']}`",
              f"Import the target with: `from {view['module']} import {view['import_name']}`"]
    if view["imports"]:
        blocks.append("Module-level imports used by the code shown:\n" + "\n".join(view["imports"]))
    if view["constants"]:
        blocks.append("Module-level constants used by the code shown:\n"
                      + "\n".join(view["constants"]))
    if view["definitions"]:
        blocks.append("Other module-level definitions used by the code shown (bodies elided):\n"
                      + "\n".join(view["definitions"]))
    if view["class_skeleton"]:
        blocks.append("Enclosing class (only members the target uses; bodies elided):\n"
                      + view["class_skeleton"])
    return "\n\n".join(blocks)


def seal(prompt: str, view: Mapping[str, Any], buggy_source: str, condition: str
         ) -> Dict[str, Any]:
    return {"builder_version": BUILDER_VERSION, "prompt_schema": PROMPT_SCHEMA_VERSION,
            "condition": condition, "target_key": view["dto"]["target_key"],
            "dto": view["dto"],
            "dto_sha256": hashlib.sha256(repr(sorted(view["dto"].items())).encode()).hexdigest(),
            "buggy_source_sha256": hashlib.sha256(buggy_source.encode("utf-8")).hexdigest(),
            "view_sha256": view["view_sha256"],
            "retained_components": view["retained_components"], "omitted": view["omitted"],
            "target_source": view["target_source"],
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}


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


# The scaffolded diagnostic condition was removed by amendment v2.1 section C: its
# body-only instruction conflicted with the production output instruction.
