"""Deterministic mechanical-repair transforms for the post-hoc exploratory v2.4 diagnostic
``posthoc_exploratory_mechanical_repair_ceiling`` (NOT a primary metric; no confirmation claim).

Every transform is a pure function of the retained candidate module (R0) and, for R3 only, the
model-visible prompt. Nothing here reads fixed code, patches, official tests or any execution
outcome, so no transform can be outcome-conditioned. Standard library only (imported by the
WSL runner).

  R0  original retained module, unchanged.
  R1  single-fence extraction: only when the module contains exactly one CLOSED fence whose
      language tag is empty, ``python`` or ``py``; the module becomes that block. Several
      fences, an unclosed fence or another language are refused (module unchanged).
  R2  pytest import: when ``pytest.`` is referenced and no import binds ``pytest``, insert
      ``import pytest``.
  R3  target import: insert the EXACT statement the prompt gives on its
      ``Import the target with: `...` `` line when the module does not already bind every name
      that statement binds. No such prompt line: not applicable (unchanged).
  R4  R1, then R2, then R3, in that fixed order, for every candidate.

Insertion point (R2/R3): the top of the module, except after any leading ``from __future__``
import lines (inserting before them would itself create a syntax error). Every transform is
idempotent.
"""
from __future__ import annotations

import ast
import hashlib
import re
from typing import Dict, List, Optional, Tuple

STUDY = "posthoc_exploratory_mechanical_repair_ceiling"
VERSION = "oneiros_repair_ceiling_transforms_v1"
CONDITIONS = ("R0", "R1", "R2", "R3", "R4")
FENCE_OPEN = re.compile(r"^[ \t]*```([^\n`]*)$", re.M)
FENCE_BLOCK = re.compile(r"^[ \t]*```([^\n`]*)\n(.*?)^[ \t]*```[ \t]*$", re.M | re.S)
PYTHON_TAGS = ("", "python", "py", "python3")
TARGET_IMPORT = re.compile(r"^Import the target with: `([^`\n]+)`[ \t]*$", re.M)
PYTEST_REF = re.compile(r"\bpytest\s*\.")
FUTURE = re.compile(r"^from __future__ import [^\n]*\n", re.M)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def r1_single_fence(module: str) -> Tuple[str, Optional[str]]:
    """(module, refusal_or_None). Unchanged with a reason when not exactly one python fence."""
    if "```" not in module:
        return module, "no_fence"
    opens = FENCE_OPEN.findall(module)
    blocks = FENCE_BLOCK.findall(module)
    if len(blocks) != 1 or len(opens) != 2:
        return module, ("unclosed_fence" if not blocks else "multiple_or_unbalanced_fences")
    tag, body = blocks[0]
    if tag.strip().lower() not in PYTHON_TAGS:
        return module, f"non_python_fence:{tag.strip()[:20]}"
    return body, None


def _bound_import_names(module: str) -> Optional[set]:
    """Names bound by import statements; None when the module does not parse."""
    try:
        tree = ast.parse(module)
    except (SyntaxError, ValueError):
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update((a.asname or a.name) for a in node.names)
    return names


def _bound_by_text(module: str, statement: str, names: set) -> bool:
    """Fallback for an unparsable module: the exact statement line, or (for pytest) a plain
    ``import pytest`` line, already present."""
    lines = {l.strip() for l in module.splitlines()}
    if statement.strip() in lines:
        return True
    return names == {"pytest"} and any(re.fullmatch(r"import\s+pytest(\s*#.*)?", l)
                                       for l in lines)


def _insert(module: str, statement: str) -> str:
    """Insert one import line at the top, after any leading __future__ imports."""
    pos = 0
    for m in FUTURE.finditer(module):
        if module[pos:m.start()].strip(" \t\r\n") != "":
            break
        pos = m.end()
    return module[:pos] + statement + "\n" + module[pos:]


def _ensure_import(module: str, statement: str) -> Tuple[str, Optional[str]]:
    wanted = _bound_import_names(statement)
    if not wanted:
        return module, "unparsable_import_statement"
    bound = _bound_import_names(module)
    present = (wanted <= bound) if bound is not None else _bound_by_text(module, statement, wanted)
    if present:
        return module, "already_imported"
    return _insert(module, statement), None


def r2_pytest_import(module: str) -> Tuple[str, Optional[str]]:
    if not PYTEST_REF.search(module):
        return module, "pytest_not_referenced"
    return _ensure_import(module, "import pytest")


def target_import_statement(prompt: str) -> Optional[str]:
    found = TARGET_IMPORT.findall(prompt)
    return found[0].strip() if len(found) == 1 else None


def r3_target_import(module: str, prompt: str) -> Tuple[str, Optional[str]]:
    statement = target_import_statement(prompt)
    if statement is None:
        return module, "no_target_import_in_prompt"
    return _ensure_import(module, statement)


def apply(condition: str, module: str, prompt: str) -> Dict[str, object]:
    """The repaired module for one condition, with the transforms applied and refusals."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    steps = {"R0": (), "R1": ("R1",), "R2": ("R2",), "R3": ("R3",),
             "R4": ("R1", "R2", "R3")}[condition]
    out, applied, reasons = module, [], {}
    for step in steps:
        if step == "R1":
            out, why = r1_single_fence(out)
        elif step == "R2":
            out, why = r2_pytest_import(out)
        else:
            out, why = r3_target_import(out, prompt)
        if why is None:
            applied.append(step)
        else:
            reasons[step] = why
    return {"condition": condition, "module": out, "changed": out != module,
            "applied": applied, "not_applied": reasons,
            "original_module_sha256": sha256(module), "repaired_module_sha256": sha256(out),
            "target_import": target_import_statement(prompt) if "R3" in steps else None}


def definition() -> Dict[str, object]:
    return {"study": STUDY, "transforms_version": VERSION, "conditions": list(CONDITIONS),
            "rules": __doc__.strip().splitlines()[4:20],
            "inputs": ["v2.4 retained candidate module (R0)",
                       "v2.4 job prompt (model-visible), for R3 only"],
            "never_used": ["fixed code", "patches", "official tests", "execution outcomes",
                           "validation or sealed records"],
            "selection": "none: every condition applied to every candidate; no best-of choice",
            "unchanged_candidates": "a candidate whose module a condition leaves byte-identical "
                                    "keeps its v2.4 execution row (same input); only changed "
                                    "modules are executed",
            "primary_metric_replacement": False, "confirmation_claims": False}


def refusal_counts(results: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        for step, why in (r.get("not_applied") or {}).items():
            key = f"{step}:{str(why).split(':')[0]}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
