"""v2.5 prompt builder for the single output type ``pytest_module_v1`` (protocol v2.5 A).

Identical model-visible information to the v2.4 builder (``native_generated_test_prompt``: the
same permitted buggy-side view and execution-context block, the same specification sanitiser
and section layout); ONLY the output contract changes:
- the format label is ``pytest_module_v1`` (never ``pytest_fragment``);
- the instruction asks for one complete importable module with the exact target import, pytest
  imported only when used, one focused test, no fences, no prose;
- a STRUCTURAL skeleton (the import line and ``def test_<name>():``) carries no oracle and no
  target-specific expected value.
The v2.4 builder is unchanged.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Mapping

from engine.test_generation_prompt import (
    PROMPT_SCHEMA_VERSION, normalize_target_symbols, sanitize_behavioral_specification)
from harness.native_generated_test_prompt import _context, permitted_view

BUILDER_VERSION = "oneiros_native_generated_test_prompt_v25_pytest_module_v1"
OUTPUT_TYPE = "pytest_module_v1"
TASK = (
    "Write one complete, importable Python test module that demonstrates one behavioral\n"
    "defect of the target. The module must start with the import statements it needs:\n"
    "the exact target import given in the execution context, and `import pytest` only if\n"
    "pytest is used. It contains one focused test function (or one tightly related\n"
    "parameterized test); setup and several assertions are allowed only when they establish\n"
    "one defect. Do not write several independent tests.")
RULES = (
    "Output only the Python module. Do not output Markdown fences, prose, a diagnosis, a\n"
    "suggested fix, corrected code, or a reference implementation. Use only the target\n"
    "module's public names and the standard library. Prefer a minimal counterexample.")


def skeleton(view: Mapping[str, Any]) -> str:
    name = re.sub(r"\W", "_", view["dto"]["qualname"].split(".")[-1]).strip("_").lower()
    return (f"from {view['module']} import {view['import_name']}\n\n\n"
            f"def test_{name}():\n    ...")


def render(view: Mapping[str, Any]) -> str:
    symbols = normalize_target_symbols([view["dto"]["qualname"]],
                                       view["dto"]["qualname"].split(".")[-1])
    spec = sanitize_behavioral_specification(view["docstring"]) or (
        "No additional behavioral specification is available. Infer intended behavior "
        "conservatively from the supplied execution context and public interface.")
    return f"""### TEST GENERATION TASK

Task mode: repository
Expected test format: {OUTPUT_TYPE}
Target symbol(s): {", ".join(f"`{s}`" for s in symbols)}

### Behavioral specification

{spec}

### Available execution context

{_context(view)}

### Code under test

{view["target_source"].strip()}

### Task

{TASK}

Module structure (fill in the test; this skeleton contains no expected value):

{skeleton(view)}

{RULES}

### Output

{OUTPUT_TYPE}"""


def build_prompt(dto: Mapping[str, Any], buggy_source: str) -> Dict[str, Any]:
    view = permitted_view(dto, buggy_source)
    prompt = render(view)
    return {"builder_version": BUILDER_VERSION, "prompt_schema": PROMPT_SCHEMA_VERSION,
            "output_type": OUTPUT_TYPE, "target_key": view["dto"]["target_key"],
            "dto": view["dto"], "view_sha256": view["view_sha256"],
            "buggy_source_sha256": hashlib.sha256(buggy_source.encode("utf-8")).hexdigest(),
            "target_source": view["target_source"], "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
