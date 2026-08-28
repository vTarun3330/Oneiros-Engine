"""Dataset-agnostic prompt contract for Oneiros test generation.

Only model-visible fields belong in this module.  Reference implementations,
patches, mutation labels, oracle outcomes, and expected completions must never
be accepted by the prompt builder.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence


PROMPT_SCHEMA_VERSION = "oneiros_unified_test_generation_v2"

SYSTEM_PROMPT = """You are an expert Python software test engineer.

Inspect the supplied behavioral specification, available execution context,
and code under test. Generate one minimal, self-contained bug-revealing test
case. The test case may contain any setup and assertions necessary to
demonstrate one behavioral defect, but do not generate multiple independent
test cases.

The test must represent the intended behavior, pass when that behavior is
correctly implemented, fail on the supplied code because of the defect, and
use only symbols available in the supplied context. Keep it deterministic,
minimal, and focused on externally observable behavior. Do not repair or
modify the code under test. Do not output an explanation, diagnosis, patch,
mutation description, corrected implementation, or Markdown fence.

The reference implementation is intentionally hidden. Infer intended behavior
only from the supplied specification and execution context. Return only the
requested Python test code."""

_PATCH_LINE_PATTERNS = (
    re.compile(r"^\s*(?:diff --git|index [0-9a-f]+\.\.[0-9a-f]+|@@ )"),
    re.compile(r"^\s*(?:---|\+\+\+)\s+(?:a/|b/|/dev/null)"),
)
_FIX_DIRECTIVE = re.compile(
    r"\b(?:apply|use)\s+(?:this|the)\s+patch\b|"
    r"\b(?:replace|change)\s+.{0,120}\s+with\s+.{0,120}\b|"
    r"\bchange\s+(?:the\s+)?line\b|"
    r"\bthe\s+correct\s+(?:expression|line|implementation)\s+is\b|"
    r"\binstead\s+of\s+.{0,80}\s+use\s+.{0,80}\b|"
    r"\b(?:gold|reference|fixed)\s+(?:patch|implementation|solution)\b",
    re.IGNORECASE,
)


def sanitize_behavioral_specification(value: str) -> str:
    """Retain behavioral intent while removing obvious patch/fix leakage.

    SWE-bench problem statements normally contain behavior and reproduction
    examples rather than gold patches.  This conservative line filter keeps
    those examples but removes diff headers and explicit repair directives.
    """
    retained: list[str] = []
    for raw_line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if any(pattern.search(raw_line) for pattern in _PATCH_LINE_PATTERNS):
            continue
        if _FIX_DIRECTIVE.search(raw_line):
            continue
        retained.append(raw_line.rstrip())
    cleaned = "\n".join(retained)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def task_mode_for_execution_mode(execution_mode: str) -> str:
    return "repository" if str(execution_mode).startswith("repository_") else "function"


def test_format_for_execution_mode(execution_mode: str) -> str:
    if execution_mode == "repository_pytest_fragment":
        return "pytest_fragment"
    if execution_mode == "repository_unittest_fragment":
        return "unittest_fragment"
    if execution_mode == "function_assertion":
        return "assert_statement"
    raise ValueError(f"Unsupported Oneiros execution mode: {execution_mode!r}")


def normalize_target_symbols(symbols: Sequence[str] | str | None, entry_point: str = "") -> list[str]:
    if isinstance(symbols, str):
        values: Iterable[str] = symbols.split(",")
    else:
        values = symbols or []
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if entry_point and entry_point not in normalized:
        normalized.insert(0, entry_point)
    return list(dict.fromkeys(normalized))


def build_unified_user_prompt(
    *,
    code_under_test: str,
    execution_mode: str,
    specification: str = "",
    support_context: str = "",
    target_symbols: Sequence[str] | str | None = None,
    entry_point: str = "",
) -> str:
    """Build the same field layout for every dataset and execution mode."""
    task_mode = task_mode_for_execution_mode(execution_mode)
    test_format = test_format_for_execution_mode(execution_mode)
    symbols = normalize_target_symbols(target_symbols, entry_point)
    symbol_text = ", ".join(f"`{symbol}`" for symbol in symbols) or "(not separately identified)"
    clean_specification = sanitize_behavioral_specification(specification)
    specification_text = clean_specification or (
        "No additional behavioral specification is available. Infer intended "
        "behavior conservatively from the supplied execution context and public interface."
    )
    context_text = str(support_context or "").strip() or "(No additional execution context.)"
    source_text = str(code_under_test or "").strip()
    if not source_text:
        raise ValueError("Unified prompt requires non-empty code_under_test")

    if test_format == "assert_statement":
        output_rule = (
            "Prefer one minimal bounded Python assert statement when sufficient. "
            "Return only one self-contained test case."
        )
    elif test_format == "pytest_fragment":
        output_rule = (
            "Return one complete pytest-compatible test case. Multiple assertions "
            "are allowed only when required to establish one behavioral defect."
        )
    else:
        output_rule = (
            "Return one complete unittest-compatible test case. Multiple assertions "
            "are allowed only when required to establish one behavioral defect."
        )

    # Keep specification near the instruction-heavy head and the target code
    # near the output suffix. The shared head/tail token compactor therefore
    # preserves the two highest-priority sections when a repository prompt is
    # too long, while support context yields first.
    return f"""### TEST GENERATION TASK

Task mode: {task_mode}
Expected test format: {test_format}
Target symbol(s): {symbol_text}

### Behavioral specification

{specification_text}

### Available execution context

{context_text}

### Code under test

{source_text}

### Task

Generate one minimal, self-contained bug-revealing Python test case. The test
case may contain setup and assertions needed to demonstrate one behavioral
defect, but it must not contain multiple independent test cases.

{output_rule}
Do not output prose, a diagnosis, a suggested fix, corrected code, hidden
reference code, or Markdown fences. Use only symbols available in the supplied
source or execution context. Prefer a minimal counterexample.

### Output

{test_format}"""


def format_chat_prompt(tokenizer, user_prompt: str) -> str:
    """Render the identical system/user turns for SFT, DPO, and inference."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
