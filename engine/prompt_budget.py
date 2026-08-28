"""Shared prompt budgeting for SFT, DPO, and live generation.

V4.1 compacts canonical semantic fields *before* rendering the chat template.
The legacy token head/tail helper remains available only for the declared D0
ablation; it is not the active training or inference strategy.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple


PROMPT_COMPACTION_STRATEGY = "section_aware_ast_units_before_chat_v4_1"
LEGACY_PROMPT_COMPACTION_STRATEGY = "priority_head_tail_preserve_spec_target_v2"

_SECTION_ORDER = (
    "### TEST GENERATION TASK",
    "### Behavioral specification",
    "### Available execution context",
    "### Code under test",
    "### Task",
    "### Output",
)


class PromptBudgetError(RuntimeError):
    """Raised instead of producing a malformed or over-budget prompt."""


@dataclass(frozen=True)
class PromptCompactionResult:
    user_prompt: str
    rendered_prompt: str
    token_ids: List[int]
    original_token_count: int
    final_token_count: int
    compacted: bool
    reduced_sections: Tuple[str, ...]
    support_units_dropped: int
    code_units_dropped: int

    def audit_dict(self) -> dict:
        return {
            "strategy": PROMPT_COMPACTION_STRATEGY,
            "original_token_count": self.original_token_count,
            "final_token_count": self.final_token_count,
            "compacted": self.compacted,
            "reduced_sections": list(self.reduced_sections),
            "support_units_dropped": self.support_units_dropped,
            "code_units_dropped": self.code_units_dropped,
        }


def _parse_sections(user_prompt: str) -> dict[str, str]:
    positions = [(heading, user_prompt.find(heading)) for heading in _SECTION_ORDER]
    if any(position < 0 for _, position in positions):
        missing = [heading for heading, position in positions if position < 0]
        raise PromptBudgetError(f"Unified prompt is missing required headings: {missing}")
    if [position for _, position in positions] != sorted(position for _, position in positions):
        raise PromptBudgetError("Unified prompt headings are out of canonical order")
    sections: dict[str, str] = {}
    for index, (heading, position) in enumerate(positions):
        start = position + len(heading)
        end = positions[index + 1][1] if index + 1 < len(positions) else len(user_prompt)
        sections[heading] = user_prompt[start:end].strip()
    return sections


def _render_sections(sections: dict[str, str]) -> str:
    return "\n\n".join(
        f"{heading}\n\n{sections[heading].strip()}" for heading in _SECTION_ORDER
    )


def _top_level_source_units(source: str) -> list[str]:
    """Return complete top-level Python units; never token fragments."""
    source = str(source or "").strip()
    if not source:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Multi-file corpus excerpts contain ``# File:`` markers. Parse each
        # file payload independently so units still remain syntactically whole.
        chunks = re.split(r"(?m)^(?=# File:)", source)
        if len(chunks) > 1:
            units: list[str] = []
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                lines = chunk.splitlines()
                marker = lines[0] if lines[0].startswith("# File:") else ""
                payload = "\n".join(lines[1:] if marker else lines).strip()
                child_units = _top_level_source_units(payload)
                if child_units:
                    units.extend(
                        [f"{marker}\n{unit}".strip() for unit in child_units]
                    )
                elif payload:
                    units.append(chunk)
            return units
        return [source]
    lines = source.splitlines()
    units: list[str] = []
    for node in tree.body:
        start = max(0, int(getattr(node, "lineno", 1)) - 1)
        end = int(getattr(node, "end_lineno", start + 1))
        unit = "\n".join(lines[start:end]).strip()
        if unit:
            units.append(unit)
    return units or [source]


def _target_names(task_header: str) -> set[str]:
    match = re.search(r"(?m)^Target symbol\(s\):\s*(.+)$", task_header)
    if not match:
        return set()
    return set(re.findall(r"`([A-Za-z_]\w*)`", match.group(1)))


def _defined_names(unit: str) -> set[str]:
    try:
        tree = ast.parse(unit)
    except SyntaxError:
        return set()
    return {
        str(getattr(node, "name"))
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _without_docstrings(source: str) -> str:
    """Produce a complete AST rendering with comments/docstrings removed."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree).strip()
    except (AttributeError, ValueError):
        return source


def compact_unified_user_prompt(
    tokenizer,
    user_prompt: str,
    max_tokens: int,
    render_chat: Callable[[object, str], str],
) -> PromptCompactionResult:
    """Compact whole semantic/source units before chat rendering.

    Support context is reduced first.  Non-target code units and docstrings are
    reduced only if necessary.  The behavioral specification, task contract,
    target symbols, expected format, and target definitions are never sliced.
    If those required fields cannot fit, the record fails closed.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    sections = _parse_sections(user_prompt)
    rendered = render_chat(tokenizer, user_prompt)
    original_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if len(original_ids) <= max_tokens:
        return PromptCompactionResult(
            user_prompt, rendered, list(original_ids), len(original_ids), len(original_ids),
            False, (), 0, 0,
        )

    reduced: list[str] = []
    support_dropped = 0
    code_dropped = 0

    def current() -> tuple[str, str, list[int]]:
        candidate_user = _render_sections(sections)
        candidate_rendered = render_chat(tokenizer, candidate_user)
        candidate_ids = tokenizer(candidate_rendered, add_special_tokens=False)["input_ids"]
        return candidate_user, candidate_rendered, list(candidate_ids)

    # Drop complete support-context units from lowest priority to highest.
    context_heading = "### Available execution context"
    context = sections[context_heading]
    context_units = _top_level_source_units(context)
    if len(context_units) > 1:
        while len(context_units) > 1:
            context_units.pop()
            support_dropped += 1
            sections[context_heading] = "\n\n".join(context_units)
            candidate_user, candidate_rendered, candidate_ids = current()
            if len(candidate_ids) <= max_tokens:
                reduced.append("support_context")
                return PromptCompactionResult(
                    candidate_user, candidate_rendered, candidate_ids,
                    len(original_ids), len(candidate_ids), True,
                    tuple(reduced), support_dropped, code_dropped,
                )
    if sections[context_heading] != "(No additional execution context.)":
        support_dropped += max(1, len(context_units))
        sections[context_heading] = "(No additional execution context.)"
        reduced.append("support_context")
        candidate_user, candidate_rendered, candidate_ids = current()
        if len(candidate_ids) <= max_tokens:
            return PromptCompactionResult(
                candidate_user, candidate_rendered, candidate_ids,
                len(original_ids), len(candidate_ids), True,
                tuple(reduced), support_dropped, code_dropped,
            )

    # Remove non-target source units only; retain every target definition.
    code_heading = "### Code under test"
    targets = _target_names(sections["### TEST GENERATION TASK"])
    code_units = _top_level_source_units(sections[code_heading])
    retained_units = list(code_units)
    for unit in reversed(code_units):
        if _defined_names(unit) & targets:
            continue
        if len(retained_units) <= 1:
            break
        retained_units.remove(unit)
        code_dropped += 1
        sections[code_heading] = "\n\n".join(retained_units)
        candidate_user, candidate_rendered, candidate_ids = current()
        if len(candidate_ids) <= max_tokens:
            reduced.append("non_target_code_units")
            return PromptCompactionResult(
                candidate_user, candidate_rendered, candidate_ids,
                len(original_ids), len(candidate_ids), True,
                tuple(dict.fromkeys(reduced)), support_dropped, code_dropped,
            )

    # Last safe compaction: re-render complete AST units without comments or
    # docstrings. No function/class is split or joined to an unrelated suffix.
    compact_code = _without_docstrings(sections[code_heading])
    if compact_code and compact_code != sections[code_heading]:
        sections[code_heading] = compact_code
        reduced.append("code_comments_and_docstrings")
        candidate_user, candidate_rendered, candidate_ids = current()
        if len(candidate_ids) <= max_tokens:
            return PromptCompactionResult(
                candidate_user, candidate_rendered, candidate_ids,
                len(original_ids), len(candidate_ids), True,
                tuple(dict.fromkeys(reduced)), support_dropped, code_dropped,
            )

    candidate_user, candidate_rendered, candidate_ids = current()
    raise PromptBudgetError(
        "Required prompt sections and complete target source exceed the declared "
        f"budget ({len(candidate_ids)} > {max_tokens}); refusing token slicing."
    )


def compact_prompt_token_ids(
    token_ids: Sequence[int], max_tokens: int,
) -> Tuple[List[int], bool]:
    """Return a deterministic head/tail token window within ``max_tokens``."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    token_ids = list(token_ids)
    if len(token_ids) <= max_tokens:
        return token_ids, False

    # Retain the instruction/signature-heavy prefix while reserving a large
    # suffix for the end of the source excerpt and assistant chat boundary.
    suffix_tokens = max(1, (max_tokens * 2) // 5)
    prefix_tokens = max_tokens - suffix_tokens
    return [*token_ids[:prefix_tokens], *token_ids[-suffix_tokens:]], True


def compact_prompt_texts(
    tokenizer, prompts: Iterable[str], max_tokens: int,
) -> Tuple[List[List[int]], int]:
    """Tokenize and compact prompts individually before batch padding."""
    compacted: List[List[int]] = []
    truncated = 0
    for prompt in prompts:
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        bounded, changed = compact_prompt_token_ids(token_ids, max_tokens)
        compacted.append(bounded)
        truncated += int(changed)
    return compacted, truncated


def compact_prompt_string(tokenizer, prompt: str, max_tokens: int) -> Tuple[str, bool, int, int]:
    """Head/tail compact text for trainers that require string columns."""
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    bounded, changed = compact_prompt_token_ids(token_ids, max_tokens)
    if not changed:
        return prompt, False, len(token_ids), len(token_ids)
    try:
        compacted = tokenizer.decode(
            bounded,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        compacted = tokenizer.decode(bounded, skip_special_tokens=False)
    round_trip = tokenizer(compacted, add_special_tokens=False)["input_ids"]
    if len(round_trip) > max_tokens:
        raise RuntimeError(
            "Tokenizer decode/encode round trip exceeded the DPO prompt limit; "
            "refusing implicit trainer truncation."
        )
    return compacted, True, len(token_ids), len(round_trip)
