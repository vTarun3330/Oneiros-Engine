"""Shared prompt-token budgeting for SFT and live generation.

Raw right truncation removes the end of a chat template, including the
assistant-generation boundary.  Keep both the instruction prefix and the
chat/code suffix so training and inference see the same bounded prompt.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


PROMPT_COMPACTION_STRATEGY = "head_tail_preserve_chat_suffix_v1"


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
