"""One generation path, shared by locked validation and the sealed final run.

This exists because I wrote a second one and it was wrong in three ways at
once. ``harness/sealed_final_loader.py`` called ``build_test_generation_prompt``
(no such function), ``Phi3Generator.generate_candidates`` (no such method), and
never set ``parse_mode``, which defaults to ``"first_assertion"`` - so the final
measurement would have been scored by the legacy parser rather than the frozen
successor protocol. None of it was caught, because every test injected a mock
generator and the real path was never executed.

The lesson is not "test harder". It is that a second implementation of a
measured path is a second thing to be wrong, and the numbers it produces are
not comparable with the numbers the first one produced even when both run. So
there is now one adapter: ``scripts/train_on_dataset.py`` delegates to it, and
the sealed final path calls the same function with the same settings object.
If this code is broken, locked validation's own path is broken too, loudly and
immediately, rather than silently on the one run that cannot be repeated.

Settings are passed explicitly rather than read from module globals, so the
sealed path cannot accidentally inherit a development default - which is
exactly how ``parse_mode`` went wrong.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

ADAPTER_VERSION = "oneiros_generation_adapter_v1"


@dataclass(frozen=True)
class GenerationSettings:
    """Everything that decides what the model emits and how it is read."""

    candidate_parse_mode: str
    retain_raw_output: bool
    candidates_per_function: int
    temperature: float
    top_p: float
    prompt_token_limit: int
    generation_completion_token_limit: int
    max_sequence_tokens: int
    seed: int

    def problems(self) -> List[str]:
        found: List[str] = []
        if self.candidate_parse_mode not in {"first_assertion", "whole_output"}:
            found.append(f"unknown candidate_parse_mode {self.candidate_parse_mode!r}")
        if self.candidates_per_function < 1:
            found.append("candidates_per_function must be positive")
        if not 0.0 < self.temperature <= 2.0:
            found.append(f"temperature out of range: {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            found.append(f"top_p out of range: {self.top_p}")
        if self.prompt_token_limit + self.generation_completion_token_limit >= self.max_sequence_tokens:
            found.append(
                f"prompt {self.prompt_token_limit} + completion "
                f"{self.generation_completion_token_limit} does not fit "
                f"max_sequence_tokens {self.max_sequence_tokens}")
        return found

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def successor_settings() -> GenerationSettings:
    """The frozen successor protocol, read from its single definition."""
    from harness.successor_protocol import SUCCESSOR_PROTOCOL as P
    from config import model_config
    return GenerationSettings(
        candidate_parse_mode=P["candidate_parse_mode"],
        retain_raw_output=P["retain_raw_output"],
        candidates_per_function=P["candidates_per_function"],
        temperature=model_config.temperature,
        top_p=model_config.top_p,
        prompt_token_limit=1024,
        generation_completion_token_limit=P["function_generation_completion_limit"],
        max_sequence_tokens=P["max_sequence_tokens"],
        seed=P["generation_seed"],
    )


def empty_slots(count: int, rank_offset: int = 0) -> List[Dict[str, Any]]:
    """Candidate slots for a record that could not be prompted at all.

    They still occupy their denominator positions, so a prompt-budget failure
    stays distinguishable from a model that generated nothing useful.
    """
    return [{
        "rank": rank_offset + rank + 1,
        "parse_valid": False,
        "code": None,
        "raw_output_sha256": None,
    } for rank in range(count)]


def build_prompts(
    tokenizer,
    records: Sequence[Dict[str, Any]],
    settings: GenerationSettings,
    build_pair_prompt: Callable[[Dict[str, Any]], str],
    prompt_additions: Optional[Dict[int, str]] = None,
):
    """Compact each record's prompt, failing closed per record.

    Returns (token_id_lists, generable_indexes, budget_failures). A record whose
    required sections cannot fit its budget is recorded rather than allowed to
    abort the batch.
    """
    from engine.prompt_budget import PromptBudgetError, compact_unified_user_prompt
    from engine.test_generation_prompt import format_chat_prompt

    prompt_additions = prompt_additions or {}
    token_ids: List[Any] = []
    generable: List[int] = []
    failures: Dict[int, str] = {}
    for index, record in enumerate(records):
        prompt = build_pair_prompt(record)
        addition = (prompt_additions.get(index) or "").strip()
        if addition:
            prompt = f"{prompt}\n\n{addition}"
        try:
            compaction = compact_unified_user_prompt(
                tokenizer, prompt, settings.prompt_token_limit, format_chat_prompt)
        except (PromptBudgetError, ValueError) as exc:
            failures[index] = str(exc)
            continue
        token_ids.append(compaction.token_ids)
        generable.append(index)
    return token_ids, generable, failures


def generate_candidate_slots(
    generator,
    records: Sequence[Dict[str, Any]],
    settings: GenerationSettings,
    build_pair_prompt: Callable[[Dict[str, Any]], str],
    prompt_additions: Optional[Dict[int, str]] = None,
    rank_offset: int = 0,
) -> Dict[int, Dict[str, Any]]:
    """Generate and parse candidates for a batch of records.

    This is the body that locked validation ran, lifted out so the sealed final
    path cannot diverge from it. Returns per-record accounting keyed by the
    record's index in ``records``.
    """
    import torch

    bad = settings.problems()
    if bad:
        raise ValueError(f"invalid generation settings: {bad}")
    if not generator.is_loaded:
        generator.load_model()

    tokenizer = generator.tokenizer
    num = settings.candidates_per_function
    token_ids, generable, failures = build_prompts(
        tokenizer, records, settings, build_pair_prompt, prompt_additions)

    accounting: Dict[int, Dict[str, Any]] = {
        index: {
            "requested_candidates": num,
            "raw_generated_sequences": 0,
            "parsed_candidates": 0,
            "generation_invalid_candidates": num,
            "candidate_slots": empty_slots(num, rank_offset),
            "prompt_budget_failure": index in failures,
            "prompt_budget_failure_reason": failures.get(index),
            "parsed_codes": [],
        }
        for index in range(len(records))
    }

    outputs: List[Any] = []
    input_length = 0
    if token_ids:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        try:
            inputs = tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in token_ids],
                padding=True, return_tensors="pt",
            ).to(generator.model.device)
            input_length = inputs.input_ids.shape[1]
            generator.model.eval()
            with torch.inference_mode():
                outputs = generator.model.generate(
                    **inputs,
                    max_new_tokens=settings.generation_completion_token_limit,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    do_sample=True,
                    num_return_sequences=num,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        finally:
            tokenizer.padding_side = original_padding_side

    # The parse mode is set explicitly, every time, from the settings object.
    # Leaving it to the attribute default is precisely how a sealed run would
    # have been scored by the legacy parser while claiming the successor
    # protocol.
    generator.parse_mode = settings.candidate_parse_mode

    for sequence_index, output in enumerate(outputs):
        position = sequence_index // num
        if position >= len(generable):
            break
        record_index = generable[position]
        accounting[record_index]["raw_generated_sequences"] += 1
        text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        slot = accounting[record_index]["candidate_slots"][sequence_index % num]
        slot["raw_output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if settings.retain_raw_output:
            slot["raw_output"] = text
        parsed = generator._parse_output(text, records[record_index]["entry_point"])
        if parsed.is_valid:
            accounting[record_index]["parsed_codes"].append(parsed.input_code)
            accounting[record_index]["parsed_candidates"] += 1
            slot["parse_valid"] = True
            slot["code"] = parsed.input_code

    for item in accounting.values():
        item["generation_invalid_candidates"] = max(
            0, item["requested_candidates"] - item["parsed_candidates"])
    return accounting


def adapter_source_hashes() -> Dict[str, str]:
    """Identity of the shared generation path."""
    from pathlib import Path
    from harness.source_identity import canonical_sha256, raw_sha256
    path = Path(__file__).resolve()
    return {
        "module": "harness/generation_adapter.py",
        "adapter_version": ADAPTER_VERSION,
        "raw_sha256": raw_sha256(path),
        "canonical_sha256": canonical_sha256(path),
    }
