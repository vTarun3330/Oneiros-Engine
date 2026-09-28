"""Phase 4 (V2 design, NOT launched): matched control/treatment examples that differ
ONLY in their label mask.

Both arms see the identical token sequence:

    prompt_ids      = tokenize(chat_template(system, user))          (masked in both)
    completion_ids  = tokenize("assert <call> == <value>" + eos_token)

mirroring engine/sft_trainer.py (prompt and ``completion + eos_token`` tokenized
separately, labels masked before the completion start).

* control labels: every completion token (assertion, call, ``==``, value, EOS);
* treatment labels: only the complete value span and EOS.

The value span is located POSITIONALLY from character offsets of the completion
(``len("assert ") + len(call) + len(" == ")``), never by substring search, so a value
that also appears in the call or prompt cannot be mis-selected.  Token boundaries must
align with the span: a supervised token may absorb only the single separator space
before the value, the last value token must end exactly at the value's end, and any
token straddling ``==`` and the value refuses the example.

Also provides a matched-manifest validator (C4).  It is exercised only on toy fixtures
in this task; no real training manifest is frozen or emitted.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from engine.test_generation_prompt import format_chat_prompt
from harness.safe_execution import classify_assertions

IGNORE = -100
DESIGN_VERSION = "oneiros_phase4_objective_masking_v2"


class AlignmentError(ValueError):
    """The value span does not align with token boundaries."""


def completion_text(call: str, value: str) -> tuple[str, int, int]:
    prefix = f"assert {call} == "
    return prefix + value, len(prefix), len(prefix) + len(value)


def build_example(tokenizer, user_prompt: str, call: str, value: str) -> dict[str, Any]:
    if value != value.strip() or not value:
        raise AlignmentError("the value must be non-empty canonical text without outer spaces")
    prompt_ids = tokenizer(format_chat_prompt(tokenizer, user_prompt),
                           add_special_tokens=False)["input_ids"]
    body, start, end = completion_text(call, value)
    encoded = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    body_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(eos_ids) != 1 or eos_ids[0] != tokenizer.eos_token_id:
        raise AlignmentError("eos_token does not tokenize to the single eos id")
    value_tokens = [i for i, (s, e) in enumerate(offsets) if e > start and s < end]
    if not value_tokens:
        raise AlignmentError("no token covers the value")
    first_start = offsets[value_tokens[0]][0]
    if first_start < start - 1 or body[first_start:start].strip():
        raise AlignmentError(f"a token straddles '==' and the value: {body[first_start:end]!r}")
    if offsets[value_tokens[-1]][1] != end:
        raise AlignmentError("the last value token does not end at the value's end")
    if value_tokens != list(range(value_tokens[0], value_tokens[-1] + 1)) or \
            value_tokens[-1] != len(body_ids) - 1:
        raise AlignmentError("the value tokens are not the contiguous end of the completion")

    input_ids = prompt_ids + body_ids + eos_ids
    completion_start = len(prompt_ids)
    control = [IGNORE] * completion_start + body_ids + eos_ids
    treatment = [IGNORE] * len(input_ids)
    for index in value_tokens:
        treatment[completion_start + index] = body_ids[index]
    treatment[-1] = eos_ids[0]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": {"control": control, "treatment": treatment},
            "completion_text": body, "completion_start": completion_start,
            "value_token_positions": [completion_start + i for i in value_tokens],
            "supervised_value_text": body[first_start:end]}


def decode_supervised(tokenizer, example: Mapping[str, Any], arm: str) -> str:
    ids = [t for t in example["labels"][arm] if t != IGNORE]
    return tokenizer.decode(ids, skip_special_tokens=False)


def sequence_hash(values: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(values)).encode("utf-8")).hexdigest()


# --- matched-manifest validation (C4) ----------------------------------------------------------

def verify_pair(reference: str, mutant: str, call: str, value: str) -> dict[str, bool]:
    """Executes ``assert call == value`` on the reference and the mutant (restricted worker)."""
    row = classify_assertions([f"assert {call} == ({value})"], reference, mutant)[0]
    return {"passes_reference": bool(row.get("valid")),
            "discriminates": bool(row.get("killed"))}


def validate_manifest(rows: Sequence[Mapping[str, Any]], *, gate_groups: Iterable[str],
                      confirmation_groups: Iterable[str], group_cap: int,
                      require_discrimination: bool = True) -> dict[str, Any]:
    """Refuse a matched C/T manifest unless every C4 rule holds; report its composition.

    Each row: record_id, group_id, call, value, execution_mode, representation,
    passes_reference, discriminates, dataset, bug_family, complexity_tier,
    input_ids_sha256, control_labels_sha256, treatment_labels_sha256,
    control_input_ids_sha256, treatment_input_ids_sha256, order_index_control,
    order_index_treatment, tokens.
    """
    problems: list[str] = []
    held_out = set(gate_groups) | set(confirmation_groups)
    seen, per_group = set(), Counter()
    for row in rows:
        key = (row["record_id"], row["call"], row["value"])
        if key in seen:
            problems.append(f"repeated row {key}")
        seen.add(key)
        per_group[row["group_id"]] += 1
        if not row.get("passes_reference"):
            problems.append(f"{row['record_id']}: call/value fails on the reference")
        if require_discrimination and not row.get("discriminates"):
            problems.append(f"{row['record_id']}: call does not discriminate buggy and fixed")
        if row.get("execution_mode", "function") != "function" and not row.get("representation"):
            problems.append(f"{row['record_id']}: non-function record without a valid "
                            "training representation")
        if row["group_id"] in held_out:
            problems.append(f"{row['record_id']}: gate/confirmation group in training")
        if row["control_input_ids_sha256"] != row["treatment_input_ids_sha256"]:
            problems.append(f"{row['record_id']}: input sequences differ between arms")
        if row["control_labels_sha256"] == row["treatment_labels_sha256"]:
            problems.append(f"{row['record_id']}: label masks are identical")
        if row["order_index_control"] != row["order_index_treatment"]:
            problems.append(f"{row['record_id']}: example order differs between arms")
    for group, count in per_group.items():
        if count > group_cap:
            problems.append(f"group {group} has {count} rows > cap {group_cap}")
    if problems:
        raise ValueError("manifest refused: " + "; ".join(problems[:10]))
    return {"rows": len(rows), "groups": len(per_group),
            "dataset": dict(Counter(r["dataset"] for r in rows)),
            "bug_family": dict(Counter(r["bug_family"] for r in rows)),
            "complexity_tier": dict(Counter(r["complexity_tier"] for r in rows)),
            "tokens": {"input": sum(r["tokens"]["input"] for r in rows),
                       "control_supervised": sum(r["tokens"]["control_supervised"] for r in rows),
                       "treatment_supervised": sum(r["tokens"]["treatment_supervised"]
                                                   for r in rows)},
            "examples_matched": True, "order_matched": True}
