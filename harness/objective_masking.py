"""Phase 4 (V2 design, NOT launched): matched control/treatment examples that differ
ONLY in their label mask -- built through the PRODUCTION SFT data path.

There is one tokenization path.  ``make_datapoint`` builds an ``SFTDataPoint`` whose
prompt is the exact fixed-call probe prompt (``harness.fixed_input_probe.build_prompt``,
level A0) and whose completion is ``assert <call> == <value>`` with the value's
character span.  ``prepare_arm`` runs ``engine.sft_trainer.OneirosSFTTrainer.
prepare_dataset`` itself (task-kind-aware prompt renderer and compaction,
``completion.strip() + eos_token`` tokenization, sequence limits) with
``objective_mode`` set to ``full_completion`` (control) or ``value_only``
(treatment).  Both arms therefore share prompts, input_ids, attention masks and
EOS handling by construction; only the label rows differ:

* control labels: every completion token (assertion, call, ``==``, value, EOS);
* treatment labels: only the complete value span and EOS.

The value span is located POSITIONALLY from character offsets, never by substring
search (``engine.sft_trainer.value_token_labels``); unalignable examples are refused.

Also provides a matched-manifest validator (C4).  It is exercised only on toy fixtures
in this task; no real training manifest is frozen or emitted.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from config import training_config
from engine.sft_trainer import (
    IGNORE_INDEX, ObjectiveAlignmentError, OneirosSFTTrainer, SFTDataPoint,
)
from harness.fixed_input_probe import build_prompt
from harness.safe_execution import classify_assertions

IGNORE = IGNORE_INDEX
AlignmentError = ObjectiveAlignmentError
DESIGN_VERSION = "oneiros_phase4_objective_masking_v3_production_path"
ARMS = {"control": "full_completion", "treatment": "value_only"}


def completion_text(call: str, value: str) -> tuple[str, int, int]:
    prefix = f"assert {call} == "
    return prefix + value, len(prefix), len(prefix) + len(value)


def make_datapoint(record: Mapping[str, Any], call: str, value: str,
                   expected_repr: str | None = None) -> SFTDataPoint:
    """One fixed-call training example (identical for both arms)."""
    if value != value.strip() or not value:
        raise AlignmentError("the value must be non-empty canonical text without outer spaces")
    user = build_prompt(record, call, expected_repr or value, "A0")["user"]
    body, start, end = completion_text(call, value)
    return SFTDataPoint(prompt=user, completion=body, function_id=str(record.get("id", "")),
                        semantic_group=str(record.get("group_id", "unknown")),
                        execution_mode="function_assertion", task_kind="test_generation",
                        supervised_char_span=(start, end))


def dataset_trainer(tokenizer, objective_mode: str | None) -> OneirosSFTTrainer:
    """A trainer object with the production data path and limits but no model (CPU)."""
    trainer = object.__new__(OneirosSFTTrainer)
    trainer.tokenizer = tokenizer
    trainer.objective_mode = objective_mode
    trainer.max_prompt_tokens = training_config.sft_prompt_token_limit
    trainer.max_repository_prompt_tokens = training_config.sft_repository_prompt_token_limit
    trainer.max_completion_tokens = training_config.sft_completion_token_limit
    trainer.max_repository_completion_tokens =         training_config.sft_repository_completion_token_limit
    trainer.dataset_stats = {}
    return trainer


def prepare_arm(tokenizer, datapoints: Sequence[SFTDataPoint], arm: str):
    """The production dataset for one arm, plus the trainer's recorded stats."""
    trainer = dataset_trainer(tokenizer, ARMS[arm])
    dataset = trainer.prepare_dataset(list(datapoints))
    return dataset, dict(trainer.dataset_stats)


def build_example(tokenizer, record: Mapping[str, Any], call: str, value: str
                  ) -> dict[str, Any]:
    """Both arms for one example via the production path (used by tests/receipts)."""
    dp = make_datapoint(record, call, value)
    control, _ = prepare_arm(tokenizer, [dp], "control")
    treatment, _ = prepare_arm(tokenizer, [dp], "treatment")
    c, t = control[0], treatment[0]
    start = c["completion_start"]
    body, v_start, v_end = completion_text(call, value)
    supervised = [i for i, x in enumerate(t["labels"]) if x != IGNORE]
    return {"input_ids": list(c["input_ids"]), "treatment_input_ids": list(t["input_ids"]),
            "attention_mask": list(c["attention_mask"]),
            "treatment_attention_mask": list(t["attention_mask"]),
            "labels": {"control": list(c["labels"]), "treatment": list(t["labels"])},
            "completion_text": body, "completion_start": start,
            "treatment_completion_start": t["completion_start"],
            "value_token_positions": supervised[:-1],
            "value_char_span": (v_start, v_end)}


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
