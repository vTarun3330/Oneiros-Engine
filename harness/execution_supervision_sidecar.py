"""Immutable data contracts for the execution-supervision pilot.

This module contains no corpus loader and cannot open evaluation data.  It
operates on already-audited train rows, freezes group-level partitions, and
constructs two example-level arms whose completion bytes are identical at
every position.  Corpus construction lives in a separate script so the
trainer can validate a small, closed artifact without knowing how it was
mined.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "oneiros_execution_supervision_ab_v1"
SPLIT_SALT = "oneiros_execution_supervision_v1"
ARM_SIZE = 1024
REPLACEMENT_COUNT = 128
TRAIN_LINEAGES = 385
PILOT_LINEAGES = 100
CONFIRM_LINEAGES = 100


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_rank(*parts: object) -> str:
    material = "|".join([SPLIT_SALT, *(str(part) for part in parts)])
    return sha256_text(material)


def freeze_lineage_split(
    rows: Iterable[dict[str, Any]],
    *,
    train_count: int = TRAIN_LINEAGES,
    pilot_count: int = PILOT_LINEAGES,
    confirm_count: int = CONFIRM_LINEAGES,
) -> dict[str, Any]:
    """Freeze a deterministic, source-stratified split of eligible lineages.

    A lineage is assigned only after upstream safety/giveaway filters.  Source
    labels must agree within a lineage.  Counts are apportioned by largest
    remainder independently for train then pilot; confirmation receives the
    exact residual.  No failed runtime row is ever backfilled across these
    boundaries later.
    """
    source_by_lineage: dict[str, str] = {}
    for row in rows:
        lineage = str(row.get("function_lineage") or "")
        source = str(row.get("source_dataset") or "unknown")
        if not lineage:
            raise ValueError("eligible row has no function_lineage")
        previous = source_by_lineage.setdefault(lineage, source)
        if previous != source:
            raise ValueError(f"lineage {lineage} crosses source datasets")
    total_requested = train_count + pilot_count + confirm_count
    if len(source_by_lineage) != total_requested:
        raise ValueError(
            f"eligible lineage universe is {len(source_by_lineage)}, expected "
            f"exactly {total_requested}; do not hide attrition or backfill"
        )
    by_source: dict[str, list[str]] = defaultdict(list)
    for lineage, source in source_by_lineage.items():
        by_source[source].append(lineage)
    for source in by_source:
        by_source[source].sort(key=lambda item: stable_rank(source, item))

    def quotas(count: int, available: dict[str, int]) -> dict[str, int]:
        denominator = sum(available.values())
        raw = {source: count * size / denominator for source, size in available.items()}
        result = {source: int(value) for source, value in raw.items()}
        left = count - sum(result.values())
        order = sorted(
            available,
            key=lambda source: (-(raw[source] - result[source]), source),
        )
        for source in order[:left]:
            result[source] += 1
        if any(result[source] > available[source] for source in result):
            raise ValueError("stratified quota exceeds available lineages")
        return result

    train_quota = quotas(train_count, {key: len(value) for key, value in by_source.items()})
    train: list[str] = []
    remainder: dict[str, list[str]] = {}
    for source, lineages in sorted(by_source.items()):
        take = train_quota[source]
        train.extend(lineages[:take])
        remainder[source] = lineages[take:]
    pilot_quota = quotas(pilot_count, {key: len(value) for key, value in remainder.items()})
    pilot: list[str] = []
    confirm: list[str] = []
    for source, lineages in sorted(remainder.items()):
        take = pilot_quota[source]
        pilot.extend(lineages[:take])
        confirm.extend(lineages[take:])
    if len(confirm) != confirm_count:
        raise ValueError("confirmation split does not have the declared exact size")

    result = {
        "schema_version": SCHEMA,
        "salt": SPLIT_SALT,
        "eligible_lineages": len(source_by_lineage),
        "train_lineages": sorted(train),
        "pilot_development_lineages": sorted(pilot),
        "unopened_confirmation_lineages": sorted(confirm),
        "source_counts": {
            "eligible": dict(sorted(Counter(source_by_lineage.values()).items())),
            "train": dict(sorted(Counter(source_by_lineage[item] for item in train).items())),
            "pilot_development": dict(sorted(
                Counter(source_by_lineage[item] for item in pilot).items()
            )),
            "unopened_confirmation": dict(sorted(
                Counter(source_by_lineage[item] for item in confirm).items()
            )),
        },
        "no_lineage_overlap": not (
            set(train) & set(pilot)
            or set(train) & set(confirm)
            or set(pilot) & set(confirm)
        ),
        "no_backfill_after_runtime_filters": True,
    }
    result["split_sha256"] = sha256_text(json.dumps(
        result, sort_keys=True, separators=(",", ":")
    ))
    return result


def _training_row(row: dict[str, Any], *, prompt: str, task_kind: str) -> dict[str, Any]:
    completion = str(row["completion"])
    return {
        "record_id": str(row["record_id"]),
        "function_lineage": str(row["function_lineage"]),
        "source_dataset": str(row.get("source_dataset") or "unknown"),
        "bug_family": str(row.get("bug_family") or "unknown"),
        "complexity_tier": str(row.get("complexity_tier") or "unknown"),
        "prompt": prompt,
        "completion": completion,
        "prompt_sha256": sha256_text(prompt),
        "completion_sha256": sha256_text(completion),
        "task_kind": task_kind,
        "execution_mode": str(row.get("execution_mode") or "function_assertion"),
    }


def assemble_controlled_arms(
    shared_rows: Sequence[dict[str, Any]],
    replacement_rows: Sequence[dict[str, Any]],
    *,
    arm_size: int = ARM_SIZE,
    replacement_count: int = REPLACEMENT_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Construct matched arms; only the declared replacement prompts differ."""
    if len(shared_rows) != arm_size - replacement_count:
        raise ValueError("shared row count does not match the frozen arm size")
    if len(replacement_rows) != replacement_count:
        raise ValueError("replacement row count does not match the frozen intervention")

    # Spread replacements deterministically through the ordered arm.  The
    # completion at every position remains byte-identical between arms.
    positions = {
        min(arm_size - 1, int((index + 0.5) * arm_size / replacement_count))
        for index in range(replacement_count)
    }
    if len(positions) != replacement_count:
        raise ValueError("replacement position construction collided")
    shared_iter = iter(shared_rows)
    replacement_iter = iter(replacement_rows)
    control: list[dict[str, Any]] = []
    treatment: list[dict[str, Any]] = []
    declared_positions: list[int] = []
    for position in range(arm_size):
        if position in positions:
            row = next(replacement_iter)
            completion = str(row["completion"])
            canonical_prompt = str(row["canonical_prompt"])
            focused_prompt = str(row["focused_prompt"])
            control.append(_training_row(
                row, prompt=canonical_prompt, task_kind="test_generation"
            ))
            treatment.append(_training_row(
                row, prompt=focused_prompt,
                task_kind="execution_output_prediction",
            ))
            if control[-1]["completion"] != completion:
                raise AssertionError("control completion changed")
            declared_positions.append(position)
        else:
            row = next(shared_iter)
            canonical_prompt = str(row["canonical_prompt"])
            item = _training_row(
                row, prompt=canonical_prompt, task_kind="test_generation"
            )
            control.append(item)
            treatment.append(dict(item))
    if len(control) != arm_size or len(treatment) != arm_size:
        raise AssertionError("arm size drift")
    differing = []
    for position, (left, right) in enumerate(zip(control, treatment)):
        left_without_prompt = {key: value for key, value in left.items()
                               if key not in {"prompt", "prompt_sha256", "task_kind"}}
        right_without_prompt = {key: value for key, value in right.items()
                                if key not in {"prompt", "prompt_sha256", "task_kind"}}
        if left_without_prompt != right_without_prompt:
            raise ValueError(f"arm metadata/completion differs at position {position}")
        if left != right:
            differing.append(position)
    if differing != declared_positions:
        raise ValueError("arms differ outside the declared replacement positions")
    report = {
        "schema_version": SCHEMA,
        "arm_size": arm_size,
        "shared_examples": len(shared_rows),
        "replacement_examples": replacement_count,
        "replacement_positions": declared_positions,
        "completion_bytes_equal_at_every_position": all(
            left["completion"] == right["completion"]
            for left, right in zip(control, treatment)
        ),
        "completion_hashes_equal_at_every_position": all(
            left["completion_sha256"] == right["completion_sha256"]
            for left, right in zip(control, treatment)
        ),
        "only_prompt_and_task_kind_change_at_replacements": True,
    }
    return control, treatment, report


def verify_arm_file(path: str | Path, expected_sha256: str) -> list[dict[str, Any]]:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"arm artifact hash mismatch: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != ARM_SIZE:
        raise ValueError("arm artifact does not contain exactly 1024 examples")
    for row in rows:
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        if sha256_text(prompt) != row.get("prompt_sha256"):
            raise ValueError("arm prompt hash mismatch")
        if sha256_text(completion) != row.get("completion_sha256"):
            raise ValueError("arm completion hash mismatch")
        if row.get("task_kind") not in {
            "test_generation", "execution_output_prediction",
        }:
            raise ValueError("arm row has an unsupported task kind")
    return rows
