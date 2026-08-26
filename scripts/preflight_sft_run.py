"""Exact, GPU-free preflight for a bounded Oneiros V3 SFT run.

This script mirrors the production SFT data path through canonical-corpus
verification, bounded selection, behavioral winner verification, the deployed
completion-token gate, exact deduplication, repository balancing, prompt
compaction accounting, and optimizer schedule planning.  It never loads a
model, submits Modal work, or mutates a corpus/checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CANONICAL_CORPUS_VERSION, model_config, training_config
from engine.prompt_budget import PROMPT_COMPACTION_STRATEGY, compact_prompt_token_ids
from engine.sft_trainer import (
    MAX_SFT_SEQUENCE_LENGTH,
    SFTDataPoint,
    plan_sft_optimizer_schedule,
    sft_completion_limit_for_execution_mode,
)
from harness.corpus import verify_corpus
from scripts.train_on_dataset import (
    FUNCTION_EXECUTION_MODE,
    _filter_overlong_repository_completions,
    _repository_fragment_tests,
    balanced_repeat_examples,
    build_prompt,
    deduplicate_sft_examples,
    evaluate_pair,
    extract_dataset_tests,
    filter_generation_compatible_sft_examples,
    format_generation_prompt,
    is_repository_execution_mode,
    load_phase3_pairs,
    select_bounded_train_pairs,
    supervision_exclusion_summary,
    summarize_train_pair_selection,
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(value or "unknown") for value in values).items()))


def preflight_gates_pass(gates: dict[str, bool]) -> bool:
    """Fail closed unless every declared full-run readiness gate passes."""
    return bool(gates) and all(value is True for value in gates.values())


def _token_summary(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {"minimum": 0, "median": 0, "p95": 0, "maximum": 0}
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        return ordered[math.ceil(fraction * len(ordered)) - 1]

    return {
        "minimum": ordered[0],
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


def _make_data_point(pair: dict, prompt: str, completion: str) -> SFTDataPoint:
    return SFTDataPoint(
        prompt=prompt,
        completion=completion,
        function_id=pair["id"],
        project=pair.get("project", "synthetic"),
        bug_family=pair.get("bug_family", "unknown"),
        semantic_group=pair.get("group_id", pair["id"]),
        execution_mode=pair.get("execution_mode", FUNCTION_EXECUTION_MODE),
    )


def build_preflight(
    corpus_version: str,
    max_pairs: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_scheduler_type: str,
    real_target_fraction: float,
    max_real_repeats: int,
    synthetic_balance_fraction: float,
    max_synthetic_repeats: int,
    prompt_token_limit: int,
    completion_token_limit: int,
    repository_completion_token_limit: int,
    warmup_steps: int,
    checkpoint_steps: int,
    minimum_monitor_checkpoints: int,
    min_function_kill_rate: float,
    local_files_only: bool,
) -> dict[str, Any]:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if not 0.0 <= real_target_fraction < 1.0:
        raise ValueError("real_target_fraction must be in [0, 1)")
    if lr_scheduler_type not in {"cosine", "constant_with_warmup"}:
        raise ValueError("Unsupported SFT LR scheduler")
    if not 0.0 <= min_function_kill_rate <= 1.0:
        raise ValueError("min_function_kill_rate must be in [0, 1]")
    if not 0 < repository_completion_token_limit < MAX_SFT_SEQUENCE_LENGTH:
        raise ValueError(
            "repository_completion_token_limit must be between 1 and the sequence limit"
        )
    started = time.time()
    corpus_dir = ROOT / "data" / "corpus" / corpus_version
    manifest = verify_corpus(corpus_dir)
    readiness_path = ROOT / "results" / f"{corpus_version}_train_readiness.json"
    if not readiness_path.exists():
        raise RuntimeError(f"Missing locked train-readiness audit: {readiness_path}")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not readiness.get("ready"):
        raise RuntimeError("Locked train-readiness audit is not ready=true")
    if readiness.get("corpus_id") != manifest.get("corpus_id"):
        raise RuntimeError("Train-readiness audit does not match the verified corpus")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        revision=model_config.model_revision,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        raise RuntimeError("Tokenizer has no EOS token")

    source_pairs = load_phase3_pairs(corpus_dir, "train")
    eligible_pairs, context_exclusions = _filter_overlong_repository_completions(
        source_pairs
    )
    locked_context_exclusions = readiness.get(
        "repository_overlong_completions_excluded", []
    )
    if len(context_exclusions) != len(locked_context_exclusions):
        raise RuntimeError(
            "Production context gate disagrees with the locked readiness audit"
        )

    compatible_repository_ids: set[str] = set()
    compatible_repository_completion_counts: Counter[str] = Counter()
    context_eligible_repository_pairs = 0
    for pair in eligible_pairs:
        if not is_repository_execution_mode(
            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        ):
            continue
        context_eligible_repository_pairs += 1
        compatible = 0
        for completion in _repository_fragment_tests(pair.get("test_cases", []))[:3]:
            completion_tokens = len(
                tokenizer(
                    completion.strip() + tokenizer.eos_token,
                    add_special_tokens=False,
                )["input_ids"]
            )
            compatible += int(
                completion_tokens <= repository_completion_token_limit
            )
        if compatible:
            compatible_repository_ids.add(pair["id"])
            compatible_repository_completion_counts[pair["id"]] = compatible

    selected_pairs = select_bounded_train_pairs(
        eligible_pairs,
        max_pairs,
        compatible_repository_ids=compatible_repository_ids,
        target_real_fraction=real_target_fraction,
        max_real_repeats=max_real_repeats,
    )
    selection_stats = summarize_train_pair_selection(selected_pairs)
    selection_sha256 = _sha256_json([pair["id"] for pair in selected_pairs])

    synthetic_examples: list[SFTDataPoint] = []
    repository_examples: list[SFTDataPoint] = []
    records_without_winners: list[str] = []
    for index, pair in enumerate(selected_pairs, start=1):
        mode = pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        if is_repository_execution_mode(mode):
            winners = _repository_fragment_tests(pair.get("test_cases", []))
        else:
            source_tests = extract_dataset_tests(
                pair.get("test_cases", []), pair["entry_point"]
            )
            winners, _ = evaluate_pair(
                source_tests, pair["golden_code"], pair["mutant_code"],
                pair["entry_point"],
            )
        if not winners:
            records_without_winners.append(pair["id"])
            continue
        prompt = build_prompt(
            pair["mutant_code"],
            pair["entry_point"],
            pair.get("specification", ""),
            mode,
        )
        target = (
            repository_examples
            if is_repository_execution_mode(mode)
            else synthetic_examples
        )
        target.extend(
            _make_data_point(pair, prompt, completion)
            for completion in winners[:3]
        )
        if index % 100 == 0 or index == len(selected_pairs):
            print(
                f"verified {index}/{len(selected_pairs)} selected records: "
                f"synthetic={len(synthetic_examples)} "
                f"repository={len(repository_examples)}",
                flush=True,
            )

    raw_synthetic_examples = len(synthetic_examples)
    raw_repository_examples = len(repository_examples)
    synthetic_examples, synthetic_generation_exclusions = (
        filter_generation_compatible_sft_examples(
            synthetic_examples,
            tokenizer,
            completion_token_limit,
            repository_completion_token_limit,
        )
    )
    repository_examples, repository_generation_exclusions = (
        filter_generation_compatible_sft_examples(
            repository_examples,
            tokenizer,
            completion_token_limit,
            repository_completion_token_limit,
        )
    )
    synthetic_examples, synthetic_deduplication = deduplicate_sft_examples(
        synthetic_examples
    )
    repository_examples, repository_deduplication = deduplicate_sft_examples(
        repository_examples
    )
    verified_supervision_exclusions = supervision_exclusion_summary(
        records_without_winners
    )
    generation_compatible_record_ids = {
        example.function_id
        for example in [*synthetic_examples, *repository_examples]
    }
    no_verified_winner_ids = set(records_without_winners)
    raw_verified_record_ids = {
        pair["id"] for pair in selected_pairs
        if pair["id"] not in no_verified_winner_ids
    }
    records_without_generation_compatible_winners = [
        pair["id"] for pair in selected_pairs
        if pair["id"] in raw_verified_record_ids
        and pair["id"] not in generation_compatible_record_ids
    ]

    effective_synthetic = list(synthetic_examples)
    synthetic_sampling = None
    if synthetic_examples and synthetic_balance_fraction:
        effective_synthetic, synthetic_sampling = balanced_repeat_examples(
            synthetic_examples,
            math.ceil(len(synthetic_examples) * (1.0 + synthetic_balance_fraction)),
            max_synthetic_repeats,
            "semantic_group",
        )

    desired_repository_examples = (
        math.ceil(
            len(effective_synthetic)
            * real_target_fraction
            / (1.0 - real_target_fraction)
        )
        if real_target_fraction
        else len(repository_examples)
    )
    effective_repository, repository_sampling = balanced_repeat_examples(
        repository_examples,
        desired_repository_examples,
        max_real_repeats,
        "project",
    )
    effective_examples = [*effective_synthetic, *effective_repository]
    effective_real_fraction = (
        len(effective_repository) / len(effective_examples)
        if effective_examples
        else 0.0
    )

    raw_prompt_lengths: list[int] = []
    retained_prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    compacted_examples: list[SFTDataPoint] = []
    sequence_overflow_examples: list[dict[str, Any]] = []
    for example in effective_examples:
        prompt_ids = tokenizer(
            format_generation_prompt(tokenizer, example.prompt),
            add_special_tokens=False,
        )["input_ids"]
        completion_ids = tokenizer(
            example.completion.strip() + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        mode_completion_limit = sft_completion_limit_for_execution_mode(
            example.execution_mode,
            completion_token_limit,
            repository_completion_token_limit,
        )
        if len(completion_ids) > mode_completion_limit:
            sequence_overflow_examples.append(
                {
                    "record_id": example.function_id,
                    "execution_mode": example.execution_mode,
                    "completion_tokens": len(completion_ids),
                    "completion_limit": mode_completion_limit,
                    "reason": "mode_completion_limit_exceeded_after_sampling",
                }
            )
        allowed_prompt_tokens = min(
            prompt_token_limit,
            MAX_SFT_SEQUENCE_LENGTH - len(completion_ids),
        )
        compacted_prompt_ids, was_compacted = compact_prompt_token_ids(
            prompt_ids, allowed_prompt_tokens
        )
        raw_prompt_lengths.append(len(prompt_ids))
        retained_prompt_lengths.append(len(compacted_prompt_ids))
        completion_lengths.append(len(completion_ids))
        if was_compacted:
            compacted_examples.append(example)
        if len(compacted_prompt_ids) + len(completion_ids) > MAX_SFT_SEQUENCE_LENGTH:
            sequence_overflow_examples.append(
                {
                    "record_id": example.function_id,
                    "prompt_tokens": len(compacted_prompt_ids),
                    "completion_tokens": len(completion_ids),
                }
            )

    schedule = plan_sft_optimizer_schedule(
        len(effective_examples),
        epochs,
        batch_size,
        warmup_steps,
        checkpoint_steps,
    )
    required_optimizer_steps = checkpoint_steps * minimum_monitor_checkpoints
    schedule["minimum_monitor_checkpoints"] = minimum_monitor_checkpoints
    schedule["minimum_monitored_optimizer_steps"] = required_optimizer_steps
    checkpoints = list(range(checkpoint_steps, schedule["planned_optimizer_steps"] + 1, checkpoint_steps))
    if schedule["planned_optimizer_steps"] not in checkpoints:
        checkpoints.append(schedule["planned_optimizer_steps"])

    selected_repository_pairs = [
        pair
        for pair in selected_pairs
        if is_repository_execution_mode(
            pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        )
    ]
    selected_generation_compatible_repository_pairs = sum(
        pair["id"] in compatible_repository_ids
        for pair in selected_repository_pairs
    )
    gates = {
        "corpus_verified": True,
        "locked_train_readiness": True,
        "verified_supervision_exclusions_accounted": (
            verified_supervision_exclusions["count"]
            == len(no_verified_winner_ids)
        ),
        "verified_synthetic_supervision_present": bool(synthetic_examples),
        "generation_compatible_supervision_present": bool(effective_examples),
        "zero_sequence_overflows": not sequence_overflow_examples,
        "minimum_monitor_schedule_reached": (
            schedule["planned_optimizer_steps"] >= required_optimizer_steps
        ),
        "repository_supervision_present": bool(repository_examples),
        "real_fraction_target_reached": (
            effective_real_fraction + 1e-12 >= real_target_fraction
        ),
    }
    ready = preflight_gates_pass(gates)
    return {
        "ready": ready,
        "mode": "exact_local_sft_preflight",
        "modal_used": False,
        "gpu_model_loaded": False,
        "corpus_modified": False,
        "checkpoint_modified": False,
        "elapsed_seconds": round(time.time() - started, 3),
        "corpus": {
            "version": corpus_version,
            "corpus_id": manifest["corpus_id"],
            "training_records": manifest["training_records"],
            "train_records": len(source_pairs),
            "context_eligible_train_records": len(eligible_pairs),
            "context_excluded_repository_completions": len(context_exclusions),
            "records_sha256": manifest["files"]["records.json"]["sha256"],
            "splits_sha256": manifest["files"]["splits.json"]["sha256"],
        },
        "selection": {
            "requested_pairs": max_pairs,
            "retained_pairs": len(selected_pairs),
            "selection_sha256": selection_sha256,
            "context_eligible_repository_pairs": context_eligible_repository_pairs,
            "generation_compatible_repository_pairs": len(compatible_repository_ids),
            "selected_generation_compatible_repository_pairs": (
                selected_generation_compatible_repository_pairs
            ),
            "selected_repository_record_ids": [
                pair["id"] for pair in selected_repository_pairs
            ],
            **selection_stats,
        },
        "supervision": {
            "raw_synthetic_examples": raw_synthetic_examples,
            "raw_repository_examples": raw_repository_examples,
            "generation_compatible_synthetic_examples": len(synthetic_examples),
            "generation_compatible_repository_examples": len(repository_examples),
            "generation_exclusions": {
                "synthetic": len(synthetic_generation_exclusions),
                "repository": len(repository_generation_exclusions),
                "details": [
                    *synthetic_generation_exclusions,
                    *repository_generation_exclusions,
                ],
            },
            "deduplication": {
                "synthetic": synthetic_deduplication,
                "repository": repository_deduplication,
            },
            "records_without_verified_winners": records_without_winners,
            "verified_supervision_exclusions": verified_supervision_exclusions,
            "records_without_generation_compatible_winners": (
                records_without_generation_compatible_winners
            ),
            "raw_repository_project_counts": _counts(
                item.project for item in repository_examples
            ),
            "raw_synthetic_bug_family_counts": _counts(
                item.bug_family for item in synthetic_examples
            ),
        },
        "sampling": {
            "target_real_fraction": real_target_fraction,
            "actual_real_fraction": round(effective_real_fraction, 6),
            "target_reached": effective_real_fraction + 1e-12 >= real_target_fraction,
            "effective_synthetic_examples": len(effective_synthetic),
            "desired_repository_examples": desired_repository_examples,
            "effective_repository_examples": len(effective_repository),
            "effective_total_examples": len(effective_examples),
            "max_real_repeats": max_real_repeats,
            "repository_project_balance": repository_sampling,
            "synthetic_balance_fraction": synthetic_balance_fraction,
            "synthetic_semantic_group_balance": synthetic_sampling,
        },
        "tokenization": {
            "model_name": model_config.model_name,
            "prompt_token_limit": prompt_token_limit,
            "completion_token_limit": completion_token_limit,
            "repository_completion_token_limit": repository_completion_token_limit,
            "sequence_token_limit": MAX_SFT_SEQUENCE_LENGTH,
            "prompt_compaction_strategy": PROMPT_COMPACTION_STRATEGY,
            "prompt_compacted_examples": len(compacted_examples),
            "prompt_compacted_fraction": round(
                len(compacted_examples) / len(effective_examples), 6
            ),
            "prompt_compacted_by_execution_mode": _counts(
                item.execution_mode for item in compacted_examples
            ),
            "raw_prompt_tokens": _token_summary(raw_prompt_lengths),
            "retained_prompt_tokens": _token_summary(retained_prompt_lengths),
            "completion_tokens": _token_summary(completion_lengths),
            "sequence_overflow_examples": sequence_overflow_examples,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_steps_requested": warmup_steps,
            "checkpoint_steps": checkpoint_steps,
            "planned_validation_checkpoints": checkpoints,
            "min_function_kill_rate": min_function_kill_rate,
            "optimizer_schedule": schedule,
        },
        "gates": gates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", default=CANONICAL_CORPUS_VERSION)
    parser.add_argument("--max-pairs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=training_config.sft_epochs)
    parser.add_argument("--batch-size", type=int, default=training_config.sft_batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=training_config.sft_learning_rate
    )
    parser.add_argument(
        "--lr-scheduler-type",
        choices=["cosine", "constant_with_warmup"],
        default=training_config.sft_lr_scheduler_type,
    )
    parser.add_argument("--real-target-fraction", type=float, default=0.20)
    parser.add_argument("--max-real-repeats", type=int, default=8)
    parser.add_argument("--synthetic-balance-fraction", type=float, default=0.0)
    parser.add_argument("--max-synthetic-repeats", type=int, default=2)
    parser.add_argument(
        "--prompt-token-limit",
        type=int,
        default=training_config.sft_prompt_token_limit,
    )
    parser.add_argument(
        "--completion-token-limit",
        type=int,
        default=training_config.sft_completion_token_limit,
    )
    parser.add_argument(
        "--repository-completion-token-limit",
        type=int,
        default=training_config.sft_repository_completion_token_limit,
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=training_config.sft_warmup_steps
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=training_config.sft_checkpoint_steps
    )
    parser.add_argument(
        "--minimum-monitor-checkpoints",
        type=int,
        default=training_config.sft_min_monitor_checkpoints,
    )
    parser.add_argument(
        "--min-function-kill-rate",
        type=float,
        default=training_config.sft_min_function_kill_rate,
    )
    parser.add_argument(
        "--allow-tokenizer-download",
        action="store_true",
        help="Allow Hugging Face network access if the tokenizer is not cached.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "v3_hardened_sft_preflight.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_preflight(
        corpus_version=args.corpus_version,
        max_pairs=args.max_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        real_target_fraction=args.real_target_fraction,
        max_real_repeats=args.max_real_repeats,
        synthetic_balance_fraction=args.synthetic_balance_fraction,
        max_synthetic_repeats=args.max_synthetic_repeats,
        prompt_token_limit=args.prompt_token_limit,
        completion_token_limit=args.completion_token_limit,
        repository_completion_token_limit=args.repository_completion_token_limit,
        warmup_steps=args.warmup_steps,
        checkpoint_steps=args.checkpoint_steps,
        minimum_monitor_checkpoints=args.minimum_monitor_checkpoints,
        min_function_kill_rate=args.min_function_kill_rate,
        local_files_only=not args.allow_tokenizer_download,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
