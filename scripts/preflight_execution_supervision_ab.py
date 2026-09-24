"""Fail-closed CPU preflight for the matched execution-supervision A/B pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.sft_trainer import (
    OneirosSFTTrainer,
    SFTDataPoint,
    plan_sft_optimizer_schedule,
)
from harness.execution_supervision_sidecar import (
    ARM_SIZE,
    REPLACEMENT_COUNT,
    SCHEMA,
    sha256_file,
    verify_arm_file,
)


MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
SEED = 42
LEARNING_RATE = 1e-5
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_STEPS = 25
CHECKPOINT_STEPS = 32


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_points(rows: list[dict[str, Any]]) -> list[SFTDataPoint]:
    return [
        SFTDataPoint(
            prompt=str(row["prompt"]),
            completion=str(row["completion"]),
            function_id=str(row["record_id"]),
            project=(
                "synthetic" if row["source_dataset"] in {
                    "mbpp", "humaneval", "manual_curated_examples"
                } else str(row["source_dataset"])
            ),
            bug_family=str(row["bug_family"]),
            semantic_group=str(row["function_lineage"]),
            execution_mode=str(row["execution_mode"]),
            dataset=str(row["source_dataset"]),
            dataset_family=(
                f"{row['source_dataset']}::{row['bug_family']}"
            ),
            task_kind=str(row["task_kind"]),
        )
        for row in rows
    ]


def _source_problems(manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for relative, expected in (manifest.get("source_files_sha256") or {}).items():
        path = ROOT / relative
        if not path.exists():
            problems.append(f"source missing: {relative}")
        elif _sha(path) != expected:
            problems.append(f"source drift: {relative}")
    return problems


def preflight(dataset_dir: Path, output: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest.get("schema_version") != SCHEMA:
        problems.append("dataset schema mismatch")
    for key in (
        "sealed_final_test_accessed", "ablation_dev_accessed",
        "validation_accessed", "test_accessed", "canonical_records_json_opened",
    ):
        if manifest.get(key) is not False:
            problems.append(f"dataset isolation field {key} is not false")
    if manifest.get("evaluation_split") != "train":
        problems.append("dataset is not train-only")
    problems.extend(_source_problems(manifest))

    arm_a_path = dataset_dir / "arm_a.control.json"
    arm_b_path = dataset_dir / "arm_b.treatment.json"
    try:
        arm_a = verify_arm_file(arm_a_path, str(manifest.get("arm_a_sha256")))
        arm_b = verify_arm_file(arm_b_path, str(manifest.get("arm_b_sha256")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        problems.append(str(exc))
        arm_a, arm_b = [], []

    replacement_positions = list(
        (manifest.get("arm_report") or {}).get("replacement_positions") or []
    )
    if len(replacement_positions) != REPLACEMENT_COUNT:
        problems.append("replacement position count is not 128")
    changed: list[int] = []
    if len(arm_a) == len(arm_b) == ARM_SIZE:
        for position, (left, right) in enumerate(zip(arm_a, arm_b)):
            if left["completion"] != right["completion"]:
                problems.append(f"completion differs at position {position}")
            if left != right:
                changed.append(position)
        if changed != replacement_positions:
            problems.append("arms differ outside the declared positions")

    # Training launch requires committed, clean source. This is intentionally
    # checked before tokenizer/model preparation so a dirty run costs nothing.
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    if status:
        problems.append("working tree is not clean")

    prepared: dict[str, Any] = {}
    if not problems:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        datasets = {}
        stats = {}
        for name, rows in (("control", arm_a), ("treatment", arm_b)):
            trainer = OneirosSFTTrainer(
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                output_dir=ROOT / "checkpoints" / f"preflight_execsup_{name}",
                learning_rate=LEARNING_RATE,
                max_prompt_tokens=1024,
                max_repository_prompt_tokens=2048,
                max_completion_tokens=128,
                max_repository_completion_tokens=1024,
                warmup_steps=WARMUP_STEPS,
                checkpoint_steps=CHECKPOINT_STEPS,
                lr_scheduler_type="constant_with_warmup",
            )
            trainer.tokenizer = tokenizer
            try:
                datasets[name] = trainer.prepare_dataset(_data_points(rows))
                stats[name] = dict(trainer.dataset_stats)
            except ValueError as exc:
                problems.append(f"{name} trainer preparation failed: {exc}")
                break
        if problems:
            prepared = {"stats": stats, "complete": False}
        elif len(datasets["control"]) != ARM_SIZE or len(datasets["treatment"]) != ARM_SIZE:
            problems.append("trainer dropped one or more examples")
        else:
            supervised_equal = True
            per_position_tokens: list[int] = []
            for left, right in zip(datasets["control"], datasets["treatment"]):
                left_supervised = list(left["input_ids"])[int(left["completion_start"]):]
                right_supervised = list(right["input_ids"])[int(right["completion_start"]):]
                if left_supervised != right_supervised:
                    supervised_equal = False
                    break
                per_position_tokens.append(len(left_supervised))
            if not supervised_equal:
                problems.append("supervised token IDs differ between arms")
            prepared = {
                "stats": stats,
                "supervised_tokens_equal_at_every_position": supervised_equal,
                "total_supervised_tokens_per_arm": sum(per_position_tokens),
                "min_supervised_tokens": min(per_position_tokens),
                "max_supervised_tokens": max(per_position_tokens),
            }

    schedule = plan_sft_optimizer_schedule(
        ARM_SIZE, EPOCHS, BATCH_SIZE, WARMUP_STEPS, CHECKPOINT_STEPS,
        GRADIENT_ACCUMULATION_STEPS,
    )
    if schedule["planned_optimizer_steps"] != 64:
        problems.append("optimizer plan is not exactly 64 steps")
    if schedule["optimizer_padding_examples"] != 0:
        problems.append("optimizer plan introduces padding examples")

    receipt = {
        "schema_version": "oneiros_execution_supervision_preflight_v1",
        "ready": not problems,
        "problems": problems,
        "git": {"branch": _git("branch", "--show-current"), "commit": head,
                "clean": not bool(status)},
        "dataset": {
            "directory": dataset_dir.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "arm_a_sha256": manifest.get("arm_a_sha256"),
            "arm_b_sha256": manifest.get("arm_b_sha256"),
            "lineage_split_sha256": manifest.get("lineage_split_sha256"),
            "sidecar_sha256": manifest.get("sidecar_sha256"),
            "pilot_development_sha256": manifest.get("pilot_development_sha256"),
            "unopened_confirmation_ids_sha256": manifest.get(
                "unopened_confirmation_ids_sha256"
            ),
            "sealed_final_test_accessed": False,
            "confirmation_opened": False,
        },
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION,
                  "attention_implementation": "sdpa", "quantization": "4bit_nf4"},
        "held_constant": {
            "seed": SEED,
            "examples": ARM_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "optimizer_steps": 64,
            "learning_rate": LEARNING_RATE,
            "lr_scheduler_type": "constant_with_warmup",
            "warmup_steps": WARMUP_STEPS,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "function_prompt_tokens": 1024,
            "repository_prompt_tokens": 2048,
            "function_completion_tokens": 128,
            "repository_completion_tokens": 1024,
            "max_sequence_tokens": 3072,
            "sft_monitor_enabled": False,
            "relearning": False,
            "o1": False,
            "multi_mutant": False,
            "dpo": False,
            "rlvr": False,
        },
        "intervention": {
            "replacement_examples": REPLACEMENT_COUNT,
            "replacement_positions": replacement_positions,
            "only_prompt_and_task_kind_change": changed == replacement_positions,
            "completion_bytes_equal_at_every_position": all(
                left.get("completion") == right.get("completion")
                for left, right in zip(arm_a, arm_b)
            ),
            "task_counts_control": dict(Counter(
                row.get("task_kind") for row in arm_a
            )),
            "task_counts_treatment": dict(Counter(
                row.get("task_kind") for row in arm_b
            )),
        },
        "optimizer_schedule": schedule,
        "trainer_preparation": prepared,
        "source_files_sha256": {
            path.relative_to(ROOT).as_posix(): _sha(path)
            for path in (
                ROOT / "engine" / "sft_trainer.py",
                ROOT / "harness" / "execution_supervision.py",
                ROOT / "harness" / "execution_supervision_sidecar.py",
                ROOT / "scripts" / "run_execution_supervision_pilot.py",
                ROOT / "scripts" / "evaluate_execution_supervision_pilot.py",
                ROOT / "scripts" / "analyse_execution_supervision_pilot.py",
                Path(__file__).resolve(),
            )
        },
        "next_permitted_step": (
            "launch Arm A alone through scripts/gpu_run.py"
            if not problems else "fix preflight problems; do not launch GPU training"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1")
    # This is a launch-time receipt, not a committed result.  Keeping it next
    # to the ignored, hash-bound arm artifacts avoids the impossible cycle in
    # which writing/committing the receipt changes the HEAD it attests to.
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1" / "preflight.json")
    arguments = parser.parse_args()
    receipt = preflight(arguments.dataset_dir, arguments.output)
    print(json.dumps({
        "ready": receipt["ready"],
        "problems": receipt["problems"],
        "optimizer_steps": receipt["optimizer_schedule"]["planned_optimizer_steps"],
        "trainer_preparation": receipt["trainer_preparation"],
        "output": str(arguments.output),
    }, indent=2))
    return 0 if receipt["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
