"""Fail-closed CPU preflight for unordered-event versus ordered-trace SFT."""
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

from engine.sft_trainer import OneirosSFTTrainer, plan_sft_optimizer_schedule
from harness.execution_supervision_sidecar import sha256_file
from scripts.preflight_execution_supervision_ab import (
    MODEL_NAME, MODEL_REVISION, SEED, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, WARMUP_STEPS, CHECKPOINT_STEPS, _data_points,
)

SCHEMA = "oneiros_execution_trace_ab_v1"
ARM_FILES = {
    "unordered_events": "arm_unordered_events.json",
    "ordered_trace": "arm_ordered_trace.json",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_arm(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"arm hash mismatch: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1024:
        raise ValueError(f"arm must contain exactly 1024 rows: {path}")
    return rows


def preflight(dataset_dir: Path, output: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest.get("schema_version") != SCHEMA:
        problems.append("dataset schema mismatch")
    for key in ("sealed_final_test_accessed", "ablation_dev_accessed",
                "validation_accessed", "test_accessed", "confirmation_opened"):
        if manifest.get(key) is not False:
            problems.append(f"isolation field {key} is not false")
    if manifest.get("evaluation_split") != "train":
        problems.append("dataset is not train-only")
    for relative, expected in (manifest.get("source_files_sha256") or {}).items():
        path = ROOT / relative
        if not path.exists() or _sha(path) != expected:
            problems.append(f"dataset source drift: {relative}")
    try:
        unordered = _load_arm(
            dataset_dir / ARM_FILES["unordered_events"],
            str(manifest.get("unordered_arm_sha256")),
        )
        ordered = _load_arm(
            dataset_dir / ARM_FILES["ordered_trace"],
            str(manifest.get("ordered_arm_sha256")),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        problems.append(str(exc))
        unordered, ordered = [], []

    positions = list(manifest.get("replacement_positions") or [])
    if len(positions) != 122:
        problems.append("replacement position count is not 122")
    position_set = set(positions)
    if len(unordered) == len(ordered) == 1024:
        for index, (left, right) in enumerate(zip(unordered, ordered)):
            if left.get("record_id") != right.get("record_id"):
                problems.append(f"record identity differs at {index}")
                break
            if (left != right) != (index in position_set):
                problems.append(f"undeclared arm difference at {index}")
                break
            if index in position_set:
                try:
                    lobj, robj = json.loads(left["completion"]), json.loads(right["completion"])
                except (KeyError, json.JSONDecodeError) as exc:
                    problems.append(f"invalid structured completion at {index}: {exc}")
                    break
                if lobj["actual"] != robj["actual"] or lobj["intended"] != robj["intended"]:
                    problems.append(f"value targets differ at {index}")
                    break
                if lobj["differs"] != robj["differs"]:
                    problems.append(f"differs target differs at {index}")
                    break
                if sorted(lobj["events"], key=lambda event: (
                    int(event["line"]), str(event["source"])
                )) != sorted(robj["events"], key=lambda event: (
                    int(event["line"]), str(event["source"])
                )):
                    problems.append(f"event multiset differs at {index}")
                    break

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
        stats, datasets = {}, {}
        for name, rows in (("unordered_events", unordered), ("ordered_trace", ordered)):
            trainer = OneirosSFTTrainer(
                model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                output_dir=ROOT / "checkpoints" / f"preflight_{name}",
                learning_rate=LEARNING_RATE, max_prompt_tokens=1024,
                max_repository_prompt_tokens=2048, max_completion_tokens=1024,
                max_repository_completion_tokens=1024, warmup_steps=WARMUP_STEPS,
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
        prepared = {"stats": stats}
        if not problems:
            if any(len(dataset) != 1024 for dataset in datasets.values()):
                problems.append("trainer dropped examples")
            token_totals = {}
            for name, dataset in datasets.items():
                token_totals[name] = sum(
                    len(row["input_ids"]) - int(row["completion_start"])
                    for row in dataset
                )
            prepared["supervised_token_totals"] = token_totals
            low, high = sorted(token_totals.values())
            prepared["supervised_token_mass_ratio"] = low / high
            if low / high < 0.999:
                problems.append("supervised token mass differs by more than 0.1%")

    schedule = plan_sft_optimizer_schedule(
        1024, EPOCHS, BATCH_SIZE, WARMUP_STEPS, CHECKPOINT_STEPS,
        GRADIENT_ACCUMULATION_STEPS,
    )
    if schedule["planned_optimizer_steps"] != 64 or schedule["optimizer_padding_examples"]:
        problems.append("optimizer schedule is not the frozen 64-step no-padding plan")
    sources = [
        ROOT / "engine" / "sft_trainer.py",
        ROOT / "harness" / "execution_supervision.py",
        ROOT / "scripts" / "build_execution_trace_ab.py",
        ROOT / "scripts" / "run_execution_trace_pilot.py",
        ROOT / "scripts" / "evaluate_execution_trace_pilot.py",
        ROOT / "scripts" / "analyse_execution_trace_pilot.py",
        Path(__file__).resolve(),
    ]
    receipt = {
        "schema_version": "oneiros_execution_trace_preflight_v1",
        "ready": not problems,
        "problems": problems,
        "git": {"branch": _git("branch", "--show-current"), "commit": head,
                "clean": not bool(status)},
        "dataset": {
            "directory": dataset_dir.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "unordered_arm_sha256": manifest.get("unordered_arm_sha256"),
            "ordered_arm_sha256": manifest.get("ordered_arm_sha256"),
            "pilot_development_sha256": manifest.get("pilot_development_sha256"),
            "unopened_confirmation_ids_sha256": manifest.get(
                "unopened_confirmation_ids_sha256"
            ),
        },
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION,
                  "attention_implementation": "sdpa", "quantization": "4bit_nf4"},
        "held_constant": {
            "seed": SEED, "examples": 1024, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "optimizer_steps": 64, "learning_rate": LEARNING_RATE,
            "lr_scheduler_type": "constant_with_warmup",
            "warmup_steps": WARMUP_STEPS, "checkpoint_steps": CHECKPOINT_STEPS,
            "function_prompt_tokens": 1024, "repository_prompt_tokens": 2048,
            "auxiliary_completion_tokens": 1024, "max_sequence_tokens": 3072,
            "monitor": None, "dpo": False, "rlvr": False,
        },
        "intervention": {
            "positions": positions,
            "examples": len(positions),
            "same_event_multiset_and_values": True,
            "only_temporal_order_and_instruction_differ": True,
            "task_counts": {
                "unordered_events": dict(Counter(row["task_kind"] for row in unordered)),
                "ordered_trace": dict(Counter(row["task_kind"] for row in ordered)),
            },
        },
        "optimizer_schedule": schedule,
        "trainer_preparation": prepared,
        "source_files_sha256": {
            path.relative_to(ROOT).as_posix(): _sha(path) for path in sources
        },
        "next_permitted_step": (
            "train unordered-events arm alone" if not problems
            else "fix preflight problems; no GPU training"
        ),
        "confirmation_opened": False,
        "validation_accessed": False,
        "sealed_final_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1" / "preflight.json")
    args = parser.parse_args()
    receipt = preflight(args.dataset_dir, args.output)
    print(json.dumps({"ready": receipt["ready"], "problems": receipt["problems"],
                      "trainer_preparation": receipt["trainer_preparation"],
                      "output": str(args.output)}, indent=2))
    return 0 if receipt["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
