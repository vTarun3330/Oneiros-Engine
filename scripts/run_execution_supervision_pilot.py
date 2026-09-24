"""Train one frozen arm of the execution-supervision pilot on the local GPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.sft_trainer import OneirosSFTTrainer, SFTDataPoint
from harness.execution_supervision_sidecar import sha256_file, verify_arm_file
from scripts.preflight_execution_supervision_ab import (
    BATCH_SIZE,
    CHECKPOINT_STEPS,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_REVISION,
    SEED,
    WARMUP_STEPS,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _points(rows: list[dict[str, Any]]) -> list[SFTDataPoint]:
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
            dataset_family=f"{row['source_dataset']}::{row['bug_family']}",
            task_kind=str(row["task_kind"]),
        )
        for row in rows
    ]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["control", "treatment"])
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1")
    parser.add_argument("--preflight", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_preflight.json")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()

    preflight = json.loads(arguments.preflight.read_text(encoding="utf-8"))
    if preflight.get("ready") is not True:
        raise SystemExit("REFUSED: execution-supervision preflight is not ready")
    head = _git("rev-parse", "HEAD")
    if head != (preflight.get("git") or {}).get("commit"):
        raise SystemExit("REFUSED: HEAD differs from the frozen preflight")
    if _git("status", "--short"):
        raise SystemExit("REFUSED: working tree is not clean")
    for relative, expected in (preflight.get("source_files_sha256") or {}).items():
        path = ROOT / relative
        if not path.exists() or _sha(path) != expected:
            raise SystemExit(f"REFUSED: preflight source drift: {relative}")

    dataset_manifest_path = arguments.dataset_dir / "manifest.json"
    if sha256_file(dataset_manifest_path) != preflight["dataset"]["manifest_sha256"]:
        raise SystemExit("REFUSED: dataset manifest drifted after preflight")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    arm_key = "arm_a_sha256" if arguments.arm == "control" else "arm_b_sha256"
    arm_file = arguments.dataset_dir / (
        "arm_a.control.json" if arguments.arm == "control"
        else "arm_b.treatment.json"
    )
    rows = verify_arm_file(arm_file, str(dataset_manifest[arm_key]))

    run_contract = {
        "schema_version": "oneiros_execution_supervision_training_v1",
        "arm": arguments.arm,
        "git_commit": head,
        "preflight_sha256": sha256_file(arguments.preflight),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "arm_sha256": sha256_file(arm_file),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 16,
        "warmup_steps": WARMUP_STEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "scheduler": "constant_with_warmup",
        "attention_implementation": "sdpa",
        "function_prompt_tokens": 1024,
        "repository_prompt_tokens": 2048,
        "function_completion_tokens": 128,
        "repository_completion_tokens": 1024,
        "max_sequence_tokens": 3072,
        "examples": len(rows),
        "monitor": None,
        "other_interventions": [],
        "source_files_sha256": {
            path.relative_to(ROOT).as_posix(): _sha(path)
            for path in (
                ROOT / "engine" / "sft_trainer.py",
                ROOT / "harness" / "execution_supervision.py",
                ROOT / "harness" / "execution_supervision_sidecar.py",
                Path(__file__).resolve(),
            )
        },
    }
    contract_path = arguments.checkpoint_dir / "run_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != run_contract:
            raise SystemExit(
                "REFUSED: checkpoint run contract differs; use a fresh directory"
            )
    else:
        _atomic_json(contract_path, run_contract)

    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except ImportError:
        pass
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    trainer = OneirosSFTTrainer(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        output_dir=arguments.checkpoint_dir,
        learning_rate=LEARNING_RATE,
        max_prompt_tokens=1024,
        max_repository_prompt_tokens=2048,
        max_completion_tokens=128,
        max_repository_completion_tokens=1024,
        warmup_steps=WARMUP_STEPS,
        checkpoint_steps=CHECKPOINT_STEPS,
        lr_scheduler_type="constant_with_warmup",
        attention_implementation="sdpa",
    )
    metrics = trainer.train(
        _points(rows), num_epochs=EPOCHS, batch_size=BATCH_SIZE,
        checkpoint_monitor=None,
    )
    trainer.save_adapter(arguments.checkpoint_dir)
    adapter_path = arguments.checkpoint_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise RuntimeError("training returned without an adapter artifact")
    result = {
        "schema_version": "oneiros_execution_supervision_training_result_v1",
        "status": "complete",
        "arm": arguments.arm,
        "run_contract": run_contract,
        "run_contract_sha256": hashlib.sha256(json.dumps(
            run_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "metrics": metrics,
        "adapter_sha256": sha256_file(adapter_path),
        "adapter_path": arguments.checkpoint_dir.relative_to(ROOT).as_posix(),
        "sealed_final_test_accessed": False,
        "validation_accessed": False,
        "ablation_dev_accessed": False,
        "confirmation_opened": False,
    }
    _atomic_json(arguments.result, result)
    print(json.dumps({
        "status": "complete",
        "arm": arguments.arm,
        "loss": metrics["loss"],
        "completed_optimizer_steps": metrics["completed_optimizer_steps"],
        "adapter_sha256": result["adapter_sha256"],
        "result": str(arguments.result),
    }, indent=2))
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
