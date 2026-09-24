"""Train one frozen unordered-event or ordered-trace arm on the local GPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.sft_trainer import OneirosSFTTrainer
from harness.execution_supervision_sidecar import sha256_file
from scripts.preflight_execution_supervision_ab import (
    MODEL_NAME, MODEL_REVISION, SEED, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    WARMUP_STEPS, CHECKPOINT_STEPS, _data_points,
)
from scripts.preflight_execution_trace_ab import ARM_FILES


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_FILES))
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1")
    parser.add_argument("--preflight", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1" / "preflight.json")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("ready") is not True:
        raise SystemExit("REFUSED: trace preflight is not ready")
    head = _git("rev-parse", "HEAD")
    if head != preflight["git"]["commit"] or _git("status", "--short"):
        raise SystemExit("REFUSED: source is dirty or differs from preflight")
    for relative, expected in preflight["source_files_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise SystemExit(f"REFUSED: preflight source drift: {relative}")
    manifest_path = args.dataset_dir / "manifest.json"
    if sha256_file(manifest_path) != preflight["dataset"]["manifest_sha256"]:
        raise SystemExit("REFUSED: trace dataset manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arm_key = "unordered_arm_sha256" if args.arm == "unordered_events" else "ordered_arm_sha256"
    arm_path = args.dataset_dir / ARM_FILES[args.arm]
    if sha256_file(arm_path) != manifest[arm_key]:
        raise SystemExit("REFUSED: trace arm hash mismatch")
    rows = json.loads(arm_path.read_text(encoding="utf-8"))
    if len(rows) != 1024:
        raise SystemExit("REFUSED: trace arm is not 1024 examples")
    contract = {
        "schema_version": "oneiros_execution_trace_training_v1",
        "arm": args.arm, "git_commit": head,
        "preflight_sha256": sha256_file(args.preflight),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "arm_sha256": sha256_file(arm_path),
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "seed": SEED, "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "gradient_accumulation_steps": 16,
        "warmup_steps": WARMUP_STEPS, "checkpoint_steps": CHECKPOINT_STEPS,
        "scheduler": "constant_with_warmup", "attention_implementation": "sdpa",
        "function_prompt_tokens": 1024, "repository_prompt_tokens": 2048,
        "auxiliary_completion_tokens": 1024, "max_sequence_tokens": 3072,
        "examples": len(rows), "monitor": None, "other_interventions": [],
        "source_files_sha256": preflight["source_files_sha256"],
    }
    contract_path = checkpoint_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise SystemExit("REFUSED: checkpoint run contract differs")
    else:
        _atomic(contract_path, contract)
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
        model_name=MODEL_NAME, model_revision=MODEL_REVISION,
        output_dir=checkpoint_dir, learning_rate=LEARNING_RATE,
        max_prompt_tokens=1024, max_repository_prompt_tokens=2048,
        max_completion_tokens=1024, max_repository_completion_tokens=1024,
        warmup_steps=WARMUP_STEPS, checkpoint_steps=CHECKPOINT_STEPS,
        lr_scheduler_type="constant_with_warmup", attention_implementation="sdpa",
    )
    metrics = trainer.train(
        _data_points(rows), num_epochs=EPOCHS, batch_size=BATCH_SIZE,
        checkpoint_monitor=None,
    )
    trainer.save_adapter(checkpoint_dir)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise RuntimeError("training returned without adapter")
    result = {
        "schema_version": "oneiros_execution_trace_training_result_v1",
        "status": "complete", "arm": args.arm,
        "run_contract": contract,
        "run_contract_sha256": hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "metrics": metrics,
        "adapter_sha256": sha256_file(adapter_path),
        "adapter_path": checkpoint_dir.relative_to(ROOT).as_posix(),
        "sealed_final_test_accessed": False, "validation_accessed": False,
        "ablation_dev_accessed": False, "confirmation_opened": False,
    }
    _atomic(args.result, result)
    print(json.dumps({"status": "complete", "arm": args.arm,
                      "loss": metrics["loss"],
                      "completed_optimizer_steps": metrics["completed_optimizer_steps"],
                      "adapter_sha256": result["adapter_sha256"],
                      "result": str(args.result)}, indent=2))
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
