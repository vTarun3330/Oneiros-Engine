"""Train the single composite execution-intervention treatment arm (local GPU).

Refuses unless the frozen dose preflight is ready and HEAD differs from its
commit only by the committed receipt.  The frozen control is not retrained.
Hyperparameters are imported from the control's own preflight module, so they
cannot drift from what trained the control.

Stopping rule (predeclared): a non-finite loss, a dropped example or any step
count other than 64 ends the experiment as a failed run.  The only permitted
retry is an identical rerun after an infrastructure crash that produced no
adapter, recorded as such; no setting may change.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.sft_trainer import OneirosSFTTrainer
from harness.execution_supervision_sidecar import sha256_file
from scripts.preflight_execution_dose_ab import (
    ARM_FILE, ARM_NAME, CHECKPOINT, DATASET_DIR, RECEIPT, TRAINER_KWARGS,
    verify_receipt_for_launch,
)
from scripts.preflight_execution_supervision_ab import (
    BATCH_SIZE, EPOCHS, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, MODEL_NAME,
    MODEL_REVISION, SEED, _data_points,
)

RESULT = DATASET_DIR / "dose_treatment_training_result.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    receipt = verify_receipt_for_launch()
    if RESULT.exists():
        raise SystemExit("REFUSED: a training result already exists; raw artifacts are "
                         "never overwritten")
    arm_path = DATASET_DIR / ARM_FILE
    if sha256_file(arm_path) != receipt["dataset"]["treatment_arm_sha256"]:
        raise SystemExit("REFUSED: treatment arm hash mismatch")
    rows = json.loads(arm_path.read_text(encoding="utf-8"))
    contract = {
        "schema_version": "oneiros_execution_dose_training_v1",
        "arm": ARM_NAME, "preflight_commit": receipt["git"]["commit"],
        "preflight_sha256": sha256_file(RECEIPT),
        "arm_sha256": sha256_file(arm_path),
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "seed": SEED, "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "trainer_kwargs": TRAINER_KWARGS, "attention_implementation": "sdpa",
        "examples": len(rows), "monitor": None,
        "source_files_sha256": receipt["source_files_sha256"],
    }
    contract_path = CHECKPOINT / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise SystemExit("REFUSED: checkpoint run contract differs")
    else:
        CHECKPOINT.mkdir(parents=True, exist_ok=True)
        contract_path.write_bytes((json.dumps(contract, indent=2) + "\n").encode("utf-8"))

    random.seed(SEED)
    import numpy as np
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    trainer = OneirosSFTTrainer(model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                                output_dir=CHECKPOINT, learning_rate=LEARNING_RATE,
                                attention_implementation="sdpa", **TRAINER_KWARGS)
    metrics = trainer.train(_data_points(rows), num_epochs=EPOCHS, batch_size=BATCH_SIZE,
                            checkpoint_monitor=None)
    if (metrics["completed_optimizer_steps"] != 64 or metrics["retained_examples"] != 1024
            or metrics["dropped_overlong_examples"]):
        raise SystemExit(f"STOPPED: run violated the frozen plan: steps="
                         f"{metrics['completed_optimizer_steps']}, retained="
                         f"{metrics['retained_examples']}")
    trainer.save_adapter(CHECKPOINT)
    adapter = CHECKPOINT / "adapter_model.safetensors"
    result = {
        "schema_version": "oneiros_execution_dose_training_result_v1",
        "status": "complete", "arm": ARM_NAME, "run_contract": contract,
        "run_contract_sha256": hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "metrics": metrics, "adapter_sha256": sha256_file(adapter),
        "adapter_path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "training_loss_is_not_a_selection_criterion": True,
        "sealed_final_test_accessed": False, "validation_accessed": False,
        "ablation_dev_accessed": False, "confirmation_opened": False,
    }
    RESULT.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"status": "complete", "steps": metrics["completed_optimizer_steps"],
                      "adapter_sha256": result["adapter_sha256"]}, indent=2))
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
