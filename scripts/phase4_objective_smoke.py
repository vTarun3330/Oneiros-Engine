"""Phase 5: bounded GPU integration and timing smoke for the objective modes.

NOT efficacy training.  One arm per invocation (``--arm control|treatment``); both arms
use the same 32 fixed-input training examples (train split, Phase 3 exposed cohort:
already-used development evidence), the same order, pinned base model, LoRA
configuration, optimiser, schedule and seed.  Two optimiser steps.  No validation, test,
sealed or reserved data; no checkpoint monitor; no evaluation of any kind.  The adapter
is disposable and never promoted; nothing is inferred from the loss.

Evidence recorded from the REAL Trainer: the production collator is wrapped so every
batch it hands the Trainer is hashed (input_ids, attention_mask, labels, supervised
token counts).  An optimiser update is proven by LoRA-B weights moving off zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
COHORT = "results/sft_root_cause_phase3a_cohort.json"
COHORT_SHA256 = "1316956a66f718ec1b900438132eb0ae3bc7e8e6da718a58b0b5fcc18d718ce7"
TRAIN = "data/corpus/v4_1_research_hardened_candidate/development_view/train.records.json"
OUT = "results/sft_root_cause/phase4_smoke"
CHECKPOINTS = "checkpoints/phase4_objective_smoke_DISPOSABLE"
EXAMPLES = 32
SEED = 42
MODES = {"control": "full_completion", "treatment": "value_only"}


def sha_rows(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([int(x) for x in row]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def examples():
    from harness.objective_masking import make_datapoint
    cohort_bytes = (ROOT / COHORT).read_bytes()
    if hashlib.sha256(cohort_bytes).hexdigest() != COHORT_SHA256:
        raise SystemExit("REFUSED: cohort changed")
    cohort = json.loads(cohort_bytes)
    exposed = sorted((f for f in cohort["functions"] if f["cohort"] == "exposed"),
                     key=lambda f: f["record_id"])[:EXAMPLES // 2]
    ids = {f["record_id"] for f in exposed}
    records = {r["id"]: r for r in json.loads((ROOT / TRAIN).read_text(encoding="utf-8"))
               if r["id"] in ids}
    points, keys = [], []
    for function in exposed:
        for item in sorted(function["items"], key=lambda i: i["input_kind"]):
            points.append(make_datapoint(records[function["record_id"]], item["call"],
                                         item["expected_repr"]))
            keys.append(f"{function['record_id']}::{item['input_kind']}")
    return points, keys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(MODES), required=True)
    args = parser.parse_args(argv)
    import torch
    import transformers
    from engine import sft_trainer as sft

    out = ROOT / OUT
    out.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / CHECKPOINTS / args.arm
    if ckpt.exists() and any(ckpt.iterdir()):
        raise SystemExit(f"REFUSED: {ckpt} is not empty (no resume for a smoke)")
    points, keys = examples()
    transformers.set_seed(SEED)

    batches = []
    original = sft.CompletionOnlyDataCollator.__call__

    def recording_call(self, features):
        batch = original(self, features)
        labels = batch["labels"]
        batches.append({
            "input_ids_sha256": sha_rows(batch["input_ids"].tolist()),
            "attention_mask_sha256": sha_rows(batch["attention_mask"].tolist()),
            "labels_sha256": sha_rows(labels.tolist()),
            "supervised_tokens": int((labels != sft.IGNORE_INDEX).sum()),
            "tokens": int(batch["attention_mask"].sum())})
        return batch

    sft.CompletionOnlyDataCollator.__call__ = recording_call
    trainer = sft.OneirosSFTTrainer(
        model_name=MODEL, model_revision=REVISION, output_dir=ckpt, learning_rate=1e-5,
        max_prompt_tokens=1024, max_repository_prompt_tokens=1024,
        max_completion_tokens=1024, max_repository_completion_tokens=1024,
        warmup_steps=1, checkpoint_steps=2, lr_scheduler_type="constant_with_warmup",
        attention_implementation="sdpa", objective_mode=MODES[args.arm])
    trainer.setup_model()

    def lora_b_norm() -> float:
        return float(sum(p.detach().float().norm() for n, p in trainer.model.named_parameters()
                         if "lora_B" in n))

    before = lora_b_norm()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train(points, num_epochs=1, batch_size=1)
    seconds = time.time() - started
    after = lora_b_norm()
    tokens = sum(b["tokens"] for b in batches)
    receipt = {
        "arm": args.arm, "objective_mode": MODES[args.arm], "model": MODEL, "revision": REVISION,
        "seed": SEED, "examples": len(points), "example_keys_sha256": hashlib.sha256(
            "\n".join(keys).encode()).hexdigest(),
        "dataset": {k: result.get(k) for k in ("objective_mode", "supervised_tokens",
                                               "input_ids_sha256", "attention_mask_sha256",
                                               "labels_sha256", "retained_examples",
                                               "prompt_truncated_examples")},
        "trainer_batches": batches,
        "trainer_batch_sequence": {
            "input_ids_sha256": hashlib.sha256("".join(b["input_ids_sha256"] for b in batches)
                                               .encode()).hexdigest(),
            "labels_sha256": hashlib.sha256("".join(b["labels_sha256"] for b in batches)
                                            .encode()).hexdigest(),
            "supervised_tokens": sum(b["supervised_tokens"] for b in batches)},
        "optimizer": {"planned_steps": result["planned_optimizer_steps"],
                      "completed_steps": result["completed_optimizer_steps"],
                      "lr": 1e-5, "scheduler": "constant_with_warmup", "warmup_steps":
                      result["warmup_steps"], "effective_batch": 16},
        "lora_b_norm": {"before": before, "after": after, "update_applied": after > before},
        "timing": {"train_seconds": round(seconds, 1),
                   "tokens_per_second": round(tokens / seconds, 1) if seconds else None,
                   "tokens_seen_by_trainer": tokens},
        "gpu": {"peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                "device": torch.cuda.get_device_name(0)},
        "loss": result["loss"], "loss_note": "NOT an efficacy signal; not interpreted",
        "runtime_profile": result["model_runtime_profile"],
        "evaluation_run": False, "promoted": False, "checkpoint_dir": str(ckpt),
    }
    (out / f"{args.arm}_receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("arm", "optimizer", "lora_b_norm", "timing", "gpu",
                                              "trainer_batch_sequence")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
