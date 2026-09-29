"""Phase 5 comparison: accept the smoke only if the arms differ in nothing but labels.

Reads the two per-arm smoke receipts (local, ignored) and the durable-run status files.
Everything must be identical except the label hashes, the supervised-token counts and
quantities that necessarily follow from them (loss value, post-update LoRA norm).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

LOCAL = ROOT / "results/sft_root_cause/phase4_smoke"
RUNS = {"control": "runs/20260929-114416-phase4_smoke_control",
        "treatment": "runs/20260929-114445-phase4_smoke_treatment"}
OUTPUT = "results/sft_root_cause_phase4_objective_smoke_v1.json"
ALLOWED_DIFFERENT = {"arm", "objective_mode", "dataset.objective_mode",
                     "dataset.labels_sha256", "dataset.supervised_tokens",
                     "trainer_batch_sequence.labels_sha256",
                     "trainer_batch_sequence.supervised_tokens", "loss",
                     "lora_b_norm.after", "timing.train_seconds", "timing.tokens_per_second",
                     "checkpoint_dir"}
BATCH_ALLOWED = {"labels_sha256", "supervised_tokens"}


def flatten(obj, prefix=""):
    out = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, path + "."))
        elif key != "trainer_batches":
            out[path] = value
    return out


def main() -> int:
    arms = {a: json.loads((LOCAL / f"{a}_receipt.json").read_text(encoding="utf-8"))
            for a in RUNS}
    status = {a: json.loads((ROOT / RUNS[a] / "status.json").read_text(encoding="utf-8"))
              for a in RUNS}
    c, t = flatten(arms["control"]), flatten(arms["treatment"])
    differing = sorted(k for k in set(c) | set(t) if c.get(k) != t.get(k))
    unexpected = [k for k in differing if k not in ALLOWED_DIFFERENT]
    cb, tb = arms["control"]["trainer_batches"], arms["treatment"]["trainer_batches"]
    batch_problems = []
    if len(cb) != len(tb):
        batch_problems.append("different number of trainer batches")
    for index, (x, y) in enumerate(zip(cb, tb)):
        for key in set(x) | set(y):
            if key not in BATCH_ALLOWED and x.get(key) != y.get(key):
                batch_problems.append(f"batch {index}: {key} differs")
        if x["labels_sha256"] == y["labels_sha256"]:
            batch_problems.append(f"batch {index}: labels identical")
    checks = {
        "both_runs_exited_zero": all(s.get("exit_code") == 0 for s in status.values()),
        "only_labels_and_derived_quantities_differ": not unexpected,
        "trainer_batches_identical_except_labels": not batch_problems,
        "control_supervises_more_tokens": (
            arms["control"]["trainer_batch_sequence"]["supervised_tokens"]
            > arms["treatment"]["trainer_batch_sequence"]["supervised_tokens"]),
        "optimizer_update_completed_both": all(
            a["optimizer"]["completed_steps"] >= 1 and a["lora_b_norm"]["update_applied"]
            for a in arms.values()),
        "no_evaluation_no_promotion": all(not a["evaluation_run"] and not a["promoted"]
                                          for a in arms.values()),
    }
    accepted = all(checks.values())
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase4_objective_smoke_v1",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "purpose": "integration and timing smoke; NOT efficacy training",
        "model_calls": "training forward/backward passes only (no generation)",
        "data": ("32 fixed-input train-split examples from the Phase 3 exposed cohort "
                 "(already-used development evidence); no validation, test, sealed or reserved "
                 "data; no evaluation"),
        "checks": checks,
        "differing_fields": differing, "unexpected_differences": unexpected,
        "batch_problems": batch_problems,
        "identical": {
            "example_keys_sha256": arms["control"]["example_keys_sha256"],
            "dataset_input_ids_sha256": arms["control"]["dataset"]["input_ids_sha256"],
            "dataset_attention_mask_sha256": arms["control"]["dataset"]["attention_mask_sha256"],
            "trainer_batch_input_ids_sequence_sha256":
                arms["control"]["trainer_batch_sequence"]["input_ids_sha256"],
            "tokens_seen_by_trainer": arms["control"]["timing"]["tokens_seen_by_trainer"],
            "optimizer": arms["control"]["optimizer"], "seed": arms["control"]["seed"],
            "model": arms["control"]["model"], "revision": arms["control"]["revision"],
            "runtime_profile": arms["control"]["runtime_profile"]},
        "per_arm": {a: {"objective_mode": r["objective_mode"],
                        "dataset_labels_sha256": r["dataset"]["labels_sha256"],
                        "trainer_labels_sequence_sha256":
                            r["trainer_batch_sequence"]["labels_sha256"],
                        "supervised_tokens": r["trainer_batch_sequence"]["supervised_tokens"],
                        "optimizer_steps_completed": r["optimizer"]["completed_steps"],
                        "lora_b_norm": r["lora_b_norm"], "timing": r["timing"], "gpu": r["gpu"],
                        "run_id": Path(RUNS[a]).name,
                        "run_duration_seconds": status[a].get("duration_seconds"),
                        "loss_not_interpreted": r["loss"]}
                    for a, r in arms.items()},
        "local_receipts_sha256": {a: hashlib.sha256((LOCAL / f"{a}_receipt.json").read_bytes())
                                  .hexdigest() for a in RUNS},
        "disposable_checkpoints": {a: r["checkpoint_dir"] for a, r in arms.items()},
        "efficacy_inferred": False,
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({k: receipt[k] for k in ("status", "checks", "differing_fields",
                                              "unexpected_differences", "batch_problems")},
                     indent=1))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
