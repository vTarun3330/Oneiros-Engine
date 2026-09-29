"""Choice B: internal, exploratory, Arm-A-exposed composite-intervention screen.

    freeze     CPU. Freeze the gate/training split and the matched training rows.
    preflight  CPU (tokenizer only). Prove the two arms differ only in the label mask
               and output directory; write the source-bound preflight receipt.
    train      GPU. One arm per invocation, launched durably:
                 python scripts/gpu_run.py start --name phase4_choice_b_<arm> -- \\
                   .venv-gpu/Scripts/python.exe scripts/phase4_choice_b.py train --arm <arm>
    evaluate   GPU. Gate look for one arm: fixed-input probe (both schemas) and canonical
               Kill@8, final adapter only.
    analyse    CPU. The one frozen analysis; refuses to run twice.

Nothing here runs without a green, committed preflight receipt, and `train` /
`evaluate` require explicit user authorisation (not given in the task that built this).
Labels: see harness.choice_b.LABELS. The old arm A is never evaluated on this panel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness import choice_b as cb
from harness.source_identity import canonical_sha256

VIEW = "data/corpus/v4_1_research_hardened_candidate/development_view"
TRAIN = f"{VIEW}/train.records.json"
COMPLEXITY = f"{VIEW}/complexity_manifest.json"
ARM_A_PREFLIGHT = "results/v4_2_armA_successor_preflight.json"
PHASE3_COHORT = "results/sft_root_cause_phase3a_cohort.json"
CENSUS = "results/sft_root_cause_phase4_cohort_census.json"
SPLIT_V1 = "results/sft_root_cause_phase4_choice_b_split_v1.json"   # superseded: token fit
SPLIT = "results/sft_root_cause_phase4_choice_b_split_v2.json"
RECEIPT = "results/sft_root_cause_phase4_choice_b_preflight_receipt.json"
RUN_DIR = "results/sft_root_cause/phase4_choice_b"          # ignored; raw outputs
TRACKED_RESULT = "results/sft_root_cause_phase4_choice_b_{what}_{arm}.json"
SCHEMA = "oneiros_phase4_choice_b_preflight_v1"

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ARMS = {"control": "full_completion", "treatment": "value_only"}
SEED = 42
LEARNING_RATE = 1e-5            # arm A's frozen value
EPOCHS = 2
BATCH_SIZE = 1
WARMUP_STEPS = 25
CHECKPOINT_STEPS = 50
TRAINER_KWARGS = {
    "max_prompt_tokens": 1024, "max_repository_prompt_tokens": 2048,
    "max_completion_tokens": 1024, "max_repository_completion_tokens": 1024,
    "warmup_steps": WARMUP_STEPS, "checkpoint_steps": CHECKPOINT_STEPS,
    "lr_scheduler_type": "constant_with_warmup",
}
SECONDS_PER_OPTIMIZER_STEP = 4.35     # objective smoke v1 (short fixed-call prompts)
BOUND_SOURCES = (
    "scripts/phase4_choice_b.py", "harness/choice_b.py", "harness/objective_masking.py",
    "harness/fixed_input_probe.py", "harness/safe_execution.py", "engine/sft_trainer.py",
    "engine/test_generation_prompt.py", "engine/generator.py", "config/settings.py",
    "harness/rehearsal_evaluator.py", "harness/generation_adapter.py",
    "harness/prompt_factory.py", "harness/sealed_final_evaluator.py",
    "harness/evaluation_admission.py", "scripts/gpu_run.py",
    "scripts/analyse_tool_assisted_pilot.py",
)


def rel_sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def checkpoint_dir(arm: str) -> Path:
    return ROOT / "checkpoints" / f"phase4_choice_b_{arm}_qwen15b_s{SEED}"


def publish_once(rel: str, payload: dict) -> None:
    data = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    target = ROOT / rel
    if target.exists() and target.read_bytes() != data:
        raise SystemExit(f"REFUSED: {rel} exists with different bytes; never overwritten")
    publish_file_atomically(target, data)


# --- freeze ---------------------------------------------------------------------------------

def build_split() -> dict:
    from harness.parallel_execution import map_jobs
    census = load(CENSUS)["pools"]["arm_a_exposed_remainder"]
    train = load(TRAIN)
    selected = set(load(ARM_A_PREFLIGHT)["selection"]["selected_record_ids"])
    phase3 = {f["group_id"] for f in load(PHASE3_COHORT)["functions"]}
    pool = cb.pool_groups(train, selected, phase3)
    if cb.ids_sha256(pool) != census["group_ids_sha256"] or len(pool) != census["groups"]:
        raise SystemExit("REFUSED: re-derived pool differs from the Phase 4 census")
    gate_candidates = map_jobs(cb.gate_job, [(g, recs) for g, recs in pool.items()])
    feasible = {f["group_id"]: f for f in gate_candidates if f}
    if (cb.ids_sha256(feasible) != census["feasible_group_ids_sha256"]
            or len(feasible) != census["groups_with_a_feasible_fixed_input_function"]):
        raise SystemExit("REFUSED: re-derived feasible groups differ from the census")
    gate = cb.assign_gate(feasible)
    training_groups = sorted(set(pool) - set(gate))
    tiers = {r["record_id"]: r["tier"] for r in load(COMPLEXITY)["records"]}
    by_id = {r["id"]: r for g in training_groups for r in pool[g]}
    records = [r for g in training_groups for r in pool[g] if cb.is_function(r)]
    raw = [row for chunk in map_jobs(cb.training_rows_for_record, records) for row in chunk]
    fitting, overlong = token_fit(raw, by_id)
    rows = []
    for row in cb.cap_and_order(fitting):
        record = by_id[row["record_id"]]
        rows.append({**row, "dataset": record["source"]["upstream"],
                     "complexity_tier": tiers.get(row["record_id"], "none"),
                     "bug_family": (record.get("provenance") or {}).get("mutation_type"),
                     "record_content_hash": record.get("content_hash")})
    gate_functions = [feasible[g] for g in gate]
    spec = cb.frozen_evaluation_spec()
    return {
        "schema_version": "oneiros_phase4_choice_b_split_v2",
        "supersedes": {"path": SPLIT_V1, "sha256": rel_sha(SPLIT_V1), "modified": False,
                       "reason": ("the v1 preflight refused v1 because 37 of 3,944 training "
                                  "prompts would be compacted at the 1,024-token prompt limit; "
                                  "v2 adds a token-fit admission rule before the group cap; "
                                  "the gate, pool and every other rule are unchanged; no "
                                  "Choice B outcome existed")},
        "design_version": cb.DESIGN_VERSION, "labels": list(cb.LABELS),
        "frozen_before_any_choice_b_outcome": True,
        "inputs": {rel: rel_sha(rel) for rel in (TRAIN, COMPLEXITY, ARM_A_PREFLIGHT,
                                                 PHASE3_COHORT, CENSUS)},
        "splits_read": ["train"],
        "pool": {"definition": "arm A training groups in neither Phase 3A cohort",
                 "groups": len(pool), "group_ids_sha256": cb.ids_sha256(pool),
                 "feasible_groups": len(feasible),
                 "feasible_group_ids_sha256": cb.ids_sha256(feasible)},
        "gate": {"groups": len(gate), "group_ids": gate, "group_ids_sha256": cb.ids_sha256(gate),
                 "order": f"stable_key({cb.GATE_ORDER_TAG!r}, group_id), first {cb.GATE_GROUPS} "
                          "feasible groups",
                 "functions": gate_functions,
                 "items": sum(len(f["items"]) for f in gate_functions)},
        "training": {"groups_in_split": len(training_groups),
                     "group_ids_sha256": cb.ids_sha256(training_groups),
                     "function_records_considered": len(records),
                     "rows_verified": len(raw),
                     "rows_excluded_by_token_fit": len(overlong),
                     "token_fit_rule": ("a row is admitted only if its A0 prompt passes the "
                                        "production prepare_dataset at max_prompt_tokens="
                                        f"{TRAINER_KWARGS['max_prompt_tokens']} with no "
                                        "compaction and no drop; applied before the group "
                                        "cap"),
                     "rows_before_cap": len(fitting), "rows": len(rows),
                     "groups_with_rows": len({r["group_id"] for r in rows}),
                     "max_calls_per_record": cb.MAX_CALLS_PER_RECORD,
                     "group_cap": cb.GROUP_CAP,
                     "composition": {key: dict(Counter(str(r[key]) for r in rows).most_common())
                                     for key in ("dataset", "complexity_tier", "bug_family")},
                     "development_functions_reserved": 0},
        "rows": rows,
        "evaluation_spec": spec, "evaluation_spec_sha256": cb.spec_sha256(spec),
    }


def token_fit(rows, records_by_id):
    """Split rows into (fits, overlong) through the production dataset preparation."""
    import transformers
    from harness.objective_masking import dataset_trainer, make_datapoint
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                           local_files_only=True)
    trainer = dataset_trainer(tokenizer, "full_completion")
    for key in ("max_prompt_tokens", "max_repository_prompt_tokens",
                "max_completion_tokens", "max_repository_completion_tokens"):
        setattr(trainer, key, TRAINER_KWARGS[key])
    fits, overlong = [], []
    for row in rows:
        trainer.prepare_dataset([make_datapoint(records_by_id[row["record_id"]], row["call"],
                                                row["value"])])
        stats = trainer.dataset_stats
        ok = stats["retained_examples"] == 1 and not stats["prompt_truncated_examples"]
        (fits if ok else overlong).append(row)
    return fits, overlong


def freeze() -> int:
    from harness.acquisition_receipt import ProtectedAccessMonitor
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    split = build_split()
    evidence = ProtectedAccessMonitor.evidence(mark)
    if evidence["protected_paths_opened"]:
        raise SystemExit(f"REFUSED: protected paths opened: {evidence['protected_paths_opened']}")
    split["protected_access_audit"] = {k: v for k, v in evidence.items() if k != "opens_checked"}
    publish_once(SPLIT, split)
    print(json.dumps({k: split[k] for k in ("pool", "training")} |
                     {"gate_groups": split["gate"]["groups"], "gate_items": split["gate"]["items"]},
                     indent=1, default=str)[:3000])
    return 0


# --- preflight ------------------------------------------------------------------------------

def datapoints(split: dict):
    from harness.objective_masking import make_datapoint
    records = {r["id"]: r for r in load(TRAIN)}
    return [make_datapoint(records[row["record_id"]], row["call"], row["value"])
            for row in split["rows"]]


def prepared_arms(split: dict):
    """Both arms through the production dataset preparation with the frozen limits."""
    import transformers
    from harness.objective_masking import dataset_trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                           local_files_only=True)
    points = datapoints(split)
    out = {}
    for arm, mode in ARMS.items():
        trainer = dataset_trainer(tokenizer, mode)
        for key in ("max_prompt_tokens", "max_repository_prompt_tokens",
                    "max_completion_tokens", "max_repository_completion_tokens"):
            setattr(trainer, key, TRAINER_KWARGS[key])
        dataset = trainer.prepare_dataset(points)
        out[arm] = (dataset, dict(trainer.dataset_stats))
    return tokenizer, points, out


def preflight() -> int:
    from engine.sft_trainer import SFT_GRADIENT_ACCUMULATION_STEPS, plan_sft_optimizer_schedule
    from harness.acquisition_receipt import ProtectedAccessMonitor
    from harness.objective_masking import sequence_hash, validate_manifest
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    problems: list[str] = []
    status = git("status", "--porcelain")
    if status:
        problems.append("working tree is not clean")
    split = load(SPLIT)
    rebuilt = build_split()
    rebuilt["protected_access_audit"] = split.get("protected_access_audit")
    if json.dumps(rebuilt, sort_keys=True) != json.dumps(split, sort_keys=True):
        problems.append("the frozen split does not re-derive byte-for-byte")
    if split["evaluation_spec_sha256"] != cb.spec_sha256(cb.frozen_evaluation_spec()):
        problems.append("evaluation spec changed after freezing")
    gate = set(split["gate"]["group_ids"])
    train_groups = {r["group_id"] for r in split["rows"]}
    gate_records = {f["record_id"] for f in split["gate"]["functions"]}
    isolation = {
        "no_gate_group_in_training": not (gate & train_groups),
        "no_gate_record_in_training": not (gate_records & {r["record_id"] for r in split["rows"]}),
        "no_phase3_group_anywhere": not ((gate | train_groups) & {
            f["group_id"] for f in load(PHASE3_COHORT)["functions"]}),
        "train_split_only": split["splits_read"] == ["train"],
        "old_arm_a_not_evaluated": split["evaluation_spec"]["old_arm_A_evaluated"] is False,
    }
    problems += [f"isolation: {k}" for k, ok in isolation.items() if not ok]

    tokenizer, points, arms = prepared_arms(split)
    (c_data, c_stats), (t_data, t_stats) = arms["control"], arms["treatment"]
    manifest_rows = []
    for index, (row, c_row, t_row) in enumerate(zip(split["rows"], c_data, t_data)):
        manifest_rows.append({
            **row, "execution_mode": "function", "representation": None,
            "passes_reference": True, "discriminates": True,
            "control_input_ids_sha256": sequence_hash(c_row["input_ids"]),
            "treatment_input_ids_sha256": sequence_hash(t_row["input_ids"]),
            "control_labels_sha256": sequence_hash(c_row["labels"]),
            "treatment_labels_sha256": sequence_hash(t_row["labels"]),
            "order_index_control": index, "order_index_treatment": index,
            "tokens": {"input": len(c_row["input_ids"]),
                       "control_supervised": sum(x != -100 for x in c_row["labels"]),
                       "treatment_supervised": sum(x != -100 for x in t_row["labels"])}})
    try:
        composition = validate_manifest(manifest_rows, gate_groups=gate, confirmation_groups=(),
                                        group_cap=cb.GROUP_CAP)
        for key in ("dataset", "bug_family", "complexity_tier"):
            composition[key] = {str(k): v for k, v in composition[key].items()}
    except ValueError as error:
        problems.append(str(error)[:500])
        composition = None
    fit = {"rows": len(points), "retained_control": len(c_data), "retained_treatment": len(t_data),
           "prompt_truncated_control": c_stats.get("prompt_truncated_examples"),
           "prompt_truncated_treatment": t_stats.get("prompt_truncated_examples")}
    if not (fit["rows"] == fit["retained_control"] == fit["retained_treatment"]):
        problems.append(f"token fit: examples dropped {fit}")
    if fit["prompt_truncated_control"] or fit["prompt_truncated_treatment"]:
        problems.append(f"token fit: prompts truncated {fit}")
    identical = {key: c_stats[key] == t_stats[key]
                 for key in ("input_ids_sha256", "attention_mask_sha256")}
    identical["labels_differ"] = c_stats["labels_sha256"] != t_stats["labels_sha256"]
    problems += [f"arm equality: {k}" for k, ok in identical.items() if not ok]

    python = ".venv-gpu/Scripts/python.exe"
    commands = {arm: cb.arm_command(python, arm) for arm in ARMS}
    difference = cb.command_difference(commands["control"], commands["treatment"])
    if [d[1:] for d in difference] != [(f"phase4_choice_b_control", "phase4_choice_b_treatment"),
                                       ("control", "treatment")]:
        problems.append(f"launch commands differ beyond the arm: {difference}")
    contracts = {arm: run_contract(arm, split_sha=rel_sha(SPLIT), receipt_sha="<receipt>")
                 for arm in ARMS}
    contract_diff = sorted(k for k in contracts["control"]
                           if contracts["control"][k] != contracts["treatment"][k])
    if contract_diff != ["arm", "objective_mode", "output_dir"]:
        problems.append(f"run contracts differ beyond arm/objective/output: {contract_diff}")

    schedule = plan_sft_optimizer_schedule(len(points), EPOCHS, BATCH_SIZE, WARMUP_STEPS,
                                           CHECKPOINT_STEPS, SFT_GRADIENT_ACCUMULATION_STEPS)
    steps = schedule["planned_optimizer_steps"]
    evidence = ProtectedAccessMonitor.evidence(mark)
    if evidence["protected_paths_opened"]:
        problems.append(f"protected paths opened: {evidence['protected_paths_opened']}")
    receipt = {
        "schema_version": SCHEMA, "ready": not problems, "problems": problems,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "clean": not status},
        "labels": list(cb.LABELS),
        "split": {"path": SPLIT, "sha256": rel_sha(SPLIT), "rows": len(points),
                  "gate_groups": len(gate), "gate_items": split["gate"]["items"],
                  "training_groups_with_rows": len(train_groups)},
        "isolation": isolation,
        "matched_manifest": composition,
        "token_fit": fit,
        "arms": {arm: {"objective_mode": ARMS[arm], "output_dir":
                       checkpoint_dir(arm).relative_to(ROOT).as_posix(),
                       "dataset_stats": {k: v for k, v in stats.items()
                                         if k in ("input_ids_sha256", "attention_mask_sha256",
                                                  "labels_sha256", "supervised_tokens",
                                                  "objective_mode", "retained_examples")},
                       "launch_command": commands[arm]}
                 for arm, (_, stats) in arms.items()},
        "arm_equality": identical,
        "command_difference": difference,
        "run_contract_differences": contract_diff,
        "estimand": "COMPOSITE: value-only masking plus reduced supervised-token mass",
        "held_constant": {"model": MODEL, "revision": REVISION, "seed": SEED,
                          "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
                          "batch_size": BATCH_SIZE,
                          "gradient_accumulation_steps": SFT_GRADIENT_ACCUMULATION_STEPS,
                          "trainer_kwargs": TRAINER_KWARGS, "attention_implementation": "sdpa",
                          "lora": "config.settings model_config defaults (r=16, alpha=32, "
                                  "dropout=0.05), weight_decay 0",
                          "order": "identical dataset order; Trainer shuffles with the same "
                                   "seed", "checkpoint_cadence": CHECKPOINT_STEPS,
                          "evaluation_inputs_and_decoding": "frozen evaluation spec"},
        "optimizer_schedule": schedule,
        "evaluation_spec_sha256": split["evaluation_spec_sha256"],
        "runtime_and_storage_estimate": {
            "planned_optimizer_steps_per_arm": steps,
            "training_minutes_per_arm": round(steps * SECONDS_PER_OPTIMIZER_STEP / 60, 1),
            "training_basis": f"{SECONDS_PER_OPTIMIZER_STEP} s/step measured in the objective "
                              "smoke on the same prompt family (RTX 4500 Ada)",
            "evaluation_minutes_per_arm": "~5 fixed-input probe (800 greedy generations) + "
                                          "~20-30 canonical Kill@8 (200 functions x 8)",
            "gpu_hours_total_estimate": round((2 * steps * SECONDS_PER_OPTIMIZER_STEP
                                               + 2 * 35 * 60) / 3600, 2),
            "storage_gb_estimate": "~0.2 per arm (LoRA checkpoints, save_total_limit=2) + "
                                   "<0.1 raw outputs"},
        "source_files_sha256": {rel: canonical_sha256(ROOT / rel) for rel in BOUND_SOURCES},
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
        "training_launched": False, "evaluation_launched": False,
        "next_permitted_step": "stop for explicit user authorisation before any GPU step",
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1, sort_keys=True)
                                             + "\n").encode("utf-8"))
    print(json.dumps({k: receipt[k] for k in ("ready", "problems", "split", "token_fit",
                                              "arm_equality", "runtime_and_storage_estimate")},
                     indent=1))
    return 0 if receipt["ready"] else 2


def verify_receipt_for_launch() -> dict:
    """HEAD may differ from the preflight commit only by the committed receipt."""
    receipt = load(RECEIPT)
    if receipt.get("ready") is not True:
        raise SystemExit("REFUSED: Choice B preflight is not ready")
    if git("status", "--porcelain"):
        raise SystemExit("REFUSED: working tree is not clean")
    commit = receipt["git"]["commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT,
                      capture_output=True).returncode != 0:
        raise SystemExit("REFUSED: preflight commit is not an ancestor of HEAD")
    changed = set(filter(None, git("diff", "--name-only", commit, "HEAD").splitlines()))
    if changed - {RECEIPT}:
        raise SystemExit(f"REFUSED: files changed since preflight: {sorted(changed - {RECEIPT})}")
    for rel, expected in receipt["source_files_sha256"].items():
        if canonical_sha256(ROOT / rel) != expected:
            raise SystemExit(f"REFUSED: bound source drift: {rel}")
    if rel_sha(SPLIT) != receipt["split"]["sha256"]:
        raise SystemExit("REFUSED: the frozen split changed")
    return receipt


# --- train ----------------------------------------------------------------------------------

def run_contract(arm: str, *, split_sha: str, receipt_sha: str) -> dict:
    return {"schema_version": "oneiros_phase4_choice_b_training_v1", "arm": arm,
            "objective_mode": ARMS[arm], "output_dir": checkpoint_dir(arm).name,
            "split_sha256": split_sha, "preflight_sha256": receipt_sha,
            "model_name": MODEL, "model_revision": REVISION, "seed": SEED,
            "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "trainer_kwargs": TRAINER_KWARGS, "attention_implementation": "sdpa",
            "monitor": None}


def train(arm: str) -> int:
    receipt = verify_receipt_for_launch()
    ckpt = checkpoint_dir(arm)
    result_path = ROOT / TRACKED_RESULT.format(what="training", arm=arm)
    if result_path.exists():
        raise SystemExit("REFUSED: a training result already exists; never overwritten")
    split = load(SPLIT)
    contract = run_contract(arm, split_sha=rel_sha(SPLIT), receipt_sha=rel_sha(RECEIPT))
    contract_path = ckpt / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise SystemExit("REFUSED: checkpoint run contract differs (no cross-arm or "
                             "changed-setting resume)")
    else:
        if ckpt.exists() and any(ckpt.iterdir()):
            raise SystemExit("REFUSED: checkpoint directory is not empty and has no contract")
        ckpt.mkdir(parents=True, exist_ok=True)
        contract_path.write_bytes((json.dumps(contract, indent=2) + "\n").encode("utf-8"))
    random.seed(SEED)
    import numpy as np
    import torch
    from engine.sft_trainer import OneirosSFTTrainer
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    points = datapoints(split)
    trainer = OneirosSFTTrainer(model_name=MODEL, model_revision=REVISION, output_dir=ckpt,
                                learning_rate=LEARNING_RATE, attention_implementation="sdpa",
                                objective_mode=ARMS[arm], **TRAINER_KWARGS)
    metrics = trainer.train(points, num_epochs=EPOCHS, batch_size=BATCH_SIZE,
                            checkpoint_monitor=None)
    planned = receipt["optimizer_schedule"]["planned_optimizer_steps"]
    expected_stats = receipt["arms"][arm]["dataset_stats"]
    if (metrics["completed_optimizer_steps"] != planned
            or metrics["retained_examples"] != len(points)
            or metrics.get("labels_sha256") != expected_stats["labels_sha256"]
            or metrics.get("input_ids_sha256") != expected_stats["input_ids_sha256"]):
        raise SystemExit("STOPPED: run violated the frozen plan")
    final = ckpt / "final_adapter"
    trainer.save_adapter(final)
    adapter_sha = hashlib.sha256((final / "adapter_model.safetensors").read_bytes()).hexdigest()
    publish_once(result_path.relative_to(ROOT).as_posix(), {
        "schema_version": "oneiros_phase4_choice_b_training_result_v1", "status": "complete",
        "labels": list(cb.LABELS), "arm": arm, "run_contract": contract,
        "metrics": {k: v for k, v in metrics.items() if k != "model_runtime_profile"},
        "adapter_sha256": adapter_sha, "adapter_path": final.relative_to(ROOT).as_posix(),
        "training_loss_is_not_a_selection_criterion": True,
        "validation_accessed": False, "sealed_final_test_accessed": False})
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


# --- evaluate -------------------------------------------------------------------------------

def final_adapter(arm: str) -> tuple[Path, str]:
    result = load(TRACKED_RESULT.format(what="training", arm=arm))
    path = ROOT / result["adapter_path"]
    digest = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
    if result["status"] != "complete" or digest != result["adapter_sha256"]:
        raise SystemExit(f"REFUSED: {arm} final adapter differs from its training result")
    return path, digest


def evaluate(arm: str) -> int:
    """Gate look for one arm: fixed-input probe on both schemas, then canonical Kill@8."""
    verify_receipt_for_launch()
    out = ROOT / RUN_DIR / arm
    envelope_rel = TRACKED_RESULT.format(what="gate_look", arm=arm)
    if (ROOT / envelope_rel).exists():
        raise SystemExit("REFUSED: the gate look for this arm already happened (one look)")
    adapter, adapter_sha = final_adapter(arm)
    split = load(SPLIT)
    spec = split["evaluation_spec"]["primary_mediator"]
    import torch
    from engine.generator import Phi3Generator
    from engine.test_generation_prompt import format_chat_prompt
    from harness.fixed_input_probe import build_prompt
    records = {r["id"]: r for r in load(TRAIN)}
    work = []
    for function in split["gate"]["functions"]:
        record = records[function["record_id"]]
        for item in function["items"]:
            for schema, level in cb.SCHEMAS.items():
                prompt = build_prompt(record, item["call"], item["expected_repr"], level)
                work.append({"key": f"{item['item_id']}::{schema}", "schema": schema, **prompt})
    out.mkdir(parents=True, exist_ok=True)
    generations = out / "fixed_input_generations.jsonl"
    done = set()
    if generations.exists():
        done = {json.loads(l)["key"] for l in generations.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    generator = Phi3Generator(model_name=MODEL, model_revision=REVISION,
                              attention_implementation="sdpa")
    generator.load_model()
    generator.load_lora_adapter(adapter)
    generator.model.eval()
    tokenizer = generator.tokenizer
    tokenizer.padding_side = "left"
    started = time.time()
    batch = spec["decoding"]["batch"]
    for schema in cb.SCHEMAS:
        todo = [w for w in work if w["schema"] == schema and w["key"] not in done]
        limit = spec["decoding"]["max_new_tokens"][schema]
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            texts = [format_chat_prompt(tokenizer, w["user"]) + w["prefill"] for w in chunk]
            encoded = tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(generator.model.device)
            with torch.no_grad():
                output = generator.model.generate(**encoded, max_new_tokens=limit,
                                                  do_sample=False,
                                                  pad_token_id=tokenizer.pad_token_id)
            width = encoded["input_ids"].shape[1]
            with generations.open("a", encoding="utf-8") as handle:
                for w, row in zip(chunk, output):
                    handle.write(json.dumps({"key": w["key"], "schema": schema, "output":
                                             tokenizer.decode(row[width:],
                                                              skip_special_tokens=True)}) + "\n")
    probe_seconds = round(time.time() - started, 1)
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    kill = kill_at_8(arm, adapter, adapter_sha, out / "kill_at_8")
    publish_once(envelope_rel, {
        "schema_version": "oneiros_phase4_choice_b_gate_look_v1", "arm": arm,
        "labels": list(cb.LABELS), "adapter_sha256": adapter_sha,
        "fixed_input_generations": {"path": generations.relative_to(ROOT).as_posix(),
                                    "sha256": hashlib.sha256(generations.read_bytes()).hexdigest(),
                                    "items": len(work), "seconds": probe_seconds},
        "kill_at_8": kill, "validation_accessed": False, "sealed_final_test_accessed": False})
    return 0


def kill_at_8(arm: str, adapter: Path, adapter_sha: str, output_dir: Path) -> dict:
    from harness.corpus_view import load_development_split
    from harness.evaluation_admission import scope_split
    from engine.generator import Phi3Generator
    from harness.generation_adapter import generate_candidate_slots, successor_settings
    from harness.generation_rng import seed_generation_rngs
    from harness.prompt_factory import prompt_factory
    from harness.rehearsal_evaluator import run_rehearsal_evaluation
    from scripts.train_on_dataset import _record_to_pair
    split = load(SPLIT)
    ids = [f["record_id"] for f in split["gate"]["functions"]]
    panel = "phase4_choice_b_gate"
    scope = scope_split({panel: ids}, load_development_split(
        ROOT / "data/corpus/v4_1_research_hardened_candidate", "train"), panel,
        adapt=_record_to_pair)
    if scope.target_count != len(ids):
        raise SystemExit("REFUSED: the gate panel did not resolve completely")
    settings = successor_settings()
    if settings.problems():
        raise SystemExit(f"REFUSED: invalid generation settings: {settings.problems()}")
    generator = Phi3Generator(model_name=settings.base_model_name,
                              model_revision=settings.base_model_revision,
                              attention_implementation=settings.attention_implementation)
    generator.temperature, generator.top_p = settings.temperature, settings.top_p
    generator.parse_mode = settings.candidate_parse_mode
    generator.load_model()
    generator.load_lora_adapter(adapter)
    build_prompt = prompt_factory(settings.prompt_settings())
    seed_record = seed_generation_rngs(settings.seed)

    def generate_batch(batch):
        accounting = generate_candidate_slots(generator, [dict(r) for r in batch], settings,
                                              build_prompt, seed_before_generation=False)
        return [[{"raw_output": s.get("raw_output", ""), "code": s.get("code")}
                 for s in accounting[i]["candidate_slots"]] for i in range(len(batch))]

    artifact = run_rehearsal_evaluation(
        load_records=lambda: scope.eligible, generate_batch=generate_batch,
        output_dir=output_dir, split_name=panel, scope_summary=scope.to_dict(),
        frozen_settings=settings.to_dict(), receipt_sha256=rel_sha(RECEIPT),
        candidates_per_target=settings.candidates_per_function,
        generation_batch_size=settings.generation_batch_size,
        allow_test_function=settings.allow_test_function_candidates,
        log=lambda m: print(m, flush=True), seed_record=seed_record,
        generator_identity={"model_name": settings.base_model_name,
                            "model_revision": settings.base_model_revision,
                            "adapter_sha256": adapter_sha, "arm": arm})
    result = output_dir / "rehearsal_result.json"
    return {"path": result.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
            "kill_at_k": {k: v["rate"] for k, v in artifact["kill_at_k"].items()},
            "wall_time_seconds": artifact["wall_time_seconds"]}


# --- analyse --------------------------------------------------------------------------------

def analyse() -> int:
    from harness.fixed_input_probe import extract_answer, score_answers
    from scripts.analyse_tool_assisted_pilot import record_metrics
    output = "results/sft_root_cause_phase4_choice_b_analysis_v1.json"
    if (ROOT / output).exists():
        raise SystemExit("REFUSED: the one frozen analysis already ran")
    verify_receipt_for_launch()
    split = load(SPLIT)
    records = {r["id"]: r for r in load(TRAIN)}
    items = {i["item_id"]: (f, i) for f in split["gate"]["functions"] for i in f["items"]}
    per_arm: dict[str, dict] = {}
    kill_rows: dict[str, dict] = {}
    for arm in ARMS:
        look = load(TRACKED_RESULT.format(what="gate_look", arm=arm))
        path = ROOT / look["fixed_input_generations"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != look["fixed_input_generations"][
                "sha256"]:
            raise SystemExit(f"REFUSED: {arm} generations changed after the gate look")
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
        scored = {}
        for schema, level in cb.SCHEMAS.items():
            batch = []
            for item_id, (function, item) in items.items():
                answer = extract_answer(rows[f"{item_id}::{schema}"]["output"], level,
                                        item["call"])
                batch.append({"item_id": item_id, "group_id": function["group_id"],
                              "record_id": function["record_id"], "call": item["call"],
                              "answer": answer["answer"]})
            by_record: dict[str, list] = {}
            for entry in batch:
                by_record.setdefault(entry["record_id"], []).append(entry)
            for record_id, entries in by_record.items():
                verdicts = score_answers(entries, str(records[record_id]["reference_code"]))
                for entry, verdict in zip(entries, verdicts):
                    scored[(entry["item_id"], schema)] = {
                        "group_id": entry["group_id"], "answered": entry["answer"] is not None,
                        "correct": verdict["correct"]}
        per_arm[arm] = scored
        rehearsal = load(look["kill_at_8"]["path"])
        kill_rows[arm] = {r["record_id"]: record_metrics(r) for r in rehearsal["function_results"]}
    schemas = {}
    for schema in cb.SCHEMAS:
        keys = sorted(k for k in per_arm["control"] if k[1] == schema)
        groups = [per_arm["control"][k]["group_id"] for k in keys]

        def series(arm, field):
            return [float(per_arm[arm][k][field]) for k in keys]
        answer = {arm: sum(series(arm, "answered")) / len(keys) * 100 for arm in ARMS}
        schemas[schema] = {
            "items": len(keys),
            "answer_rate_points": {arm: round(v, 3) for arm, v in answer.items()},
            "answer_rate_gate": answer["treatment"] >= answer["control"]
            - cb.ANSWER_RATE_MARGIN_POINTS,
            "accuracy": cb.cluster_bootstrap_difference(groups, series("control", "correct"),
                                                        series("treatment", "correct"))}
    shared = sorted(set(kill_rows["control"]) & set(kill_rows["treatment"]))
    gid = {f["record_id"]: f["group_id"] for f in split["gate"]["functions"]}

    def metric(name):
        return (cb.cluster_bootstrap_difference(
            [gid[r] for r in shared], [kill_rows["control"][r][name] for r in shared],
            [kill_rows["treatment"][r][name] for r in shared]))
    validity = metric("reference_valid_per_requested")
    c_div = sum(kill_rows["control"][r]["exact_unique_ratio"] for r in shared) / len(shared)
    t_div = sum(kill_rows["treatment"][r]["exact_unique_ratio"] for r in shared) / len(shared)
    guardrails = {
        "validity": validity["point"] >= -cb.VALIDITY_MARGIN_POINTS,
        "diversity": (t_div >= c_div * (1 - cb.DIVERSITY_MAX_RELATIVE_LOSS)) if c_div else True}
    decision = cb.decide(schemas, guardrails)
    publish_once(output, {
        "schema_version": "oneiros_phase4_choice_b_analysis_v1", "labels": list(cb.LABELS),
        "evaluation_spec_sha256": split["evaluation_spec_sha256"], "schemas": schemas,
        "kill_at_8": metric("kill_at_8"), "reference_validity": validity,
        "exact_unique_ratio": {"control": round(c_div, 4), "treatment": round(t_div, 4)},
        "decision": decision, "root_cause_established": False,
        "efficacy_established": False})
    print(json.dumps(decision, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("freeze", "preflight", "train", "evaluate",
                                            "analyse"))
    parser.add_argument("--arm", choices=sorted(ARMS))
    args = parser.parse_args(argv)
    if args.command in ("train", "evaluate") and not args.arm:
        parser.error("--arm is required")
    if args.command == "freeze":
        return freeze()
    if args.command == "preflight":
        return preflight()
    if args.command == "train":
        return train(args.arm)
    if args.command == "evaluate":
        return evaluate(args.arm)
    return analyse()


if __name__ == "__main__":
    raise SystemExit(main())
