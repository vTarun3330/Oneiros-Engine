"""Fail-closed CPU preflight for the execution-dose-and-retention experiment.

Produces the frozen receipt that the training runner and both evaluators
require.  It is run on a clean, committed tree; the receipt is then committed
on its own, and the runner accepts only a HEAD whose difference from the
preflight commit is that receipt and nothing else.

Checks, all before any GPU work:

* the treatment arm and its manifest match the tracked manifest copy and the
  frozen control byte-for-byte outside the declared positions;
* leakage: no treatment lineage in the pilot-development or confirmation
  split, no overlap with the mechanism panel or the retention panel;
* the training code and configuration are identical to the commit that
  trained the frozen control, so seed, learning rate, LoRA, quantisation,
  scheduler and regularisation are the control's;
* the real trainer prepares all 1,024 examples with none dropped and no
  malformed prompt; its section-aware prompt compaction (which the trainer
  records as ``prompt_truncated_examples``) and the support/context and code
  units it removes are copied into the receipt verbatim, and the
  supervised-token totals are measured, not estimated;
* the token-matched control study is present and its verdict is the one this
  design rests on (infeasible at 1%, 2% and 5%, hence a composite pilot);
* the optimizer plan is the frozen 64-step, no-padding schedule.

CPU only.  No model weights are loaded; the tokenizer is.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.sft_trainer import OneirosSFTTrainer, plan_sft_optimizer_schedule
from harness.execution_dose import INFERENCE_LIMITATION, INTERVENTION_LABEL
from harness.source_identity import canonical_sha256
from harness.execution_supervision_sidecar import sha256_file
from scripts.preflight_execution_supervision_ab import (
    BATCH_SIZE, CHECKPOINT_STEPS, EPOCHS, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE,
    MODEL_NAME, MODEL_REVISION, SEED, WARMUP_STEPS, _data_points,
)

SCHEMA = "oneiros_execution_dose_preflight_v1"
DATASET_DIR = ROOT / "results" / "v4_3_execution_dose_v1"
SOURCE_DIR = ROOT / "results" / "v4_3_execution_supervision_v1"
RECEIPT = ROOT / "results" / "v4_3_execution_dose_preflight_receipt.json"
MANIFEST_COPY = ROOT / "results" / "v4_3_execution_dose_dataset_manifest.json"
PANEL = ROOT / "results" / "v4_3_execution_dose_retention_panel.json"
MATCHED_CONTROL_REPORT = ROOT / "results" / "v4_3_execution_dose_matched_control.json"
ARM_FILE = "arm_dose_treatment.json"
ARM_NAME = "dose_treatment"
CHECKPOINT = ROOT / "checkpoints" / "v4_3_execdose_d25_qwen15b_s42"
CONTROL_ADAPTER = ROOT / "checkpoints" / "v4_3_execsup_a_control_qwen15b_s42"
CONTROL_RESULT = SOURCE_DIR / "arm_a_training_result.json"
MAX_MASS_RATIO = 1.25
TRAINER_KWARGS = {
    "max_prompt_tokens": 1024, "max_repository_prompt_tokens": 2048,
    "max_completion_tokens": 1024, "max_repository_completion_tokens": 1024,
    "warmup_steps": WARMUP_STEPS, "checkpoint_steps": CHECKPOINT_STEPS,
    "lr_scheduler_type": "constant_with_warmup",
}
#: Files that decide the treatment's training or evaluation.  Any change after
#: the receipt is frozen refuses the run.
BOUND_SOURCES = (
    "engine/sft_trainer.py", "engine/generator.py", "engine/test_generation_prompt.py",
    "config/settings.py", "harness/execution_supervision.py", "harness/execution_dose.py",
    "harness/generation_adapter.py", "harness/rehearsal_evaluator.py",
    "harness/sealed_final_evaluator.py", "harness/evaluation_admission.py",
    "scripts/census_execution_dose_pool.py", "scripts/build_execution_dose_ab.py",
    "scripts/build_execution_dose_matched_control.py",
    "scripts/build_execution_trace_ab.py", "scripts/freeze_execution_dose_retention_panel.py",
    "scripts/preflight_execution_dose_ab.py", "scripts/run_execution_dose_pilot.py",
    "scripts/evaluate_execution_dose_mechanism.py",
    "scripts/evaluate_execution_dose_retention.py",
    "scripts/evaluate_execution_supervision_pilot.py",
    "scripts/analyse_execution_dose_pilot.py", "scripts/diagnose_execution_trace_pilot.py",
)
#: Training-path code must be byte-identical to the commit that trained the
#: frozen control; otherwise "held constant" would be a claim, not a fact.
CONTROL_IDENTICAL_PATHS = ("engine", "config", "harness/execution_supervision.py",
                           "harness/model_identity.py")
#: Measured on this machine by the earlier pilots (durable runner status files).
#: Training time tracks all tokens (prompt + completion), so the estimate
#: scales the slowest comparable 1,024-example run by the treatment's total
#: tokens relative to the control's.
RUNTIME_BASIS = {
    "training_seconds_slowest_1024_example_arm": 496.6,
    "mechanism_eval_seconds": 60.2,
    "kill_at_8_seconds_per_record": 577.753 / 542,
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True).returncode == 0


def arm_problems(treatment: list[dict], control: list[dict], manifest: dict) -> list[str]:
    problems = []
    if len(treatment) != 1024 or len(control) != 1024:
        return ["arms are not exactly 1,024 examples"]
    positions = set(manifest["replacement_positions"])
    if len(positions) != manifest["execution_examples"] or len(positions) != 256:
        problems.append("declared replacement positions are not 256 unique positions")
    for index, (left, right) in enumerate(zip(control, treatment)):
        if (index in positions) != (left != right):
            problems.append(f"undeclared arm difference at {index}")
            break
        if index in positions:
            if right.get("task_kind") != "execution_output_prediction":
                problems.append(f"replacement {index} is not an execution example")
                break
            payload = json.loads(right["completion"])
            if list(payload) != ["events", "actual", "intended", "differs"]:
                problems.append(f"replacement {index} target schema differs")
                break
    kinds = Counter(row["task_kind"] for row in treatment)
    if kinds != Counter({"test_generation": 768, "execution_output_prediction": 256}):
        problems.append(f"task mix is {dict(kinds)}")
    return problems


def leakage_problems(treatment: list[dict]) -> list[str]:
    split = json.loads((SOURCE_DIR / "lineage_split.json").read_text(encoding="utf-8"))
    held_out = set(split["pilot_development_lineages"]) | set(
        split["unopened_confirmation_lineages"])
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    mechanism = json.loads((SOURCE_DIR / "pilot_development.execution.json").read_text(
        encoding="utf-8"))
    lineages = {str(row["function_lineage"]) for row in treatment}
    ids = {str(row["record_id"]) for row in treatment}
    checks = {
        "held-out lineage in treatment": bool(lineages & held_out),
        "retention panel record in treatment": bool(ids & set(panel["record_ids"])),
        "retention panel lineage in treatment": bool(
            lineages & set(panel["record_lineages"].values())),
        "mechanism panel record in treatment": bool(
            ids & {str(row["record_id"]) for row in mechanism}),
    }
    return [name for name, leaked in checks.items() if leaked]


def prepare(rows: list[dict], tokenizer, name: str) -> tuple[Any, dict]:
    trainer = OneirosSFTTrainer(model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                                output_dir=ROOT / "checkpoints" / f"preflight_dose_{name}",
                                learning_rate=LEARNING_RATE, **TRAINER_KWARGS)
    trainer.tokenizer = tokenizer
    dataset = trainer.prepare_dataset(_data_points(rows))
    return dataset, dict(trainer.dataset_stats)


def preflight() -> dict[str, Any]:
    problems: list[str] = []
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if status:
        problems.append("working tree is not clean")
    manifest_path = DATASET_DIR / "manifest.json"
    if sha256_file(manifest_path) != sha256_file(MANIFEST_COPY):
        problems.append("local dataset manifest differs from the tracked copy")
    manifest = json.loads(MANIFEST_COPY.read_text(encoding="utf-8"))
    for key in ("sealed_final_test_accessed", "ablation_dev_accessed",
                "validation_accessed", "test_accessed", "confirmation_opened"):
        if manifest.get(key) is not False:
            problems.append(f"dataset isolation field {key} is not false")
    for relative, expected in manifest["source_files_sha256"].items():
        if canonical_sha256(ROOT / relative) != expected:
            problems.append(f"dataset source drift: {relative}")
    arm_path = DATASET_DIR / ARM_FILE
    if sha256_file(arm_path) != manifest["treatment_arm_sha256"]:
        problems.append("treatment arm hash differs from its manifest")
    control_path = SOURCE_DIR / "arm_a.control.json"
    if sha256_file(control_path) != manifest["control_arm_sha256"]:
        problems.append("frozen control arm hash differs")
    treatment = json.loads(arm_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    problems.extend(arm_problems(treatment, control, manifest))
    matched = json.loads(MATCHED_CONTROL_REPORT.read_text(encoding="utf-8"))
    if matched.get("feasible_tolerances") != [] or matched.get("primary_tolerance") is not None:
        problems.append("matched-control study no longer says infeasible; the composite "
                        "design must be re-decided")
    if manifest.get("inference_limitation") != INFERENCE_LIMITATION:
        problems.append("dataset manifest does not carry the frozen inference limitation")
    problems.extend(f"leakage: {item}" for item in leakage_problems(treatment))

    control_result = json.loads(CONTROL_RESULT.read_text(encoding="utf-8"))
    control_commit = control_result["run_contract"]["git_commit"]
    if sha256_file(CONTROL_ADAPTER / "adapter_model.safetensors") != control_result[
            "adapter_sha256"]:
        problems.append("frozen control adapter differs from its training result")
    if not git_ok("merge-base", "--is-ancestor", control_commit, "HEAD"):
        problems.append("control commit is not an ancestor of HEAD")
    if not git_ok("diff", "--quiet", control_commit, "HEAD", "--", *CONTROL_IDENTICAL_PATHS):
        problems.append("training code differs from the commit that trained the control")
    control_contract = control_result["run_contract"]
    for field, value in (("seed", SEED), ("learning_rate", LEARNING_RATE), ("epochs", EPOCHS),
                         ("batch_size", BATCH_SIZE),
                         ("gradient_accumulation_steps", GRADIENT_ACCUMULATION_STEPS),
                         ("warmup_steps", WARMUP_STEPS), ("checkpoint_steps", CHECKPOINT_STEPS),
                         ("model_revision", MODEL_REVISION), ("examples", 1024)):
        if control_contract.get(field) != value:
            problems.append(f"control contract {field} is {control_contract.get(field)!r}")
    if CHECKPOINT.exists() and any(CHECKPOINT.iterdir()):
        problems.append(f"treatment checkpoint directory is not empty: {CHECKPOINT}")

    prepared: dict[str, Any] = {}
    if not problems:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                                  trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        totals, stats = {}, {}
        for name, rows in (("control", control), (ARM_NAME, treatment)):
            try:
                dataset, stats[name] = prepare(rows, tokenizer, name)
            except ValueError as exc:
                problems.append(f"{name} trainer preparation failed: {exc}")
                break
            if len(dataset) != 1024 or stats[name]["dropped_overlong_examples"]:
                problems.append(f"{name}: trainer dropped examples")
            totals[name] = {
                "supervised_tokens": sum(len(row["input_ids"]) - int(row["completion_start"])
                                         for row in dataset),
                "all_tokens": sum(len(row["input_ids"]) for row in dataset),
            }
        if not problems:
            ratio = (totals[ARM_NAME]["supervised_tokens"]
                     / totals["control"]["supervised_tokens"])
            if ratio > MAX_MASS_RATIO:
                problems.append(f"supervised token mass ratio {ratio:.4f} > {MAX_MASS_RATIO}")
            if totals[ARM_NAME]["supervised_tokens"] != manifest["dose"][
                    "treatment_supervised_tokens"]:
                problems.append("trainer token mass differs from the builder's count")
            prepared = {"stats": stats, "totals": totals,
                        "supervised_token_mass_ratio": ratio}
            # Retention prompts through the canonical builder, so a budget
            # failure surfaces here rather than as a Kill@8 zero on the GPU.
            from harness.generation_adapter import build_prompts, successor_settings
            from harness.prompt_factory import prompt_factory
            from scripts.evaluate_execution_dose_retention import load_panel_scope
            _, scope, _ = load_panel_scope()
            settings = successor_settings()
            _, generable, failures = build_prompts(
                tokenizer, scope.eligible, settings, prompt_factory(settings.prompt_settings()))
            if failures or len(generable) != 613:
                problems.append(f"retention prompt budget failures: {len(failures)}")
            prepared["retention_panel"] = {
                "records": scope.target_count, "prompt_budget_failures": len(failures),
                "admission_scope_sha256": scope.scope_sha256(),
                "generation_settings": settings.to_dict()}

    schedule = plan_sft_optimizer_schedule(1024, EPOCHS, BATCH_SIZE, WARMUP_STEPS,
                                           CHECKPOINT_STEPS, GRADIENT_ACCUMULATION_STEPS)
    if schedule["planned_optimizer_steps"] != 64 or schedule["optimizer_padding_examples"]:
        problems.append("optimizer schedule is not the frozen 64-step no-padding plan")

    estimate = None
    if prepared:
        scale = (prepared["totals"][ARM_NAME]["all_tokens"]
                 / prepared["totals"]["control"]["all_tokens"])
        training = RUNTIME_BASIS["training_seconds_slowest_1024_example_arm"] * max(1.0, scale)
        retention = RUNTIME_BASIS["kill_at_8_seconds_per_record"] * 613
        estimate = {
            "basis": RUNTIME_BASIS,
            "treatment_training_seconds": round(training),
            "mechanism_eval_seconds": RUNTIME_BASIS["mechanism_eval_seconds"],
            "retention_seconds_per_arm": round(retention),
            "retention_arms": ["control", ARM_NAME, "base"],
            "total_gpu_seconds": round(training + RUNTIME_BASIS["mechanism_eval_seconds"]
                                       + 3 * retention),
            "gpu": "one local RTX 4500 Ada (24 GB), jobs strictly sequential",
            "storage_bytes": {
                "treatment_checkpoint_including_resume_state": 171_000_000,
                "mechanism_eval_artifact": 160_000,
                "retention_artifacts_three_arms": 3 * 4_100_000,
                "treatment_arm_and_manifest": 8_000_000,
            },
        }

    return {
        "schema_version": SCHEMA,
        "ready": not problems,
        "problems": problems,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": git("branch", "--show-current"), "commit": head,
                "clean": not status},
        "dataset": {"manifest_sha256": sha256_file(MANIFEST_COPY),
                    "treatment_arm_sha256": manifest["treatment_arm_sha256"],
                    "control_arm_sha256": manifest["control_arm_sha256"],
                    "chosen_design": manifest["chosen_design"],
                    "replacement_positions": manifest["replacement_positions"],
                    "pilot_development_sha256": manifest["pilot_development_sha256"],
                    "retention_panel_sha256": sha256_file(PANEL),
                    "design_report_sha256": sha256_file(
                        ROOT / "results" / "v4_3_execution_dose_design.json"),
                    "census_sha256": sha256_file(
                        ROOT / "results" / "v4_3_execution_dose_census.json")},
        "control": {"training_result_sha256": sha256_file(CONTROL_RESULT),
                    "adapter_sha256": control_result["adapter_sha256"],
                    "adapter_path": CONTROL_ADAPTER.relative_to(ROOT).as_posix(),
                    "trained_at_commit": control_commit,
                    "retrained": False},
        "treatment_checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION,
                  "attention_implementation": "sdpa", "quantization": "4bit_nf4"},
        "held_constant": {
            "seed": SEED, "examples": 1024, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "optimizer_steps": 64, "learning_rate": LEARNING_RATE,
            "lr_scheduler_type": "constant_with_warmup", "warmup_steps": WARMUP_STEPS,
            "checkpoint_steps": CHECKPOINT_STEPS, "function_prompt_tokens": 1024,
            "repository_prompt_tokens": 2048, "repository_completion_tokens": 1024,
            "max_sequence_tokens": 3072,
            "training_code_identical_to_control_commit": list(CONTROL_IDENTICAL_PATHS),
            "monitor": None, "dpo": False, "rlvr": False,
        },
        "intervention_label": INTERVENTION_LABEL,
        "inference_limitation": INFERENCE_LIMITATION,
        "design_class": "composite efficacy pilot (token-matched control infeasible)",
        "matched_control_study": {
            "report_sha256": sha256_file(MATCHED_CONTROL_REPORT),
            "feasible_tolerances": matched["feasible_tolerances"],
            "best_ratio_to_treatment": matched["tolerances"]["1pct"]["ratio_to_treatment"],
            "upper_bound_ratio_valid_construction": matched["upper_bounds"][
                "intervention_positions_only"]["ratio_to_treatment"]},
        "replay_balance": {
            "no_category_removed": manifest["replay_balance"]["no_category_removed"],
            "minimum_kept_fraction": manifest["replay_balance"]["minimum_kept_fraction"],
            "integer_granularity_note": manifest["replay_balance"][
                "integer_granularity_note"]},
        "prompt_compaction_evidence": {
            name: {key: prepared["stats"][name][key] for key in (
                "retained_examples", "dropped_overlong_examples",
                "malformed_prompt_examples", "prompt_compacted_examples",
                "prompt_truncated_examples", "support_units_dropped",
                "code_units_dropped", "prompt_compaction_strategy")}
            for name in (prepared.get("stats") or {})} or None,
        "declared_differences_from_control": [
            "256 canonical rows replaced by ordered execution-trace examples (the intervention)",
            "auxiliary execution targets use the 1,024-token completion ceiling the 12% "
            "arm used; no canonical function completion exceeds 128 tokens, so canonical "
            "rows are prepared identically",
            "supervised token mass is 1.205306x the control's; the effect of this "
            "difference cannot be separated from the supervision type",
        ],
        "optimizer_schedule": schedule,
        "trainer_preparation": prepared,
        "runtime_and_storage_estimate": estimate,
        "gpu_sequence_after_approval": [
            f"train {ARM_NAME} (scripts/run_execution_dose_pilot.py)",
            f"mechanism eval {ARM_NAME} (scripts/evaluate_execution_dose_mechanism.py)",
            "retention Kill@8: control, then dose_treatment, then base "
            "(scripts/evaluate_execution_dose_retention.py)",
            "frozen analysis (scripts/analyse_execution_dose_pilot.py, CPU)",
        ],
        "source_files_sha256": {path: canonical_sha256(ROOT / path) for path in BOUND_SOURCES},
        "leakage": {"validation_accessed": False, "ablation_dev_accessed": False,
                    "test_accessed": False, "sealed_final_test_accessed": False,
                    "confirmation_opened": False},
        "training_launched": False,
        "next_permitted_step": ("stop for explicit approval before any GPU training"
                                if not problems else "fix preflight problems; no GPU work"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECEIPT)
    args = parser.parse_args(argv)
    receipt = preflight()
    args.output.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: receipt[key] for key in (
        "ready", "problems", "trainer_preparation", "runtime_and_storage_estimate")},
        indent=2))
    return 0 if receipt["ready"] else 2


def verify_receipt_for_launch(receipt_path: Path = RECEIPT) -> dict[str, Any]:
    """Shared launch gate for the runner and both evaluators.

    The receipt names the commit it was built at.  HEAD may differ from that
    commit only by the receipt itself (it is committed afterwards), and every
    bound source must still hash to what the receipt recorded.
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("ready") is not True:
        raise SystemExit("REFUSED: dose preflight is not ready")
    if git("status", "--porcelain"):
        raise SystemExit("REFUSED: working tree is not clean")
    commit = receipt["git"]["commit"]
    if not git_ok("merge-base", "--is-ancestor", commit, "HEAD"):
        raise SystemExit("REFUSED: preflight commit is not an ancestor of HEAD")
    changed = set(filter(None, git("diff", "--name-only", commit, "HEAD").splitlines()))
    allowed = {receipt_path.relative_to(ROOT).as_posix()}
    if changed - allowed:
        raise SystemExit(f"REFUSED: files changed since preflight: {sorted(changed - allowed)}")
    for relative, expected in receipt["source_files_sha256"].items():
        if canonical_sha256(ROOT / relative) != expected:
            raise SystemExit(f"REFUSED: bound source drift: {relative}")
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
