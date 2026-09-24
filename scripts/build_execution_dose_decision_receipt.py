"""Freeze the decision receipt for the composite execution-intervention pilot.

Small and tracked.  Everything it points at is large and stays in ignored
local storage on the GPU machine, bound here by SHA-256: the training result
and adapter, the mechanism artifact, the three retention envelopes and their
raw rehearsal results, the durable run records and the source files.

It refuses to write a receipt that disagrees with its inputs: every artifact
the frozen analysis read must still hash to what the analysis recorded, the
adapter must match its training result, and each run must have completed with
exit code 0.

It also records one exploratory, non-gating descriptor: how many mechanism
outputs are byte-identical between control and treatment.

CPU only.  Reads nothing outside the train-derived artifacts of this pilot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_dose import INFERENCE_LIMITATION, INTERVENTION_LABEL
from harness.source_identity import canonical_sha256

RECEIPT_VERSION = "oneiros_execution_dose_decision_receipt_v1"
DOSE = "results/v4_3_execution_dose_v1"
TRAINING = {"result": f"{DOSE}/dose_treatment_training_result.json",
            "adapter": "checkpoints/v4_3_execdose_d25_qwen15b_s42",
            "run": "runs/20260924-194214-execdose-d25-treatment-qwen15b-s42"}
EVALUATION_RUNS = {
    "mechanism_dose_treatment": "runs/20260924-195055-execdose-mechanism-qwen15b-s42",
    "retention_control": "runs/20260924-195156-execdose-retention-control-qwen15b-s42",
    "retention_dose_treatment": "runs/20260924-200500-execdose-retention-treatment-qwen15b-s42",
    "retention_base": "runs/20260924-201642-execdose-retention-base-qwen15b-s42",
}
TRACKED = ("results/v4_3_execution_dose_census.json",
           "results/v4_3_execution_dose_design.json",
           "results/v4_3_execution_dose_dataset_manifest.json",
           "results/v4_3_execution_dose_matched_control.json",
           "results/v4_3_execution_dose_retention_panel.json",
           "results/v4_3_execution_dose_preflight_receipt.json")
SOURCES = ("scripts/census_execution_dose_pool.py", "scripts/build_execution_dose_ab.py",
           "scripts/build_execution_dose_matched_control.py",
           "scripts/freeze_execution_dose_retention_panel.py",
           "scripts/preflight_execution_dose_ab.py", "scripts/run_execution_dose_pilot.py",
           "scripts/evaluate_execution_dose_mechanism.py",
           "scripts/evaluate_execution_dose_retention.py",
           "scripts/analyse_execution_dose_pilot.py",
           "scripts/build_execution_dose_decision_receipt.py",
           "harness/execution_dose.py", "engine/sft_trainer.py",
           "harness/rehearsal_evaluator.py", "harness/generation_adapter.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def run_record(relative: str) -> dict:
    status = json.loads((ROOT / relative / "status.json").read_text(encoding="utf-8"))
    return {"run_id": Path(relative).name, "state": status["state"],
            "exit_code": status.get("exit_code"),
            "duration_seconds": status.get("duration_seconds"),
            "start_utc": status.get("start_utc"), "end_utc": status.get("end_utc")}


def identical_outputs(control: dict, treatment: dict) -> dict:
    """Exploratory: byte-identical raw outputs per condition (not a gate)."""
    result = {}
    for condition in ("intended_output", "shown_actual_output"):
        left = {row["record_id"]: row["raw"] for row in control["detail"][condition]}
        right = {row["record_id"]: row["raw"] for row in treatment["detail"][condition]}
        same = sum(left[key] == right[key] for key in left)
        result[condition] = {"identical": same, "items": len(left),
                             "fraction": round(same / len(left), 4)}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="results/v4_3_execution_dose_analysis.json")
    parser.add_argument("--out", default="results/v4_3_execution_dose_decision_receipt.json")
    args = parser.parse_args(argv)

    analysis_path = ROOT / args.analysis
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    for name, entry in analysis["inputs"].items():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            print(f"REFUSED: {name} no longer matches the frozen analysis")
            return 2
    result = json.loads((ROOT / TRAINING["result"]).read_text(encoding="utf-8"))
    adapter = sha(ROOT / TRAINING["adapter"] / "adapter_model.safetensors")
    if adapter != result["adapter_sha256"]:
        print("REFUSED: adapter on disk differs from its training receipt")
        return 2
    runs = {"training": run_record(TRAINING["run"])}
    runs.update({name: run_record(path) for name, path in EVALUATION_RUNS.items()})
    if any(run["state"] != "completed" or run["exit_code"] != 0 for run in runs.values()):
        print(f"REFUSED: a run did not complete cleanly: {runs}")
        return 2

    retention = {}
    for arm in ("control", "dose_treatment", "base"):
        envelope_path = f"{DOSE}/retention_{arm}.json"
        envelope = json.loads((ROOT / envelope_path).read_text(encoding="utf-8"))
        raw = ROOT / envelope["rehearsal_result"]["path"]
        if sha(raw) != envelope["rehearsal_result"]["sha256"]:
            print(f"REFUSED: retention {arm} raw result hash mismatch")
            return 2
        retention[arm] = {"envelope": meta(envelope_path),
                          "rehearsal_result": meta(envelope["rehearsal_result"]["path"]),
                          "adapter_sha256": envelope["adapter_sha256"],
                          "kill_at_k": envelope["kill_at_k"],
                          "wall_time_seconds": envelope["wall_time_seconds"]}

    control_eval = json.loads((ROOT / analysis["inputs"]["control_mechanism"]["path"])
                              .read_text(encoding="utf-8"))
    treatment_eval = json.loads((ROOT / analysis["inputs"]["treatment_mechanism"]["path"])
                                .read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    preflight = json.loads((ROOT / TRACKED[-1]).read_text(encoding="utf-8"))
    metrics = result["metrics"]
    decision = analysis["decision"]
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "label": ("train-derived mechanism and retention diagnostic; not generalisation, "
                  "not model selection, not a final-test result"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "intervention": INTERVENTION_LABEL,
        "design_class": preflight["design_class"],
        "inference_limitation": INFERENCE_LIMITATION,
        "experiment_commit": result["run_contract"]["preflight_commit"],
        "launched_at_commit": preflight["git"]["commit"],
        "receipt_built_at_commit": head,
        "training": {
            "result": meta(TRAINING["result"]), "adapter_sha256": adapter,
            "run_contract_sha256": result["run_contract_sha256"],
            "retained_examples": metrics["retained_examples"],
            "dropped_overlong_examples": metrics["dropped_overlong_examples"],
            "malformed_prompt_examples": metrics["malformed_prompt_examples"],
            "prompt_compacted_examples": metrics["prompt_compacted_examples"],
            "prompt_truncated_examples": metrics["prompt_truncated_examples"],
            "support_units_dropped": metrics["support_units_dropped"],
            "code_units_dropped": metrics["code_units_dropped"],
            "completed_optimizer_steps": metrics["completed_optimizer_steps"],
            "completed_epochs": metrics["completed_epochs"],
            "task_kind_counts": metrics["task_kind_counts"],
            "training_loss_not_used_for_any_decision": metrics["loss"],
        },
        "mechanism_artifact": meta(analysis["inputs"]["treatment_mechanism"]["path"]),
        "control_mechanism_artifact": meta(analysis["inputs"]["control_mechanism"]["path"]),
        "retention": retention,
        "runs": runs,
        "tracked_design_artifacts": {relative: sha(ROOT / relative) for relative in TRACKED},
        "analysis": {**meta(args.analysis),
                     "mechanism_checks": analysis["mechanism"]["checks"],
                     "mechanism_passed": analysis["mechanism"]["passed"],
                     "mechanism_treatment_minus_control":
                         analysis["mechanism"]["treatment_minus_control"],
                     "retention_verdict": analysis["retention"]["verdict"],
                     "retention_treatment_minus_control":
                         analysis["retention"]["treatment_minus_control"]},
        "exploratory_not_gating": {
            "byte_identical_mechanism_outputs_control_vs_treatment":
                identical_outputs(control_eval, treatment_eval)},
        "decision": {
            **decision,
            "retained_reference": "frozen control (adapter c1fe9e52...)",
            "not_permitted": [
                "attributing any effect to execution supervision alone",
                "dose escalation, extra epochs, mixture change or re-thresholding",
                "promotion of the treatment adapter",
                "opening the 100-lineage confirmation panel",
                "validation, ablation_dev, test, sealed-final or canonical records.json "
                "access",
            ],
        },
        "leakage": {"sealed_final_test_accessed": False, "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "confirmation_opened": False, "canonical_records_json_opened": False},
        "source_files_sha256": {relative: canonical_sha256(ROOT / relative)
                                for relative in SOURCES},
    }
    out = ROOT / args.out
    out.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"outcome": decision["outcome"],
                      "exploratory": receipt["exploratory_not_gating"],
                      "written": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
