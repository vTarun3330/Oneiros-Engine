"""Freeze the decision receipt for the execution-order (trace) pilot.

Small and tracked; everything it points at is large and stays in ignored local
storage, bound here by SHA-256. It records the frozen gate verdict exactly as
the frozen analysis produced it, the post-gate diagnosis that explains it, and
the explicit decision taken on that basis.

It refuses to write a receipt that disagrees with its inputs: every artifact
the frozen analysis read must hash to what the analysis recorded, the diagnosis
must be bound to that same analysis, and both training results must still match
their adapters on disk.

CPU only. Reads nothing outside the train-derived mechanism panel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

RECEIPT_VERSION = "oneiros_execution_trace_decision_receipt_v1"

TRAINING = {
    "unordered_events": {
        "result": "results/v4_3_execution_trace_v1/unordered_training_result.json",
        "adapter": "checkpoints/v4_3_execorder_unordered_qwen15b_s42",
        "run": "runs/20260924-153249-execorder-unordered-qwen15b-s42",
    },
    "ordered_trace": {
        "result": "results/v4_3_execution_trace_v1/ordered_training_result.json",
        "adapter": "checkpoints/v4_3_execorder_ordered_qwen15b_s42",
        "run": "runs/20260924-154111-execorder-ordered-qwen15b-s42",
    },
}
EVALUATION_RUNS = {
    "unordered_events": "runs/20260924-155542-execorder-eval-unordered-qwen15b-s42",
    "ordered_trace": "runs/20260924-155717-execorder-eval-ordered-qwen15b-s42",
}
EVALUATIONS = {
    "control": "results/v4_3_execution_supervision_v1/eval_control.json",
    "unordered_events": "results/v4_3_execution_trace_v1/eval_unordered_events.json",
    "ordered_trace": "results/v4_3_execution_trace_v1/eval_ordered_trace.json",
}
SUPPORTING = {
    "trace_preflight": "results/v4_3_execution_trace_v1/preflight.json",
    "trace_manifest": "results/v4_3_execution_trace_v1/manifest.json",
    "pilot_panel": "results/v4_3_execution_supervision_v1/pilot_development.execution.json",
}
SOURCES = (
    "scripts/build_execution_trace_ab.py",
    "scripts/preflight_execution_trace_ab.py",
    "scripts/run_execution_trace_pilot.py",
    "scripts/evaluate_execution_trace_pilot.py",
    "scripts/evaluate_execution_supervision_pilot.py",
    "scripts/analyse_execution_trace_pilot.py",
    "scripts/diagnose_execution_trace_pilot.py",
    "engine/sft_trainer.py",
    "harness/execution_supervision.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="results/v4_3_execution_trace_pilot_analysis.json")
    parser.add_argument("--diagnosis", default="results/v4_3_execution_trace_pilot_diagnosis.json")
    parser.add_argument("--out", default="results/v4_3_execution_trace_pilot_decision_receipt.json")
    args = parser.parse_args(argv)

    analysis_path, diagnosis_path = ROOT / args.analysis, ROOT / args.diagnosis
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))

    for arm, relative in EVALUATIONS.items():
        if analysis["inputs"][arm]["sha256"] != sha(ROOT / relative):
            print(f"REFUSED: {arm} evaluation no longer matches the frozen analysis")
            return 2
    if diagnosis["inputs"]["analysis_sha256"] != sha(analysis_path):
        print("REFUSED: diagnosis is not bound to this analysis")
        return 2

    training = {}
    for arm, paths in TRAINING.items():
        result = json.loads((ROOT / paths["result"]).read_text(encoding="utf-8"))
        adapter = sha(ROOT / paths["adapter"] / "adapter_model.safetensors")
        if adapter != result["adapter_sha256"]:
            print(f"REFUSED: {arm} adapter on disk differs from its training receipt")
            return 2
        status = json.loads((ROOT / paths["run"] / "status.json").read_text(encoding="utf-8"))
        metrics = result["metrics"]
        training[arm] = {
            "result": meta(paths["result"]),
            "adapter_sha256": adapter,
            "run_id": Path(paths["run"]).name,
            "run_state": status["state"], "run_exit_code": status.get("exit_code"),
            "run_duration_seconds": status.get("duration_seconds"),
            "completed_optimizer_steps": metrics["completed_optimizer_steps"],
            "completed_epochs": metrics["completed_epochs"],
            "retained_examples": metrics["retained_examples"],
            "task_kind_counts": metrics["task_kind_counts"],
            "training_loss": metrics["loss"],
            "run_contract_sha256": result["run_contract_sha256"],
        }

    evaluations = {}
    for arm, relative in EVALUATIONS.items():
        artifact = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        entry = {"artifact": meta(relative), "adapter_sha256": artifact.get("adapter_sha256"),
                 "evaluator_source_sha256": artifact.get("source_sha256"),
                 "decoding": artifact.get("decoding"), "summary": artifact["summary"]}
        if arm in EVALUATION_RUNS:
            status = json.loads((ROOT / EVALUATION_RUNS[arm] / "status.json").read_text(
                encoding="utf-8"))
            entry.update(run_id=Path(EVALUATION_RUNS[arm]).name, run_state=status["state"],
                         run_exit_code=status.get("exit_code"),
                         run_duration_seconds=status.get("duration_seconds"))
        evaluations[arm] = entry

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    # The receipt is tracked, so it is always built before the commit that
    # contains it. Record that honestly instead of implying HEAD produced it.
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip())
    contract = json.loads((ROOT / TRAINING["ordered_trace"]["result"]).read_text(
        encoding="utf-8"))["run_contract"]
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "label": ("train-derived mechanism diagnostic; not generalisation, not model "
                  "selection, not a final-test result"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_commit": contract["git_commit"],
        "receipt_built_at_commit": head,
        "worktree_clean_at_build": not dirty,
        "model": contract["model_name"],
        "model_revision": contract["model_revision"],
        "comparison": ("unordered_events vs ordered_trace: identical 122-example execution "
                       "event multiset, temporal order removed vs preserved; both judged "
                       "against the earlier prompt-only control"),
        "training": training,
        "evaluations": evaluations,
        "supporting_artifacts": {name: meta(rel) for name, rel in SUPPORTING.items()},
        "analysis": {**meta(args.analysis),
                     "predeclared_gates": analysis["predeclared_gates"],
                     "comparisons_vs_control": analysis["comparisons_vs_control"],
                     "ordered_minus_unordered": analysis["ordered_minus_unordered"],
                     "gate_checks": analysis["gate_checks"],
                     "mechanism_gate_passed": analysis["mechanism_gate_passed"],
                     "passing_arms": analysis["passing_arms"]},
        "diagnosis": {**meta(args.diagnosis),
                      "answer_tracking_where_shown_differs":
                          diagnosis["answer_tracking_where_shown_differs"],
                      "min_gain_exclusion_intended_output":
                          diagnosis["min_gain_exclusion_intended_output"],
                      "min_gain_exclusion_shown_actual_output":
                          diagnosis["min_gain_exclusion_shown_actual_output"],
                      "gate_mechanics_findings": diagnosis["gate_mechanics_findings"]},
        "source_files_sha256": {relative: sha(ROOT / relative) for relative in SOURCES},
        "decision": {
            "advance": False,
            "reason": "frozen mechanism gate failed for both arms",
            "retained_reference": "prior prompt-only control (eval_control.json)",
            "not_permitted": [
                "opening the 100-lineage confirmation panel",
                "validation, ablation_dev, test or sealed-final access",
                "canonical Kill@8 for either trace arm",
                "further training, extra epochs, mixture changes or re-thresholding "
                "of this intervention",
            ],
        },
        "leakage": {"sealed_final_test_accessed": False, "validation_accessed": False,
                    "ablation_dev_accessed": False, "confirmation_opened": False,
                    "evaluation_panel": "97 train-derived lineages, frozen before training"},
    }
    out = ROOT / args.out
    # LF bytes so the hash of this tracked file survives a fresh checkout
    # (.gitattributes stores *.json as LF; Windows text mode would write CRLF).
    out.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(f"decision: advance={receipt['decision']['advance']}  "
          f"gate_passed={receipt['analysis']['mechanism_gate_passed']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
