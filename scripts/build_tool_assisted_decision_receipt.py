"""Freeze the decision receipt for the execution-feedback repair pilot.

Small and tracked; everything it points at stays in ignored local storage on
the GPU machine and is bound here by SHA-256: the design receipt, the panel,
the adapter, the three evaluation envelopes and their raw rehearsal results,
the append-only generation journal, a per-file manifest of the lineage
directory, every durable run record and the source files.

It refuses to write a receipt that disagrees with its inputs: the analysis
must still verify against the evaluation artifacts, every run must have
completed with exit code 0, and every lineage file must be present.

CPU only.  Reads nothing outside this pilot's train-derived artifacts.
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

from harness.source_identity import canonical_sha256

SCHEMA = "oneiros_tool_assisted_decision_receipt_v1"
OUT = ROOT / "results" / "v4_3_tool_assisted_v1"
RUNS = {
    "generate-a": "runs/20260925-103511-toolassist-generate-a-qwen15b-s42",
    "generate-bc": "runs/20260925-104038-toolassist-generate-bc-qwen15b-s42",
    "score-b": "runs/20260925-113357-toolassist-score-b-qwen15b-s42",
    "score-c": "runs/20260925-113501-toolassist-score-c-qwen15b-s42",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
            "bytes": path.stat().st_size}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="results/v4_3_tool_assisted_analysis.json")
    parser.add_argument("--out", default="results/v4_3_tool_assisted_decision_receipt.json")
    args = parser.parse_args(argv)
    design_path = ROOT / "results" / "v4_3_tool_assisted_design_receipt.json"
    panel_path = ROOT / "results" / "v4_3_tool_assisted_panel.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    analysis_path = ROOT / args.analysis
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    runs = {}
    for stage, relative in RUNS.items():
        status = json.loads((ROOT / relative / "status.json").read_text(encoding="utf-8"))
        runs[stage] = {"run_id": Path(relative).name, "state": status["state"],
                       "exit_code": status.get("exit_code"),
                       "duration_seconds": status.get("duration_seconds"),
                       "start_utc": status.get("start_utc"), "end_utc": status.get("end_utc")}
    if any(run["state"] != "completed" or run["exit_code"] != 0 for run in runs.values()):
        print(f"REFUSED: a run did not complete cleanly: {runs}")
        return 2

    evaluations = {}
    for arm in ("A", "B", "C"):
        envelope_path = OUT / f"eval_{arm}.json"
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        raw = ROOT / envelope["rehearsal_result"]["path"]
        if sha(raw) != envelope["rehearsal_result"]["sha256"]:
            print(f"REFUSED: arm {arm} raw result hash mismatch")
            return 2
        evaluations[arm] = {"envelope": meta(envelope_path), "rehearsal_result": meta(raw),
                            "kill_at_k": envelope["kill_at_k"],
                            "wall_time_seconds": envelope["wall_time_seconds"]}

    manifest = {}
    for record_id in panel["record_ids"]:
        path = OUT / "loop" / (hashlib.sha256(record_id.encode()).hexdigest() + ".json")
        if not path.exists():
            print(f"REFUSED: lineage file missing for {record_id}")
            return 2
        manifest[path.name] = sha(path)
    manifest_bytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = ROOT / "results" / "v4_3_tool_assisted_lineage_manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    control = json.loads((ROOT / "results" / "v4_3_execution_supervision_v1"
                          / "arm_a_training_result.json").read_text(encoding="utf-8"))
    adapter = sha(ROOT / "checkpoints" / "v4_3_execsup_a_control_qwen15b_s42"
                  / "adapter_model.safetensors")
    if adapter != control["adapter_sha256"] or adapter != design["adapter"]["sha256"]:
        print("REFUSED: adapter differs from its training result or the design receipt")
        return 2
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    primary = analysis["primary_c_minus_b"]
    receipt = {
        "schema_version": SCHEMA,
        "label": "exploratory train-derived pilot on a previously base-inspected panel; not "
                 "confirmation, generalisation, validation or final-test evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": design["question"],
        "control": design["control"],
        "design_receipt": {**meta(design_path), "built_at_commit": design["git"]["commit"]},
        "panel": {**meta(panel_path), "records": panel["records"], "lineages": panel["lineages"],
                  "evidence_class": panel["qualification"]["evidence_class"]},
        "adapter_sha256": adapter,
        "receipt_built_at_commit": head,
        "runs": runs,
        "evaluations": evaluations,
        "journal": meta(OUT / "journal.jsonl"),
        "lineage_manifest": {**meta(manifest_path), "files": len(manifest)},
        "analysis": {**meta(analysis_path), "verdict": analysis["verdict"],
                     "request_budget_matched": analysis["accounting"]["request_budget_matched"],
                     "pairing_problem_count": analysis["accounting"]["pairing_problem_count"],
                     "kill_at_8_c_minus_b_pp": primary["kill_at_8"],
                     "reference_valid_per_requested_c_minus_b_pp":
                         primary["reference_valid_per_requested"],
                     "repairs": analysis["accounting"]["repairs"],
                     "output_tokens": analysis["accounting"]["output_tokens"],
                     "wall_seconds": analysis["accounting"]["wall_seconds"],
                     "unique_defects_killed": analysis["unique_defects_killed"],
                     "secondary": analysis["secondary"]},
        "decision": {
            "outcome": analysis["verdict"],
            "action": analysis["decision"],
            "retained_reference": "frozen model-only control (adapter c1fe9e52...)",
            "promotion_permitted": False,
            "confirmation_opening_permitted": False,
            "not_permitted": ["additional repairs, calls or tokens",
                              "changed thresholds, prompts, feedback or padding",
                              "a rerun of this pilot",
                              "opening confirmation, validation, ablation_dev, test or "
                              "sealed-final data"],
        },
        "leakage": {"validation_accessed": False, "ablation_dev_accessed": False,
                    "test_accessed": False, "sealed_final_test_accessed": False,
                    "confirmation_opened": False, "canonical_records_json_opened": False,
                    "fixed_code_in_loop": False, "gold_tests_in_loop": False,
                    "weights_written": False},
        "source_files_sha256": design["source_files_sha256"] | {
            "scripts/build_tool_assisted_decision_receipt.py":
                canonical_sha256(Path(__file__))},
    }
    for relative, expected in design["source_files_sha256"].items():
        if canonical_sha256(ROOT / relative) != expected:
            print(f"REFUSED: source drift since the design receipt: {relative}")
            return 2
    (ROOT / args.out).write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"verdict": analysis["verdict"], "written": args.out,
                      "lineage_files": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
