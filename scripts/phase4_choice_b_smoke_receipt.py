"""Tracked receipt for the Choice B v2 GPU integration smoke (operational evidence only).

Embeds the ignored smoke report and the durable run's status and manifest (repository
root normalised) with their original hashes. Not efficacy evidence; no gate data.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.evidence_bundle import SCHEMA_VERSION, bundle_problems, embed, load_json_text

OUTPUT = "results/sft_root_cause_phase4_choice_b_smoke_v2.json"


def main(argv=None) -> int:
    run_id = (argv or sys.argv[1:])[0]
    run = f"runs/{run_id}"
    files = {"smoke_report.json": embed(ROOT, "results/sft_root_cause/phase4_choice_b_v2_SMOKE/"
                                              "smoke_report.json", ["repository_root"]),
             "run_status.json": embed(ROOT, f"{run}/status.json", ["repository_root"]),
             "run_manifest.json": embed(ROOT, f"{run}/manifest.json", ["repository_root"])}
    report = load_json_text(files["smoke_report.json"])
    status = load_json_text(files["run_status.json"])
    manifest = load_json_text(files["run_manifest.json"])
    receipt = {"schema_version": SCHEMA_VERSION, "bundle": "choice_b_v2_gpu_smoke",
               "purpose": "operational integration evidence only; NOT efficacy; no gate data",
               "run_id": run_id, "run_git_commit": manifest["git"]["commit"],
               "run_git_dirty": manifest["git"]["dirty"], "state": status["state"],
               "exit_code": status["exit_code"], "duration_seconds": status["duration_seconds"],
               "passed": report["passed"], "checks": report["checks"], "files": files}
    problems = bundle_problems(ROOT, receipt)
    if problems or not report["passed"] or status["exit_code"] != 0:
        raise SystemExit(f"REFUSED: {problems or 'smoke did not pass'}")
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({"passed": receipt["passed"], "commit": receipt["run_git_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
