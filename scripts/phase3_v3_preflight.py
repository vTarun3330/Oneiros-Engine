"""Phase 0 of the Phase 3 v3 repair: hash every immutable input before editing.

CPU only, read-only except for its own receipt.  Records branch/HEAD/origin,
the tree state, and SHA-256 of every artifact and source the repair depends
on.  ``--verify`` re-hashes against an existing receipt and fails on any
change to an immutable artifact (sources that the repair is declared to change
are reported, not failed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

RECEIPT = "results/sft_root_cause_phase3_v3_preflight.json"

IMMUTABLE_ARTIFACTS = [
    "results/sft_root_cause_phase3a_result_receipt.json",
    "results/sft_root_cause_phase3c_result_receipt.json",
    "results/sft_root_cause_phase3a_result_receipt_v2.json",
    "results/sft_root_cause_phase3c_result_receipt_v2.json",
    "results/sft_root_cause_phase3_interpretation_correction_2026-09-28.json",
    "results/sft_root_cause_phase3a_cohort.json",
    "results/sft_root_cause_phase3a_design_receipt.json",
    "results/sft_root_cause_phase3c_design_receipt.json",
    "results/sft_root_cause_phase3_cohort_census.json",
    "results/sft_root_cause_phase4_cohort_census.json",
    "results/sft_root_cause/phase3a/generations_base.jsonl",
    "results/sft_root_cause/phase3a/generations_arm_a_431.jsonl",
    "results/sft_root_cause/phase3c/generations_qwen7b_base.jsonl",
    "results/sft_root_cause/phase3a/completion_base.json",
    "results/sft_root_cause/phase3a/completion_arm_a_431.json",
    "results/sft_root_cause/phase3c/completion_qwen7b_base.json",
    "docs/SFT_ROOT_CAUSE_PHASE4_DESIGN_DRAFT.md",
]
SOURCES = [
    "harness/fixed_input_probe.py",
    "harness/probe_statistics.py",
    "harness/safe_execution.py",
    "engine/test_generation_prompt.py",
    "scripts/analyze_fixed_input_probe.py",
    "scripts/analyze_fixed_input_probe_v2.py",
    "scripts/analyze_capacity_probe.py",
    "scripts/analyze_capacity_probe_v2.py",
    "scripts/freeze_fixed_input_cohort.py",
    "scripts/correct_phase3_interpretation.py",
    "docs/SFT_ROOT_CAUSE_PROTOCOL.md",
]
#: Sources this repair is declared to change (everything else must stay byte-identical).
WILL_CHANGE = [
    "harness/probe_statistics.py",
    "harness/fixed_input_probe.py",
    "scripts/freeze_fixed_input_cohort.py",
    "docs/SFT_ROOT_CAUSE_PROTOCOL.md",
    "docs/SFT_ROOT_CAUSE_PHASE4_DESIGN_DRAFT.md (superseded notice appended at top only if "
    "declared in Phase C; otherwise unchanged)",
    "results/sft_root_cause_hypotheses.json (additive: withdrawals and new evidence)",
    "results/sft_root_cause_state.json (additive: current/history sections)",
]
#: v1 LEVEL_DEFINITIONS were written into the frozen cohort at this commit.
V1_PROBE_COMMIT = "c9c89608c3f2be188b2003a3bf126d124809c41d"


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    hashes = {p: sha(p) for p in IMMUTABLE_ARTIFACTS}
    if args.verify:
        frozen = json.loads((ROOT / RECEIPT).read_text(encoding="utf-8"))
        changed = [p for p, h in frozen["immutable_artifacts"].items() if hashes.get(p) != h]
        print(json.dumps({"immutable_changed": changed,
                          "sources_changed": [p for p, h in frozen["sources"].items()
                                              if sha(p) != h]}, indent=1))
        return 1 if changed else 0
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase3_v3_preflight_v1",
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "origin_head": git("rev-parse", "origin/experiment/research-eval-ablations"),
        # The only permitted pre-existing change is this preflight script itself.
        "working_tree_changes_at_start": [
            line for line in git("status", "--porcelain").splitlines()
            if not line.endswith(("scripts/phase3_v3_preflight.py", RECEIPT))],
        "model_calls": 0, "gpu_used": False, "training": False,
        "restricted_data_isolation": ("this preflight reads only the listed Phase 3 artifacts "
                                      "and sources; it opens no validation, ablation_dev, test, "
                                      "sealed-final, reserved-confirmation or A-prime record"),
        "immutable_artifacts": hashes,
        "sources": {p: sha(p) for p in SOURCES},
        "sources_declared_to_change": WILL_CHANGE,
        "v1_probe_definitions_commit": V1_PROBE_COMMIT,
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1) + "\n")
                            .encode("utf-8"))
    print(json.dumps({k: receipt[k] for k in ("branch", "head", "origin_head",
                                              "working_tree_changes_at_start")}, indent=1))
    print(f"{len(hashes)} immutable artifacts and {len(SOURCES)} sources hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
