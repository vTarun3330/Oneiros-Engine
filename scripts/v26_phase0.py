"""v2.6 Phase 0: read-only reverification of the completed v2.5 state (751539e).

Opens no corpus, validation, confirmation-only or sealed path. Rehashes the key v2.5 protocols
and receipts, byte-verifies every external archive against its tracked manifest, measures the
current canonical executable-tree hash and compares it with the last full-suite receipt, and
records v2.5 as a completed NEGATIVE FEASIBILITY result (not failed model training).

    python scripts/v26_phase0.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v26_phase0_reverification.json"
EXPECTED_HEAD = "751539ef1adbea52a0fa432a5524634b2f399c79"
ARCHIVE_ROOT = Path(r"C:\Users\Student2\oneiros_archive")
ARCHIVES = ("results/sft_root_cause_v25_stage1_archive_manifest.json",
            "results/sft_root_cause_v25_stage1_r2_archive_manifest.json",
            "results/sft_root_cause_v25_cpu2_archive_manifest.json",
            "results/sft_root_cause_v25_cpu2_archive_manifest_r2.json")
SUITE = "results/sft_root_cause/native_v25_cpu2_full_suite_fb04648.json"
KEY = ("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5.md",
       "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_1.md",
       "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_2.md",
       "results/sft_root_cause_v25_repository_verification_r2.json",
       "results/sft_root_cause_v25_attainable_gate_decision.json",
       "results/sft_root_cause_v25_acquisition_projection.json",
       "results/sft_root_cause_v25_cpu2_preflight_decision.json",
       "results/sft_root_cause_v25_universe_partition.json",
       "results/sft_root_cause_v25_atheris_readiness.json", SUITE)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    from harness.native_launch_gate import source_identity
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    issues = []
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/experiment/research-eval-ablations")
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=all").splitlines()
             if not l.endswith("scripts/v26_phase0.py")]
    if head != EXPECTED_HEAD or head != origin or dirty:
        issues.append({"head": head, "origin": origin, "dirty": dirty})
    key = {k: sha(ROOT / k) for k in KEY}
    for k in KEY:
        committed = subprocess.run(["git", "show", f"HEAD:{k}"], cwd=ROOT,
                                   capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest() != key[k]:
            issues.append(f"{k} differs from the committed blob")
    archives = {}
    for rel in ARCHIVES:
        man = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        base = ARCHIVE_ROOT / man["archive_directory_name"]
        bad = [i["file"] for i in man["items"]
               if not (base / i["file"]).is_file() or sha(base / i["file"]) != i["sha256"]]
        archives[man["archive_directory_name"]] = {"items": len(man["items"]),
                                                   "byte_verified": len(man["items"]) - len(bad)}
        issues += [f"archive {man['archive_directory_name']}/{b}" for b in bad]
    suite = json.loads((ROOT / SUITE).read_text(encoding="utf-8"))
    current = source_identity(ROOT)["executable_tree_sha256"]
    receipt = {
        "schema_version": "oneiros_v26_phase0_reverification_v1",
        "head": head, "head_equals_origin": head == origin, "worktree_clean": not dirty,
        "key_artifacts_sha256": key, "archives": archives,
        "full_suite_currentness": {
            "receipt": SUITE, "receipt_source_commit": suite["source_commit"],
            "receipt_executable_tree_sha256": suite.get("executable_tree_sha256"),
            "current_executable_tree_sha256": current,
            "certifies_current_source": suite.get("executable_tree_sha256") == current,
            "note": "751539e added scripts/v25_cpu2_preflight_decision.py after the suite "
                    "ran at fb04648; a new suite run is required at the final source"},
        "v25_status": "COMPLETED NEGATIVE FEASIBILITY RESULT: the frozen 150/8/60 training "
                      "gate is not reachable from the 13 existing train repositories (28 "
                      "verified unique tests, 3 repositories, 28 lineages); no SFT was trained, "
                      "so this is not a failed model training",
        "protected_access": "none (no corpus, validation, confirmation-only or sealed path "
                            "opened)",
        "issues": issues}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "issues": issues, "archives": archives,
                      "full_suite_currentness": receipt["full_suite_currentness"],
                      "sha256": sha(ROOT / RECEIPT)}, indent=1))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
