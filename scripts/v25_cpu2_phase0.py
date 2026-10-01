"""v2.5 CPU program 2, Phase 0: read-only reverification of the accepted state at a5be32e.

Opens no corpus, validation, reserved or sealed path. Rehashes every receipt added after the
Stage-1 commit, byte-verifies the Stage-1 and Stage-1-r2 external archives against their tracked
manifests, checks the full-suite receipt is bound to 66e7397 and that HEAD is its receipt-only
descendant, and publishes a new receipt (never overwriting).

    python scripts/v25_cpu2_phase0.py
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

RECEIPT = "results/sft_root_cause_v25_cpu2_phase0_reverification.json"
EXPECTED_HEAD = "a5be32ea09e5f0899099b634dcb0702603803fbe"
STAGE1_COMMIT = "6b35b20"
SUITE_COMMIT = "66e7397440bf38f19f4a629cefe847fd808ecad6"
ARCHIVE_ROOT = Path(r"C:\Users\Student2\oneiros_archive")
ARCHIVE_MANIFESTS = ("results/sft_root_cause_v25_stage1_archive_manifest.json",
                     "results/sft_root_cause_v25_stage1_r2_archive_manifest.json")
HEALTH = "results/sft_root_cause/native_v25_full_suite_health_66e7397.json"
STOP = "results/sft_root_cause_v25_stop_decision.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    issues = []
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/" + git("rev-parse", "--abbrev-ref", "HEAD"))
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=all").splitlines()
             if not l.endswith("scripts/v25_cpu2_phase0.py")]   # this script, before commit
    if head != EXPECTED_HEAD:
        issues.append(f"head {head} != expected")
    if head != origin:
        issues.append("head != origin")
    if dirty:
        issues.append("worktree not clean")
    receipts = {}
    for rel in git("diff", "--name-only", "--diff-filter=AM", STAGE1_COMMIT, head,
                   "--", "results").splitlines():
        blob = git("show", f"{head}:{rel}").encode("utf-8") + b"\n"
        disk = (ROOT / rel).read_bytes()
        receipts[rel] = sha(ROOT / rel)
        if hashlib.sha256(disk).hexdigest() != receipts[rel] or \
                hashlib.sha256(blob).hexdigest() != receipts[rel]:
            issues.append(f"receipt {rel} differs from committed blob")
    archives = {}
    for rel in ARCHIVE_MANIFESTS:
        man = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        base = ARCHIVE_ROOT / man["archive_directory_name"]
        ok = 0
        for item in man["items"]:
            path = base / item["file"]
            if not path.is_file() or sha(path) != item["sha256"] or \
                    path.stat().st_size != item["bytes"]:
                issues.append(f"archive {man['archive_directory_name']}/{item['file']} differs")
            else:
                ok += 1
        archives[man["archive_directory_name"]] = {"items": len(man["items"]),
                                                   "byte_verified": ok}
    health = json.loads((ROOT / HEALTH).read_text(encoding="utf-8"))
    stop = json.loads((ROOT / STOP).read_text(encoding="utf-8"))
    descendant = git("diff", "--name-only", SUITE_COMMIT, head).splitlines()
    if health["source_commit"] != SUITE_COMMIT or health["failed"] or health["exit"]:
        issues.append("full-suite receipt not a green run of 66e7397")
    if sorted(descendant) != sorted([HEALTH, STOP]):
        issues.append(f"HEAD is not a receipt-only descendant of 66e7397: {descendant}")
    if stop["health"]["sha256"] != sha(ROOT / HEALTH):
        issues.append("stop decision does not bind the health receipt")
    locks = ROOT / "runs" / ".exclusive"
    active_locks = sorted(p.name for p in locks.iterdir()) if locks.is_dir() else []
    receipt = {
        "schema_version": "oneiros_v25_cpu2_phase0_reverification_v1",
        "head": head, "head_equals_origin": head == origin, "worktree_clean": not dirty,
        "receipts_after_stage1": receipts,
        "archives": archives,
        "full_suite": {"receipt": HEALTH, "sha256": sha(ROOT / HEALTH),
                       "source_commit": health["source_commit"], "passed": health["passed"],
                       "failed": health["failed"], "skipped": health["skipped"],
                       "head_is_receipt_only_descendant": sorted(descendant) ==
                       sorted([HEALTH, STOP])},
        "active_exclusive_locks": active_locks,
        "protected_access_in_phase0": "none (no corpus, validation, reserved or sealed path "
                                      "opened)",
        "issues": issues}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "issues": issues, "receipts": len(receipts),
                      "archives": archives, "sha256": sha(ROOT / RECEIPT)}, indent=1))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
