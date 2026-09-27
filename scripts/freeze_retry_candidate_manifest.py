"""Freeze the exact retry sample: every candidate identity of the refused
A-prime confirmation pilot, unfiltered by outcome.

For each of the refused journal's candidates it records the canonical
repository, the fixed commit, its buggy parent (read from retained git objects,
never fetched), the linked issue/PR number and the commit class, plus the
refused run's status for later comparison.  No candidate is removed, and no
newer commit may replace one.  Offline: git lazy fetching is disabled.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.github_acquisition import ContentStore, Journal
from harness.repository_isolation import candidate_from_sidecar, parse_commit

SOURCE = "data/repository_native/aprime_confirmation_pilot"
SOURCE_JOURNAL_SHA256 = "a065e9138da96e40129088f849f62e38c70b3f0a836e94e2f4de6cc8a957326a"
REFUSED_RECORD = "results/v4_3_repository_native_aprime_confirmation_pilot_REFUSED.json"
OUTPUT = "results/v4_3_repository_native_aprime_retry_candidate_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_from_clone(clone: Path, oid: str) -> str | None:
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    result = subprocess.run(["git", "--git-dir", str(clone), "cat-file", "commit", oid],
                            env=env, capture_output=True)
    if result.returncode != 0:
        return None
    parents = parse_commit(result.stdout)["parents"]
    return parents[0] if parents else None


def main() -> int:
    source = ROOT / SOURCE
    journal_path = source / "journal.jsonl"
    if sha256(journal_path) != SOURCE_JOURNAL_SHA256:
        raise SystemExit("REFUSED: the refused journal changed")
    store = ContentStore(source / "objects")
    entries = Journal(journal_path).entries()
    scans = {e["repository"]: e for e in entries if e["key"].startswith("scan:")}
    candidates = []
    for entry in entries:
        if not entry["key"].startswith("cand:"):
            continue
        repository, oid = entry["key"][5:].rsplit("@", 1)
        item = next(i for i in scans[repository]["selected"] if i["oid"] == oid)
        parent = None
        if entry.get("addresses"):
            candidate = candidate_from_sidecar(json.loads(store.get_raw(
                entry["addresses"]["sidecar"])))
            parent = candidate.buggy_commit
        if parent is None:
            parent = parent_from_clone(source / "repos" / (repository.replace("/", "__")
                                                           + ".git"), oid)
        outcome = entry.get("outcome") or {}
        candidates.append({
            "repository": repository, "fixed_commit": oid, "buggy_commit": parent,
            "linked_number": item["number"], "commit_class": item["reason"],
            "refused_run_status": ("ADMITTED" if outcome.get("admitted") else
                                   entry.get("pre_pipeline_exclusion") or entry.get("failure")
                                   or (outcome.get("reasons") or ["?"])[0])})
    missing_parent = [c["fixed_commit"] for c in candidates if not c["buggy_commit"]]
    if missing_parent:
        raise SystemExit(f"REFUSED: buggy parent unrecoverable offline for {missing_parent}")
    repositories = list(dict.fromkeys(c["repository"] for c in candidates))
    manifest = {
        "schema_version": "oneiros_retry_candidate_manifest_v1",
        "label": ("exact retry sample: ALL candidate identities of the refused A-prime "
                  "confirmation pilot, unfiltered by outcome; the retry must evaluate exactly "
                  "these, with no rediscovery, substitution or early stop"),
        "source_store": SOURCE, "source_journal_sha256": SOURCE_JOURNAL_SHA256,
        "refused_record": REFUSED_RECORD,
        "refused_record_sha256": sha256(ROOT / REFUSED_RECORD),
        "candidate_count": len(candidates), "repository_count": len(repositories),
        "repositories": repositories, "candidates": candidates}
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(manifest, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({"candidates": len(candidates), "repositories": len(repositories),
                      "sha256": sha256(ROOT / OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
