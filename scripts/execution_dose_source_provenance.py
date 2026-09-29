"""Additive source-drift / provenance receipt for the v4.3 execution-dose artifacts.

The historical dataset manifest and preflight receipt bind LF-canonical source
hashes. They were built at the frozen commit named in the preflight receipt
(``git.commit``), and every binding reproduces there. Later, intentional Phase 4
work (commit efa8d9a) changed ``engine/sft_trainer.py``, so the working tree no
longer matches. This receipt records that drift explicitly. It does not edit,
regenerate or re-validate the historical artifacts, and it does not make them
current: a new execution-dose run needs a new source-bound preflight.

CPU only; deterministic apart from the HEAD it was recorded at; refuses to
overwrite an existing receipt with different bytes (write a successor instead).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.historical_source_provenance import (
    DRIFT_SCHEMA_VERSION, changing_commits, current_drift, resolve_commit, verify_at_commit)
from harness.source_identity import HASH_SCHEME_VERSION

OUTPUT = "results/v4_3_execution_dose_source_provenance_v1.json"
MANIFEST = "results/v4_3_execution_dose_dataset_manifest.json"
PREFLIGHT = "results/v4_3_execution_dose_preflight_receipt.json"
HISTORICAL = (MANIFEST, PREFLIGHT)


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def build() -> dict:
    commit = resolve_commit(ROOT, load(PREFLIGHT)["git"]["commit"])
    artifacts, drift = [], {}
    for rel in HISTORICAL:
        bindings = load(rel)["source_files_sha256"]
        report = verify_at_commit(ROOT, commit, bindings)
        if not all(entry["matches"] for entry in report.values()):
            raise SystemExit(f"REFUSED: {rel} does not reproduce at {commit}")
        artifacts.append({"path": rel, "sha256": sha(rel), "bound_sources": len(bindings),
                          "all_bindings_reproduce_at_frozen_commit": True})
        for relative, current in current_drift(ROOT, bindings).items():
            drift.setdefault(relative, {
                "path": relative, "frozen_canonical_sha256": bindings[relative],
                "current_canonical_sha256": current,
                "source_changing_commits": changing_commits(ROOT, commit, relative),
                "bound_by": []})["bound_by"].append(rel)
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "hash_scheme": HASH_SCHEME_VERSION,
        "frozen_source_commit": commit,
        "frozen_source_commit_read_from": f"{PREFLIGHT} :: git.commit",
        "historical_artifacts": artifacts,
        "current_source_drift": sorted(drift.values(), key=lambda entry: entry["path"]),
        "status": {
            "historical": True,
            "valid_for_original_run": True,
            "current_run_ready": False,
            "statement": ("The v4.3 execution-dose artifacts are historical records, valid "
                          "for the run they described: every bound source reproduces "
                          "exactly at the frozen commit. They are NOT ready for a new run, "
                          "because the current source differs; the existing launch guard "
                          "(scripts/preflight_execution_dose_ab.verify_receipt_for_launch) "
                          "refuses them and is not relaxed.")},
        "new_run_requirement": ("Any new execution-dose run requires a new source-bound "
                                "preflight receipt built at the then-current commit. This "
                                "receipt does not authorise reuse of the historical one."),
        "historical_artifacts_modified": False,
    }


def main() -> int:
    receipt = build()
    data = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode("utf-8")
    target = ROOT / OUTPUT
    if target.exists() and target.read_bytes() != data:
        raise SystemExit(f"REFUSED: {OUTPUT} exists with different bytes; write a successor")
    publish_file_atomically(target, data)
    print(json.dumps({"drift": [e["path"] for e in receipt["current_source_drift"]],
                      "commit": receipt["frozen_source_commit"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
