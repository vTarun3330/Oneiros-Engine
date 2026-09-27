"""Offline revalidation of the frozen 127-candidate retry sample under the
current receipt and policy (Step 6), compared candidate by candidate with the
refused run.

Evidence comes ONLY from the refused run's retained content store and local
clones (git lazy fetching disabled) and its retained issue/PR and repository
responses; no REST call and no network git.  The audit scope covers the whole
process.  The result decides whether the correction was outcome-neutral on the
frozen sample: any substantive change means the refused sample cannot confirm
the corrected policy.  This artifact is diagnostic, not a confirmation pilot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.acquisition_receipt import ProtectedAccessMonitor
from harness.github_acquisition import (
    AcquisitionFailure, ContentStore, IntegrityViolation, Journal, LocalGitRepository, utc_now,
)
from harness.repository_isolation import ApiResponse, candidate_from_sidecar, git_object_id

CONFIG = {
    "label": ("OFFLINE REVALIDATION (diagnostic) of the frozen 127-candidate retry sample under "
              "isolation v6, compared with the refused run; not a confirmation pilot"),
    "repositories_file": "docs/repository_native_confirmation_repositories.json",
    "store": "data/repository_native/aprime_sample_revalidation_v6",
    "report": "results/v4_3_repository_native_aprime_sample_revalidation_v6.json",
    "candidate_manifest": "results/v4_3_repository_native_aprime_retry_candidate_manifest.json",
    "reference_journal": "data/repository_native/aprime_confirmation_pilot/journal.jsonl",
    "pool_files": ["docs/repository_native_candidate_repositories.json",
                   "docs/repository_native_confirmation_repositories.json"],
    "mode": "offline revalidation from retained evidence; no REST, no network git"}
SOURCE = "data/repository_native/aprime_confirmation_pilot"


class OfflineSource:
    """Retained store first, then the retained clone with lazy fetching DISABLED."""

    def __init__(self, store: ContentStore, clone: Path):
        self.store, self.clone = store, clone
        self.missing: list[str] = []

    def read(self, oid: str):
        try:
            return self.store.get_git(oid)
        except (FileNotFoundError, IntegrityViolation):
            pass
        env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
        kind = subprocess.run(["git", "--git-dir", str(self.clone), "cat-file", "-t", oid],
                              env=env, capture_output=True)
        if kind.returncode != 0:
            self.missing.append(oid)
            return None
        kind_name = kind.stdout.decode().strip()
        body = subprocess.run(["git", "--git-dir", str(self.clone), "cat-file", kind_name, oid],
                              env=env, capture_output=True).stdout
        if git_object_id(kind_name, body) != oid:
            raise AcquisitionFailure("hash_mismatch", oid)
        return kind_name, body


class RetainedClient:
    """Serves only retained API responses; any other request is refused (no REST)."""

    def __init__(self, responses: dict[str, ApiResponse]):
        self.responses, self.calls, self.missing = responses, 0, []

    def get(self, url: str):
        self.calls += 1
        if url in self.responses:
            return self.responses[url], None
        self.missing.append(url)
        raise AcquisitionFailure("not_found", f"not retained (no REST call made): {url}")


def main() -> int:
    ProtectedAccessMonitor.install(ROOT)
    scope = ProtectedAccessMonitor.scope()
    scope.session = f"{scope.start_utc}:{os.getpid()}"
    from scripts import run_repository_native_acquisition_pilot as pilot
    manifest = json.loads((ROOT / CONFIG["candidate_manifest"]).read_text(encoding="utf-8"))
    store_root = ROOT / CONFIG["store"]
    store_root.mkdir(parents=True, exist_ok=True)
    Journal(store_root / "journal.jsonl").record(f"audit_start:{scope.session}",
                                                 {"scope_start_utc": scope.start_utc})
    frozen, identity = pilot.load_frozen()
    identity["reference_universe_receipt_sha256"] = frozen.receipt_sha256
    source_root = ROOT / SOURCE
    source_store = ContentStore(source_root / "objects")
    source_entries = Journal(source_root / "journal.jsonl").entries()
    store = ContentStore(store_root / "objects")
    journal = pilot.AuditedJournal(Journal(store_root / "journal.jsonl"), scope)
    responses: dict[str, ApiResponse] = {}
    for entry in source_entries:
        if entry.get("addresses"):
            candidate = candidate_from_sidecar(json.loads(source_store.get_raw(
                entry["addresses"]["sidecar"])))
            responses[candidate.issue_metadata.url] = candidate.issue_metadata
    client = RetainedClient(responses)
    repos = {e["repository"]: e for e in source_entries if e["key"].startswith("repo:")}
    for entry in source_entries:
        if entry["key"].startswith(("repo:", "scan:")) and not journal.done(entry["key"]):
            if entry.get("metadata_address"):
                store.put_raw(source_store.get_raw(entry["metadata_address"]))
            journal.record(entry["key"], {k: v for k, v in entry.items()
                                          if k not in ("key", "recorded_utc", "audit_snapshot")})
    started = utc_now()
    began = time.time()
    missing_objects: dict[str, list[str]] = {}
    for item in manifest["candidates"]:
        repository, oid = item["repository"], item["fixed_commit"]
        key = f"cand:{repository}@{oid}"
        if journal.done(key):
            continue
        repo = repos[repository]
        context = pilot._context(source_store, repo)
        clone = source_root / "repos" / (repository.replace("/", "__") + ".git")
        source = OfflineSource(source_store, clone)
        result = pilot.acquire_and_evaluate(
            repository, context, {"oid": oid, "number": item["linked_number"],
                                  "reason": item["commit_class"]}, source, client, store, frozen)
        if source.missing:
            missing_objects[key] = source.missing
            result["offline_missing_objects"] = source.missing
        journal.record(key, result)
    journal.record(f"session:{started}:{os.getpid()}", {
        "start_utc": started, "end_utc": utc_now(),
        "elapsed_seconds": round(time.time() - began, 1), "api_calls": 0, "api_retries": 0,
        "api_seconds": 0.0, "rate_limit_and_backoff_wait_seconds": 0.0,
        "authenticated_requests": False, "git_network": False,
        "retained_response_lookups": client.calls, "unretained_urls": client.missing})
    report = pilot.build_report(store_root, CONFIG, None, frozen_identity=identity,
                                label=CONFIG["label"])
    report["offline"] = {"rest_calls": 0, "network_git": False,
                         "unretained_urls": client.missing,
                         "candidates_with_missing_objects": missing_objects}
    pilot.close_audit(scope, store_root)
    report = pilot.finalise_report(report, store_root)
    pilot.publish(report, ROOT / CONFIG["report"])
    comparison = report["candidate_comparison_with_reference"]
    print(json.dumps({"counts": report["counts"],
                      "compared": comparison["candidates_compared"],
                      "substantive_outcome_changes": comparison["substantive_outcome_changes"],
                      "classification_changes": comparison["classification_changes"],
                      "gate": report["gate"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
