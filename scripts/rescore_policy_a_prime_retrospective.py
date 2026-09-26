"""Retrospective DEVELOPMENT rescore of the failed policy-A pilot under policy A-prime.

The 124 candidates of the policy-A pilot were used to DESIGN policy A-prime, so
this rescore is policy-development evidence only: it is not an independent
pilot and not confirmatory.  The historical policy-A report and store are read,
never modified.

Every candidate is rebuilt and run through the COMPLETE A-prime pipeline
(schema, authentication including the A-prime changed-path rules, temporal
rule, overlap including vendored/generated detection, sidecar revalidation and
licence validation) against the current frozen universe.  Nothing is copied
from the old outcomes.

Evidence sources, in order: the retained content store of the policy-A pilot;
then the retained local bare git clones.  Policy A never needed the blobs of
auxiliary (test/documentation) files, so A-prime may need objects that were not
retained; those are fetched lazily by git (read-only, no REST call) and counted
separately.  Issue/PR responses are ONLY the retained ones: no REST call is made,
and a candidate that would need an unretained response is reported as such.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.acquisition_receipt import ProtectedAccessMonitor
from harness.github_acquisition import (
    AcquisitionFailure, ContentStore, IntegrityViolation, Journal, LocalGitRepository, utc_now,
)
from harness.repository_isolation import ApiResponse, candidate_from_sidecar, git_object_id
from harness.repository_native_candidates import RepositoryContext
from scripts.run_repository_native_acquisition_pilot import (
    acquire_and_evaluate, build_report, load_frozen, publish,
)

SOURCE_STORE = "data/repository_native/acquisition_pilot"
STORE = "data/repository_native/a_prime_retrospective"
REPORT = "results/v4_3_repository_native_a_prime_retrospective.json"
LABEL = ("RETROSPECTIVE DEVELOPMENT RESCORE under policy A-prime of the failed policy-A pilot's "
         "124 candidates, which were used to design A-prime; not an independent pilot and not "
         "confirmatory evidence")


class RetainedFirstSource:
    """Git objects from the retained store, then the retained clone (lazy fetch counted)."""

    def __init__(self, store: ContentStore, clone: LocalGitRepository):
        self.store, self.clone = store, clone
        self.from_store = self.from_clone = self.fetched = 0

    def _local(self, oid: str) -> bool:
        env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
        result = subprocess.run(["git", "--git-dir", str(self.clone.path), "cat-file", "-e", oid],
                                env=env, capture_output=True)
        return result.returncode == 0

    def read(self, oid: str):
        try:
            found = self.store.get_git(oid)
            self.from_store += 1
            return found
        except (FileNotFoundError, IntegrityViolation):
            pass
        local = self._local(oid)
        found = self.clone.read(oid)
        if found is not None:
            if local:
                self.from_clone += 1
            else:
                self.fetched += 1
        return found


class RetainedIssueClient:
    """Serves only issue/PR responses retained in the policy-A sidecars; no REST call."""

    def __init__(self, responses: dict[str, ApiResponse]):
        self.responses = responses
        self.calls = 0
        self.missing: list[str] = []

    def get(self, url: str):
        self.calls += 1
        if url in self.responses:
            return self.responses[url], None
        self.missing.append(url)
        raise AcquisitionFailure("not_found", f"not retained (no REST call made): {url}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    ProtectedAccessMonitor.install()
    since = ProtectedAccessMonitor.mark()
    source_root, store_root = ROOT / SOURCE_STORE, ROOT / STORE
    source_store = ContentStore(source_root / "objects")
    source_entries = Journal(source_root / "journal.jsonl").entries()
    frozen, frozen_identity = load_frozen()
    frozen_identity["reference_universe_receipt_sha256"] = frozen.receipt_sha256
    config = {"label": LABEL, "repositories_file": "docs/repository_native_candidate_repositories.json",
              "store": STORE, "report": REPORT, "source_store": SOURCE_STORE,
              "source_journal_sha256": __import__("hashlib").sha256(
                  (source_root / "journal.jsonl").read_bytes()).hexdigest(),
              "per_repository_cap": 10, "candidates_since": "2025-01-01",
              "mode": "retrospective development rescore; no REST calls"}
    if not args.report_only:
        store_root.mkdir(parents=True, exist_ok=True)
        store = ContentStore(store_root / "objects")
        journal = Journal(store_root / "journal.jsonl")
        started = utc_now()
        # Retained issue/PR responses, keyed by their request URL.
        responses: dict[str, ApiResponse] = {}
        for entry in source_entries:
            if entry.get("addresses"):
                candidate = candidate_from_sidecar(json.loads(
                    source_store.get_raw(entry["addresses"]["sidecar"])))
                responses[candidate.issue_metadata.url] = candidate.issue_metadata
        client = RetainedIssueClient(responses)
        totals = {"from_store": 0, "from_clone": 0, "fetched": 0}
        for entry in source_entries:
            key = entry["key"]
            if key.startswith(("repo:", "scan:")) and not journal.done(key):
                journal.record(key, {k: v for k, v in entry.items()
                                     if k not in ("key", "recorded_utc")})
        # Every raw object a copied journal entry references must live in THIS store.
        for entry in source_entries:
            if entry.get("metadata_address"):
                store.put_raw(source_store.get_raw(entry["metadata_address"]))
        repos = {e["repository"]: e for e in source_entries if e["key"].startswith("repo:")
                 and not e.get("failure") and not e.get("screen_problems")}
        for entry in source_entries:
            key = entry["key"]
            if not key.startswith("cand:") or journal.done(key):
                continue
            repository, oid = key[5:].rsplit("@", 1)
            repo = repos[repository]
            metadata_body = source_store.get_raw(repo["metadata_address"])
            context = RepositoryContext(repository, ApiResponse(
                url=repo["metadata_url"], body=metadata_body, sha256=repo["metadata_sha256"],
                retrieved_utc=repo["retrieved_utc"], etag=repo["etag"]),
                json.loads(metadata_body), repo["renames"])
            clone = LocalGitRepository(source_root / "repos" / (repository.replace("/", "__")
                                                                + ".git"),
                                       f"https://github.com/{repository}.git")
            source = RetainedFirstSource(source_store, clone)
            scan = next(e for e in source_entries if e["key"] == f"scan:{repository}")
            item = next(i for i in scan["selected"] if i["oid"] == oid)
            result = acquire_and_evaluate(repository, context, item, source, client, store,
                                          frozen)
            result["evidence_sources"] = {"from_retained_store": source.from_store,
                                          "from_retained_clone": source.from_clone,
                                          "fetched_by_git": source.fetched}
            for name, value in result["evidence_sources"].items():
                totals[{"from_retained_store": "from_store", "from_retained_clone": "from_clone",
                        "fetched_by_git": "fetched"}[name]] += value
            if result.get("failure") == "not_found" and "not retained" in result.get("detail", ""):
                result["failure_note"] = "issue/PR response not retained; not rescorable offline"
            journal.record(key, result)
            print(f"{key[5:60]}: " + ("ADMITTED" if result.get("outcome", {}).get("admitted")
                                      else result.get("pre_pipeline_exclusion")
                                      or result.get("failure")
                                      or result["outcome"]["reasons"][0]), flush=True)
        journal.record(f"session:{started}:{os.getpid()}", {
            "start_utc": started, "end_utc": utc_now(), "elapsed_seconds": 0.0,
            "api_calls": 0, "api_retries": 0, "api_seconds": 0.0,
            "rate_limit_and_backoff_wait_seconds": 0.0, "authenticated_requests": False,
            "retained_issue_lookups": client.calls, "unretained_issue_urls": client.missing,
            "git_objects": totals,
            "protected_access_evidence": ProtectedAccessMonitor.evidence(since)})
    report = build_report(store_root, config, None, frozen_identity=frozen_identity,
                          label=LABEL)
    sessions = [e for e in Journal(store_root / "journal.jsonl").entries()
                if e["key"].startswith("session:")]
    report["retrospective"] = {
        "development_only": True, "not_confirmatory": True,
        "source_report_sha256": "119fae38e1d7be0c8109e689ce5206ba61568a25316e896fba01f2b5444244ea",
        "source_journal_sha256": config["source_journal_sha256"],
        "rest_calls": 0,
        "git_objects": sessions[-1]["git_objects"] if sessions else None,
        "unretained_issue_urls": sessions[-1].get("unretained_issue_urls") if sessions else None,
        "provisional_estimate_superseded": "the earlier provisional 32/124 is replaced by this "
                                           "complete A-prime rescore"}
    publish(report, ROOT / REPORT)
    print(json.dumps({"counts": report["counts"], "yield": report["yield"],
                      "admitted_per_repository": report["admitted_per_repository"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
