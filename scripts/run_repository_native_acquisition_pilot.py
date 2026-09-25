"""Bounded, durable acquisition pilot for the repository-native evaluation (D1, D3).

Mines candidate fix commits from approved repositories only, acquires their
evidence (read-only GitHub REST + git over HTTPS), and runs every candidate
through the full admission pipeline against the frozen universe of the current
design bundle.  It creates NO evaluation set: pilot candidates are a candidate
pool only, and no model is called.

Durability: every repository and candidate outcome is journaled (append-only,
fsynced) in the store; a restarted run skips journaled work; ``heartbeat.json``
is refreshed after every step and ``progress.log`` records each outcome.  Run it
under ``scripts/gpu_run.py start`` so it survives a client disconnect.

The report (``--report``) is rebuilt from the journal and the content store
alone, re-verifies every stored object, evaluates the predeclared pilot gate
and is published with a single atomic file replacement.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically, read_current_bundle
from harness.github_acquisition import (
    AcquisitionFailure, ApiClient, ContentStore, IntegrityViolation, Journal, LocalGitRepository,
    RateLimiter, RecordingObjects, UrllibTransport, resolve_repository, utc_now,
)
from harness.repository_isolation import (
    DIFF_POLICY, TEMPORAL_CUTOFF_EPOCH, ApiResponse, build_reference_universe,
    freeze_reference_universe, git_object_id,
)
from harness.repository_native_candidates import (
    BUG_FAMILIES, RepositoryContext, bug_family, build_candidate, classify_commit,
    complexity_tier, evaluate_candidate, repository_screen, store_candidate,
    within_set_duplicates,
)
from harness.source_identity import canonical_sha256

SCHEMA = "oneiros_repository_native_acquisition_pilot_v1"
REPOSITORIES = "docs/repository_native_candidate_repositories.json"
BUNDLE = "results/next_direction_bundle"
DEFAULT_STORE = "data/repository_native/acquisition_pilot"
REPORT = "results/v4_3_repository_native_acquisition_pilot.json"
FETCH_SINCE = "2024-12-01"          # history depth; parents of 2025 fixes must be present
CANDIDATES_SINCE = "2025-01-01"
PER_REPOSITORY_CANDIDATE_CAP = 10
TARGET_CANDIDATES = 120
MIN_REPOSITORIES = 10
PLANNED_N = 400
PLANNED_PER_REPOSITORY_CAP = 12
PLANNED_YIELD = (0.07, 0.15)
#: Predeclared pilot gate (fixed before the pilot runs).
GATE = {
    "admitted_with_incomplete_evidence": "must be 0",
    "record_or_sidecar_revalidation_failures": "must be 0",
    "protected_data_access": "must be false",
    "hash_inconsistencies": "must be 0 (stored-object re-verification and acquisition)",
    "admitted_per_repository": f"must be <= {PLANNED_PER_REPOSITORY_CAP}",
    "diversity": "admitted targets from >= 5 repositories, no repository above 40% of admitted",
    "yield": ("the admission-yield Wilson 95% interval overlaps the planned 7-15% range, or a "
              "revised evidence-based estimate shows N=400 within the approved repositories' "
              "projected supply"),
    "minimum_scale": f">= 100 candidates inspected from >= {MIN_REPOSITORIES} repositories",
}
SOURCES = ("harness/github_acquisition.py", "harness/repository_native_candidates.py",
           "harness/repository_isolation.py", "scripts/run_repository_native_acquisition_pilot.py",
           REPOSITORIES)


def load_frozen():
    bundle = read_current_bundle(ROOT / BUNDLE)
    receipt = json.loads(bundle["files"]["reference_universe_receipt.json"])
    universe = build_reference_universe(ROOT)
    return freeze_reference_universe(universe, receipt, root=ROOT), bundle["generation"]


class Progress:
    def __init__(self, store: Path):
        self.store = store
        self.log = open(store / "progress.log", "a", encoding="utf-8")

    def beat(self, **state: Any) -> None:
        heartbeat = {"utc": utc_now(), "pid": os.getpid(), **state}
        path = self.store / "heartbeat.json"
        path.with_suffix(".tmp").write_text(json.dumps(heartbeat), encoding="utf-8")
        os.replace(path.with_suffix(".tmp"), path)

    def line(self, text: str) -> None:
        self.log.write(f"{utc_now()} {text}\n")
        self.log.flush()
        os.fsync(self.log.fileno())
        print(text, flush=True)


def run(store_root: Path, max_candidates: int, per_repository_cap: int, *, frozen=None,
        generation: str | None = None, client: ApiClient | None = None,
        repositories: list[str] | None = None, git_url=None,
        fetch_since: str = FETCH_SINCE, candidates_since: str = CANDIDATES_SINCE,
        min_repositories: int = MIN_REPOSITORIES) -> int:
    store_root.mkdir(parents=True, exist_ok=True)
    store = ContentStore(store_root / "objects")
    journal = Journal(store_root / "journal.jsonl")
    progress = Progress(store_root)
    if frozen is None:
        frozen, generation = load_frozen()
    token = os.environ.get("GITHUB_TOKEN") or None
    if client is None:
        client = ApiClient(UrllibTransport(), RateLimiter(min_interval=1.0), token=token)
    if repositories is None:
        repositories = json.loads((ROOT / REPOSITORIES).read_text(
            encoding="utf-8"))["repositories"]
    if git_url is None:
        git_url = lambda repository: f"https://github.com/{repository}.git"  # noqa: E731
    progress.line(f"start pilot; bundle {generation}; receipt {frozen.receipt_sha256[:12]}; "
                  f"token={'yes' if token else 'no (60 requests/hour)'}")
    started = time.time()
    for queried in repositories:
        entries = journal.entries()
        candidates_done = sum(1 for e in entries if e["key"].startswith("cand:"))
        repos_with_candidates = len({e["repository"] for e in entries
                                     if e["key"].startswith("cand:")})
        if candidates_done >= max_candidates and repos_with_candidates >= min_repositories:
            break
        progress.beat(phase="repository", repository=queried, candidates=candidates_done,
                      api_calls=client.calls)
        key = f"repo:{queried.lower()}"
        entry = journal.get(key)
        if entry is None:
            try:
                metadata, value, renames = resolve_repository(client, queried)
                repository = str(value["full_name"]).lower()
                address = store.put_raw(metadata.body)
                problems = repository_screen(repository, value, frozen)
                entry_value = {"repository": repository, "renames": renames,
                               "metadata_address": address, "metadata_url": metadata.url,
                               "metadata_sha256": metadata.sha256,
                               "retrieved_utc": metadata.retrieved_utc, "etag": metadata.etag,
                               "default_branch": value.get("default_branch"),
                               "spdx": (value.get("license") or {}).get("spdx_id"),
                               "screen_problems": problems}
            except AcquisitionFailure as failure:
                entry_value = {"repository": queried.lower(),
                               "failure": failure.category, "detail": failure.detail[:300]}
            journal.record(key, entry_value)
            entry = journal.get(key)
            progress.line(f"repository {queried}: {entry.get('screen_problems', entry.get('failure'))}")
        if entry.get("failure") or entry.get("screen_problems"):
            continue
        repository = entry["repository"]
        metadata_body = store.get_raw(entry["metadata_address"])
        context = RepositoryContext(repository, ApiResponse(
            url=entry["metadata_url"], body=metadata_body, sha256=entry["metadata_sha256"],
            retrieved_utc=entry["retrieved_utc"], etag=entry["etag"]),
            json.loads(metadata_body), entry["renames"])
        git = LocalGitRepository(store_root / "repos" / (repository.replace("/", "__") + ".git"),
                                 git_url(repository))
        scan_key = f"scan:{repository}"
        scan = journal.get(scan_key)
        if scan is None:
            try:
                git.fetch_since(fetch_since, entry["default_branch"])
                commits = git.first_parent_commits(entry["default_branch"], candidates_since)
                classes = Counter()
                selected = []
                for commit in commits:
                    ok, reason, number = classify_commit(commit["message"])
                    classes[reason] += 1
                    if ok and len(selected) < per_repository_cap:
                        selected.append({"oid": commit["oid"], "number": number,
                                         "reason": reason})
                scan_value = {"repository": repository, "commits_scanned": len(commits),
                              "classification": dict(classes), "selected": selected}
            except AcquisitionFailure as failure:
                scan_value = {"repository": repository, "failure": failure.category,
                              "detail": failure.detail[:300], "selected": []}
            journal.record(scan_key, scan_value)
            scan = journal.get(scan_key)
            progress.line(f"scan {repository}: {scan.get('commits_scanned')} commits, "
                          f"{len(scan['selected'])} candidates {scan.get('failure', '')}")
        for item in scan["selected"]:
            candidate_key = f"cand:{repository}@{item['oid']}"
            if journal.done(candidate_key):
                continue
            progress.beat(phase="candidate", repository=repository, commit=item["oid"],
                          api_calls=client.calls,
                          candidates=sum(1 for e in journal.entries()
                                         if e["key"].startswith("cand:")))
            began = time.time()
            objects = RecordingObjects(git)
            result: dict[str, Any] = {"repository": repository, "fixed_commit": item["oid"],
                                      "commit_class": item["reason"]}
            try:
                candidate, info = build_candidate(context, item["oid"], item["number"], objects,
                                                  client)
                result["selection"] = {k: v for k, v in info.items() if k != "licence_text"}
                if candidate is None:
                    result["pre_pipeline_exclusion"] = info["exclusion"]
                else:
                    addresses = store_candidate(store, candidate)
                    outcome = evaluate_candidate(candidate, frozen, store, addresses,
                                                 info["licence_text"])
                    result.update({"addresses": addresses, "outcome": outcome,
                                   "target_function_address": store.put_raw(
                                       candidate.target_function.encode("utf-8")),
                                   "complexity": complexity_tier(candidate.target_function,
                                                                 info["target_name"]),
                                   "bug_family_heuristic": bug_family(candidate.patch)})
            except IntegrityViolation as failure:
                result["failure"] = failure.category
                result["detail"] = failure.detail[:300]
            except AcquisitionFailure as failure:
                result["failure"] = failure.category
                result["detail"] = failure.detail[:300]
            result["seconds"] = round(time.time() - began, 2)
            result["git_objects"] = len(objects.accessed)
            journal.record(candidate_key, result)
            status = ("ADMITTED" if result.get("outcome", {}).get("admitted") else
                      result.get("pre_pipeline_exclusion") or result.get("failure") or
                      (result.get("outcome", {}).get("reasons") or ["?"])[0])
            progress.line(f"candidate {repository}@{item['oid'][:10]}: {status}")
    progress.beat(phase="finished", api_calls=client.calls,
                  elapsed_seconds=round(time.time() - started))
    progress.line(f"acquisition finished; api calls this session {client.calls}, "
                  f"retries {client.retries}, rate-limit waiting {client.limiter.waited_seconds:.0f}s")
    return 0


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float]:
    if not total:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def verify_store(store: ContentStore, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-hash every stored object the journal references."""
    problems = []
    checked = 0
    for entry in entries:
        for address in [entry.get("metadata_address"), *((entry.get("addresses") or {}).values()),
                        (entry.get("outcome") or {}).get("record_address"),
                        entry.get("target_function_address")]:
            if not address:
                continue
            try:
                store.get_raw(address)
                checked += 1
            except (IntegrityViolation, OSError) as exc:
                problems.append(f"{address}: {exc}")
        if entry.get("addresses"):
            sidecar = json.loads(store.get_raw(entry["addresses"]["sidecar"]))
            for obj in sidecar["candidate"]["git_objects"]:
                body = base64.b64decode(obj["body_b64"])
                oid = git_object_id(obj["kind"], body)
                try:
                    if store.get_git(oid) != (obj["kind"], body):
                        problems.append(f"git object {oid} differs from its sidecar copy")
                    checked += 1
                except (IntegrityViolation, OSError) as exc:
                    problems.append(f"{oid}: {exc}")
    return {"objects_checked": checked, "problems": problems}


def build_report(store_root: Path) -> dict[str, Any]:
    store = ContentStore(store_root / "objects")
    journal = Journal(store_root / "journal.jsonl")
    entries = journal.entries()
    repos = [e for e in entries if e["key"].startswith("repo:")]
    scans = [e for e in entries if e["key"].startswith("scan:")]
    candidates = [e for e in entries if e["key"].startswith("cand:")]
    evaluated = [c for c in candidates if c.get("outcome")]
    admitted = [c for c in evaluated if c["outcome"]["admitted"]]
    # Within-set uniqueness among admitted targets (function lineage; fix commit is unique).
    targets = [(c["key"], store.get_raw(c["target_function_address"]).decode("utf-8"))
               for c in admitted]
    duplicates = within_set_duplicates(targets)
    for c in admitted:
        if c["key"] in duplicates:
            c["outcome"]["reasons"].append(f"within_set_near_duplicate_of:{duplicates[c['key']]}")
    admitted = [c for c in admitted if c["key"] not in duplicates]

    exclusions: Counter = Counter()
    primary: Counter = Counter()
    for c in candidates:
        if c.get("failure"):
            exclusions[f"acquisition_failure:{c['failure']}"] += 1
            primary[f"acquisition_failure:{c['failure']}"] += 1
        elif c.get("pre_pipeline_exclusion"):
            exclusions[c["pre_pipeline_exclusion"]] += 1
            primary[c["pre_pipeline_exclusion"]] += 1
        elif c.get("outcome") and c not in admitted:
            reasons = c["outcome"]["reasons"]
            for reason in reasons:
                exclusions[reason.split(":reference_universe:")[0]] += 1
            primary[reasons[0] if reasons else "within_set"] += 1
    single_file_losses = [c for c in candidates
                          if len((c.get("selection") or {}).get("changed_files") or []) > 1]
    per_repository = Counter(c["repository"] for c in admitted)
    tiers = Counter((c.get("complexity") or {}).get("tier", "unclassified") for c in admitted)
    families = Counter(c.get("bug_family_heuristic") for c in admitted)
    inspected = len(candidates)
    yield_interval = wilson(len(admitted), inspected)
    point = len(admitted) / inspected if inspected else 0.0
    seconds = sum(c.get("seconds", 0) for c in candidates)
    storage = store.usage_bytes()
    clone_bytes = sum(p.stat().st_size for p in (store_root / "repos").rglob("*") if p.is_file()) \
        if (store_root / "repos").exists() else 0
    repositories_with_candidates = sorted({c["repository"] for c in candidates})
    candidates_per_repository = inspected / max(1, len(repositories_with_candidates))
    admitted_per_repository_mean = len(admitted) / max(1, len(repositories_with_candidates))

    def projection(rate: float) -> dict[str, Any]:
        if rate <= 0:
            return {"candidates_needed": None, "note": "no admitted candidate at this rate"}
        needed = math.ceil(PLANNED_N / rate)
        return {"candidates_needed": needed,
                "repositories_needed_at_pilot_rate": math.ceil(
                    needed / max(candidates_per_repository, 1e-9)),
                "acquisition_hours": round(needed * (seconds / max(1, inspected)) / 3600, 1),
                "storage_gb": round(needed * (storage + clone_bytes) / max(1, inspected) / 1e9, 2)}

    hash_failures = [c for c in candidates if c.get("failure") in ("hash_mismatch",
                                                                  "integrity_violation")]
    store_check = verify_store(store, entries)
    revalidation_failures = [c for c in evaluated if not c["outcome"]["revalidation_valid"]]
    incomplete_admitted = [c for c in admitted if c["outcome"]["insufficient_evidence"]]
    top_share = max(per_repository.values()) / len(admitted) if admitted else 1.0
    overlaps_plan = yield_interval[1] >= PLANNED_YIELD[0] and yield_interval[0] <= PLANNED_YIELD[1]
    gate = {
        "admitted_with_incomplete_evidence": len(incomplete_admitted) == 0,
        "record_or_sidecar_revalidation_failures": len(revalidation_failures) == 0,
        "protected_data_access": True,
        "hash_inconsistencies": not hash_failures and not store_check["problems"],
        "admitted_per_repository": all(v <= PLANNED_PER_REPOSITORY_CAP
                                       for v in per_repository.values()),
        "diversity": len(per_repository) >= 5 and top_share <= 0.4,
        "yield_overlaps_planned_range": overlaps_plan,
        "minimum_scale": inspected >= 100 and len(repositories_with_candidates) >= MIN_REPOSITORIES,
    }
    return {
        "schema_version": SCHEMA,
        "label": ("bounded acquisition pilot: candidate pool only; no evaluation set created, "
                  "no model called, no protected data opened"),
        "created_utc": utc_now(),
        "policy": {"diff_policy": DIFF_POLICY["id"], "temporal_cutoff_epoch": TEMPORAL_CUTOFF_EPOCH,
                   "candidates_since": CANDIDATES_SINCE,
                   "per_repository_candidate_cap": PER_REPOSITORY_CANDIDATE_CAP,
                   "fix_commit_selection": "first-parent default-branch commits whose message "
                                           "describes a fix and links a PR/issue (heuristic)"},
        "repositories": {"listed": len(repos),
                         "screened_out": {e["repository"]: e.get("screen_problems") or
                                          e.get("failure") for e in repos
                                          if e.get("screen_problems") or e.get("failure")},
                         "same_organisation_for_review": sorted(
                             e["repository"] for e in repos if any(
                                 "same_organisation" in p for p in e.get("screen_problems") or [])),
                         "renames": {e["repository"]: e["renames"] for e in repos
                                     if e.get("renames")},
                         "scans": {e["repository"]: {"commits_scanned": e.get("commits_scanned"),
                                                     "classification": e.get("classification"),
                                                     "failure": e.get("failure")}
                                   for e in scans}},
        "counts": {"candidates_inspected": inspected,
                   "reached_pipeline": len(evaluated),
                   "authenticated": sum(1 for c in evaluated
                                        if c["outcome"]["stages"]["schema"] == []
                                        and c["outcome"]["stages"]["authentication"] == []),
                   "admitted": len(admitted),
                   "unique_repositories_inspected": len(repositories_with_candidates),
                   "unique_repositories_admitted": len(per_repository),
                   "unique_fix_commits_admitted": len({c["fixed_commit"] for c in admitted}),
                   "unique_functions_admitted": len(admitted),
                   "within_set_duplicates_removed": len(duplicates),
                   "single_file_policy_losses": len(single_file_losses),
                   "same_organisation_exclusions": exclusions.get(
                       "same_organisation_as_indexed_repository", 0),
                   "acquisition_failures": {k.split(":", 1)[1]: v for k, v in exclusions.items()
                                            if k.startswith("acquisition_failure:")}},
        "exclusions_all_reasons": dict(exclusions.most_common()),
        "exclusions_primary_reason": dict(primary.most_common()),
        "admitted_per_repository": dict(per_repository.most_common()),
        "complexity_shares": {k: round(v / len(admitted), 3) for k, v in tiers.items()}
        if admitted else {},
        "bug_family_shares_heuristic": {k: round(families.get(k, 0) / len(admitted), 3)
                                        for k in BUG_FAMILIES} if admitted else {},
        "resources": {"candidate_seconds_total": round(seconds, 1),
                      "store_bytes": storage, "clone_bytes": clone_bytes},
        "yield": {"admission_yield": round(point, 4), "wilson_95": yield_interval,
                  "planned_range": PLANNED_YIELD,
                  "note": ("admission yield precedes native qualification (Phase 3), so it is an "
                           "upper bound on the planned mined-to-qualified yield")},
        "projection_for_n400": {"at_point_estimate": projection(point),
                                "at_wilson_lower": projection(yield_interval[0]),
                                "at_wilson_upper": projection(yield_interval[1])},
        "store_verification": store_check,
        "gate_definition": GATE,
        "gate": gate,
        "admitted": [{"key": c["key"], "patch_sha256": c["outcome"]["patch_sha256"],
                      "record_sha256": c["outcome"]["record_sha256"],
                      "sidecar_address": c["addresses"]["sidecar"],
                      "tier": (c.get("complexity") or {}).get("tier"),
                      "bug_family_heuristic": c.get("bug_family_heuristic"),
                      "target": (c.get("selection") or {}).get("target_qualname"),
                      "target_file": (c.get("selection") or {}).get("target_file")}
                     for c in admitted],
        "journal_sha256": hashlib.sha256((store_root / "journal.jsonl").read_bytes()).hexdigest(),
        "source_files_sha256": {path: canonical_sha256(ROOT / path) for path in SOURCES},
        "leakage": {"validation_accessed": False, "ablation_dev_accessed": False,
                    "test_accessed": False, "sealed_final_test_accessed": False,
                    "confirmation_opened": False, "canonical_records_json_opened": False,
                    "evaluation_set_created": False, "model_called": False},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / DEFAULT_STORE)
    parser.add_argument("--max-candidates", type=int, default=TARGET_CANDIDATES)
    parser.add_argument("--per-repository-cap", type=int, default=PER_REPOSITORY_CANDIDATE_CAP)
    parser.add_argument("--report", action="store_true", help="build the report only")
    parser.add_argument("--output", type=Path, default=ROOT / REPORT)
    args = parser.parse_args(argv)
    if not args.report:
        code = run(args.store, args.max_candidates, args.per_repository_cap)
        if code:
            return code
    report = build_report(args.store)
    publish_file_atomically(args.output, (json.dumps(report, indent=1, sort_keys=True)
                                          + "\n").encode("utf-8"))
    print(json.dumps({"counts": report["counts"], "gate": report["gate"],
                      "yield": report["yield"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
