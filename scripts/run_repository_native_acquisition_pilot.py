"""Durable acquisition pilots for the repository-native evaluation (D1, D3).

Mines candidate fix commits from a FROZEN repository list, acquires their
evidence (read-only GitHub REST + git over HTTPS) and runs every candidate
through the full admission pipeline against the frozen universe of the current
design bundle.  It creates NO evaluation set and calls no model.

Durability: every repository, scan, candidate and session is journaled
(append-only, fsynced) in the store; a restarted run skips journaled work;
``heartbeat.json`` is refreshed after every step and ``progress.log`` records
each outcome.  Run it under ``scripts/gpu_run.py start``.

Reports use the acquisition receipt schema (``harness.acquisition_receipt``):
they are rebuilt from the journal and content store alone, re-verify every
stored object, evaluate a gate FROZEN in a hash-bound file before the pilot
runs, and are refused publication if any identity is missing or any event and
gate contradict.  The failed policy-A pilot report
(results/v4_3_repository_native_acquisition_pilot.json) predates this schema and
is left untouched.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.acquisition_receipt import (
    SCHEMA as RECEIPT_SCHEMA, ProtectedAccessMonitor, sha256_file, source_tree_identity,
    tool_sources, validate_receipt,
)
from harness.atomic_publish import publish_file_atomically, read_current_bundle
from harness.github_acquisition import (
    AcquisitionFailure, ApiClient, ContentStore, IntegrityViolation, Journal, LocalGitRepository,
    RateLimiter, RecordingObjects, UrllibTransport, resolve_repository, utc_now,
)
from harness.repository_isolation import (
    DIFF_POLICY, ISOLATION_VERSION, TEMPORAL_CUTOFF_EPOCH, ApiResponse, build_reference_universe,
    freeze_reference_universe, git_object_id,
)
from harness.repository_native_candidates import (
    BUG_FAMILIES, RepositoryContext, bug_family, build_candidate, classify_commit,
    complexity_tier, evaluate_candidate, repository_screen, store_candidate,
    within_set_duplicates,
)

REPORT_SCHEMA = "oneiros_repository_native_acquisition_report_v2"
BUNDLE = "results/next_direction_bundle"
FETCH_SINCE = "2024-12-01"          # history depth; parents of 2025 fixes must be present
CANDIDATES_SINCE = "2025-01-01"
PLANNED_N = 400
PLANNED_PER_REPOSITORY_CAP = 12
PLANNED_MIN_REPOSITORIES = 34
TOOL_SOURCES = ("harness/github_acquisition.py", "harness/repository_native_candidates.py",
                "harness/repository_isolation.py", "harness/acquisition_receipt.py",
                "harness/vendored_code_exclusions.json",
                "scripts/run_repository_native_acquisition_pilot.py")


def load_frozen():
    bundle = read_current_bundle(ROOT / BUNDLE)
    receipt = json.loads(bundle["files"]["reference_universe_receipt.json"])
    universe = build_reference_universe(ROOT)
    frozen = freeze_reference_universe(universe, receipt, root=ROOT)
    return frozen, {"bundle_generation": bundle["generation"],
                    "bundle_manifest_sha256": bundle["manifest_sha256"]}


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


def prior_commits(journals: list[Path]) -> set[str]:
    found: set[str] = set()
    for path in journals:
        for entry in Journal(path).entries():
            if entry["key"].startswith("cand:"):
                found.add(entry["key"].rsplit("@", 1)[1])
    return found


def load_list(path: Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["repositories"]


def run(store_root: Path, config: dict[str, Any], *, frozen=None,
        client: ApiClient | None = None, git_url=None) -> int:
    ProtectedAccessMonitor.install()
    since = ProtectedAccessMonitor.mark()
    store_root.mkdir(parents=True, exist_ok=True)
    store = ContentStore(store_root / "objects")
    journal = Journal(store_root / "journal.jsonl")
    progress = Progress(store_root)
    if frozen is None:
        frozen, _ = load_frozen()
    if client is None:
        token = os.environ.get("GITHUB_TOKEN") or None   # used, never printed or stored
        client = ApiClient(UrllibTransport(), RateLimiter(min_interval=1.0), token=token)
    if git_url is None:
        git_url = lambda repository: f"https://github.com/{repository}.git"  # noqa: E731
    repositories = load_list(ROOT / config["repositories_file"])
    used = {name.lower() for path in config.get("exclude_repository_files", [])
            for name in load_list(ROOT / path)}
    overlap = sorted(name for name in repositories if name.lower() in used)
    if overlap:
        raise SystemExit(f"repository list reuses previously used repositories: {overlap}")
    excluded_commits = prior_commits([ROOT / path for path in config.get("exclude_journals", [])])
    session_start = utc_now()
    started = time.time()
    progress.line(f"start {config['label']}; receipt {frozen.receipt_sha256[:12]}; "
                  f"policy {DIFF_POLICY['id']}; token={'yes' if client.token else 'no'}")
    for queried in repositories:
        entries = journal.entries()
        candidates_done = sum(1 for e in entries if e["key"].startswith("cand:"))
        repos_with_candidates = len({e["repository"] for e in entries
                                     if e["key"].startswith("cand:")})
        if candidates_done >= config["max_candidates"] and \
                repos_with_candidates >= config["min_repositories"]:
            break
        progress.beat(phase="repository", repository=queried, candidates=candidates_done,
                      api_calls=client.calls)
        key = f"repo:{queried.lower()}"
        entry = journal.get(key)
        if entry is None:
            try:
                metadata, value, renames = resolve_repository(client, queried)
                repository = str(value["full_name"]).lower()
                problems = repository_screen(repository, value, frozen)
                if repository in used and repository != queried.lower():
                    problems.append("renamed_to_a_previously_used_repository")
                entry_value = {"repository": repository, "renames": renames,
                               "metadata_address": store.put_raw(metadata.body),
                               "metadata_url": metadata.url, "metadata_sha256": metadata.sha256,
                               "retrieved_utc": metadata.retrieved_utc, "etag": metadata.etag,
                               "default_branch": value.get("default_branch"),
                               "spdx": (value.get("license") or {}).get("spdx_id"),
                               "screen_problems": problems}
            except AcquisitionFailure as failure:
                entry_value = {"repository": queried.lower(), "failure": failure.category,
                               "detail": failure.detail[:300]}
            journal.record(key, entry_value)
            entry = journal.get(key)
            progress.line(f"repository {queried}: "
                          f"{entry.get('screen_problems', entry.get('failure'))}")
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
                git.fetch_since(config["fetch_since"], entry["default_branch"])
                commits = git.first_parent_commits(entry["default_branch"],
                                                   config["candidates_since"])
                classes: Counter = Counter()
                selected, skipped_prior = [], 0
                for commit in commits:
                    ok, reason, number = classify_commit(commit["message"])
                    classes[reason] += 1
                    if not ok:
                        continue
                    if commit["oid"] in excluded_commits:
                        skipped_prior += 1
                        continue
                    if len(selected) < config["per_repository_cap"]:
                        selected.append({"oid": commit["oid"], "number": number,
                                         "reason": reason})
                scan_value = {"repository": repository, "commits_scanned": len(commits),
                              "classification": dict(classes), "selected": selected,
                              "skipped_prior_commits": skipped_prior,
                              "fix_candidate_supply": classes["fix_candidate"]
                              + classes["fix_merge_commit"]}
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
            result = acquire_and_evaluate(repository, context, item, git, client, store,
                                          frozen)
            journal.record(candidate_key, result)
            progress.line(f"candidate {repository}@{item['oid'][:10]}: {status_of(result)}")
    session = {"start_utc": session_start, "end_utc": utc_now(),
               "elapsed_seconds": round(time.time() - started, 1), "api_calls": client.calls,
               "api_retries": client.retries, "api_seconds": round(client.api_seconds, 1),
               "rate_limit_and_backoff_wait_seconds": round(client.limiter.waited_seconds, 1),
               "authenticated_requests": bool(client.token), "git_network": True,
               "protected_access_evidence": ProtectedAccessMonitor.evidence(since)}
    journal.record(f"session:{session_start}:{os.getpid()}", session)
    progress.beat(phase="finished", api_calls=client.calls)
    progress.line(f"acquisition finished; api calls {client.calls}, retries {client.retries}, "
                  f"api {client.api_seconds:.0f}s, waiting {client.limiter.waited_seconds:.0f}s")
    return 0


def status_of(result: dict[str, Any]) -> str:
    return ("ADMITTED" if result.get("outcome", {}).get("admitted") else
            result.get("pre_pipeline_exclusion") or result.get("failure") or
            (result.get("outcome", {}).get("reasons") or ["?"])[0])


def acquire_and_evaluate(repository, context, item, source, client, store, frozen
                         ) -> dict[str, Any]:
    began = time.time()
    objects = RecordingObjects(source)
    result: dict[str, Any] = {"repository": repository, "fixed_commit": item["oid"],
                              "commit_class": item["reason"]}
    try:
        candidate, info = build_candidate(context, item["oid"], item["number"], objects, client)
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
    except AcquisitionFailure as failure:
        result["failure"] = failure.category
        result["detail"] = failure.detail[:300]
    result["seconds"] = round(time.time() - began, 2)
    result["git_objects"] = len(objects.accessed)
    return result


# --- reports -------------------------------------------------------------------------

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


def _reason_family(reason: str) -> str:
    reason = reason.replace("insufficient_isolation_evidence:authentication:", "auth:")
    for prefix in ("auth:changed_production_source_outside_target",
                   "auth:changed_runtime_or_config_outside_target",
                   "auth:auxiliary_evidence_missing", "auth:auxiliary_unsupported_entry_mode",
                   "vendored_or_generated:", "repository_in_reference_universe",
                   "fork_parent_in_reference_universe"):
        if reason.startswith(prefix):
            return prefix.rstrip(":")
    return reason.split(":reference_universe:")[0]


def summarise(store_root: Path) -> dict[str, Any]:
    """Every count a report needs, from the journal and store alone."""
    store = ContentStore(store_root / "objects")
    journal = Journal(store_root / "journal.jsonl")
    entries = journal.entries()
    repos = [e for e in entries if e["key"].startswith("repo:")]
    scans = [e for e in entries if e["key"].startswith("scan:")]
    sessions = [e for e in entries if e["key"].startswith("session:")]
    candidates = [e for e in entries if e["key"].startswith("cand:")]
    evaluated = [c for c in candidates if c.get("outcome")]
    admitted = [c for c in evaluated if c["outcome"]["admitted"]]
    targets = [(c["key"], store.get_raw(c["target_function_address"]).decode("utf-8"))
               for c in admitted]
    duplicates = within_set_duplicates(targets)
    admitted = [c for c in admitted if c["key"] not in duplicates]
    all_reasons: Counter = Counter()
    families: Counter = Counter()
    primary: Counter = Counter()
    for c in candidates:
        if c.get("failure"):
            reasons = [f"acquisition_failure:{c['failure']}"]
        elif c.get("pre_pipeline_exclusion"):
            reasons = [c["pre_pipeline_exclusion"]]
        elif c["key"] in duplicates:
            reasons = [f"within_set_near_duplicate_of:{duplicates[c['key']]}"]
        elif not c["outcome"]["admitted"]:
            reasons = c["outcome"]["reasons"]
        else:
            continue
        primary[_reason_family(reasons[0])] += 1
        for reason in reasons:
            all_reasons[reason] += 1
            families[_reason_family(reason)] += 1
    categories: Counter = Counter()
    for c in candidates:
        for category in ((c.get("selection") or {}).get("changed_file_categories") or {}).values():
            categories[category] += 1
    per_repository = Counter(c["repository"] for c in admitted)
    inspected = len(candidates)
    tested = [c for c in admitted
              if c["outcome"]["derived"].get("regression_test_changed")]
    return {
        "entries": entries, "repos": repos, "scans": scans, "sessions": sessions,
        "candidates": candidates, "evaluated": evaluated, "admitted": admitted,
        "duplicates": duplicates, "store": store,
        "counts": {
            "candidates_inspected": inspected, "reached_pipeline": len(evaluated),
            "authenticated": sum(1 for c in evaluated if c["outcome"]["stages"]["schema"] == []
                                 and c["outcome"]["stages"]["authentication"] == []),
            "admitted": len(admitted),
            "unique_repositories_inspected": len({c["repository"] for c in candidates}),
            "unique_repositories_admitted": len(per_repository),
            "unique_fix_commits_admitted": len({c["fixed_commit"] for c in admitted}),
            "unique_functions_admitted": len(admitted),
            "within_set_duplicates_removed": len(duplicates),
            "admitted_with_authenticated_regression_test": len(tested),
            "other_production_source_exclusions": families[
                "auth:changed_production_source_outside_target"],
            "pyi_or_other_source_file_exclusions": sum(
                v for k, v in all_reasons.items()
                if k.startswith("insufficient_isolation_evidence:authentication:changed_"
                                "production_source_outside_target:") and not k.endswith(".py")),
            "runtime_or_config_exclusions": families[
                "auth:changed_runtime_or_config_outside_target"],
            "module_level_or_out_of_function_hunk_exclusions": all_reasons[
                "insufficient_isolation_evidence:authentication:"
                "production_hunk_outside_target_function"],
            "vendored_or_generated_exclusions": families["vendored_or_generated"],
            "auxiliary_evidence_failures": families["auth:auxiliary_evidence_missing"]
            + families["auth:auxiliary_unsupported_entry_mode"],
            "near_duplicate_patch_exclusions": all_reasons["patch_near_duplicate_of_known_bug"],
            "near_duplicate_function_exclusions": all_reasons[
                "function_near_duplicate_of_reference"],
            "same_organisation_exclusions": all_reasons[
                "same_organisation_as_indexed_repository"],
            "acquisition_failures": {k.split(":", 1)[1]: v for k, v in primary.items()
                                     if k.startswith("acquisition_failure:")}},
        "exclusions_primary_family": dict(primary.most_common()),
        "exclusions_all_reasons": dict(all_reasons.most_common()),
        "changed_file_categories": dict(categories.most_common()),
        "per_repository": per_repository,
    }


def projection(summary: dict[str, Any], rate: float, pool: list[str],
               budget: dict[str, Any]) -> dict[str, Any]:
    """Supply-based N=400 projection over the frozen repository pool."""
    scans = {e["repository"]: e for e in summary["scans"] if not e.get("failure")}
    screened = {e["repository"] for e in summary["repos"]
                if e.get("screen_problems") or e.get("failure")}
    queried_to_canonical = {e["key"][5:]: e["repository"] for e in summary["repos"]}
    supplies = [scan.get("fix_candidate_supply", 0) for scan in scans.values()]
    median_supply = statistics.median(supplies) if supplies else 0
    pass_fraction = (len(scans) / max(1, len(summary["repos"]))) if summary["repos"] else 0.0
    expected, used_candidates = [], 0
    seen = set()
    for name in pool:
        canonical = queried_to_canonical.get(name.lower(), name.lower())
        if canonical in seen:
            continue
        seen.add(canonical)
        if canonical in screened:
            continue
        if canonical in scans:
            supply, weight = scans[canonical].get("fix_candidate_supply", 0), 1.0
        else:
            supply, weight = median_supply, pass_fraction
        admitted = min(PLANNED_PER_REPOSITORY_CAP, supply * rate) * weight
        expected.append(admitted)
        if admitted > 0 and rate > 0:
            used_candidates += weight * min(supply, math.ceil(PLANNED_PER_REPOSITORY_CAP / rate))
    total = sum(expected)
    contributing = sum(1 for value in expected if value >= 1)
    seconds_per_candidate = (sum(c.get("seconds", 0) for c in summary["candidates"])
                             / max(1, len(summary["candidates"])))
    storage_per_candidate = summary["store"].usage_bytes() / max(1, len(summary["candidates"]))
    hours = used_candidates * seconds_per_candidate / 3600
    storage_gb = used_candidates * storage_per_candidate / 1e9
    feasible = (total >= PLANNED_N and contributing >= PLANNED_MIN_REPOSITORIES
                and used_candidates <= budget["max_candidates_to_inspect"]
                and hours <= budget["max_acquisition_hours"]
                and storage_gb <= budget["max_acquisition_storage_gb"])
    return {"admission_rate": rate, "pool_repositories": len(seen),
            "projected_admitted_with_cap": round(total, 1),
            "repositories_contributing_at_least_one": contributing,
            "candidates_to_inspect": math.ceil(used_candidates),
            "acquisition_hours_at_measured_rate": round(hours, 1),
            "acquisition_storage_gb": round(storage_gb, 2),
            "median_fix_candidate_supply_per_repository": median_supply,
            "feasible_within_pool_and_budget": feasible}


def build_report(store_root: Path, config: dict[str, Any], gate_spec: dict[str, Any] | None,
                 *, frozen_identity: dict[str, Any], label: str) -> dict[str, Any]:
    summary = summarise(store_root)
    counts = summary["counts"]
    admitted = summary["admitted"]
    per_repository = summary["per_repository"]
    inspected = counts["candidates_inspected"]
    interval = wilson(len(admitted), inspected)
    point = len(admitted) / inspected if inspected else 0.0
    store_check = verify_store(summary["store"], summary["entries"])
    sessions = summary["sessions"]
    evidence_sessions = [s["protected_access_evidence"] for s in sessions
                         if s.get("protected_access_evidence")]
    opened = sorted({path for ev in evidence_sessions for path in ev["protected_paths_opened"]})
    hash_failures = [c for c in summary["candidates"]
                     if c.get("failure") in ("hash_mismatch", "integrity_violation")]
    revalidation_failures = [c for c in summary["evaluated"]
                             if not c["outcome"]["revalidation_valid"]]
    incomplete_admitted = [c for c in admitted if c["outcome"]["insufficient_evidence"]]
    vendored_admitted = [c for c in admitted if any(
        r.startswith("vendored_or_generated") for r in c["outcome"]["reasons"])]
    top_share = max(per_repository.values()) / len(admitted) if admitted else 1.0
    pool = [name for path in config.get("pool_files", [config["repositories_file"]])
            for name in load_list(ROOT / path)]
    budget = (gate_spec or {}).get("budget", {"max_candidates_to_inspect": 10**9,
                                               "max_acquisition_hours": 10**9,
                                               "max_acquisition_storage_gb": 10**9})
    projections = {"at_wilson_lower": projection(summary, interval[0], pool, budget),
                   "at_point_estimate": projection(summary, point, pool, budget),
                   "at_wilson_upper": projection(summary, interval[1], pool, budget)}
    events = {"protected_data_access": bool(opened), "model_called": False,
              "evaluation_set_created": False, "network_accessed": any(
                  s.get("api_calls") or s.get("git_network")
                  or (s.get("git_objects") or {}).get("fetched") for s in sessions)}
    gate: dict[str, bool] = {"no_protected_data_access": not events["protected_data_access"]}
    if gate_spec is not None:
        spec = gate_spec["thresholds"]
        new_repositories = counts["unique_repositories_inspected"]
        gate.update({
            "minimum_scale": inspected >= spec["min_candidates"]
            and new_repositories >= spec["min_new_repositories"],
            "zero_incomplete_evidence_admissions": not incomplete_admitted,
            "zero_revalidation_failures": not revalidation_failures,
            "zero_hash_inconsistencies": not hash_failures and not store_check["problems"],
            "zero_vendored_or_generated_admissions": not vendored_admitted,
            "admitted_repository_diversity": len(per_repository) >= spec["min_admitted_repositories"],
            "maximum_repository_share": top_share <= spec["max_repository_share"],
            "yield_or_projection": interval[0] >= spec["min_wilson_lower_yield"]
            or projections["at_wilson_lower"]["feasible_within_pool_and_budget"],
            # Always required: at the conservative (Wilson-lower) rate the frozen pool
            # yields >= 400 targets from >= 34 repositories with <= 12 per repository.
            "n400_projection_respects_repository_rules":
                projections["at_wilson_lower"]["projected_admitted_with_cap"] >= PLANNED_N
                and projections["at_wilson_lower"]["repositories_contributing_at_least_one"]
                >= PLANNED_MIN_REPOSITORIES,
        })
    api = {"sessions": len(sessions),
           "api_calls": sum(s.get("api_calls", 0) for s in sessions),
           "api_retries": sum(s.get("api_retries", 0) for s in sessions),
           "api_seconds": round(sum(s.get("api_seconds", 0) for s in sessions), 1),
           "rate_limit_and_backoff_wait_seconds": round(sum(
               s.get("rate_limit_and_backoff_wait_seconds", 0) for s in sessions), 1),
           "elapsed_seconds": round(sum(s.get("elapsed_seconds", 0) for s in sessions), 1),
           "authenticated_requests": any(s.get("authenticated_requests") for s in sessions)}
    identity = {**source_tree_identity(ROOT), **frozen_identity,
                "isolation_version": ISOLATION_VERSION, "diff_policy_id": DIFF_POLICY["id"],
                "candidate_repository_list_sha256": sha256_file(ROOT / config["repositories_file"]),
                "tool_source_sha256": tool_sources(ROOT, TOOL_SOURCES),
                "journal_sha256": sha256_file(store_root / "journal.jsonl"),
                "store_verification": {"objects_checked": store_check["objects_checked"],
                                       "problems": len(store_check["problems"])},
                "command": config.get("command") or "scripts/run_repository_native_acquisition_"
                                                    "pilot.py",
                "configuration": {k: v for k, v in config.items() if k != "command"},
                "start_utc": min((s["start_utc"] for s in sessions), default=""),
                "end_utc": max((s["end_utc"] for s in sessions), default=""),
                "api": api}
    if gate_spec is not None:
        identity["gate_sha256"] = config.get("gate_sha256")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "report_schema": REPORT_SCHEMA,
        "label": label,
        "created_utc": utc_now(),
        "identity": identity,
        "events": events,
        "protected_access_evidence": {"installed": bool(evidence_sessions) and all(
            ev["installed"] for ev in evidence_sessions),
            "opens_checked": sum(ev["opens_checked"] for ev in evidence_sessions),
            "protected_paths_opened": opened,
            "method": evidence_sessions[0]["method"] if evidence_sessions else None},
        "policy": {"diff_policy": DIFF_POLICY["id"],
                   "temporal_cutoff_epoch": TEMPORAL_CUTOFF_EPOCH,
                   "candidates_since": config.get("candidates_since"),
                   "per_repository_candidate_cap": config.get("per_repository_cap"),
                   "fix_commit_selection": "first-parent default-branch commits whose message "
                                           "describes a fix and links a PR/issue (heuristic)"},
        "repositories": {
            "listed": len(load_list(ROOT / config["repositories_file"])),
            "reached": len(summary["repos"]),
            "screened_out": {e["repository"]: e.get("screen_problems") or e.get("failure")
                             for e in summary["repos"]
                             if e.get("screen_problems") or e.get("failure")},
            "same_organisation_for_review": sorted(
                e["repository"] for e in summary["repos"] if any(
                    "same_organisation" in p for p in e.get("screen_problems") or [])),
            "renames": {e["repository"]: e["renames"] for e in summary["repos"]
                        if e.get("renames")},
            "scans": {e["repository"]: {k: e.get(k) for k in (
                "commits_scanned", "fix_candidate_supply", "skipped_prior_commits", "failure")}
                for e in summary["scans"]}},
        "counts": counts,
        "exclusions_primary_family": summary["exclusions_primary_family"],
        "exclusions_all_reasons": summary["exclusions_all_reasons"],
        "changed_file_categories": summary["changed_file_categories"],
        "admitted_per_repository": dict(per_repository.most_common()),
        "maximum_repository_share": round(top_share, 3) if admitted else None,
        "complexity_shares": _shares(Counter((c.get("complexity") or {}).get(
            "tier", "unclassified") for c in admitted)),
        "bug_family_shares_heuristic": {k: round(sum(
            1 for c in admitted if c.get("bug_family_heuristic") == k)
            / len(admitted), 3) for k in BUG_FAMILIES} if admitted else {},
        "yield": {"admission_yield": round(point, 4), "wilson_95": interval,
                  "note": "admission yield precedes native qualification (Phase 3); it is an "
                          "upper bound on the mined-to-qualified yield"},
        "projection_for_n400": projections,
        "native_qualification_workload": {
            "admitted_targets": len(admitted),
            "with_authenticated_regression_test": counts[
                "admitted_with_authenticated_regression_test"],
            "official_test_runs_needed": len(admitted) * 2 * 3,
            "note": "qualification is NOT performed here; 3 runs x 2 revisions per target"},
        "store_verification": store_check,
        "gate_definition": gate_spec,
        "gate": gate,
        "admitted": [{"key": c["key"], "patch_sha256": c["outcome"]["patch_sha256"],
                      "record_sha256": c["outcome"]["record_sha256"],
                      "sidecar_address": c["addresses"]["sidecar"],
                      "tier": (c.get("complexity") or {}).get("tier"),
                      "bug_family_heuristic": c.get("bug_family_heuristic"),
                      "target": (c.get("selection") or {}).get("target_qualname"),
                      "target_file": (c.get("selection") or {}).get("target_file"),
                      "auxiliary": [(a["path"], a["category"], a["diff_sha256"]) for a in
                                    c["outcome"]["derived"].get("auxiliary_changes") or []]}
                     for c in admitted],
    }


def _shares(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {k: round(v / total, 3) for k, v in counter.items()} if total else {}


def publish(report: dict[str, Any], output: Path) -> None:
    problems = validate_receipt(report)
    if problems:
        raise SystemExit("REFUSED publication: " + "; ".join(problems))
    publish_file_atomically(output, (json.dumps(report, indent=1, sort_keys=True)
                                     + "\n").encode("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="pilot configuration JSON (repository list, store, gate, caps)")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["command"] = "python scripts/run_repository_native_acquisition_pilot.py --config " \
        + args.config.as_posix()
    gate_spec = None
    if config.get("gate_file"):
        gate_path = ROOT / config["gate_file"]
        config["gate_sha256"] = sha256_file(gate_path)
        if config.get("expected_gate_sha256") and \
                config["expected_gate_sha256"] != config["gate_sha256"]:
            raise SystemExit("REFUSED: the frozen gate file changed")
        gate_spec = json.loads(gate_path.read_text(encoding="utf-8"))
    if config.get("expected_repository_list_sha256") and sha256_file(
            ROOT / config["repositories_file"]) != config["expected_repository_list_sha256"]:
        raise SystemExit("REFUSED: the frozen repository list changed")
    store_root = ROOT / config["store"]
    ProtectedAccessMonitor.install()
    frozen, frozen_identity = load_frozen()
    frozen_identity["reference_universe_receipt_sha256"] = frozen.receipt_sha256
    if not args.report_only:
        run(store_root, config, frozen=frozen)
    report = build_report(store_root, config, gate_spec, frozen_identity=frozen_identity,
                          label=config["label"])
    publish(report, ROOT / config["report"])
    print(json.dumps({"counts": report["counts"], "gate": report["gate"],
                      "yield": report["yield"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
