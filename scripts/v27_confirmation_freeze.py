"""v2.7 Phase 1: freeze the confirmation-only repository roles, overlap audit, acquisition
configuration and panel gate BEFORE any confirmation candidate is acquired.

Sources (committed, non-protected): the frozen universe partition (70 confirmation-only keys),
the confirmation-panel specification (exclusion ledger, training-function fingerprints), the
earlier acquisition journals (repository-level contact only) and the previously-scanned list.

Outputs (all hash-bound in the preflight receipt):
  docs/repository_native_v27_confirmation_repositories.json   flat list of 70 canonical names
  docs/repository_native_v27_confirmation_acquisition.json    runner configuration
  docs/repository_native_v27_confirmation_panel_gate.json     panel selection/gate rules
  results/sft_root_cause_v27_confirmation_role_registry.json  immutable roles
  results/sft_root_cause_v27_confirmation_overlap_pre.json    pre-acquisition overlap audit
  results/sft_root_cause_v27_confirmation_preflight.json      binds everything above

    python scripts/v27_confirmation_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARTITION = "results/sft_root_cause_v25_universe_partition.json"
PANEL_SPEC = "results/sft_root_cause_v25_confirmation_panel_spec.json"
PREV_SCANNED = "docs/repository_native_previously_scanned_repositories.json"
REPOS = "docs/repository_native_v27_confirmation_repositories.json"
CONFIG = "docs/repository_native_v27_confirmation_acquisition.json"
GATE = "docs/repository_native_v27_confirmation_panel_gate.json"
REGISTRY = "results/sft_root_cause_v27_confirmation_role_registry.json"
OVERLAP = "results/sft_root_cause_v27_confirmation_overlap_pre.json"
PREFLIGHT = "results/sft_root_cause_v27_confirmation_preflight.json"
STORE = "data/repository_native/v27_confirmation"
EARLIER_STORES = ("acquisition_pilot", "aprime_confirmation_pilot",
                  "aprime_fresh_confirmation_pilot", "a_prime_retrospective",
                  "aprime_sample_revalidation_v6")
ROLE = "CONFIRMATION_ONLY"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def write_once(rel: str, payload) -> str:
    data = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    path = ROOT / rel
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"REFUSED: {rel} exists with different content")
        return "verified_reproduction"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "written"


def earlier_contacts(keys: set) -> dict:
    """Repository-level records of earlier acquisition journals touching a confirmation key."""
    from scripts.v25_confirmation_panel_spec import repo_key
    out = {}
    for d in EARLIER_STORES:
        path = ROOT / "data" / "repository_native" / d / "journal.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            repo = r.get("repository")
            if not isinstance(repo, str) or "/" not in repo or repo_key(repo) not in keys:
                continue
            kind = str(r.get("key", "")).split(":")[0]
            out.setdefault(repo_key(repo), []).append(
                {"store": d, "record": kind, "screen_problems": r.get("screen_problems")})
    return out


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    from scripts.v25_confirmation_panel_spec import repo_key
    part = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    spec = json.loads((ROOT / PANEL_SPEC).read_text(encoding="utf-8"))
    conf = part["confirmation_only_pool"]
    training_keys = set(part["training_expansion_pool"])
    ledger = set(spec["exclusion_ledger"]["excluded_repository_keys"])
    names = sorted(n for v in conf.values() for n in v)
    if len(conf) != 70 or len(names) != 70:
        raise SystemExit(f"REFUSED: expected 70 confirmation repositories, got {len(conf)}")
    prev = {repo_key(x) for x in json.loads((ROOT / PREV_SCANNED).read_text(encoding="utf-8"))}
    contacts = earlier_contacts(set(conf))
    problems = []
    for k in conf:
        if k in training_keys:
            problems.append(f"{k}: also in the training-expansion pool")
        if k in ledger:
            problems.append(f"{k}: in the exclusion ledger")
        if k in prev:
            problems.append(f"{k}: previously scanned")
        for c in contacts.get(k, []):
            if c["record"] != "repo":
                problems.append(f"{k}: earlier {c['record']} record in {c['store']}")
    if problems:
        raise SystemExit(f"REFUSED (fail closed): unresolved overlap {problems[:5]}")
    registry = {"schema_version": "oneiros_v27_confirmation_role_registry_v1",
                "role": ROLE, "training_prohibited": True,
                "source_partition": {"path": PARTITION, "sha256": sha(PARTITION)},
                "repositories": [{"repository": n, "repository_key": repo_key(n),
                                  "url": f"https://github.com/{n}", "role": ROLE,
                                  "training_prohibited": True,
                                  "aliases": sorted(conf[repo_key(n)]),
                                  "status": "listed"} for n in names]}
    overlap = {
        "schema_version": "oneiros_v27_confirmation_overlap_pre_v1",
        "checked_before_acquisition": True,
        "repository_overlap": {"training_expansion_pool": 0, "exclusion_ledger": 0,
                               "previously_scanned_list": 0,
                               "exclusion_ledger_keys": len(ledger),
                               "ledger_covers": "every training repository, every BugsInPy / "
                                                "SWE-bench repository, the v2.4 diagnostic "
                                                "panel and the earlier pilots' cloned "
                                                "repositories"},
        "earlier_metadata_only_contacts": {
            k: v for k, v in sorted(contacts.items())},
        "earlier_contact_interpretation": "repository-level metadata screening only (licence "
                                          "screen); no commit, candidate, code or test of these "
                                          "repositories was ever inspected; no reuse",
        "candidate_level_checks_after_acquisition": [
            "fix commit and issue identities vs every earlier journal",
            "target-function canonical fingerprint vs the frozen training fingerprints "
            f"({spec['near_duplicate_gate']['training_function_fingerprints']} functions, "
            f"sha256 {spec['near_duplicate_gate']['training_function_fingerprints_sha256']})",
            "lineage (repository key, fix commit, function fingerprint)",
            "renames / aliases resolved by the acquisition API"],
        "protected_material": "no validation, reserved-confirmation or sealed record opened"}
    gate = {
        "schema_version": "oneiros_v27_confirmation_panel_gate_v1",
        "frozen_before": "any v2.7 confirmation candidate is acquired",
        "selection": {"targets": 80, "min_repositories": 10, "max_per_repository": 8,
                      "complexity_strata": ["simple", "moderate", "complex"],
                      "min_defect_families": 4,
                      "order": "structural only: per repository, admitted candidates by fixed "
                               "commit date then sha; never model, repair or Atheris outcomes"},
        "admission": ["policy-A single-target fix with authenticated regression evidence",
                      "native environment builds twice identically",
                      "official regression test fails 3/3 on buggy, passes 3/3 on fixed",
                      "target reached", "prompt-leakage audit passes",
                      "zero training overlap (repository, issue, commit, lineage, function "
                      "fingerprint, near duplicate)"],
        "underpowered_rule": "fewer than 80 targets or 10 repositories after the bounded pool "
                             "is exhausted -> freeze as built, CONFIRMATION_BUILT_UNDERPOWERED: "
                             "exploratory estimation only",
        "use": "confirmation only; never training, replay, relearning, tuning, checkpoint "
               "selection or repair"}
    status = {REPOS: write_once(REPOS, names)}
    config = {
        "label": "v2.7 CONFIRMATION-ONLY acquisition over the frozen 70-repository pool; "
                 "candidate pool only, no evaluation set, no model, no GPU",
        "repositories_file": REPOS, "expected_repository_list_sha256": sha(REPOS),
        "store": STORE, "report": "results/sft_root_cause_v27_confirmation_acquisition_report.json",
        "per_repository_cap": 30, "max_candidates": 1_000_000, "min_repositories": 10_000,
        "fetch_since": "2024-12-01", "candidates_since": "2025-01-01",
        "exclude_journals": [f"data/repository_native/{d}/journal.jsonl"
                             for d in EARLIER_STORES
                             if (ROOT / "data/repository_native" / d / "journal.jsonl").is_file()],
        "exclude_repository_files": [PREV_SCANNED],
        "stopping_rule": "every repository of the frozen list is processed (max_candidates and "
                         "min_repositories are unreachable sentinels); at most 30 classified "
                         "fix commits per repository, newest-first as the runner lists them; "
                         "independent of outcomes",
        "role": ROLE, "training_prohibited": True}
    status[CONFIG] = write_once(CONFIG, config)
    status[GATE] = write_once(GATE, gate)
    status[REGISTRY] = publish_once(REGISTRY, registry)
    status[OVERLAP] = publish_once(OVERLAP, overlap)
    preflight = {"schema_version": "oneiros_v27_confirmation_preflight_v1",
                 "frozen_before_any_result": True,
                 "artifacts_sha256": {p: sha(p) for p in (REPOS, CONFIG, GATE, REGISTRY,
                                                          OVERLAP, PARTITION, PANEL_SPEC)},
                 "repositories": 70, "role": ROLE,
                 "command": "python scripts/v27_acquire.py --config " + CONFIG}
    status[PREFLIGHT] = publish_once(PREFLIGHT, preflight)
    print(json.dumps({"status": status, "earlier_metadata_contacts": len(contacts),
                      "preflight_sha256": sha(PREFLIGHT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
