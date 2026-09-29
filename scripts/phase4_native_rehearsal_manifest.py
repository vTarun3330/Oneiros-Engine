"""Phase 4 (bounded preparation): repository inventory and the frozen native-rehearsal manifest.

CPU only; no model, no GPU.  Runs inside the protected-location audit
(``harness.acquisition_receipt.ProtectedAccessMonitor``), so the receipt carries
evidence that no protected dataset was opened.

Inventory sources (all committed, non-protected):
* the frozen reference-universe bundle receipt: ``repository_full_names`` of every
  repository lineage in the corpus, ALL splits, names only (protected lineages are
  covered conservatively without opening any protected record);
* the TRAIN shard: repository projects of training-eligible records and of arm A's
  2,400-record selection;
* the three A-prime acquisition stores' committed receipts.

Rehearsal pool: the 26 admitted targets of the A-prime RETROSPECTIVE (development
rescore).  Confirmation-derived stores (the refused confirmation sample and its v6
revalidation, and the fresh confirmation pilot) are EXCLUDED from the pool and used
only for repository-name overlap checks.  The manifest is frozen (hashed) before any
target is executed; selection uses only admission metadata, never model output.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.acquisition_receipt import ProtectedAccessMonitor
from harness.atomic_publish import publish_file_atomically, read_current_bundle
from harness.github_acquisition import ContentStore
from harness.repository_isolation import candidate_from_sidecar

OUTPUT = "results/sft_root_cause_phase4_native_rehearsal_manifest_v1.json"
BUNDLE = "results/next_direction_bundle"
TRAIN = "data/corpus/v4_1_research_hardened_candidate/development_view/train.records.json"
PREFLIGHT = "results/v4_2_armA_successor_preflight.json"
POOL = {"receipt": "results/v4_3_repository_native_a_prime_retrospective.json",
        "store": "data/repository_native/a_prime_retrospective/objects"}
EXCLUDED = {
    "revalidation_v6_of_refused_confirmation_sample":
        "results/v4_3_repository_native_aprime_sample_revalidation_v6.json",
    "fresh_confirmation_pilot": "results/v4_3_repository_native_aprime_fresh_confirmation_pilot.json",
}
MAX_PER_REPOSITORY = 12


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def repo_of(key: str) -> str:
    return key.split("@")[0].replace("cand:", "")


def owner(full: str) -> str:
    return full.split("/")[0].lower()


def main() -> int:
    ProtectedAccessMonitor.install(ROOT)
    scope = ProtectedAccessMonitor.scope()
    bundle = read_current_bundle(ROOT / BUNDLE)
    universe = json.loads(bundle["files"]["reference_universe_receipt.json"])
    corpus_repos = sorted(universe["collections"]["repository_full_names"])

    train = load(TRAIN)
    selected = set(load(PREFLIGHT)["selection"]["selected_record_ids"])
    repo_records = [r for r in train
                    if (r.get("quality") or {}).get("execution_mode", "function") != "function"]

    def project(r):
        p = r.get("provenance") or {}
        return str(p.get("repository") or p.get("project") or "unknown")

    training_projects = sorted({project(r) for r in repo_records})
    arm_a_projects = sorted({project(r) for r in repo_records if r["id"] in selected})

    pool_receipt = load(POOL["receipt"])
    store = ContentStore(ROOT / POOL["store"])
    targets = []
    for admitted in pool_receipt["admitted"]:
        cand = candidate_from_sidecar(json.loads(store.get_raw(admitted["sidecar_address"])))
        tests = [a[0] for a in admitted.get("auxiliary", []) if a[1] == "test"]
        targets.append({
            "key": admitted["key"], "repository": cand.repository,
            "repository_url": cand.repository_url, "buggy_commit": cand.buggy_commit,
            "fixed_commit": cand.fixed_commit, "target_file": cand.target_file,
            "target": admitted["target"], "tier": admitted.get("tier"),
            "bug_family_heuristic": admitted.get("bug_family_heuristic"),
            "regression_test_files": tests, "licence_spdx": cand.licence_spdx,
            "patch_sha256": admitted["patch_sha256"], "record_sha256": admitted["record_sha256"]})
    targets.sort(key=lambda t: t["key"])

    excluded_repos = {name: sorted({repo_of(a["key"]) for a in load(rel)["admitted"]})
                      for name, rel in EXCLUDED.items()}
    pool_repos = Counter(t["repository"] for t in targets)
    corpus_owners = {owner(r) for r in corpus_repos}
    training_owners = {owner(p) for p in training_projects if "/" in p}
    checks = {
        "max_targets_per_repository": max(pool_repos.values()),
        "per_repository_cap_respected": max(pool_repos.values()) <= MAX_PER_REPOSITORY,
        "pool_repositories_in_corpus_lineages": sorted(r for r in pool_repos
                                                       if r in corpus_repos),
        "pool_owners_shared_with_corpus": sorted(r for r in pool_repos
                                                 if owner(r) in corpus_owners),
        "pool_owners_shared_with_training_projects": sorted(r for r in pool_repos
                                                            if owner(r) in training_owners),
        "pool_repositories_shared_with_excluded_confirmation_stores": {
            name: sorted(set(pool_repos) & set(repos)) for name, repos in excluded_repos.items()},
        "duplicate_patches_in_pool": len(targets) - len({t["patch_sha256"] for t in targets}),
        "targets_with_regression_test_file": sum(1 for t in targets if t["regression_test_files"]),
    }
    evidence = scope.close()
    passed = (checks["per_repository_cap_respected"]
              and not checks["pool_repositories_in_corpus_lineages"]
              and not checks["pool_owners_shared_with_corpus"]
              and not checks["pool_owners_shared_with_training_projects"]
              and checks["duplicate_patches_in_pool"] == 0
              and not evidence["protected_accesses"])
    manifest_body = json.dumps(targets, sort_keys=True).encode("utf-8")
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase4_native_rehearsal_manifest_v1",
        "model_calls": 0, "gpu_used": False, "training": False,
        "selection": ("all 26 admitted targets of the A-prime retrospective (development); "
                      "selected by admission metadata only, never by model output"),
        "isolation_caveat": ("retrospective admissions were made under isolation bundle "
                             f"{pool_receipt['identity']['bundle_generation']} (pre-v6); they "
                             "have not been revalidated under isolation v6"),
        "inventory": {
            "corpus_repository_lineages_all_splits": {
                "source": f"{BUNDLE} reference_universe_receipt.json (names only; no protected "
                          "record opened)",
                "count": len(corpus_repos), "names": corpus_repos},
            "training_split_repository_projects": training_projects,
            "arm_a_selected_repository_projects": arm_a_projects,
            "phase3_panels": "MBPP function records only (no repositories)",
            "harvested_stores": {
                "a_prime_retrospective (REHEARSAL POOL)": {"admitted": len(targets),
                                                           "repositories": dict(pool_repos)},
                **{f"{name} (EXCLUDED: confirmation-derived)": {"repositories": repos}
                   for name, repos in excluded_repos.items()}},
            "policy_a_acquisition_pilot": "1 admitted under policy A; superseded by the "
                                          "retrospective rescore of the same candidates"},
        "boundary_checks": checks,
        "boundary_checks_passed": passed,
        "inputs": {p: sha(p) for p in (TRAIN, PREFLIGHT, POOL["receipt"], *EXCLUDED.values())},
        "bundle": {"generation": bundle["generation"], "manifest_sha256": bundle["manifest_sha256"]},
        "protected_access_audit": {"opens_checked": evidence["opens_checked"],
                                   "protected_accesses": evidence["protected_accesses"],
                                   "method": evidence["method"]},
        "targets": targets,
        "targets_sha256": hashlib.sha256(manifest_body).hexdigest(),
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({k: receipt[k] for k in ("boundary_checks", "boundary_checks_passed",
                                              "protected_access_audit", "targets_sha256")},
                     indent=1))
    print("training projects:", training_projects[:40])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
