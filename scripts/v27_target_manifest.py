"""v2.7: confirmation rehearsal-target manifest from an acquisition report + store, with the
post-acquisition candidate-level overlap audit (fail closed).

Each admitted candidate becomes a target in the format of scripts/native_rehearsal_wsl.py,
tagged role CONFIRMATION_ONLY. Before freezing, every candidate is checked against:
- the frozen training-function fingerprints (canonical AST of the buggy target function);
- every commit recorded by any earlier acquisition journal;
- the training repository keys (alias/rename-safe) via harness/v27_roles.py.
An overlapping candidate is EXCLUDED with its reason (never silently kept or dropped).

    python scripts/v27_target_manifest.py --report <acquisition report> --store <store dir>
        --out <manifest json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FINGERPRINTS = "results/sft_root_cause/v25_panel/training_function_fingerprints.json"
PANEL_SPEC = "results/sft_root_cause_v25_confirmation_panel_spec.json"
PARTITION = "results/sft_root_cause_v25_universe_partition.json"
EARLIER = ("acquisition_pilot", "aprime_confirmation_pilot", "aprime_fresh_confirmation_pilot",
           "a_prime_retrospective", "aprime_sample_revalidation_v6")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    from harness.github_acquisition import ContentStore
    from harness.repository_isolation import candidate_from_sidecar
    from harness.v27_roles import CONFIRMATION, admit_to_confirmation, candidate_overlap
    from scripts.run_repository_native_acquisition_pilot import prior_commits
    from scripts.v25_confirmation_panel_spec import function_fingerprint, repo_key
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    store = ContentStore(ROOT / args.store / "objects")
    fps = set(json.loads((ROOT / FINGERPRINTS).read_text(encoding="utf-8")))
    spec = json.loads((ROOT / PANEL_SPEC).read_text(encoding="utf-8"))
    if sha(ROOT / FINGERPRINTS) != spec["near_duplicate_gate"]["training_function_fingerprints_sha256"]:
        raise SystemExit("REFUSED: training fingerprints differ from the frozen specification")
    part = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    training_keys = set(part["training_expansion_pool"]) | \
        set(spec["exclusion_ledger"]["excluded_repository_keys"])
    commits = prior_commits([ROOT / "data/repository_native" / d / "journal.jsonl"
                             for d in EARLIER
                             if (ROOT / "data/repository_native" / d / "journal.jsonl").is_file()])
    targets, excluded = [], []
    for a in report["admitted"]:
        cand = candidate_from_sidecar(json.loads(store.get_raw(a["sidecar_address"])))
        try:
            fp = function_fingerprint(cand.target_function)
        except Exception as exc:                       # noqa: BLE001 - recorded
            excluded.append({"key": a["key"], "reason": f"fingerprint_failed:{exc}"[:120]})
            continue
        t = {"key": a["key"], "role": CONFIRMATION, "repository": cand.repository,
             "repository_key": repo_key(cand.repository),
             "repository_url": cand.repository_url, "buggy_commit": cand.buggy_commit,
             "fixed_commit": cand.fixed_commit, "target_file": cand.target_file,
             "target": a["target"], "tier": a.get("tier"),
             "bug_family_heuristic": a.get("bug_family_heuristic"),
             "regression_test_files": [x[0] for x in a.get("auxiliary", []) if x[1] == "test"],
             "licence_spdx": cand.licence_spdx, "patch_sha256": a["patch_sha256"],
             "record_sha256": a["record_sha256"], "function_fingerprint": fp,
             "lineage": f"{repo_key(cand.repository)}@{cand.fixed_commit}#{fp[:16]}"}
        hits = candidate_overlap([t], training_fingerprints=fps, prior_commits=commits)
        if hits:
            excluded.append({"key": a["key"], "reason": "training_or_prior_overlap",
                             "overlap": hits[0]["overlap"]})
            continue
        if not t["regression_test_files"]:
            excluded.append({"key": a["key"], "reason": "no_regression_test_file"})
            continue
        targets.append(t)
    admit_to_confirmation(targets, training_keys=training_keys)      # fail closed on crossing
    dup = len(targets) - len({t["function_fingerprint"] for t in targets})
    manifest = {"schema_version": "oneiros_v27_confirmation_rehearsal_manifest_v1",
                "role": CONFIRMATION, "training_prohibited": True,
                "source": {"report": args.report, "report_sha256": sha(ROOT / args.report)},
                "overlap_audit": {"training_function_fingerprints": len(fps),
                                  "prior_commits_checked": len(commits),
                                  "repository_keys_checked": len(training_keys),
                                  "within_manifest_duplicate_functions": dup},
                "targets": sorted(targets, key=lambda t: t["key"]), "excluded": excluded}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"targets": len(targets), "excluded": excluded,
                      "sha256": sha(out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
