"""v2.7 Phase 5: freeze the actual confirmation panel (selection is structural only; no model,
repair or Atheris outcome exists or is used).

Admission (per target): role CONFIRMATION_ONLY; zero training / prior overlap (target manifest);
official difference-exposing tests requalified 3/3 (prep); target reached on both revisions
natively (native reach); prompt-leakage audit and token fit through the REAL v2.5 job builder
(harness/native_generated_test_job_v25.build_item with the verifier-only material).
Selection order (frozen in docs/repository_native_v27_confirmation_panel_gate.json): per
repository by fixed-commit committer date, then sha; at most 8 per repository; one target per
function fingerprint and per lineage. Status by harness/v26_readiness.panel_status:
CONFIRMATION_READY, or BUILT_UNDERPOWERED (exploratory only), or INVALID.

The model-visible bundle (prompts + buggy-side DTOs) and the hidden evaluator bundle (verifier
exports: fixed source, patch, official tests) are written to SEPARATE files and hashed
independently.

    python scripts/v27_panel_freeze.py --name full
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D = "results/sft_root_cause/v27_confirmation"
GATE = "docs/repository_native_v27_confirmation_panel_gate.json"
FINGERPRINTS = "results/sft_root_cause/v25_panel/training_function_fingerprints.json"
PARTITION = "results/sft_root_cause_v25_universe_partition.json"
PANEL_SPEC = "results/sft_root_cause_v25_confirmation_panel_spec.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p) -> str:
    return sha_bytes((ROOT / p).read_bytes())


def jsonl(p) -> list:
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def last_by_key(rows: list) -> dict:
    out = {}
    for r in rows:
        out[r["key"]] = r
    return out


def commit_time(store, sidecar_address: str, fixed_commit: str) -> int:
    from harness.repository_isolation import candidate_from_sidecar, git_object_id, parse_commit
    cand = candidate_from_sidecar(json.loads(store.get_raw(sidecar_address)))
    for obj in cand.git_objects:
        if obj.kind == "commit" and git_object_id("commit", obj.body) == fixed_commit:
            info = parse_commit(obj.body)
            return int(info.get("committer_epoch") or info.get("author_epoch") or 0)
    raise SystemExit(f"REFUSED: fixed commit object missing from the evidence sidecar: "
                     f"{fixed_commit}")


def main(argv=None) -> int:
    from harness import native_generated_test_job_v25 as jobs
    from harness.github_acquisition import ContentStore
    from harness.v26_readiness import panel_status
    from scripts.native_generated_tests_generate import CONTRACT
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="canary | full")
    parser.add_argument("--store", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    n = args.name
    manifest = json.loads((ROOT / f"{D}/{n}_rehearsal_manifest.json").read_text(encoding="utf-8"))
    qual = json.loads((ROOT / f"{D}/{n}_qualification_manifest.json").read_text(encoding="utf-8"))
    rehearsal = last_by_key(jsonl(f"{D}/{n}_rehearsal/records.jsonl"))
    prep = last_by_key(jsonl(f"{D}/{n}_prep/records.jsonl"))
    reach = last_by_key(jsonl(f"{D}/{n}_native_reach.jsonl"))
    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    store = ContentStore(ROOT / args.store / "objects")
    sidecars = {a["key"]: a["sidecar_address"] for a in report["admitted"]}
    count = jobs.chat_token_counter(CONTRACT["base_model"], CONTRACT["base_revision"])
    qual_by = {t["key"]: t for t in qual["targets"]}
    qual_keys = set(qual_by)
    funnel = Counter()
    exclusions = []
    eligible = []
    for t in manifest["targets"]:
        key = t["key"]
        funnel["with_regression_test_and_no_overlap"] += 1

        def drop(reason):
            exclusions.append({"key": key, "repository": t["repository"], "reason": reason})
        r = rehearsal.get(key)
        if not r or r["category"] != "natively_qualified" or key not in qual_keys:
            drop(f"rehearsal:{(r or {}).get('category', 'not_rehearsed')}")
            continue
        funnel["natively_qualified"] += 1
        p = prep.get(key)
        if not p or p["category"] != "requalified":
            drop(f"requalification:{(p or {}).get('category', 'not_prepared')}")
            continue
        funnel["requalified_3_of_3"] += 1
        if (reach.get(key) or {}).get("reach") != "reach_verified":
            drop(f"target_not_reached:{(reach.get(key) or {}).get('reach')}")
            continue
        funnel["target_reached"] += 1
        bv = json.loads((ROOT / f"{D}/{n}_prep/buggy_view/{p['tag']}.json").read_text(
            encoding="utf-8"))
        vf = json.loads((ROOT / f"{D}/{n}_prep/verifier/{p['tag']}.json").read_text(
            encoding="utf-8"))
        built = jobs.build_item(bv["dto"], bv["buggy_source"], vf, count)
        if "refused" in built:
            drop("prompt:" + "+".join(built["refused"]["reasons"]))
            continue
        funnel["prompt_leakage_and_token_fit_passed"] += 1
        eligible.append({**t, "tag": p["tag"], "item": built["item"], "verifier": vf,
                         "difference_exposing_tests": qual_by[key]["difference_exposing_tests"],
                         "environment_lock_sha256": sha_bytes(json.dumps(
                             p["environment_lock"], sort_keys=True).encode()),
                         "fixed_time": commit_time(store, sidecars[key], t["fixed_commit"])})
    selected, per_repo, seen_fp, seen_lin = [], Counter(), set(), set()
    for e in sorted(eligible, key=lambda e: (e["repository"], e["fixed_time"],
                                             e["fixed_commit"])):
        if e["function_fingerprint"] in seen_fp or e["lineage"] in seen_lin:
            exclusions.append({"key": e["key"], "repository": e["repository"],
                               "reason": "duplicate_function_or_lineage"})
            continue
        if per_repo[e["repository"]] >= 8:
            exclusions.append({"key": e["key"], "repository": e["repository"],
                               "reason": "per_repository_cap"})
            continue
        seen_fp.add(e["function_fingerprint"])
        seen_lin.add(e["lineage"])
        per_repo[e["repository"]] += 1
        selected.append(e)
    funnel["selected"] = len(selected)
    out_dir = ROOT / D / f"{n}_panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    visible = {e["key"]: {"dto": e["item"]["target"], "prompt": e["item"]["prompt"],
                          "prompt_sha256": e["item"]["prompt_sha256"],
                          "prompt_tokens": e["item"]["prompt_tokens"]} for e in selected}
    hidden = {e["key"]: e["verifier"] for e in selected}
    vis_bytes = (json.dumps(visible, indent=1, sort_keys=True) + "\n").encode()
    hid_bytes = (json.dumps(hidden, indent=1, sort_keys=True) + "\n").encode()
    for name, data in (("model_visible_bundle.json", vis_bytes),
                       ("hidden_evaluator_bundle.json", hid_bytes)):
        path = out_dir / name
        if path.exists() and path.read_bytes() != data:
            raise SystemExit(f"REFUSED: {name} exists with different content")
        path.write_bytes(data)
    panel_targets = [{"target_id": e["key"], "repository": e["repository"],
                      "lineage": e["lineage"], "function_fingerprint": e["function_fingerprint"],
                      "buggy_commit": e["buggy_commit"], "fixed_commit": e["fixed_commit"],
                      "environment_lock_sha256": e["environment_lock_sha256"],
                      "complexity": e.get("tier"), "defect_family": e.get("bug_family_heuristic"),
                      "target_file": e["target_file"], "target": e["target"],
                      "licence_spdx": e.get("licence_spdx"), "tag": e["tag"],
                      "difference_exposing_tests": e["difference_exposing_tests"]}
                     for e in selected]
    targets_sha = sha_bytes(json.dumps(panel_targets, sort_keys=True).encode())
    part = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    spec = json.loads((ROOT / PANEL_SPEC).read_text(encoding="utf-8"))
    status = panel_status({"frozen": True, "targets_sha256": targets_sha,
                           "targets": panel_targets},
                          training_fingerprints=json.loads((ROOT / FINGERPRINTS).read_text(
                              encoding="utf-8")),
                          training_repositories=set(part["training_expansion_pool"])
                          | set(spec["exclusion_ledger"]["excluded_repository_keys"]))
    label = {"CONFIRMATION_READY": "CONFIRMATION_READY",
             "BUILT_UNDERPOWERED": "CONFIRMATION_BUILT_UNDERPOWERED"}.get(status["status"],
                                                                         "INVALID")
    receipt = {
        "schema_version": "oneiros_v27_confirmation_panel_v1", "name": n,
        "status": label, "panel_status": status,
        "frozen_before_any_model_or_atheris_outcome": True,
        "gate": {"path": GATE, "sha256": sha(GATE)},
        "acquisition_counts": report["counts"],
        "funnel": dict(funnel), "exclusions_by_reason": dict(Counter(
            x["reason"].split(":")[0] for x in exclusions)),
        "exclusions": exclusions,
        "per_repository": dict(per_repo),
        "complexity": dict(Counter(t["complexity"] for t in panel_targets)),
        "defect_families": dict(Counter(t["defect_family"] for t in panel_targets)),
        "targets": panel_targets, "targets_sha256": targets_sha,
        "model_visible_bundle": {"path": f"{D}/{n}_panel/model_visible_bundle.json",
                                 "sha256": sha_bytes(vis_bytes)},
        "hidden_evaluator_bundle": {"path": f"{D}/{n}_panel/hidden_evaluator_bundle.json",
                                    "sha256": sha_bytes(hid_bytes)},
        "inputs_sha256": {p: sha(p) for p in (
            f"{D}/{n}_rehearsal_manifest.json", f"{D}/{n}_qualification_manifest.json",
            f"{D}/{n}_rehearsal/records.jsonl", f"{D}/{n}_prep/records.jsonl",
            f"{D}/{n}_native_reach.jsonl", args.report)},
        "use": "confirmation only; never training, replay, relearning, tuning, checkpoint "
               "selection or repair",
        "claims_allowed": status.get("claim_scope")}
    rel = f"results/sft_root_cause_v27_confirmation_panel_{n}.json"
    print(publish_once(rel, receipt))
    print(json.dumps({"status": label, "funnel": dict(funnel), "per_repository": dict(per_repo),
                      "exclusions_by_reason": receipt["exclusions_by_reason"],
                      "sha256": sha(rel)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
