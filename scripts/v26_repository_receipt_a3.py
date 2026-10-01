"""v2.6 Phase 2-3: repository verification receipt for attempt a3 (frozen generic fixes F1, F2,
R1-R4) and the OBSERVED training gate (harness/v26_readiness.py: observed rows only,
deduplicated, fail closed), with the caps of addendum 1 applied first.

Byte identity and normalised identity of environment and verification runs are reported
separately.

    python scripts/v26_repository_receipt_a3.py
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIR = "results/sft_root_cause/v25_native_r2"
VERIFY = "results/sft_root_cause/v26_verify_a3"
RECEIPT = "results/sft_root_cause_v26_repository_verification_a3.json"
THEFUCK_ENVS = "results/sft_root_cause/v25_native/envs_thefuck_run3.jsonl"
ATTEMPT = "a3"
CAPS = {"per_function": 3, "per_lineage": 3, "per_repository": 40}
SCRIPTS = ("scripts/native_generated_tests_execute_wsl.py", "scripts/native_sandbox_inner.sh",
           "scripts/v25_view_rule.py", "scripts/v25_native_env_wsl.py",
           "scripts/v26_verify_repository_wsl.py", "scripts/v26_funnel_diag_wsl.py",
           "scripts/v25_django_layer_wsl.py", "scripts/v25_native_targets_r2.py",
           "scripts/v25_native_r2_build.sh", "harness/v26_readiness.py")
ENV_KEYS = ("category", "environment_lock", "environment_reproducible", "view_manifest_sha256",
            "module_sha256", "qualification", "view_rule", "kept_testing_packages",
            "official_tests_absent_from_views", "module")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonl(p) -> list:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    from harness.v26_readiness import observed_training_gate
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    root, verify = ROOT / DIR, ROOT / VERIFY
    projects = sorted(p.name for p in root.iterdir()
                      if (p / f"envs_{ATTEMPT}_run1.jsonl").exists())
    env_rep, env_files, final = {}, {}, {}
    for p in projects:
        r1, r2 = root / p / f"envs_{ATTEMPT}_run1.jsonl", root / p / f"envs_{ATTEMPT}_run2.jsonl"
        a = {r["task"]: {k: r.get(k) for k in ENV_KEYS} for r in jsonl(r1)}
        b = {r["task"]: {k: r.get(k) for k in ENV_KEYS} for r in jsonl(r2)}
        env_files[p] = [sha(r1), sha(r2)]
        env_rep[p] = {"byte_identical": sha(r1) == sha(r2), "normalised_identical": a == b}
        for r in jsonl(r2):
            final[r["task"]] = r
    for r in jsonl(ROOT / THEFUCK_ENVS):
        final[r["task"]] = r
    runs = [jsonl(verify / f"repository_verification_run{i}.jsonl") for i in (1, 2)]
    idx = [{v["index"]: v for v in run} for run in runs]
    verified = [v for v in runs[0] if v["accepted"] and idx[1].get(v["index"]) == v]
    cands = {c["index"]: c for c in jsonl(root / "converted_candidates_r2r.jsonl")}
    targets = {t["task"]: t for t in jsonl(root / f"targets_{ATTEMPT}.jsonl")}
    kept, per_fn, per_lin, per_repo, seen = [], Counter(), Counter(), Counter(), set()
    for v in sorted(verified, key=lambda v: cands[v["index"]]["conversion"]["identities"]
                    ["canonical_ast"]):
        canon = cands[v["index"]]["conversion"]["identities"]["canonical_ast"]
        fn = (v["project"], targets[v["task"]]["target_file"], v["qualname"])
        if canon in seen or per_fn[fn] >= CAPS["per_function"] or \
                per_lin[v["task"]] >= CAPS["per_lineage"] or \
                per_repo[v["project"]] >= CAPS["per_repository"]:
            continue
        seen.add(canon)
        per_fn[fn] += 1
        per_lin[v["task"]] += 1
        per_repo[v["project"]] += 1
        kept.append({"canonical_test": canon, "repository": v["project"],
                     "lineage": v["task"], "class": v["class"], "index": v["index"]})
    gate = observed_training_gate(kept)
    qualified = Counter(r["project"] for r in final.values() if r["category"] == "qualified")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    receipt = {
        "schema_version": "oneiros_v26_repository_verification_a3_v1",
        "source_commit": head, "attempt": ATTEMPT,
        "rules": {"view_rule": "oneiros_v26_view_rule_v3",
                  "target_rule": "oneiros_v26_target_any_changed_function_v1",
                  "runner_fixes": ["R1", "R2", "R3a", "R3b", "R4"],
                  "frozen_at": "1febae4 (before these outcomes)"},
        "inputs": {"targets": sha(root / f"targets_{ATTEMPT}.jsonl"),
                   "candidates": sha(root / "converted_candidates_r2r.jsonl"),
                   "environment_results": env_files, "thefuck_environments": sha(ROOT / THEFUCK_ENVS),
                   "verifier_outputs": [sha(verify / f"repository_verification_run{i}.jsonl")
                                        for i in (1, 2)]},
        "scripts_sha256": {s: sha(ROOT / s) for s in SCRIPTS},
        "environment_categories": {p: dict(Counter(r["category"] for r in final.values()
                                                   if r["project"] == p))
                                   for p in sorted({r["project"] for r in final.values()})},
        "qualified_environments": {"by_repository": dict(qualified),
                                   "repositories": len(qualified),
                                   "lineages": sum(qualified.values())},
        "environment_repeatability": env_rep,
        "verification_repeatability": {"byte_identical": sha(verify / "repository_verification_run1.jsonl")
                                       == sha(verify / "repository_verification_run2.jsonl"),
                                       "normalised_identical": idx[0] == idx[1]},
        "verification_by_project": {p: dict(Counter(v["status"] + (":" + v["class"]
                                                                   if v.get("class") else "")
                                                    for v in runs[0] if v["project"] == p))
                                    for p in sorted({v["project"] for v in runs[0]})},
        "verified_positive_rows": len(verified),
        "kills": dict(Counter(k["class"] for k in kept)),
        "observed_training_gate": gate,
        "historical_v25": {"repository_tests": 28, "repositories": 3, "lineages": 28}}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "gate": gate, "kills": receipt["kills"],
                      "qualified": receipt["qualified_environments"],
                      "env_rep": {p: (v["byte_identical"], v["normalised_identical"])
                                  for p, v in env_rep.items()},
                      "verification_repeatability": receipt["verification_repeatability"],
                      "sha256": sha(ROOT / RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
