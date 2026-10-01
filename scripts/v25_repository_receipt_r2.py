"""v2.5 CPU program 2, Phases 6D-7: repository verification receipt r2 (provenance-bound) and
the recomputed attainable training gate.

Binds (2C): converted-candidate overlay, stage1_r2 receipt, policy-recovery receipt, target
manifest, patch manifest, environment-result files (both runs of every project), verifier
outputs (both runs), Django layer records, executor / sandbox / candidate-policy / view-rule /
builder / verifier script hashes, access audit, source commit, environment locks (per qualified
target), normalised repeatability identities, and the archived artifact roles.

Repeatability is reported two ways: BYTE identity of files, and NORMALISED identity of the
fields that define an environment or a verdict (wall-clock fields excluded).

    python scripts/v25_repository_receipt_r2.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIR = "results/sft_root_cause/v25_native_r2"
RECEIPT = "results/sft_root_cause_v25_repository_verification_r2.json"
ATTEMPT = "a2"
ENV_KEYS = ("category", "environment_lock", "environment_reproducible", "view_manifest_sha256",
            "module_sha256", "qualification", "view_rule", "kept_testing_packages",
            "official_tests_absent_from_views", "module")
SCRIPTS = ("scripts/native_generated_tests_execute_wsl.py", "scripts/native_sandbox_inner.sh",
           "scripts/v25_view_rule.py", "scripts/v25_native_env_wsl.py",
           "scripts/v25_verify_repository_wsl.py", "scripts/v25_django_layer_wsl.py",
           "scripts/v25_native_targets_r2.py", "scripts/v25_policy_recovery.py",
           "scripts/v25_module_conversion.py", "scripts/v25_converted_corpus.py",
           "scripts/v25_native_r2_build.sh")
THEFUCK_VERIFY = "results/sft_root_cause/v25_native/verify_thefuck"
THEFUCK_ENVS = "results/sft_root_cause/v25_native/envs_thefuck_run3.jsonl"
GATE = {"repository_tests": 150, "repositories": 8, "lineages": 60}
CAPS = {"per_function": 3, "per_lineage": 3, "per_repository": 40}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path: Path) -> list:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def norm_env(row: dict) -> dict:
    return {k: row.get(k) for k in ENV_KEYS}


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    root = ROOT / DIR
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    projects = sorted(p.name for p in root.iterdir() if p.is_dir() and
                      (p / f"envs_{ATTEMPT}_run1.jsonl").exists())
    env_files, env_rep, envs_final = {}, {}, {}
    for p in projects:
        r1, r2 = root / p / f"envs_{ATTEMPT}_run1.jsonl", root / p / f"envs_{ATTEMPT}_run2.jsonl"
        env_files[p] = [sha(r1), sha(r2) if r2.exists() else None]
        a = {r["task"]: norm_env(r) for r in jsonl(r1)}
        b = {r["task"]: norm_env(r) for r in jsonl(r2)} if r2.exists() else {}
        env_rep[p] = {"byte_identical": r2.exists() and sha(r1) == sha(r2),
                      "normalised_identical": a == b,
                      "differing_tasks": sorted(t for t in set(a) | set(b)
                                                if a.get(t) != b.get(t))}
        for r in jsonl(r2 if r2.exists() else r1):
            envs_final[r["task"]] = r
    verify = root / f"verify_{ATTEMPT}"
    runs = [jsonl(verify / f"repository_verification_run{i}.jsonl") for i in (1, 2)]
    # thefuck was qualified and verified in v2.5 Phase 6-7 (no recovered rows, no testing
    # package) and is not rebuilt: its two verification runs are merged here, bound by hash
    old = ROOT / THEFUCK_VERIFY
    for i, run in enumerate(runs, 1):
        run[:] = [v for v in run if v["project"] != "thefuck"] + \
            [v for v in jsonl(old / f"repository_verification_run{i}.jsonl")
             if v["project"] == "thefuck"]
    env_files["thefuck (v2.5 Phase 6)"] = [sha(ROOT / THEFUCK_ENVS)]
    for r in jsonl(ROOT / THEFUCK_ENVS):
        envs_final[r["task"]] = r
    by_index = [{v["index"]: v for v in run} for run in runs]
    verified = [v for v in runs[0] if v["accepted"] and by_index[1].get(v["index"]) == v]
    cands = {c["index"]: c for c in jsonl(root / "converted_candidates_r2r.jsonl")}
    targets = {t["task"]: t for t in jsonl(root / f"targets_{ATTEMPT}.jsonl")}
    # attainable gate (unique canonical tests; caps; no repeated executions counted)
    seen, per_fn, per_lin, per_repo, kept = set(), Counter(), Counter(), Counter(), []
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
        kept.append(v)
    attained = {"repository_tests": len(kept),
                "repositories": len({v["project"] for v in kept}),
                "lineages": len({v["task"] for v in kept}),
                "functions": len({(v["project"], v["qualname"]) for v in kept}),
                "by_project": dict(Counter(v["project"] for v in kept)),
                "recovered_rows_verified": sum("recovery" in cands[v["index"]]["conversion"]
                                               for v in kept)}
    passed = all(attained[k] >= GATE[k] for k in GATE)
    patches = sorted((root / "patches").glob("*.json"))
    receipt = {
        "schema_version": "oneiros_v25_repository_verification_receipt_r2",
        "source_commit": head,
        "inputs": {
            "converted_candidates_r2r": sha(root / "converted_candidates_r2r.jsonl"),
            "stage1_r2_receipt": sha(ROOT / "results/sft_root_cause_v25_converted_corpus_stage1_r2.json"),
            "policy_recovery_receipt": sha(ROOT / "results/sft_root_cause_v25_policy_recovery_audit.json"),
            "attempt_spec": sha(ROOT / "results/sft_root_cause_v25_native_attempt_spec.json"),
            "targets": sha(root / f"targets_{ATTEMPT}.jsonl"),
            "patch_manifest": {"files": len(patches), "sha256": hashlib.sha256(json.dumps(
                {p.name: sha(p) for p in patches}, sort_keys=True).encode()).hexdigest()},
            "environment_results": env_files,
            "verifier_outputs": [sha(verify / f"repository_verification_run{i}.jsonl")
                                 for i in (1, 2)],
            "verifier_outputs_thefuck_phase6": [
                sha(ROOT / THEFUCK_VERIFY / f"repository_verification_run{i}.jsonl")
                for i in (1, 2)],
            "django_layers": {f.name: sha(f) for f in sorted(verify.glob("django_layers_run*.json"))}},
        "scripts_sha256": {s: sha(ROOT / s) for s in SCRIPTS},
        "environment_locks": {t: {"lock": r.get("environment_lock"),
                                  "interpreter": r.get("interpreter_version"),
                                  "view_manifest_sha256": r.get("view_manifest_sha256")}
                              for t, r in sorted(envs_final.items())
                              if r["category"] == "qualified"},
        "environment_categories": {p: dict(Counter(r["category"] for r in envs_final.values()
                                                   if r["project"] == p)) for p in projects},
        "environment_repeatability": env_rep,
        "verification_repeatability": {"byte_identical": sha(verify / "repository_verification_run1.jsonl")
                                       == sha(verify / "repository_verification_run2.jsonl"),
                                       "normalised_identical": by_index[0] == by_index[1]},
        "verification_by_project": {p: dict(Counter(
            v["status"] + (":" + v["class"] if v.get("class") else "")
            for v in runs[0] if v["project"] == p)) for p in sorted({v["project"]
                                                                     for v in runs[0]})},
        "verified_positive_rows": len(verified),
        "attainable_gate": {**attained, "gate": GATE, "passed": passed,
                            "shortfall": {k: max(0, GATE[k] - attained[k]) for k in GATE}},
        "archived_artifacts": "byte-verified external archive manifest (later phase)",
        "claims": {"real_repository_success": False, "training_ready": passed}}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "attainable_gate": receipt["attainable_gate"],
                      "environment_repeatability": {p: (v["byte_identical"],
                                                        v["normalised_identical"])
                                                    for p, v in env_rep.items()},
                      "verification_repeatability": receipt["verification_repeatability"],
                      "sha256": sha(ROOT / RECEIPT)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
