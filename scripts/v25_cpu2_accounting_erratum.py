"""v2.5 CPU program 2, Phase 2A/2B: additive accounting erratum and version-identity record.

2A reconciles the SWE-bench patch denominators ("11", "12" and "13" were used in different
contexts) from the local target manifest, patch cache, environment and verification runs.
2B records the successor versions of every component whose behaviour changed. Historical
receipts are NOT rewritten; this receipt supersedes their wording only.

    python scripts/v25_cpu2_accounting_erratum.py
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v25_cpu2_accounting_erratum.json"
NATIVE = ROOT / "results/sft_root_cause/v25_native"
ENVS = NATIVE / "envs_swebench_run2.jsonl"
VERIFY = NATIVE / "verify_swebench/repository_verification_run1.jsonl"


def rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    from scripts import v25_module_conversion as conv
    from scripts import native_generated_tests_generate as gen
    from scripts.native_generation_io import TELEMETRY_SCHEMA
    from harness import native_launch_gate as gate
    from harness import native_generated_test_job_v25 as jobs
    targets = rows(NATIVE / "targets.jsonl")
    swe = [t for t in targets if t["dataset"] != "BugsInPy"]
    cached = {p.stem for p in (NATIVE / "patches").glob("*.json")}
    missing = sorted(t["task"] for t in swe if t["task"] not in cached)
    project = {t["task"]: t["project"] for t in targets}
    envs = rows(ENVS)
    verify = rows(VERIFY)
    env_missing = Counter(e["project"] for e in envs if e["category"] == "patch_unavailable")
    ver_missing = Counter(v["project"] for v in verify
                          if v["status"] == "environment:patch_unavailable")
    frag = Counter(project[t["task"]] for t in swe if t["task"] in missing
                   for _ in t["fragments"])
    by_project = {p: {"required_instances": sum(project[t["task"]] == p for t in swe),
                      "cached_and_hash_verified": sum(project[t["task"]] == p and
                                                      t["task"] in cached for t in swe),
                      "missing_instances": sum(project[m] == p for m in missing),
                      "repository_fragment_rows_affected": frag.get(p, 0),
                      "environment_attempts_affected": env_missing.get(p, 0),
                      "verification_rows_affected": ver_missing.get(p, 0)}
                  for p in sorted({t["project"] for t in swe})}
    by_project["thefuck (BugsInPy)"] = {"required_instances": 0, "note": "BugsInPy: no "
                                        "SWE-bench patch needed (buggy/fixed commits)"}
    table = {"unique_required_patch_instances": len(swe),
             "unique_cached_and_hash_verified": len(cached & {t["task"] for t in swe}),
             "unique_missing_instances": len(missing), "missing": missing,
             "repository_fragment_rows_affected": sum(frag.values()),
             "environment_attempts_affected": sum(env_missing.values()),
             "verification_rows_affected": sum(ver_missing.values()),
             "by_project": by_project}
    ok = (table["unique_required_patch_instances"] == 71
          and table["unique_cached_and_hash_verified"] == 60
          and table["unique_missing_instances"] == 11
          and table["environment_attempts_affected"] == 11)
    receipt = {
        "schema_version": "oneiros_v25_cpu2_accounting_erratum_v1",
        "patch_denominators": table,
        "wording_errata": [
            {"receipt": "results/sft_root_cause_v25_repository_verification.json",
             "field": "main_causes[3]", "said": "12 SWE-bench patches unavailable",
             "correct": "11 unique SWE-bench instances lack a hash-verified patch (10 django, "
                        "1 sympy); they affect 11 environment attempts and 13 verification "
                        "rows (12 django, 1 sympy) because some instances carry several "
                        "repository fragments. '12' was the Django verification-row count."},
            {"receipt": "results/sft_root_cause_v25_stop_decision.json",
             "field": "next_source_work_before_single_freeze[3]",
             "said": "the remaining 11 SWE-bench patches", "correct": "unchanged (11 unique "
                                                                      "instances)"}],
        "patch_rule": "a patch is accepted only if BOTH sha256 equal the train record's "
                      "gold_patch_sha256 / test_patch_sha256; no relaxation, no approximate "
                      "matching",
        "versions": {
            "module_conversion": {
                "current": conv.VERSION,
                "history": {"oneiros_v25_module_conversion_v1": {
                    "commit": "6b35b20", "file_sha256": "7e06390e57af4415ccaefdd8d01c7a070a6dbc"
                                                        "c0cc961df0e66e9716ff2dcc80",
                    "artifacts": "Stage 1 (receipt bf3f9828)"},
                    "oneiros_v25_module_conversion_v2": {
                    "commit": "d2ac726", "file_sha256": "2900192d5610b0b20838bf973710c3f7a6ce1c"
                                                        "e0fe9ea7cae74936e5cb46a9be",
                    "artifacts": "stage1_r2 (receipt a29c4972); that file still carried the "
                                 "v1 string, which this erratum corrects"}}},
            "generator": {"current": gen.GENERATOR_VERSION,
                          "change": "v5 runs only v2.5 jobs (job schema bound into every arm "
                                    "contract); v4 outputs, jobs and preflights stay readable "
                                    "but cannot resume or launch"},
            "job_schema": {"current": jobs.JOB_SCHEMA,
                           "change": "new: pytest_module_v1 items built only by "
                                     "harness/native_generated_test_job_v25.py"},
            "telemetry_schema": {"current": TELEMETRY_SCHEMA,
                                 "change": "unchanged: per-row semantics are identical; the "
                                           "row identity hash binds the arm contract, which "
                                           "now carries the job schema and builder hashes"},
            "preflight_schema": {"current": gate.PREFLIGHT_SCHEMA,
                                 "change": "successor introduced at 66e7397; no instance "
                                           "exists"},
            "authorization_schema": {"current": gate.AUTH_SCHEMA,
                                     "change": "successor introduced at 66e7397; no instance "
                                               "exists; v2 authorisations never match"}},
        "historical_receipts_rewritten": False,
        "consistent": ok}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "consistent": ok,
                      "table": {k: v for k, v in table.items() if k != "by_project"},
                      "sha256": hashlib.sha256((ROOT / RECEIPT).read_bytes()).hexdigest()},
                     indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
