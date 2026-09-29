"""Phase 4: summarise the bounded native rehearsal and apply the predeclared stop rules.

CPU only (reads the WSL runner's records).  Rates carry Wilson 95% intervals; the
projection for the gate plus Phase 6 cohorts is a RANGE.  Environment failures are
reported separately and never counted as semantic negatives or positives.

Stop rules (from the task, applied as written):
* protected data touched                         -> stop
* admitted yield projects below 2%               -> stop
* natively-qualified / admitted below 50%        -> stop
* repository-diversity or lineage gates fail     -> stop
* native execution cannot reliably distinguish buggy and fixed behaviour -> stop
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

MANIFEST = "results/sft_root_cause_phase4_native_rehearsal_manifest_v1.json"
RECORDS = "results/sft_root_cause/phase4_native_rehearsal/records.jsonl"
RUN = "results/sft_root_cause/phase4_native_rehearsal/run_summary.json"
RETRO = "results/v4_3_repository_native_a_prime_retrospective.json"
FRESH = "results/v4_3_repository_native_aprime_fresh_confirmation_pilot.json"
ADDENDUM = "results/sft_root_cause_reporting_addendum_2026-09-29_v1.json"
OUTPUT = "results/sft_root_cause_phase4_native_rehearsal_receipt_v1.json"
TARGET = 350
PER_REPO_CAP, MIN_REPOS = 12, 34
Z = 1.959963984540054


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "point": round(k / n, 4) if n else None, "wilson_95": wilson(k, n)}


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    return {"median": round(statistics.median(values), 1),
            "p90": round(values[min(len(values) - 1, int(math.ceil(0.9 * len(values))) - 1)], 1),
            "max": round(values[-1], 1), "n": len(values)}


def main() -> int:
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (ROOT / RECORDS).read_text(encoding="utf-8")
               .splitlines() if line.strip()]
    keys = [r["key"] for r in records]
    expected = {t["key"] for t in manifest["targets"]}
    complete = set(keys) == expected and len(keys) == len(set(keys))
    run_summary = json.loads((ROOT / RUN).read_text(encoding="utf-8")) \
        if (ROOT / RUN).exists() else {}

    categories = Counter(r["category"] for r in records)
    env_fail = [r for r in records if r.get("environment_failure")]
    evaluable = [r for r in records if not r.get("environment_failure")]
    qualified = [r for r in records if r["category"] == "natively_qualified"]
    safe_call = [r for r in qualified
                 if (r.get("fixed_call_probe") or {}).get("safe_fixed_call_constructed")]

    retro = json.loads((ROOT / RETRO).read_text(encoding="utf-8"))["counts"]
    fresh = json.loads((ROOT / FRESH).read_text(encoding="utf-8"))["counts"]
    admitted_rates = {
        "independent_fresh_pilot": rate(fresh["admitted"], fresh["candidates_inspected"]),
        "development_retrospective_optimistic": rate(retro["admitted"],
                                                     retro["candidates_inspected"]),
        "note": ("the retrospective rescored the candidates used to DESIGN A-prime, so its "
                 "yield is optimistic; the independent fresh pilot rate is primary")}
    qual_rates = {
        "qualified_over_evaluable_admitted": rate(len(qualified), len(evaluable)),
        "qualified_over_all_admitted": rate(len(qualified), len(records)),
        "environment_success": rate(len(records) - len(env_fail), len(records)),
        "safe_fixed_call_over_qualified": rate(len(safe_call), len(qualified))}

    admit_lo, admit_hi = admitted_rates["independent_fresh_pilot"]["wilson_95"]
    admit_pt = admitted_rates["independent_fresh_pilot"]["point"]
    q = qual_rates["qualified_over_all_admitted"]
    q_pt, (q_lo, q_hi) = q["point"] or 0.0, q["wilson_95"]
    calls_per_candidate = json.loads((ROOT / ADDENDUM).read_text(encoding="utf-8"))[
        "acquisition_arithmetic"]["observed"]["api_calls_per_candidate"]
    walls = [r["wall_seconds"] for r in records if "wall_seconds" in r]

    def candidates(admit, qual):
        return None if not admit or not qual else math.ceil(TARGET / (admit * qual))

    per = {"point": candidates(admit_pt, q_pt),
           "optimistic": candidates(admit_hi, q_hi), "pessimistic": candidates(admit_lo, q_lo)}
    both = {k: (2 * v if v else None) for k, v in per.items()}
    median_wall = statistics.median(walls) if walls else None
    admitted_needed = {k: (math.ceil(TARGET / qv) if qv else None)
                       for k, qv in (("point", q_pt), ("optimistic", q_hi), ("pessimistic", q_lo))}
    projection = {
        "status": "PROJECTION from a 26-target rehearsal; ranges combine the Wilson bounds of "
                  "the independent admission rate and the qualified/admitted rate",
        "qualified_functions_per_cohort": TARGET,
        "mined_candidates_per_cohort": per, "mined_candidates_gate_plus_phase6": both,
        "admitted_targets_to_rehearse_per_cohort": admitted_needed,
        "api_calls_gate_plus_phase6": {k: (math.ceil(v * calls_per_candidate) if v else None)
                                       for k, v in both.items()},
        "unauthenticated_api_hours_gate_plus_phase6": {
            k: (round(v * calls_per_candidate / 60, 1) if v else None) for k, v in both.items()},
        "native_qualification_cpu_hours_gate_plus_phase6": {
            k: (round(2 * v * median_wall / 3600, 1) if v and median_wall else None)
            for k, v in admitted_needed.items()},
        "native_qualification_cpu_note": ("median per-target wall time x admitted targets to "
                                          "rehearse; the runner used 4 parallel repositories"),
        "repositories_per_cohort": max(math.ceil(TARGET / PER_REPO_CAP), MIN_REPOS),
        "distinct_repositories_gate_plus_phase6": 2 * max(math.ceil(TARGET / PER_REPO_CAP),
                                                          MIN_REPOS),
        "environment_failures_counted_as_losses": True}

    qual_share = q_pt
    reliable = (len(evaluable) > 0 and len(qualified) / len(evaluable) >= 0.5)
    stop = {
        "protected_data_touched": bool(manifest["protected_access_audit"]["protected_accesses"]),
        "admitted_yield_projects_below_2pct": admit_pt < 0.02,
        "admitted_yield_lower_wilson_below_2pct_note": admit_lo < 0.02,
        "qualified_over_admitted_below_50pct": qual_share < 0.5,
        "repository_diversity_or_lineage_gate_failed": not manifest["boundary_checks_passed"],
        "native_execution_not_reliable": not reliable,
        "rehearsal_incomplete": not complete,
    }
    stop_triggered = [k for k, v in stop.items()
                      if v and k != "admitted_yield_lower_wilson_below_2pct_note"]
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase4_native_rehearsal_v1",
        "model_calls": 0, "gpu_used": False, "training": False,
        "inputs": {MANIFEST: sha(MANIFEST), RECORDS: sha(RECORDS), RETRO: sha(RETRO),
                   FRESH: sha(FRESH), ADDENDUM: sha(ADDENDUM)},
        "run_environment": run_summary,
        "targets": len(records), "complete": complete,
        "categories": dict(categories),
        "environment_failures": {"count": len(env_fail),
                                 "by_category": dict(Counter(r["category"] for r in env_fail)),
                                 "note": "never counted as semantic negatives or positives"},
        "admitted_over_mined": admitted_rates,
        "rates": qual_rates,
        "by_repository": {repo: dict(Counter(r["category"] for r in records
                                             if r["repository"] == repo))
                          for repo in sorted({r["repository"] for r in records})},
        "qualified_bug_family": dict(Counter(r.get("bug_family") for r in qualified)),
        "qualified_repositories": len({r["repository"] for r in qualified}),
        "timing_seconds": {
            "wall_per_target": quantiles(walls),
            "environment_setup": quantiles([r["steps"]["environment"]["seconds"]
                                            for r in records
                                            if "environment" in r.get("steps", {})]),
            "tests_buggy_plus_fixed": quantiles([
                r["tests"]["buggy"]["seconds"] + r["tests"]["fixed"]["seconds"]
                for r in records if "tests" in r]),
            "rehearsal_total_wall": run_summary.get("wall_seconds")},
        "python_used": dict(Counter((r.get("steps", {}).get("environment") or {}).get("python")
                                    for r in records)),
        "projection": projection,
        "stop_rules": stop, "stop_triggered": stop_triggered,
        "gate_passed": not stop_triggered,
        "caveats": [
            "the rehearsed targets are the A-prime DEVELOPMENT retrospective (candidates used to "
            "design A-prime): mostly popular pure-Python libraries; the qualified/admitted rate is "
            "likely optimistic for independently mined repositories",
            "environments were fast because uv reused its package cache and no target needed a "
            "compiled extension; the native-qualification CPU projection is likely optimistic",
            "retrospective admissions were made under a pre-v6 isolation bundle and were not "
            "revalidated under isolation v6",
            "the admitted-yield point estimate (4.17%) projects above 2%, but its lower Wilson "
            "bound (1.79%) is below 2%"],
        "blockers": [
            {"blocker": "fixed_call_construction",
             "observed": f"{len(safe_call)}/{len(qualified)} qualified targets yielded a safe "
                         "fixed call from literal-argument calls in their regression tests",
             "consequence": ("repository-native targets cannot yet supply fixed discriminating "
                             "inputs for the Phase 4 fixed-input mediator; regression tests reach "
                             "the target through parametrisation, objects and wrappers"),
             "next_step": ("capture the target's actual arguments by tracing it during the "
                           "difference-exposing official test on both revisions, and keep only "
                           "short round-trippable literal inputs whose results differ")}],
        "per_target": [{k: r.get(k) for k in ("key", "repository", "category",
                                              "environment_failure", "wall_seconds",
                                              "difference_exposing_count", "tests",
                                              "fixed_call_probe", "detail")}
                       for r in sorted(records, key=lambda r: r["key"])],
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True,
                                                       default=str) + "\n").encode("utf-8"))
    print(json.dumps({k: receipt[k] for k in ("categories", "rates", "admitted_over_mined",
                                              "timing_seconds", "projection", "stop_rules",
                                              "stop_triggered", "gate_passed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
