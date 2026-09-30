"""Frozen analysis for native generated tests (protocol v2 s2/s8 + amendment v2.1 section E).

    analyse --manifest M --results R --condition primary_whole_module
            --study-mode {engineering_dress_rehearsal,confirmation} --out FILE [--atheris A]

Unit: the TARGET. kill(t, s, m) = 1 if any of the first k slots is a kill; K(t, m) is the mean
over the three seeds; the primary estimand is mean over targets of K(t, SFT) - K(t, base).
Target x seed observations are never pooled as independent samples.

Infrastructure (harness/dependency/environment) rows are never counted as parsed, collected,
executed or reached. A target with any infrastructure row in EITHER arm is excluded from the
paired comparison for BOTH arms, but only if its same-environment canary failed (proving the
failure arm-independent); an infrastructure row whose canary passed refuses the analysis.
Requested and eligible denominators are reported separately. Fixed validity comes from row
evidence (``fixed_valid``), never from a class label alone; nondeterminism is never valid.

study_mode ``engineering_dress_rehearsal`` reports descriptive pipeline metrics only (no
intervals, significance, non-inferiority or promotion decision); ``confirmation`` runs the
frozen inferential analysis (repository-clustered bootstrap, leave-one-repository-out,
validity non-inferiority on the lower bound).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSIS_VERSION = "oneiros_native_generated_tests_analyse_v2"
ARMS = ("base", "sft")
SEEDS = (42, 43, 44)
SLOTS = 8
KILLS = {"semantic_kill", "crash_kill"}
INFRA = {"harness_failure", "dependency_failure", "environment_failure"}
PARSED_FAIL = {"syntax_failure", "over_limits", "policy_refused"}
COLLECT_FAIL = PARSED_FAIL | {"fabricated_import", "collection_failure", "no_tests_collected"}
EXEC_FAIL = COLLECT_FAIL | {"skipped_or_xfail", "timeout"}
REACH_FAIL = EXEC_FAIL | {"target_not_reached"}
BOOTSTRAP = {"resamples": 10_000, "seed": 20260930, "level": 0.95}
VALIDITY_MARGIN = 3.0
STUDY_MODES = ("engineering_dress_rehearsal", "confirmation")


class AnalysisRefused(ValueError):
    pass


def index(records: Iterable[Mapping[str, Any]], targets: Sequence[str]) -> Dict[tuple, dict]:
    grid: Dict[tuple, dict] = {}
    for r in records:
        key = (r["arm"], int(r["seed"]), r["target_key"], int(r["slot"]))
        if key in grid:
            raise AnalysisRefused(f"duplicate record {key}")
        grid[key] = dict(r)
    expected = {(a, s, t, k) for a in ARMS for s in SEEDS for t in targets for k in range(SLOTS)}
    if set(grid) != expected:
        missing, extra = expected - set(grid), set(grid) - expected
        raise AnalysisRefused(f"incomplete grid: {len(missing)} missing, {len(extra)} extra")
    return grid


def infrastructure_exclusions(grid, targets) -> Dict[str, Any]:
    excluded, refusals = [], []
    for t in targets:
        rows = [grid[(a, s, t, k)] for a in ARMS for s in SEEDS for k in range(SLOTS)]
        infra = [r for r in rows if r["class"] in INFRA]
        if not infra:
            continue
        if all(r.get("canary_failed") is True for r in infra):
            excluded.append(t)
        else:
            refusals.append(t)
    if refusals:
        raise AnalysisRefused(f"infrastructure rows without a failed canary (not proven "
                              f"arm-independent): {refusals[:5]}")
    return {"excluded_targets": excluded,
            "rule": "any infrastructure row in either arm excludes the target from both arms"}


def kill_at(grid, arm, seed, target, k) -> float:
    return float(any(grid[(arm, seed, target, i)]["class"] in KILLS for i in range(k)))


def fixed_valid(row) -> bool:
    return row.get("fixed_valid") is True and row["class"] != "nondeterminism"


def denominators(grid, requested, eligible) -> Dict[str, Dict[str, int]]:
    out = {}
    for arm in ARMS:
        req = [grid[(arm, s, t, i)] for s in SEEDS for t in requested for i in range(SLOTS)]
        cells = [grid[(arm, s, t, i)] for s in SEEDS for t in eligible for i in range(SLOTS)]
        classes = [c["class"] for c in cells]
        out[arm] = {"requested": len(req), "eligible": len(cells),
                    "infrastructure_excluded": len(req) - len(cells),
                    "parsed": sum(c not in PARSED_FAIL for c in classes),
                    "collected": sum(c not in COLLECT_FAIL for c in classes),
                    "executed": sum(c not in EXEC_FAIL for c in classes),
                    "reached": sum(c not in REACH_FAIL for c in classes),
                    "fixed_valid": sum(fixed_valid(c) for c in cells),
                    "semantic_kills": classes.count("semantic_kill"),
                    "crash_kills": classes.count("crash_kill"),
                    "classes": dict(Counter(classes))}
    return out


def clustered(diffs: Mapping[str, float], repo_of: Mapping[str, str]) -> Dict[str, Any]:
    repos = sorted({repo_of[t] for t in diffs})
    sums = np.array([sum(d for t, d in diffs.items() if repo_of[t] == r) for r in repos])
    counts = np.array([sum(1 for t in diffs if repo_of[t] == r) for r in repos])
    point = float(sums.sum() / counts.sum() * 100)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    pick = rng.integers(0, len(repos), size=(BOOTSTRAP["resamples"], len(repos)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1) * 100
    alpha = (1 - BOOTSTRAP["level"]) / 2 * 100
    low, high = np.percentile(draws, [alpha, 100 - alpha])
    loo = {r: round(float((sums.sum() - sums[i]) / (counts.sum() - counts[i]) * 100), 3)
           for i, r in enumerate(repos) if counts.sum() > counts[i]}
    return {"point": round(point, 3), "low": round(float(low), 3), "high": round(float(high), 3),
            "targets": int(counts.sum()), "repositories": len(repos),
            "leave_one_repository_out": loo}


def analyse(records: Iterable[Mapping[str, Any]], targets: Sequence[str],
            repo_of: Mapping[str, str], study_mode: str,
            atheris: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    if study_mode not in STUDY_MODES:
        raise AnalysisRefused(f"unknown study mode {study_mode!r}")
    targets = sorted(targets)
    grid = index(records, targets)
    infra = infrastructure_exclusions(grid, targets)
    eligible = [t for t in targets if t not in infra["excluded_targets"]]
    if not eligible:
        raise AnalysisRefused("no eligible targets")
    result: Dict[str, Any] = {"analysis_version": ANALYSIS_VERSION, "study_mode": study_mode,
                              "unit": "target (seeds averaged; never pooled)",
                              "requested_targets": len(targets), "eligible_targets": len(eligible),
                              "infrastructure": infra,
                              "denominators": denominators(grid, targets, eligible)}
    for k in (1, 4, 8):
        scores = {arm: {t: float(np.mean([kill_at(grid, arm, s, t, k) for s in SEEDS]))
                        for t in eligible} for arm in ARMS}
        entry = {arm: round(float(np.mean(list(scores[arm].values()))) * 100, 3) for arm in ARMS}
        entry["per_seed"] = {str(s): {arm: round(float(np.mean([kill_at(grid, arm, s, t, k)
                                                                 for t in eligible])) * 100, 3)
                                      for arm in ARMS} for s in SEEDS}
        if study_mode == "confirmation":
            entry["sft_minus_base_points"] = clustered(
                {t: scores["sft"][t] - scores["base"][t] for t in eligible}, repo_of)
        result[f"kill_at_{k}"] = entry
    result["unique_bugs_killed"] = {arm: sum(any(kill_at(grid, arm, s, t, SLOTS) for s in SEEDS)
                                             for t in eligible) for arm in ARMS}
    validity = {arm: {t: np.mean([fixed_valid(grid[(arm, s, t, i)]) for s in SEEDS
                                  for i in range(SLOTS)]) for t in eligible} for arm in ARMS}
    result["fixed_valid_rate"] = {arm: round(float(np.mean(list(validity[arm].values()))) * 100, 3)
                                  for arm in ARMS}
    if study_mode == "confirmation":
        result["primary"] = result["kill_at_8"]["sft_minus_base_points"]
        vi = clustered({t: validity["sft"][t] - validity["base"][t] for t in eligible}, repo_of)
        result["validity_non_inferiority"] = {
            **vi, "margin_points": -VALIDITY_MARGIN,
            "non_inferiority_shown": vi["low"] >= -VALIDITY_MARGIN,
            "operational_stop": vi["point"] < -VALIDITY_MARGIN and vi["low"] < -VALIDITY_MARGIN}
    else:
        result["decisions"] = ("SUPPRESSED: engineering dress rehearsal - descriptive pipeline "
                               "metrics only; no significance, non-inferiority or promotion")
    if atheris:
        rows = [a for a in atheris if a["target_key"] in eligible]
        joint = sorted({a["target_key"] for a in rows if a["eligible"]})
        result["atheris_jointly_eligible"] = {
            "targets": len(joint),
            "ineligible_reasons": dict(Counter(a.get("reason") for a in rows if not a["eligible"])),
            "oneiros_unique_kills": {arm: sum(any(kill_at(grid, arm, s, t, SLOTS) for s in SEEDS)
                                              for t in joint) for arm in ARMS},
            "atheris_unique_kills": {m: sum(any(a["kill"] for a in rows if a["eligible"]
                                                and a["mode"] == m and a["target_key"] == t)
                                            for t in joint)
                                     for m in sorted({a["mode"] for a in rows})},
            "replay_errors": sum(int(a.get("replay_errors") or 0) for a in rows)}
    result["root_cause_established"] = False
    result["generalization_established"] = False
    return result


def main(argv=None) -> int:
    from harness.atomic_publish import publish_file_atomically
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("analyse",))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--condition", required=True, choices=("primary_whole_module",))
    parser.add_argument("--study-mode", required=True, choices=STUDY_MODES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--atheris", default=None)
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    contracts = {r.get("contract_sha256") for r in rows}
    if len(contracts) != 1:
        raise AnalysisRefused("results mix execution contracts (stale rows)")
    targets = manifest["kept_targets"]
    repo_of = {t["key"]: t["repository"] for t in manifest["targets"]}
    atheris = []
    if args.atheris:
        atheris = [json.loads(l) for l in Path(args.atheris).read_text(encoding="utf-8")
                   .splitlines() if l.strip()]
    result = analyse(rows, targets, repo_of, args.study_mode, atheris)
    result["inputs"] = {name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for name, p in (("manifest", args.manifest), ("results", args.results),
                                        ("atheris", args.atheris)) if p}
    result["execution_contract_sha256"] = contracts.pop()
    result["condition"] = args.condition
    publish_file_atomically(Path(args.out), (json.dumps(result, indent=1, sort_keys=True)
                                             + "\n").encode("utf-8"))
    print(json.dumps({"out": args.out, "eligible_targets": result["eligible_targets"],
                      "study_mode": args.study_mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
