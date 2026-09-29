"""Frozen analysis for native generated tests (protocol v2, sections 2 and 8). Pure CPU.

Unit of analysis: the TARGET. For model m, seed s and target t, kill(t, s, m) = 1 if any of
the first k candidate slots is a kill. The target-level score is K(t, m) = mean over the
three seeds. The primary estimand is the mean over targets of K(t, SFT) - K(t, base).
Target x seed observations are never pooled as independent samples. Inference: a two-sided
95% percentile interval from a repository-clustered paired bootstrap (10,000 resamples, seed
20260930), plus leave-one-repository-out sensitivity. The grid must be complete (every
target x seed x arm has exactly 8 slots) or the analysis refuses.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ANALYSIS_VERSION = "oneiros_native_generated_tests_analyse_v1"
ARMS = ("base", "sft")
SEEDS = (42, 43, 44)
SLOTS = 8
KILLS = {"semantic_kill", "crash_kill"}
FIXED_VALID = {"pass_both", "semantic_kill", "crash_kill", "buggy_test_error", "nondeterminism"}
INFRA = {"harness_failure", "dependency_failure", "environment_failure"}
PARSED_FAIL = {"syntax_failure", "over_limits"}
COLLECT_FAIL = PARSED_FAIL | {"fabricated_import", "collection_failure", "no_tests_collected"}
EXEC_FAIL = COLLECT_FAIL | {"skipped_or_xfail", "timeout"}
REACH_FAIL = EXEC_FAIL | {"target_not_reached"}
BOOTSTRAP = {"resamples": 10_000, "seed": 20260930, "level": 0.95}
VALIDITY_MARGIN = 3.0


class AnalysisRefused(ValueError):
    pass


def index(records: Iterable[Mapping[str, Any]], targets: Sequence[str]) -> Dict[tuple, str]:
    grid: Dict[tuple, str] = {}
    for r in records:
        key = (r["arm"], int(r["seed"]), r["target_key"], int(r["slot"]))
        if key in grid:
            raise AnalysisRefused(f"duplicate record {key}")
        grid[key] = r["class"]
    expected = {(a, s, t, k) for a in ARMS for s in SEEDS for t in targets for k in range(SLOTS)}
    if set(grid) != expected:
        missing, extra = expected - set(grid), set(grid) - expected
        raise AnalysisRefused(f"incomplete grid: {len(missing)} missing, {len(extra)} extra")
    return grid


def kill_at(grid, arm, seed, target, k) -> float:
    return float(any(grid[(arm, seed, target, i)] in KILLS for i in range(k)))


def target_scores(grid, targets, k) -> Dict[str, Dict[str, float]]:
    return {arm: {t: float(np.mean([kill_at(grid, arm, s, t, k) for s in SEEDS]))
                  for t in targets} for arm in ARMS}


def fixed_valid_rate(grid, arm, target) -> float:
    cells = [grid[(arm, s, target, i)] for s in SEEDS for i in range(SLOTS)]
    return sum(c in FIXED_VALID for c in cells) / len(cells)


def clustered(diffs: Mapping[str, float], repo_of: Mapping[str, str]) -> Dict[str, Any]:
    """Repository-clustered bootstrap of the mean target-level difference (in points)."""
    repos = sorted({repo_of[t] for t in diffs})
    by_repo = {r: [diffs[t] for t in diffs if repo_of[t] == r] for r in repos}
    sums = np.array([sum(v) for v in by_repo.values()])
    counts = np.array([len(v) for v in by_repo.values()])
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


def denominators(grid, targets) -> Dict[str, Dict[str, int]]:
    out = {}
    for arm in ARMS:
        cells = [grid[(arm, s, t, i)] for s in SEEDS for t in targets for i in range(SLOTS)]
        out[arm] = {"requested": len(cells),
                    "parsed": sum(c not in PARSED_FAIL for c in cells),
                    "collected": sum(c not in COLLECT_FAIL for c in cells),
                    "executed": sum(c not in EXEC_FAIL for c in cells),
                    "reached": sum(c not in REACH_FAIL for c in cells),
                    "fixed_valid": sum(c in FIXED_VALID for c in cells),
                    "semantic_kills": sum(c == "semantic_kill" for c in cells),
                    "crash_kills": sum(c == "crash_kill" for c in cells),
                    "infrastructure_excluded": sum(c in INFRA for c in cells),
                    "classes": dict(Counter(cells))}
    return out


def analyse(records: Iterable[Mapping[str, Any]], targets: Sequence[str],
            repo_of: Mapping[str, str], atheris: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    targets = sorted(targets)
    grid = index(records, targets)
    result: Dict[str, Any] = {"analysis_version": ANALYSIS_VERSION,
                              "unit": "target (seeds averaged; never pooled)",
                              "denominators": denominators(grid, targets)}
    for k in (1, 4, 8):
        scores = target_scores(grid, targets, k)
        diffs = {t: scores["sft"][t] - scores["base"][t] for t in targets}
        result[f"kill_at_{k}"] = {
            "base": round(float(np.mean(list(scores["base"].values()))) * 100, 3),
            "sft": round(float(np.mean(list(scores["sft"].values()))) * 100, 3),
            "sft_minus_base_points": clustered(diffs, repo_of),
            "per_seed": {str(s): {arm: round(float(np.mean([kill_at(grid, arm, s, t, k)
                                                            for t in targets])) * 100, 3)
                                  for arm in ARMS} for s in SEEDS}}
    result["primary"] = result["kill_at_8"]["sft_minus_base_points"]
    validity = {t: fixed_valid_rate(grid, "sft", t) - fixed_valid_rate(grid, "base", t)
                for t in targets}
    vi = clustered(validity, repo_of)
    result["validity_non_inferiority"] = {
        **vi, "margin_points": -VALIDITY_MARGIN,
        "non_inferiority_shown": vi["low"] >= -VALIDITY_MARGIN,
        "operational_stop": vi["point"] < -VALIDITY_MARGIN and vi["low"] < -VALIDITY_MARGIN}
    result["unique_bugs_killed"] = {arm: sum(any(kill_at(grid, arm, s, t, SLOTS) for s in SEEDS)
                                             for t in targets) for arm in ARMS}
    if atheris:
        eligible = sorted({a["target_key"] for a in atheris if a["eligible"]} & set(targets))
        modes = sorted({a["mode"] for a in atheris if a["eligible"]})
        result["atheris_jointly_eligible"] = {
            "targets": len(eligible),
            "ineligible_reasons": dict(Counter(a.get("reason") for a in atheris
                                               if not a["eligible"])),
            "oneiros_unique_kills": {arm: sum(any(kill_at(grid, arm, s, t, SLOTS) for s in SEEDS)
                                              for t in eligible) for arm in ARMS},
            "atheris_unique_kills": {m: sum(any(a["kill"] for a in atheris if a["eligible"]
                                                and a["mode"] == m and a["target_key"] == t)
                                            for t in eligible) for m in modes}}
    return result
