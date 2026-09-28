"""Phase 2 of the SFT root-cause loop: where in the pipeline does SFT change things?

Every quantity is a ratio of two sums (numerator, denominator) accumulated per
semantic cluster (``group_id``).  A paired cluster bootstrap resamples clusters,
with the SAME draws for both arms, and recomputes each ratio from the resampled
sums, so an interval reflects that mutants of one function are not independent.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260928

Row = Mapping[str, Any]


def _cand(predicate: Callable[[Row], bool], condition: Callable[[Row], bool]):
    return ("candidate", predicate, condition)


def _func(predicate: Callable[[list[Row]], bool]):
    return ("function", predicate, lambda rows: True)


def _killed_within(k: int) -> Callable[[list[Row]], bool]:
    return lambda rows: any(r["killed"] and r["rank"] <= k for r in rows)


def _executed(r: Row) -> bool:
    return bool(r["execution_valid"])


def _equality(r: Row) -> bool:
    return bool(r["assertion_forms"]) and r["assertion_forms"][0] == "compare_Eq"


#: name -> (level, numerator predicate, denominator predicate).  A candidate
#: metric counts rows with ``condition`` in the denominator and rows with both in
#: the numerator.  ``_unique`` variants exclude duplicates.
METRICS: dict[str, tuple] = {
    "p_parse": _cand(lambda r: r["parse_valid"], lambda r: True),
    "p_policy_valid_given_parse": _cand(lambda r: r["policy_valid"], lambda r: r["parse_valid"]),
    "p_execute_given_parse": _cand(_executed, lambda r: r["parse_valid"]),
    "p_disc_given_execute": _cand(lambda r: r["discriminating_input"], _executed),
    "p_correct_oracle_given_disc": _cand(
        lambda r: r["reference_valid"], lambda r: _executed(r) and r["discriminating_input"]),
    "p_correct_oracle_given_non_disc": _cand(
        lambda r: r["reference_valid"],
        lambda r: _executed(r) and not r["discriminating_input"]),
    "p_reference_valid": _cand(lambda r: r["reference_valid"], lambda r: True),
    "p_kill_given_reference_valid": _cand(lambda r: r["killed"], lambda r: r["reference_valid"]),
    "p_kill_given_valid_and_disc": _cand(
        lambda r: r["killed"], lambda r: r["reference_valid"] and r["discriminating_input"]),
    "p_disc_given_execute_unique": _cand(
        lambda r: r["discriminating_input"], lambda r: _executed(r) and not r["duplicate"]),
    "p_correct_oracle_given_disc_unique": _cand(
        lambda r: r["reference_valid"],
        lambda r: _executed(r) and r["discriminating_input"] and not r["duplicate"]),
    "duplicate_rate_given_parse": _cand(lambda r: r["duplicate"], lambda r: r["parse_valid"]),
    "test_function_shape_given_policy_valid": _cand(
        lambda r: r["candidate_shape"] == "test_function", lambda r: r["policy_valid"]),
    "equality_oracle_given_policy_valid": _cand(
        lambda r: bool(r["assertion_forms"]) and r["assertion_forms"][0] == "compare_Eq",
        lambda r: r["policy_valid"]),
    # Form-stratified oracle accuracy.  ADDED AFTER THE FIRST PHASE 2 LOOK: SFT
    # moved almost every assertion to exact equality, and a weak oracle (`!=`,
    # `>`) is "correct" far more often than an exact value, so the unstratified
    # conditional rate mixes value-prediction ability with assertion form.
    "p_correct_oracle_given_disc_equality": _cand(
        lambda r: r["reference_valid"],
        lambda r: _executed(r) and r["discriminating_input"] and _equality(r)),
    "p_correct_oracle_given_disc_equality_unique": _cand(
        lambda r: r["reference_valid"],
        lambda r: _executed(r) and r["discriminating_input"] and _equality(r)
        and not r["duplicate"]),
    "p_correct_oracle_given_disc_non_equality": _cand(
        lambda r: r["reference_valid"],
        lambda r: _executed(r) and r["discriminating_input"] and not _equality(r)),
    "kill_at_1": _func(_killed_within(1)),
    "kill_at_4": _func(_killed_within(4)),
    "kill_at_8": _func(_killed_within(8)),
    "function_disc_recall_at_8": _func(
        lambda rows: any(_executed(r) and r["discriminating_input"] for r in rows)),
    "function_has_valid_disc_candidate": _func(
        lambda rows: any(r["reference_valid"] and r["discriminating_input"] for r in rows)),
    "function_oracle_recoverable_unkilled": _func(
        lambda rows: not any(r["killed"] for r in rows) and any(
            r["terminal_category"] == "discriminating_wrong_oracle" for r in rows)),
}

#: Candidate-level sums that are not ratios over the same rows.
DIVERSITY = ("unique_code_per_parsed", "unique_call_signatures_per_parsed")

#: Metrics whose change is expressed in percentage points and compared with the
#: predeclared minimum meaningful effects.
THRESHOLDS = {"kill_at_8": 3.0, "p_correct_oracle_given_disc": 5.0,
              "p_correct_oracle_given_disc_unique": 5.0,
              "p_correct_oracle_given_disc_equality": 5.0,
              "p_correct_oracle_given_disc_equality_unique": 5.0}
VALIDITY_REGRESSION_LIMIT = 3.0


def group_functions(rows: Iterable[Row]) -> dict[str, list[Row]]:
    functions: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        functions[row["record_id"]].append(row)
    return functions


def cluster_sums(rows: Sequence[Row], clusters: Sequence[str]) -> np.ndarray:
    """Array [cluster, metric, (numerator, denominator)] for one arm."""
    index = {cluster: i for i, cluster in enumerate(clusters)}
    names = list(METRICS) + list(DIVERSITY)
    sums = np.zeros((len(clusters), len(names), 2))
    for record_rows in group_functions(rows).values():
        c = index[record_rows[0]["group_id"]]
        for m, name in enumerate(METRICS):
            level, predicate, condition = METRICS[name]
            if level == "function":
                sums[c, m, 0] += bool(predicate(record_rows))
                sums[c, m, 1] += 1
            else:
                for row in record_rows:
                    if condition(row):
                        sums[c, m, 1] += 1
                        sums[c, m, 0] += bool(predicate(row))
        parsed = [r for r in record_rows if r["parse_valid"]]
        base = len(METRICS)
        sums[c, base, 0] += len({r["normalised_code_sha256"] for r in parsed})
        sums[c, base, 1] += len(parsed)
        sums[c, base + 1, 0] += len({tuple(r["call_signatures"]) for r in parsed})
        sums[c, base + 1, 1] += len(parsed)
    return sums


def _ratio(sums: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums[..., 0] / sums[..., 1]


def _region(low: float, high: float, threshold: float | None) -> str:
    if threshold is not None and low >= threshold:
        return "supported"
    if low > 0:
        return "directional"
    if high < 0:
        return "reversed"
    return "null_compatible"


def paired_comparison(rows_a: Sequence[Row], rows_b: Sequence[Row],
                      resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """B minus A for every metric, with paired cluster-bootstrap intervals."""
    clusters = sorted({r["group_id"] for r in rows_a} | {r["group_id"] for r in rows_b})
    a, b = cluster_sums(rows_a, clusters), cluster_sums(rows_b, clusters)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(resamples, len(clusters)))
    weights = np.zeros((resamples, len(clusters)))
    np.add.at(weights, (np.repeat(np.arange(resamples), len(clusters)), draws.ravel()), 1)
    shape = a.shape[1:]
    boot_a = _ratio((weights @ a.reshape(len(clusters), -1)).reshape(resamples, *shape))
    boot_b = _ratio((weights @ b.reshape(len(clusters), -1)).reshape(resamples, *shape))
    point_a, point_b = _ratio(a.sum(axis=0)), _ratio(b.sum(axis=0))
    out = {}
    for m, name in enumerate(list(METRICS) + list(DIVERSITY)):
        diff = (boot_b[:, m] - boot_a[:, m]) * 100
        diff = diff[~np.isnan(diff)]
        low, high = (np.percentile(diff, [2.5, 97.5]) if len(diff) else (np.nan, np.nan))
        out[name] = {
            "a": round(float(point_a[m]), 6), "b": round(float(point_b[m]), 6),
            "a_counts": [int(a[:, m, 0].sum()), int(a[:, m, 1].sum())],
            "b_counts": [int(b[:, m, 0].sum()), int(b[:, m, 1].sum())],
            "difference_points": round(float((point_b[m] - point_a[m]) * 100), 4),
            "ci95_points": [round(float(low), 4), round(float(high), 4)],
            "bootstrap_se_points": round(float(np.std(diff)), 4) if len(diff) else None,
            "region": _region(float(low), float(high), THRESHOLDS.get(name)),
        }
    return {"clusters": len(clusters), "resamples": resamples, "seed": seed, "metrics": out}


def function_transitions(rows_a: Sequence[Row], rows_b: Sequence[Row]) -> dict[str, Any]:
    """Paired function-level transitions A -> B for the mediators that matter."""
    fa, fb = group_functions(rows_a), group_functions(rows_b)
    shared = sorted(set(fa) & set(fb))
    checks = {
        "killed": _killed_within(8),
        "has_executed_disc_candidate": METRICS["function_disc_recall_at_8"][1],
        "has_valid_disc_candidate": METRICS["function_has_valid_disc_candidate"][1],
    }
    out: dict[str, Any] = {"paired_functions": len(shared)}
    for name, check in checks.items():
        cells = {"both": 0, "neither": 0, "gained": 0, "lost": 0}
        for record in shared:
            x, y = bool(check(fa[record])), bool(check(fb[record]))
            cells["both" if x and y else "neither" if not (x or y)
                  else "gained" if y else "lost"] += 1
        out[name] = cells
    return out


def category_distribution(rows: Sequence[Row], categories: Sequence[str]) -> dict[str, Any]:
    total = len(rows)
    counts = {c: sum(1 for r in rows if r["terminal_category"] == c) for c in categories}
    return {"candidates": total, "counts": counts,
            "shares": {c: round(v / total, 6) if total else None for c, v in counts.items()}}
