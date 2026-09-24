"""Frozen analysis for the execution-feedback pilot (written before any generation).

Primary comparison: arm C (execution-feedback repair) versus arm B
(compute-matched resampling), paired by record, 90% percentile intervals from
a lineage-cluster bootstrap (10,000 replicates, fixed seed).

PASS requires all of:
* Kill@8 gain C - B >= +5 pp (the project's existing minimum practically
  important improvement) and its bootstrap lower bound > 0;
* reference-valid candidates per requested candidate: C - B lower bound
  >= -3 pp (the project's existing noninferiority margin);
* candidate diversity (exact-unique ratio): C - B mean >= -0.05;
* wall-clock: C <= 1.5 x B;
* exact compute matching (identical sequences and token cap per target).

FAIL when the Kill@8 upper bound is below +5 pp (the minimum gain is
excluded), or the reference-validity upper bound is below -3 pp.
Anything else is INCONCLUSIVE and does not pass.

A syntax-only improvement cannot pass: the gate is on Kill@8, a semantic
outcome, and parse/execution rates are reported but never gate.  If compute
matching fails, the analysis refuses and the comparison is not called
controlled.  Arm A (canonical model-only control) comparisons are secondary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "oneiros_tool_assisted_analysis_v1"
MIN_GAIN_PP = 5.0
VALIDITY_MARGIN_PP = 3.0
MAX_DIVERSITY_LOSS = 0.05
MAX_WALL_RATIO = 1.5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260925
OUTPUT_DIR = ROOT / "results" / "v4_3_tool_assisted_v1"
PANEL = ROOT / "results" / "v4_3_tool_assisted_panel.json"


def cluster_bootstrap(pairs: Sequence[tuple[str, float, float]], *, replicates: int,
                      seed: int, confidence: float = 0.90) -> dict[str, Any]:
    """Percentile interval for mean(b) - mean(a), resampling whole lineages."""
    clusters: dict[str, list[float]] = {}
    for cluster, first, second in pairs:
        clusters.setdefault(str(cluster), []).append(float(second) - float(first))
    keys = sorted(clusters)
    sums = [(sum(clusters[key]), len(clusters[key])) for key in keys]
    units = sum(size for _, size in sums)
    point = sum(delta for delta, _ in sums) / units
    generator = random.Random(seed)
    draws = []
    for _ in range(replicates):
        delta = size = 0.0
        for _ in keys:
            d, n = sums[generator.randrange(len(sums))]
            delta += d
            size += n
        draws.append(delta / size)
    draws.sort()
    tail = (1 - confidence) / 2
    return {"difference": point, "low": draws[max(0, math.floor(tail * replicates))],
            "high": draws[min(replicates - 1, math.ceil((1 - tail) * replicates) - 1)],
            "clusters": len(keys), "units": units, "replicates": replicates, "seed": seed}


def record_metrics(item: Mapping[str, Any]) -> dict[str, float]:
    outcomes = item["candidate_outcomes"]
    requested = len(outcomes)

    def prefix(k: int, field: str) -> float:
        return float(any(o.get(field) for o in outcomes if int(o.get("rank", 0)) <= k))
    return {
        "kill_at_1": prefix(1, "killed"), "kill_at_4": prefix(4, "killed"),
        "kill_at_8": prefix(8, "killed"),
        "reference_valid_per_requested": sum(bool(o.get("reference_valid"))
                                             for o in outcomes) / requested,
        "function_has_reference_valid": prefix(8, "reference_valid"),
        "parse_success": sum(bool(o.get("parse_valid")) for o in outcomes) / requested,
        "execution_success": sum(bool(o.get("execution_valid")) for o in outcomes) / requested,
        "exact_unique_ratio": float((item.get("diversity") or {}).get(
            "exact_unique_ratio", 0.0)),
    }


def load_arm(arm: str, panel: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    envelope = json.loads((OUTPUT_DIR / f"eval_{arm}.json").read_text(encoding="utf-8"))
    if envelope.get("status") != "complete" or envelope.get("arm") != arm:
        raise ValueError(f"arm {arm} artifact is not complete")
    if envelope["panel_record_ids_sha256"] != panel["record_ids_sha256"]:
        raise ValueError(f"arm {arm} ran on a different panel")
    path = ROOT / envelope["rehearsal_result"]["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != envelope["rehearsal_result"]["sha256"]:
        raise ValueError(f"arm {arm} raw result hash mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result["raw_output_integrity"]["complete"]:
        raise ValueError(f"arm {arm} raw-output integrity failed")
    metrics = {str(item["record_id"]): record_metrics(item)
               for item in result["function_results"]}
    if sorted(metrics) != sorted(panel["record_ids"]):
        raise ValueError(f"arm {arm} does not cover exactly the frozen panel")
    return metrics


def loop_accounting(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Compute matching and runtime from the per-target lineage files."""
    totals = {arm: Counter() for arm in ("B", "C")}
    transitions: Counter = Counter()
    unmatched = []
    caps = set()
    for record_id in panel["record_ids"]:
        target = OUTPUT_DIR / "loop" / (hashlib.sha256(record_id.encode()).hexdigest() + ".json")
        data = json.loads(target.read_text(encoding="utf-8"))
        budget = data["budget"]
        caps.add(budget["max_new_tokens"])
        if (budget["sequences"]["B"] != budget["sequences"]["C"]
                or budget["model_calls"]["B"] != budget["model_calls"]["C"]):
            unmatched.append(record_id)
        for arm in ("B", "C"):
            totals[arm]["sequences"] += budget["sequences"][arm]
            totals[arm]["model_calls"] += budget["model_calls"][arm]
            totals[arm]["input_tokens"] += budget["tokens"][arm]["input"]
            totals[arm]["output_tokens"] += budget["tokens"][arm]["output"]
            totals[arm]["generation_wall_seconds"] += sum(
                c["wall_seconds"] for c in data["candidates"][arm])
            totals[arm]["duplicates_in_final"] += len(data["final"][arm]) - len(
                {c["code"] for c in data["final"][arm] if c["code"]})
        totals["C"]["repairs"] += budget["repairs"]
        totals["C"]["repairs_undelivered"] += len(
            budget.get("repairs_undelivered_prompt_over_budget", []))
        children = {c["slot"]: c for c in data["candidates"]["C"] if c["round"] == 2}
        for parent in data["candidates"]["C"]:
            if parent["round"] == 1 and children[parent["slot"]]["feedback"] is not None:
                transitions[f"{parent['category']} -> {children[parent['slot']]['category']}"] += 1
    return {"totals": {arm: dict(value) for arm, value in totals.items()},
            "unmatched_targets": unmatched,
            "compute_matched": not unmatched and len(caps) == 1,
            "max_new_tokens": sorted(caps),
            "repair_category_transitions": dict(transitions.most_common())}


def compare(first: dict, second: dict, lineages: Mapping[str, str], metric: str) -> dict:
    ids = sorted(first)
    return cluster_bootstrap([(lineages[i], first[i][metric], second[i][metric]) for i in ids],
                             replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)


def verdict(kill: Mapping[str, float], validity: Mapping[str, float], diversity: float,
            wall_ratio: float) -> str:
    """Three-way outcome on percentage-point intervals (inputs already in pp)."""
    if kill["high"] < MIN_GAIN_PP or validity["high"] < -VALIDITY_MARGIN_PP:
        return "fail"
    if (kill["difference"] >= MIN_GAIN_PP and kill["low"] > 0
            and validity["low"] >= -VALIDITY_MARGIN_PP and diversity >= -MAX_DIVERSITY_LOSS
            and wall_ratio <= MAX_WALL_RATIO):
        return "pass"
    return "inconclusive"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_tool_assisted_analysis.json")
    args = parser.parse_args(argv)
    missing = [arm for arm in ("A", "B", "C") if not (OUTPUT_DIR / f"eval_{arm}.json").exists()]
    if missing:
        print(f"REFUSED: arms not yet complete: {missing}; no partial analysis")
        return 2
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    lineages = panel["record_lineages"]
    arms = {arm: load_arm(arm, panel) for arm in ("A", "B", "C")}
    accounting = loop_accounting(panel)
    if not accounting["compute_matched"]:
        print("REFUSED: compute is not matched between B and C; not a controlled comparison")
        return 2
    metrics = ("kill_at_1", "kill_at_4", "kill_at_8", "reference_valid_per_requested",
               "function_has_reference_valid", "parse_success", "execution_success",
               "exact_unique_ratio")

    def to_pp(interval: dict) -> dict:
        return {**interval, **{key: 100 * interval[key]
                               for key in ("difference", "low", "high")}}
    primary = {metric: compare(arms["B"], arms["C"], lineages, metric) for metric in metrics}
    for metric in metrics:
        if metric != "exact_unique_ratio":
            primary[metric] = to_pp(primary[metric])
    secondary = {f"{second}_minus_{first}": {m: to_pp(compare(arms[first], arms[second],
                                                              lineages, m))
                                             for m in ("kill_at_8",
                                                       "reference_valid_per_requested")}
                 for first, second in (("A", "B"), ("A", "C"))}
    totals = accounting["totals"]
    wall_ratio = (totals["C"]["generation_wall_seconds"]
                  / max(totals["B"]["generation_wall_seconds"], 1e-9))
    outcome = verdict(primary["kill_at_8"], primary["reference_valid_per_requested"],
                      primary["exact_unique_ratio"]["difference"], wall_ratio)
    means = {arm: {metric: sum(values[metric] for values in data.values()) / len(data)
                   for metric in metrics} for arm, data in arms.items()}
    report = {
        "schema_version": SCHEMA,
        "label": "train-derived exploratory pilot on a previously base-inspected panel; "
                 "not generalisation, not confirmatory, not a final-test result",
        "predeclared": {"min_gain_pp": MIN_GAIN_PP, "validity_margin_pp": VALIDITY_MARGIN_PP,
                        "max_diversity_loss": MAX_DIVERSITY_LOSS,
                        "max_wall_ratio": MAX_WALL_RATIO,
                        "interval": f"lineage-cluster percentile bootstrap, 90%, "
                                    f"{BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}"},
        "arm_means": means,
        "unique_defects_killed": {arm: int(sum(v["kill_at_8"] for v in data.values()))
                                  for arm, data in arms.items()},
        "primary_c_minus_b": primary,
        "secondary": secondary,
        "accounting": accounting,
        "wall_ratio_c_over_b": wall_ratio,
        "verdict": outcome,
        "decision": {
            "pass": "stop; request explicit authorization to design a confirmation on "
                    "untouched data; nothing is opened automatically",
            "fail": "stop this line; no re-thresholding, extra rounds or prompt changes",
            "inconclusive": "stop; not a pass; any follow-up needs a new explicit decision",
        }[outcome],
        "promotion_permitted": False,
        "confirmation_opening_permitted": False,
    }
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"verdict": outcome, "kill_at_8": primary["kill_at_8"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
