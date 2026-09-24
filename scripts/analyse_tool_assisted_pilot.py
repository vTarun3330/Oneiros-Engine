"""Frozen analysis for the execution-feedback pilot (written before any generation).

Primary comparison: arm C (execution-feedback repair) versus arm B (sham
feedback: a call-, sequence-, cap- and rendered-input-token-matched control),
paired by record, 90% percentile intervals from
a lineage-cluster bootstrap (10,000 replicates, fixed seed).

PASS requires all of:
* Kill@8 gain C - B >= +5 pp (the project's existing minimum practically
  important improvement) and its bootstrap lower bound > 0;
* reference-valid candidates per requested candidate: C - B lower bound
  >= -3 pp (the project's existing noninferiority margin);
* candidate diversity (exact-unique ratio): C - B mean >= -0.05;
* wall-clock: C <= 1.5 x B;
* a request-budget-matched pair on every target: equal model calls, sequences
  and max-new-token caps, and for every C repair a B sham that echoes the same
  parent with an identical rendered input-token count.

FAIL when the Kill@8 upper bound is below +5 pp (the minimum gain is
excluded), or the reference-validity upper bound is below -3 pp.
Anything else is INCONCLUSIVE and does not pass.

A syntax-only improvement cannot pass: the gate is on Kill@8, a semantic
outcome, and parse/execution rates are reported but never gate.  If compute
pairing fails anywhere, the analysis refuses and the comparison is not called
controlled.  Actual output tokens and wall-clock time are measured outcomes;
their ratios are reported and are never described as equal compute.  Arm A (canonical model-only control) comparisons are secondary.
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


def pairing_problems(data: Mapping[str, Any]) -> list[str]:
    """Every reason one target's B/C pair is not a request-budget-matched pair."""
    from harness.execution_feedback import CATEGORIES, _TEMPLATES, sham_is_neutral

    problems: list[str] = []
    record_id = data["record_id"]
    budget = data["budget"]
    for field in ("sequences", "model_calls"):
        if budget[field]["B"] != budget[field]["C"]:
            problems.append(f"{record_id}: {field} differ")
    round1 = {c["slot"]: c for c in data["candidates"]["C"] if c["round"] == 1}
    shared = {c["slot"]: c for c in data["candidates"]["B"] if c["round"] == 1}
    for slot, candidate in round1.items():
        if candidate["raw_output"] != shared[slot]["raw_output"]:
            problems.append(f"{record_id}: round-1 slot {slot} is not shared")
    c2 = {c["slot"]: c for c in data["candidates"]["C"] if c["round"] == 2}
    b2 = {c["slot"]: c for c in data["candidates"]["B"] if c["round"] == 2}
    if sorted(c2) != sorted(b2):
        problems.append(f"{record_id}: round-2 slots differ")
        return problems
    feedback_fragments = [text.split("{")[0][:30].lower() for text in _TEMPLATES.values()]
    repair_slots = []
    for slot in sorted(c2):
        c, b = c2[slot], b2[slot]
        if c["status"] == "repair":
            repair_slots.append(slot)
            parent = round1[slot]
            echo = parent["code"] if parent["code"] else parent["raw_output"]
            expected_parent = f"{record_id}|r1|{slot}"
            if b["status"] != "sham":
                problems.append(f"{record_id}: repair slot {slot} has no sham")
                continue
            if not (c["parent_candidate"] == b["parent_candidate"] == expected_parent):
                problems.append(f"{record_id}: slot {slot} echoes different parents")
            if not (c["input_tokens"] == b["input_tokens"]
                    == (c["matched_input_tokens"] or {}).get("repair")
                    == (b["matched_input_tokens"] or {}).get("sham")):
                problems.append(f"{record_id}: slot {slot} rendered input tokens differ")
            addition = str(b["prompt_addition"] or "")
            if not sham_is_neutral(addition, echo):
                problems.append(f"{record_id}: slot {slot} sham is not the neutral template")
            echoed = echo.strip()[:1500]
            lowered = (addition.split(echoed, 1)[1] if echoed in addition
                       else addition).lower()
            if any(fragment and fragment in lowered for fragment in feedback_fragments) or any(
                    category in lowered for category in CATEGORIES):
                problems.append(f"{record_id}: slot {slot} sham carries diagnostic text")
            if not str(c["prompt_addition"] or "").startswith(
                    "You previously answered:\n\n" + echo.strip()[:1500]):
                problems.append(f"{record_id}: slot {slot} repair echo differs")
        else:
            if b["status"] != "resample" or c["status"] != "resample":
                problems.append(f"{record_id}: slot {slot} is unmatched ({b['status']}/"
                                f"{c['status']})")
            elif (b["raw_output"] != c["raw_output"] or b["input_tokens"] != c["input_tokens"]):
                problems.append(f"{record_id}: non-repair slot {slot} is not shared")
    if repair_slots != sorted(budget["repairs_delivered"]):
        problems.append(f"{record_id}: delivered repairs do not match repair slots")
    for slot in budget["repairs_skipped_match_infeasible"]:
        if c2[int(slot)]["status"] != "resample" or b2[int(slot)]["status"] != "resample":
            problems.append(f"{record_id}: skipped slot {slot} was not a shared fallback")
    return problems


def loop_accounting(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Pairing integrity, request budgets and measured outcomes per arm."""
    totals = {arm: Counter() for arm in ("B", "C")}
    transitions: Counter = Counter()
    problems: list[str] = []
    caps = set()
    attempted = delivered = skipped = 0
    for record_id in panel["record_ids"]:
        target = OUTPUT_DIR / "loop" / (hashlib.sha256(record_id.encode()).hexdigest() + ".json")
        data = json.loads(target.read_text(encoding="utf-8"))
        budget = data["budget"]
        caps.add(budget["max_new_tokens"])
        problems.extend(pairing_problems(data))
        attempted += len(budget["repairs_attempted"])
        delivered += len(budget["repairs_delivered"])
        skipped += len(budget["repairs_skipped_match_infeasible"])
        for arm in ("B", "C"):
            totals[arm]["sequences"] += budget["sequences"][arm]
            totals[arm]["model_calls"] += budget["model_calls"][arm]
            totals[arm]["input_tokens"] += budget["tokens"][arm]["input"]
            totals[arm]["output_tokens"] += budget["tokens"][arm]["output"]
            totals[arm]["generation_wall_seconds"] += sum(
                c["wall_seconds"] for c in data["candidates"][arm])
            totals[arm]["duplicates_in_final"] += len(data["final"][arm]) - len(
                {c["code"] for c in data["final"][arm] if c["code"]})
        children = {c["slot"]: c for c in data["candidates"]["C"] if c["round"] == 2}
        for parent in data["candidates"]["C"]:
            if parent["round"] == 1 and children[parent["slot"]]["status"] == "repair":
                transitions[f"{parent['category']} -> {children[parent['slot']]['category']}"] += 1
    if len(caps) != 1:
        problems.append(f"max-new-token caps differ across targets: {sorted(caps)}")
    b, c = totals["B"], totals["C"]
    return {
        "control": "call-, sequence-, cap- and rendered-input-token-matched sham-feedback "
                   "control (request-budget-matched)",
        "request_budget_matched": not problems,
        "pairing_problems": problems[:50],
        "pairing_problem_count": len(problems),
        "max_new_tokens": sorted(caps),
        "totals": {arm: dict(value) for arm, value in totals.items()},
        "input_tokens": {"B": b["input_tokens"], "C": c["input_tokens"]},
        "output_tokens": {"B": b["output_tokens"], "C": c["output_tokens"],
                          "ratio_c_over_b": c["output_tokens"] / max(b["output_tokens"], 1)},
        "wall_seconds": {"B": b["generation_wall_seconds"], "C": c["generation_wall_seconds"],
                         "ratio_c_over_b": c["generation_wall_seconds"]
                         / max(b["generation_wall_seconds"], 1e-9)},
        "repairs": {"attempted": attempted, "delivered": delivered,
                    "skipped_match_infeasible": skipped},
        "measured_outcomes_not_matched": ["actual output tokens (early EOS is a model "
                                          "outcome)", "generation wall-clock time"],
        "repair_category_transitions": dict(transitions.most_common()),
    }


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
    if not accounting["request_budget_matched"]:
        print(f"REFUSED: B and C are not request-budget-matched "
              f"({accounting['pairing_problem_count']} problems); not a controlled comparison")
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
    wall_ratio = accounting["wall_seconds"]["ratio_c_over_b"]
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
        "limitations": ["actual output tokens and wall-clock time are measured outcomes and "
                        "may differ between B and C; the output-token ratio is reported, not "
                        "matched"],
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
