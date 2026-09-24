"""Frozen analysis for the unordered-event and ordered-trace pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = ("intended_output", "shown_actual_output")
MIN_GAIN_PP = 5.0
MAX_STRICT_ANSWER_REGRESSION_PP = 2.0


def paired(pairs, z=1.644853627):
    n = len(pairs)
    gain = sum(1 for a, b in pairs if b and not a)
    loss = sum(1 for a, b in pairs if a and not b)
    diff = (gain - loss) / n
    var = (gain + loss - (gain - loss) ** 2 / n) / (n * n)
    half = z * math.sqrt(max(var, 0.0))
    return {"difference_pp": 100 * diff, "ci90_low_pp": 100 * (diff-half),
            "ci90_high_pp": 100 * (diff+half), "gained": gain, "lost": loss}


def _load(path: Path, arm: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or data.get("arm") != arm:
        raise ValueError(f"invalid artifact for {arm}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--unordered-events", type=Path, required=True)
    parser.add_argument("--ordered-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"control": args.control, "unordered_events": args.unordered_events,
             "ordered_trace": args.ordered_trace}
    data = {arm: _load(path, arm) for arm, path in paths.items()}
    bound = ("model", "model_revision", "pilot_development_sha256", "items", "conditions")
    for field in bound:
        if len({json.dumps(value.get(field), sort_keys=True) for value in data.values()}) != 1:
            raise ValueError(f"artifacts differ on frozen field {field}")
    indexed = {arm: {condition: {row["record_id"]: row
                                 for row in value["detail"][condition]}
                     for condition in CONDITIONS}
               for arm, value in data.items()}
    ids = sorted(indexed["control"]["intended_output"])
    for arm in data:
        for condition in CONDITIONS:
            if sorted(indexed[arm][condition]) != ids:
                raise ValueError(f"panel mismatch for {arm}/{condition}")
    comparisons = {}
    gates = {}
    for arm in ("unordered_events", "ordered_trace"):
        comparisons[arm] = {}
        for condition in CONDITIONS:
            pairs = [(
                indexed["control"][condition][item]["lenient"]["verdict"] == "correct",
                indexed[arm][condition][item]["lenient"]["verdict"] == "correct",
            ) for item in ids]
            comparisons[arm][condition] = paired(pairs)
        strict_change = 100 * (
            data[arm]["summary"]["intended_output"]["strict_answer_rate"]
            - data["control"]["summary"]["intended_output"]["strict_answer_rate"]
        )
        gates[arm] = {
            "intended_semantic_gain": comparisons[arm]["intended_output"]["difference_pp"] >= MIN_GAIN_PP,
            "intended_direction_90ci": comparisons[arm]["intended_output"]["ci90_low_pp"] >= 0,
            "actual_execution_gain": comparisons[arm]["shown_actual_output"]["difference_pp"] >= MIN_GAIN_PP,
            "actual_direction_90ci": comparisons[arm]["shown_actual_output"]["ci90_low_pp"] >= 0,
            "strict_answer_noninferiority": strict_change >= -MAX_STRICT_ANSWER_REGRESSION_PP,
            "no_completion_limit_hits": all(
                data[arm]["summary"][condition]["completion_limit_hits"] == 0
                for condition in CONDITIONS
            ),
        }
        gates[arm]["passed"] = all(gates[arm].values())
    order_comparison = {
        condition: paired([(
            indexed["unordered_events"][condition][item]["lenient"]["verdict"] == "correct",
            indexed["ordered_trace"][condition][item]["lenient"]["verdict"] == "correct",
        ) for item in ids]) for condition in CONDITIONS
    }
    passed = [arm for arm in gates if gates[arm]["passed"]]
    report = {
        "schema_version": "oneiros_execution_trace_mechanism_analysis_v1",
        "label": "train-derived mechanism diagnostic; not generalisation or model selection",
        "items": len(ids),
        "predeclared_gates": {"minimum_lenient_gain_pp_each_condition": MIN_GAIN_PP,
                              "maximum_strict_answer_regression_pp": MAX_STRICT_ANSWER_REGRESSION_PP,
                              "paired_ci": "90%, lower bound must be non-negative"},
        "inputs": {arm: {"path": str(path), "sha256": hashlib.sha256(
            path.read_bytes()).hexdigest()} for arm, path in paths.items()},
        "arms": {arm: value["summary"] for arm, value in data.items()},
        "comparisons_vs_control": comparisons,
        "ordered_minus_unordered": order_comparison,
        "gate_checks": gates,
        "mechanism_gate_passed": bool(passed),
        "passing_arms": passed,
        "next_step": (
            "freeze a separate train-derived canonical Kill@8 noninferiority protocol"
            if passed else
            "stop this intervention; retain prior control and do not open confirmation"
        ),
        "promotion_permitted": False,
        "confirmation_opening_permitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
