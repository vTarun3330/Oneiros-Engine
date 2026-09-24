"""Analyse the frozen train-only execution-supervision mechanism pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS = ("base", "control", "treatment")
CONDITIONS = ("intended_output", "shown_actual_output")

# Declared before either adapter was trained or evaluated.
MIN_INTENDED_GAIN_PP = 5.0
MAX_ACTUAL_REGRESSION_PP = 5.0
MAX_ANSWER_RATE_REGRESSION_PP = 2.0


def paired_difference(pairs: list[tuple[bool, bool]], z: float = 1.644853627):
    n = len(pairs)
    gained = sum(1 for left, right in pairs if right and not left)
    lost = sum(1 for left, right in pairs if left and not right)
    difference = (gained - lost) / n
    variance = (gained + lost - (gained - lost) ** 2 / n) / (n * n)
    half = z * math.sqrt(max(variance, 0.0))
    return 100 * difference, 100 * (difference - half), 100 * (difference + half), gained, lost


def mcnemar_exact(gained: int, lost: int) -> float:
    discordant = gained + lost
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def _load(path: Path, expected_arm: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or data.get("arm") != expected_arm:
        raise ValueError(f"incomplete or mismatched artifact for {expected_arm}")
    if data.get("conditions") != list(CONDITIONS):
        raise ValueError(f"condition contract mismatch for {expected_arm}")
    for condition in CONDITIONS:
        seen = set()
        for row in data["detail"][condition]:
            if row["record_id"] in seen:
                raise ValueError(f"duplicate record in {expected_arm}/{condition}")
            seen.add(row["record_id"])
            raw = row.get("raw")
            if not isinstance(raw, str) or row.get("raw_sha256") != hashlib.sha256(
                raw.encode("utf-8")
            ).hexdigest() or row.get("raw_chars") != len(raw):
                raise ValueError(f"raw evidence invalid in {expected_arm}/{condition}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {arm: getattr(args, arm) for arm in ARMS}
    artifacts = {arm: _load(paths[arm], arm) for arm in ARMS}
    identity = {
        (row["record_id"], row["function_lineage"])
        for row in artifacts["base"]["detail"][CONDITIONS[0]]
    }
    for arm in ARMS:
        for condition in CONDITIONS:
            current = {(row["record_id"], row["function_lineage"])
                       for row in artifacts[arm]["detail"][condition]}
            if current != identity:
                raise ValueError(f"item panel differs for {arm}/{condition}")
    bound_fields = ("model", "model_revision", "git_commit", "preflight_sha256",
                    "pilot_development_sha256", "items")
    for field in bound_fields:
        values = {json.dumps(artifacts[arm].get(field), sort_keys=True) for arm in ARMS}
        if len(values) != 1:
            raise ValueError(f"arm artifacts differ on frozen field: {field}")
    indexed = {
        arm: {condition: {row["record_id"]: row for row in artifacts[arm]["detail"][condition]}
              for condition in CONDITIONS}
        for arm in ARMS
    }
    ids = sorted(row_id for row_id, _ in identity)
    report: dict[str, Any] = {
        "schema_version": "oneiros_execution_supervision_mechanism_analysis_v1",
        "label": "train-derived mechanism diagnostic; not generalisation or model selection",
        "items": len(ids),
        "predeclared_gates": {
            "minimum_treatment_minus_control_intended_accuracy_pp": MIN_INTENDED_GAIN_PP,
            "maximum_treatment_minus_control_actual_accuracy_regression_pp": MAX_ACTUAL_REGRESSION_PP,
            "maximum_treatment_minus_control_answer_rate_regression_pp": MAX_ANSWER_RATE_REGRESSION_PP,
        },
        "inputs": {arm: {"path": str(paths[arm]), "sha256": hashlib.sha256(
            paths[arm].read_bytes()).hexdigest()} for arm in ARMS},
        "arms": {arm: artifacts[arm]["summary"] for arm in ARMS},
    }
    comparisons = {}
    for condition in CONDITIONS:
        pairs = [(
            indexed["control"][condition][row_id]["strict"]["verdict"] == "correct",
            indexed["treatment"][condition][row_id]["strict"]["verdict"] == "correct",
        ) for row_id in ids]
        diff, low90, high90, gained, lost = paired_difference(pairs)
        comparisons[condition] = {
            "treatment_minus_control_accuracy_pp": diff,
            "paired_ci90_low_pp": low90,
            "paired_ci90_high_pp": high90,
            "gained": gained,
            "lost": lost,
            "mcnemar_exact_p": mcnemar_exact(gained, lost),
        }
    intended_gain = comparisons["intended_output"]["treatment_minus_control_accuracy_pp"]
    actual_change = comparisons["shown_actual_output"]["treatment_minus_control_accuracy_pp"]
    control_answer = artifacts["control"]["summary"]["intended_output"]["strict_answer_rate"]
    treatment_answer = artifacts["treatment"]["summary"]["intended_output"]["strict_answer_rate"]
    answer_change = 100 * (treatment_answer - control_answer)
    gate_checks = {
        "intended_gain": intended_gain >= MIN_INTENDED_GAIN_PP,
        "intended_direction_90ci": comparisons["intended_output"]["paired_ci90_low_pp"] >= 0,
        "actual_noninferiority": actual_change >= -MAX_ACTUAL_REGRESSION_PP,
        "answer_rate_noninferiority": answer_change >= -MAX_ANSWER_RATE_REGRESSION_PP,
        "no_completion_limit_hits": all(
            artifacts[arm]["summary"][condition]["completion_limit_hits"] == 0
            for arm in ARMS for condition in CONDITIONS
        ),
    }
    report["comparisons"] = comparisons
    report["answer_rate_change_pp"] = answer_change
    report["gate_checks"] = gate_checks
    report["mechanism_gate_passed"] = all(gate_checks.values())
    report["next_step"] = (
        "run a separately frozen canonical Kill@8 noninferiority check on train-derived pilot lineages"
        if report["mechanism_gate_passed"] else
        "retain the control; diagnose failures without opening confirmation or validation"
    )
    report["promotion_permitted"] = False
    report["confirmation_opening_permitted"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
