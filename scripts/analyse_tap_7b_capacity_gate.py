"""Analyse the predeclared 7B-versus-1.5B train-only TAP capacity gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyse_tap_capacity_gate import (
    ANSWERED,
    Z95,
    mcnemar_exact,
    paired_difference,
    validate_artifact,
)
from scripts.run_tap_7b_capacity_gate import MODEL_NAME, MODEL_REVISION

MIN_TAP_MUT_GAIN_PP = 5.0
MIN_CI95_LOW_PP = 0.0
MAX_ANSWER_RATE_REGRESSION_PP = 5.0
MAX_CAP_HIT_RATE = 0.02
EXPECTED_CANDIDATE_KEYS = ("base7b::TAP-ref", "base7b::TAP-mut")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_candidate(candidate: dict, baseline: dict, split: dict) -> dict[str, dict]:
    if candidate.get("status") != "complete":
        raise ValueError("7B artifact is not marked complete")
    if candidate.get("model") != MODEL_NAME or candidate.get("model_revision") != MODEL_REVISION:
        raise ValueError("7B artifact model identity does not match the frozen gate")
    if candidate.get("training_performed") is not False or candidate.get("weights_written") is not False:
        raise ValueError("capacity artifact must be base-model inference only")
    contract = candidate.get("run_contract", {})
    if contract.get("permitted_split") != "train":
        raise ValueError("7B artifact is not explicitly train-only")
    if contract.get("items_file_sha256") != split.get("items_file_sha256"):
        raise ValueError("7B artifact and pilot split bind different item files")
    if contract.get("items_file_sha256") != baseline.get("run_contract", {}).get("items_file_sha256"):
        raise ValueError("7B artifact and 1.5B baseline bind different item files")
    detail = candidate.get("detail", {})
    if tuple(sorted(detail)) != tuple(sorted(EXPECTED_CANDIDATE_KEYS)):
        raise ValueError("7B artifact conditions mismatch")

    indexed: dict[str, dict] = {}
    canonical_ids: set[str] | None = None
    for key in EXPECTED_CANDIDATE_KEYS:
        by_id: dict[str, dict] = {}
        for row in detail[key]:
            item_id = row.get("id")
            if item_id in by_id:
                raise ValueError(f"duplicate item ID in {key}: {item_id}")
            raw = row.get("raw")
            if not isinstance(raw, str):
                raise ValueError(f"missing raw output in {key}/{item_id}")
            if row.get("raw_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest():
                raise ValueError(f"raw hash mismatch in {key}/{item_id}")
            if row.get("raw_chars") != len(raw):
                raise ValueError(f"raw length mismatch in {key}/{item_id}")
            by_id[item_id] = row
        if canonical_ids is None:
            canonical_ids = set(by_id)
        elif set(by_id) != canonical_ids:
            raise ValueError("7B item IDs differ across conditions")
        indexed[key] = by_id
    if len(canonical_ids or ()) != 600:
        raise ValueError("7B artifact does not contain the exact 600-item panel")
    return indexed


def arm_metrics(rows: dict[str, dict], ids: list[str]) -> dict:
    correct = sum(rows[item_id]["verdict"] == "correct" for item_id in ids)
    answered = sum(rows[item_id]["verdict"] in ANSWERED for item_id in ids)
    cap_hits = sum(bool(rows[item_id].get("truncated")) for item_id in ids)
    n = len(ids)
    return {
        "n": n,
        "correct": correct,
        "answered": answered,
        "unanswered": n - answered,
        "per_requested_accuracy": correct / n,
        "answer_rate": answered / n,
        "cap_hits": cap_hits,
        "cap_hit_rate": cap_hits / n,
        "worst_case_lower": correct / n,
        "worst_case_upper": (correct + n - answered) / n,
    }


def paired_comparison(baseline_rows: dict[str, dict], candidate_rows: dict[str, dict],
                      ids: list[str]) -> dict:
    pairs = [
        (
            baseline_rows[item_id]["verdict"] == "correct",
            candidate_rows[item_id]["verdict"] == "correct",
        )
        for item_id in ids
    ]
    difference, low, high, gained, lost = paired_difference(pairs, z=Z95)
    return {
        "difference_pp": difference,
        "ci95_low_pp": low,
        "ci95_high_pp": high,
        "gained": gained,
        "lost": lost,
        "mcnemar_exact_p": mcnemar_exact(gained, lost),
    }


def gate_checks(primary_result: dict) -> dict[str, bool]:
    """Apply only the criteria frozen before generation."""
    paired = primary_result["paired_7b_minus_1_5b"]
    return {
        "tap_mut_gain_at_least_5pp": paired["difference_pp"] >= MIN_TAP_MUT_GAIN_PP,
        "tap_mut_ci95_lower_positive": paired["ci95_low_pp"] > MIN_CI95_LOW_PP,
        "tap_mut_answer_rate_noninferior": (
            primary_result["answer_rate_difference_pp"] >= -MAX_ANSWER_RATE_REGRESSION_PP
        ),
        "tap_mut_cap_hit_rate_within_ceiling": (
            primary_result["base7b"]["cap_hit_rate"] <= MAX_CAP_HIT_RATE
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="results/tap_7b_capacity.json")
    parser.add_argument("--baseline", default="results/tap_adapter_compare.json")
    parser.add_argument("--split", default="results/tap_pilot_split.json")
    parser.add_argument("--protocol", default="results/tap_7b_capacity_protocol.json")
    parser.add_argument("--out", default="results/tap_7b_capacity_analysis.json")
    args = parser.parse_args(argv)

    paths = {name: Path(value) for name, value in vars(args).items() if name != "out"}
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    candidate = json.loads(payloads["candidate"])
    baseline = json.loads(payloads["baseline"])
    split = json.loads(payloads["split"])
    protocol = json.loads(payloads["protocol"])

    baseline_indexed = validate_artifact(baseline, split)
    candidate_indexed = validate_candidate(candidate, baseline, split)
    pilot = set(split["pilot_ids"])
    primary = sorted(set(baseline_indexed["base::TAP-ref"]) - pilot)
    if len(primary) != 584:
        raise ValueError(f"expected 584 primary items, got {len(primary)}")
    if set(candidate_indexed["base7b::TAP-ref"]) != set(
            baseline_indexed["base::TAP-ref"]):
        raise ValueError("7B and 1.5B artifacts do not contain the same item IDs")

    expected_policy = {
        "primary_endpoint": "TAP-mut per-requested accuracy, paired 7B minus 1.5B",
        "minimum_gain_pp": MIN_TAP_MUT_GAIN_PP,
        "minimum_ci95_low_pp_exclusive": MIN_CI95_LOW_PP,
        "maximum_answer_rate_regression_pp": MAX_ANSWER_RATE_REGRESSION_PP,
        "maximum_cap_hit_rate": MAX_CAP_HIT_RATE,
    }
    if protocol.get("acceptance_policy") != expected_policy:
        raise ValueError("protocol acceptance policy does not match the predeclared analyzer")
    if protocol.get("candidate", {}).get("model_revision") != MODEL_REVISION:
        raise ValueError("protocol does not bind the exact 7B revision")

    report = {
        "schema_version": "oneiros_tap_7b_capacity_analysis_v1",
        "label": (
            "train-only quantized capacity diagnostic; not generalization, "
            "validation, final-test, real-repository, or SFT evidence"
        ),
        "primary_n": len(primary),
        "pilot_n": len(pilot),
        "inputs": {
            name: {"path": str(paths[name]), "sha256": sha256_bytes(data)}
            for name, data in payloads.items()
        },
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acceptance_policy": expected_policy,
        "conditions": {},
        "strata": {},
    }

    for condition in ("TAP-ref", "TAP-mut"):
        base_rows = baseline_indexed[f"base::{condition}"]
        seven_rows = candidate_indexed[f"base7b::{condition}"]
        base_metrics = arm_metrics(base_rows, primary)
        seven_metrics = arm_metrics(seven_rows, primary)
        paired = paired_comparison(base_rows, seven_rows, primary)
        answer_rate_difference = 100 * (
            seven_metrics["answer_rate"] - base_metrics["answer_rate"]
        )
        report["conditions"][condition] = {
            "base1_5b": base_metrics,
            "base7b": seven_metrics,
            "paired_7b_minus_1_5b": paired,
            "answer_rate_difference_pp": answer_rate_difference,
        }

    for benchmark in ("humaneval", "mbpp"):
        ids = [
            item_id for item_id in primary
            if baseline_indexed["base::TAP-ref"][item_id].get("benchmark") == benchmark
        ]
        report["strata"][benchmark] = {"n": len(ids), "conditions": {}}
        for condition in ("TAP-ref", "TAP-mut"):
            report["strata"][benchmark]["conditions"][condition] = {
                "base1_5b": arm_metrics(baseline_indexed[f"base::{condition}"], ids),
                "base7b": arm_metrics(candidate_indexed[f"base7b::{condition}"], ids),
                "paired_7b_minus_1_5b": paired_comparison(
                    baseline_indexed[f"base::{condition}"],
                    candidate_indexed[f"base7b::{condition}"],
                    ids,
                ),
            }

    primary_result = report["conditions"]["TAP-mut"]
    checks = gate_checks(primary_result)
    report["decision"] = {
        "checks": checks,
        "capacity_gate_passed": all(checks.values()),
        "selected_base_for_execution_supervision": (
            MODEL_NAME if all(checks.values())
            else "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        ),
        "scope": "mechanism/pilot model choice only; no promotion authority",
    }

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["conditions"], indent=2))
    print(json.dumps(report["decision"], indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
