"""Choice B v2 evaluation: strict raw-output integrity, compact evidence, frozen statistics.

Integrity (a gate look is refused rather than silently repaired):

* every raw generation line carries the SHA-256 of the immutable evaluation contract;
  a resumed file is accepted only if every line parses, ends in a newline, carries the
  same contract, has an expected key and no key repeats; anything else is QUARANTINED
  (moved aside, never deleted, never merged) and that output is regenerated;
* the fixed-input key set must be exactly gate groups x {upstream, novel} x both schemas
  (800 keys for 200 groups), unique; missing, extra, duplicate or substituted keys refuse;
* Kill@8 function rows must be exactly the expected gate record IDs, once each, in both
  arms; no intersection is ever analysed.

Statistics (evaluation spec v2, frozen before any Choice B outcome):

* primary: fixed-input exact-output accuracy, treatment minus control, per answer schema,
  paired semantic-group cluster bootstrap (10,000 resamples, seed 20260928, percentile);
* guardrails are POINT-ESTIMATE OPERATIONAL SAFETY THRESHOLDS (answer rate -2 points,
  reference validity per requested candidate -3 points, exact-unique ratio -10%
  relative); their paired bootstrap intervals are reported alongside;
* outcome precedence: statistically supported adverse evidence > operational guardrail
  stop > promising > inconclusive power. A point-estimate threshold failure is an
  operational stop, never "statistically proven harm".
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from harness.choice_b import cluster_bootstrap_difference

SCHEMA_LEVELS = {"prefill_assertion": "A0", "answer_schema": "A0_answer_schema"}
INPUT_KINDS = ("upstream", "novel")
BOOTSTRAP = {"resamples": 10_000, "seed": 20260928, "interval": "percentile_95"}
MARGINS = {"answer_rate_points": 2.0, "reference_validity_points": 3.0,
           "exact_unique_relative": 0.10}
OUTCOMES = ("promising", "statistically_supported_adverse", "operational_guardrail_stop",
            "inconclusive_power")
LEGACY_OUTCOME = {"promising": "promising", "statistically_supported_adverse": "harm",
                  "operational_guardrail_stop": "harm", "inconclusive_power": "inconclusive_power"}
SLICES = {"dataset": "record source upstream (mbpp, humaneval, manual_curated_examples)",
          "input_kind": "upstream test call versus novel perturbation",
          "complexity_tier": "complexity manifest tier",
          "bug_family": "mutation operator family"}
ESTIMAND = ("COMPOSITE objective intervention: value_only versus full_completion changes WHICH "
            "completion tokens are supervised, HOW MANY target tokens are supervised, their "
            "POSITIONS, and the RELATIVE per-example and per-token weighting of the loss (the "
            "loss averages over supervised tokens, so each value token's share changes); it is "
            "not a scalar gradient-dose reduction and does not isolate value-token location")


class IntegrityError(RuntimeError):
    pass


# --- raw generation files -----------------------------------------------------------------

def expected_fixed_keys(gate_functions: Sequence[Mapping[str, Any]]) -> List[str]:
    keys = []
    for function in gate_functions:
        kinds = [i["input_kind"] for i in function["items"]]
        if sorted(kinds) != sorted(INPUT_KINDS):
            raise IntegrityError(f"{function['record_id']}: items are {kinds}")
        for item in function["items"]:
            for schema in SCHEMA_LEVELS:
                keys.append(f"{item['item_id']}::{schema}")
    if len(keys) != len(set(keys)):
        raise IntegrityError("gate panel produces duplicate keys")
    return keys


def read_generation_file(path: Path, contract_sha: str, expected: Iterable[str]
                         ) -> Tuple[Dict[str, dict], List[str]]:
    """Strictly read a resumable JSONL. Returns (rows by key, problems)."""
    expected = set(expected)
    rows: Dict[str, dict] = {}
    problems: List[str] = []
    if not path.exists():
        return rows, problems
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        problems.append("partial final line")
    for number, line in enumerate(data.decode("utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            problems.append(f"blank line {number}")
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"malformed line {number}")
            continue
        key = row.get("key")
        if row.get("contract_sha256") != contract_sha:
            problems.append(f"line {number}: different contract")
        elif key not in expected:
            problems.append(f"line {number}: unexpected key {key!r}")
        elif key in rows:
            problems.append(f"line {number}: duplicate key {key!r}")
        else:
            rows[key] = row
    return rows, problems


def quarantine(path: Path, reason: Sequence[str], log: Path) -> Path:
    """Move a file aside (never delete, never merge) and log why."""
    target_dir = path.parent / "quarantine"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / f"{int(time.time())}_{path.name}"
    shutil.move(str(path), str(target))
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"quarantined": path.name, "to": target.name,
                                 "reasons": list(reason)[:20],
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                     + "\n")
    return target


def require_complete(rows: Mapping[str, dict], expected: Sequence[str]) -> None:
    missing = sorted(set(expected) - set(rows))
    extra = sorted(set(rows) - set(expected))
    if missing or extra or len(rows) != len(expected):
        raise IntegrityError(f"fixed-input keys incomplete: missing {len(missing)}, "
                             f"extra {len(extra)}")


def require_function_rows(results: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
                          ) -> Dict[str, Mapping[str, Any]]:
    counts = Counter(str(r["record_id"]) for r in results)
    duplicates = sorted(k for k, v in counts.items() if v > 1)
    missing = sorted(set(expected_ids) - set(counts))
    extra = sorted(set(counts) - set(expected_ids))
    if duplicates or missing or extra:
        raise IntegrityError(f"Kill@8 rows: duplicates {duplicates[:3]}, missing "
                             f"{missing[:3]}, extra {extra[:3]}")
    return {str(r["record_id"]): r for r in results}


# --- compact per-function evidence ----------------------------------------------------------

def is_exact_equality(code: str | None) -> bool:
    """A single assertion of the form ``assert <expr> == <expr>``."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    body = tree.body
    return (len(body) == 1 and isinstance(body[0], ast.Assert)
            and isinstance(body[0].test, ast.Compare) and len(body[0].test.ops) == 1
            and isinstance(body[0].test.ops[0], ast.Eq))


def compact_function(row: Mapping[str, Any]) -> Dict[str, Any]:
    outcomes = sorted(row["candidate_outcomes"], key=lambda o: int(o.get("rank", 0)))
    return {
        "record_id": str(row["record_id"]), "requested": len(outcomes),
        "candidates": [{"rank": int(o.get("rank", 0)),
                        "parse_valid": bool(o.get("parse_valid")),
                        "execution_valid": bool(o.get("execution_valid")),
                        "reference_valid": bool(o.get("reference_valid")),
                        "killed": bool(o.get("killed")),
                        "exact_equality": is_exact_equality(o.get("code")),
                        "code_sha256": hashlib.sha256(str(o.get("code") or "")
                                                      .encode("utf-8")).hexdigest()}
                       for o in outcomes],
        "exact_unique_ratio": float((row.get("diversity") or {}).get("exact_unique_ratio", 0.0)),
    }


def function_metrics(compact: Mapping[str, Any]) -> Dict[str, float]:
    cands = compact["candidates"]
    requested = compact["requested"]

    def prefix(k: int, flag: str) -> float:
        return float(any(c[flag] for c in cands if c["rank"] <= k))
    return {
        "kill_at_1": prefix(1, "killed"), "kill_at_4": prefix(4, "killed"),
        "kill_at_8": prefix(8, "killed"),
        "parse_success": sum(c["parse_valid"] for c in cands) / requested,
        "execution_success": sum(c["execution_valid"] for c in cands) / requested,
        "reference_validity_per_requested": sum(c["reference_valid"] for c in cands) / requested,
        "exact_equality_validity_per_requested":
            sum(c["reference_valid"] and c["exact_equality"] for c in cands) / requested,
        "has_valid_killing_candidate": float(any(c["reference_valid"] and c["killed"]
                                                 for c in cands)),
        "exact_unique_ratio": compact["exact_unique_ratio"],
    }


# --- statistics -----------------------------------------------------------------------------

def relative_change_bootstrap(groups: Sequence[str], control: Sequence[float],
                              treatment: Sequence[float]) -> Dict[str, float]:
    """Paired cluster bootstrap of mean(T)/mean(C) - 1."""
    labels = sorted(set(groups))
    index = {g: i for i, g in enumerate(labels)}
    c_g, t_g = np.zeros(len(labels)), np.zeros(len(labels))
    for g, c, t in zip(groups, control, treatment):
        c_g[index[g]] += c
        t_g[index[g]] += t
    point = t_g.sum() / c_g.sum() - 1 if c_g.sum() else 0.0
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = np.empty(BOOTSTRAP["resamples"])
    for start in range(0, BOOTSTRAP["resamples"], 1000):
        stop = min(start + 1000, BOOTSTRAP["resamples"])
        pick = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        cs = c_g[pick].sum(axis=1)
        draws[start:stop] = np.where(cs > 0, t_g[pick].sum(axis=1) / np.where(cs > 0, cs, 1)
                                     - 1, 0.0)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"point": round(float(point), 5), "low": round(float(low), 5),
            "high": round(float(high), 5), "groups": len(labels)}


def paired(groups, control, treatment) -> Dict[str, Any]:
    return cluster_bootstrap_difference(groups, control, treatment,
                                        resamples=BOOTSTRAP["resamples"], seed=BOOTSTRAP["seed"])


def guardrail(name: str, interval: Mapping[str, float]) -> Dict[str, Any]:
    bound = -MARGINS[name]
    return {"interval": dict(interval), "threshold": bound,
            "operational_pass": interval["point"] >= bound,
            "adverse_supported": interval["high"] < bound,
            "non_inferiority_shown": interval["low"] > bound}


def classify(primary: Mapping[str, Mapping[str, float]],
             guardrails: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    points = [p["point"] for p in primary.values()]
    flip = min(points) < 0 < max(points)
    reasons = []
    for schema, interval in primary.items():
        if interval["high"] < 0:
            reasons.append(f"primary_{schema}_upper_bound_below_zero")
    for name, g in guardrails.items():
        if g["adverse_supported"]:
            reasons.append(f"guardrail_{name}_adverse_supported")
    if reasons:
        outcome = "statistically_supported_adverse"
    else:
        stops = [f"guardrail_{n}_point_beyond_threshold" for n, g in guardrails.items()
                 if not g["operational_pass"]]
        if stops:
            outcome, reasons = "operational_guardrail_stop", stops
        elif all(p["low"] > 0 for p in primary.values()) and not flip:
            outcome, reasons = "promising", ["primary_lower_bounds_above_zero_both_schemas"]
        else:
            outcome = "inconclusive_power"
            reasons = ["schema_sign_flip"] if flip else ["primary_interval_includes_zero"]
    return {"outcome": outcome, "legacy_outcome": LEGACY_OUTCOME[outcome],
            "reason_codes": reasons, "schema_sign_flip": flip}


def evaluation_spec_v2(v1_spec: Mapping[str, Any], v1_spec_sha: str,
                       power_receipt: Mapping[str, Any], power_sha: str) -> Dict[str, Any]:
    return {
        "schema_version": "oneiros_choice_b_evaluation_spec_v2",
        "supersedes": {"evaluation_spec_v1_sha256": v1_spec_sha,
                       "why": ["v1 called point-estimate guardrail failures 'harm'",
                               "v1 estimand wording was incomplete",
                               "v1 did not require exact key sets or list every metric"]},
        "frozen_before_any_choice_b_outcome": True,
        "labels": ["internal", "exploratory", "Arm-A-exposed", "not untouched", "not Phase 6",
                   "not cross-dataset or repository generalisation",
                   "single seed cannot establish efficacy"],
        "estimand": ESTIMAND,
        "checkpoint": v1_spec["checkpoint"], "looks": v1_spec["looks"],
        "arms_evaluated": ["control", "treatment"], "old_arm_A_evaluated": False,
        "primary": {"metric": "fixed-input exact-output accuracy (assert CALL == ANSWER passes "
                              "on the reference)",
                    "schemas": SCHEMA_LEVELS, "decoding": v1_spec["primary_mediator"]["decoding"],
                    "keys": "every gate group x {upstream, novel} x both schemas, exactly once",
                    "inference": {"method": "paired semantic-group cluster bootstrap",
                                  **BOOTSTRAP}},
        "guardrails": {
            "status": "POINT-ESTIMATE OPERATIONAL SAFETY THRESHOLDS; paired cluster-bootstrap "
                      "intervals reported alongside",
            "answer_rate": f"T - C answer rate >= -{MARGINS['answer_rate_points']} points, per "
                           "schema",
            "reference_validity": f"T - C mean per-requested-candidate reference validity >= "
                                  f"-{MARGINS['reference_validity_points']} points",
            "exact_unique_ratio": f"T / C - 1 >= -{MARGINS['exact_unique_relative']}"},
        "outcomes": {
            "statistically_supported_adverse": "a primary upper bound < 0 under either schema, "
                                               "or a guardrail's upper bound below its threshold",
            "operational_guardrail_stop": "otherwise, any guardrail point estimate beyond its "
                                          "threshold (not a statistical claim of harm)",
            "promising": "otherwise, primary lower bound > 0 under both schemas with no sign "
                         "flip",
            "inconclusive_power": "anything else; never 'no effect', never rejects",
            "precedence": list(OUTCOMES)},
        "reported_metrics": ["fixed-input answer rate", "fixed-input nonanswer rate",
                             "fixed-input exact-output accuracy", "Kill@1", "Kill@4", "Kill@8",
                             "candidate parse success", "candidate execution success",
                             "reference validity per requested candidate",
                             "exact-equality validity per requested candidate",
                             "function-level P(at least one valid killing candidate)",
                             "exact-unique diversity ratio"],
        "descriptive_slices": {**SLICES, "status": "DESCRIPTIVE, likely underpowered; never "
                                                   "cross-dataset generalisation"},
        "power_at_gate": {"path": "results/sft_root_cause_phase4_choice_b_power_200_v1.json",
                          "sha256": power_sha,
                          "positive_effect_probability_by_true_effect_points":
                              power_receipt["exact_positive_effect_probability_by_effect"],
                          "reading": "a five-point true effect has low useful detection power "
                                     "at 200 groups; detection is not impossible"},
        "claims_forbidden": ["root cause", "generalisation", "value-location causality",
                             "efficacy from training loss", "confirmation of a >5-point effect"],
    }
