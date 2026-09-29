"""Choice B (Phase 4 internal screen): frozen data split, matched manifest and analysis.

INTERNAL, EXPLORATORY, ARM-A-EXPOSED. Not untouched, not Phase 6, not cross-dataset
or repository generalisation; a single seed cannot establish efficacy. The estimand is
the COMPOSITE intervention "value-only masking plus reduced supervised-token mass"
(decision receipt v2, interpretation A).

Data (train split of the development view only):

* pool: the arm-A-exposed remainder, i.e. semantic groups arm A trained on that are
  in neither Phase 3A cohort (re-derived and checked against the Phase 4 census);
* gate: the first ``GATE_GROUPS`` feasible pool groups in a seeded hash order, fixed
  BEFORE any Choice B outcome exists; one function per group with one discriminating
  upstream call and one discriminating novel perturbation (the Phase 3A
  construction);
* training: every function-mode record of every other pool group, each verified
  discriminating upstream call with its reference value, capped per record and per
  group. No gate group, no repeated row, no development split beyond train.

Analysis rules are frozen here, before launch: both answer schemas; paired
semantic-group cluster bootstrap (10,000 resamples, seed 20260928, percentile 95%);
answer-rate non-inferiority T >= C - 2 points per schema before accuracy is read;
the original 3-point reference-validity guardrail; exact-unique-ratio diversity may
fall at most 10% relative; Kill@8 reported; final checkpoint only; one look; outcomes
``promising`` / ``harm`` / ``inconclusive_power``.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from harness.fixed_input_probe import (
    CONTROL_LEVEL, LEVELS, build_prompt, select_inputs, stable_key, upstream_calls,
    verify_calls,
)

DESIGN_VERSION = "oneiros_phase4_choice_b_v1"
GATE_GROUPS = 200
GATE_ORDER_TAG = "phase4_choice_b_gate"
ROW_ORDER_TAG = "phase4_choice_b_row"
MAX_CALLS_PER_RECORD = 3
GROUP_CAP = 40
SCHEMAS = {"prefill_assertion": "A0", "answer_schema": CONTROL_LEVEL}
BOOTSTRAP = {"resamples": 10_000, "seed": 20260928, "interval": "percentile_95"}
ANSWER_RATE_MARGIN_POINTS = 2.0
VALIDITY_MARGIN_POINTS = 3.0
DIVERSITY_MAX_RELATIVE_LOSS = 0.10
OUTCOMES = ("promising", "harm", "inconclusive_power")
LABELS = ("internal", "exploratory", "Arm-A-exposed", "not untouched", "not Phase 6",
          "not cross-dataset or repository generalisation",
          "single seed cannot establish efficacy",
          "composite intervention: value-only masking plus reduced supervised-token mass")


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def is_function(record: Mapping[str, Any]) -> bool:
    return (record.get("quality") or {}).get("execution_mode", "function") == "function"


def group_upstream(records: Sequence[Mapping[str, Any]]) -> set:
    calls: set = set()
    for record in records:
        calls.update(upstream_calls(record.get("tests") or [], record["entry_point"]))
    return calls


def pool_groups(train: Sequence[Mapping[str, Any]], arm_a_selected: set,
                phase3_groups: set) -> Dict[str, List[Mapping[str, Any]]]:
    """Arm-A-exposed remainder: groups arm A trained on, minus both Phase 3A cohorts."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for record in train:
        groups.setdefault(record["group_id"], []).append(record)
    arm_a = {r["group_id"] for r in train if r["id"] in arm_a_selected}
    return {g: groups[g] for g in sorted(arm_a - phase3_groups)}


def gate_function(group_id: str, records: Sequence[Mapping[str, Any]]) -> dict | None:
    """The Phase 3A / census construction: first record with an upstream AND a novel item."""
    calls = group_upstream(records)
    for record in sorted(records, key=lambda r: stable_key("record", r["id"])):
        if not is_function(record):
            continue
        picked = select_inputs(record, calls)
        if not (picked["upstream"] and picked["novel"]):
            continue
        items = []
        try:
            for kind in ("upstream", "novel"):
                chosen = picked[kind]
                for level in (*LEVELS, CONTROL_LEVEL):
                    build_prompt(record, chosen["call"], chosen["expected_repr"], level)
                items.append({"input_kind": kind, "call": chosen["call"],
                              "expected_repr": chosen["expected_repr"],
                              "buggy_repr": chosen["buggy_repr"],
                              "item_id": f"{record['id']}::{kind}"})
        except ValueError:
            continue
        return {"group_id": group_id, "record_id": record["id"],
                "entry_point": record["entry_point"], "items": items}
    return None


def assign_gate(feasible: Mapping[str, dict], count: int = GATE_GROUPS) -> List[str]:
    """Seeded hash order over feasible groups; depends on no outcome of any arm."""
    order = sorted(feasible, key=lambda g: stable_key(GATE_ORDER_TAG, g))
    if len(order) < count:
        raise ValueError(f"only {len(order)} feasible groups for a {count}-group gate")
    return sorted(order[:count])


def training_rows_for_record(record: Mapping[str, Any]) -> List[dict]:
    """Verified discriminating upstream calls of one record, with reference values."""
    if not is_function(record):
        return []
    own = upstream_calls(record.get("tests") or [], record["entry_point"])
    rows = []
    for checked in verify_calls(own, str(record["reference_code"]),
                                str(record["code_under_test"])):
        if not checked["usable"]:
            continue
        try:
            build_prompt(record, checked["call"], checked["expected_repr"], "A0")
        except ValueError:
            continue
        rows.append({"record_id": record["id"], "group_id": record["group_id"],
                     "call": checked["call"], "value": checked["expected_repr"],
                     "buggy_repr": checked["buggy_repr"]})
        if len(rows) >= MAX_CALLS_PER_RECORD:
            break
    return rows


def cap_and_order(rows: Sequence[dict]) -> List[dict]:
    """Drop repeats, cap per group in a seeded order, then fix one global order."""
    seen, out, per_group = set(), [], Counter()
    for row in sorted(rows, key=lambda r: stable_key(ROW_ORDER_TAG, r["record_id"], r["call"])):
        key = (row["record_id"], row["call"], row["value"])
        if key in seen or per_group[row["group_id"]] >= GROUP_CAP:
            continue
        seen.add(key)
        per_group[row["group_id"]] += 1
        out.append(row)
    return out


# --- analysis (frozen before launch) ---------------------------------------------------------

def cluster_bootstrap_difference(groups: Sequence[str], control: Sequence[float],
                                 treatment: Sequence[float], *,
                                 resamples: int = BOOTSTRAP["resamples"],
                                 seed: int = BOOTSTRAP["seed"]) -> dict:
    """Paired semantic-group cluster bootstrap of mean(T) - mean(C), in points."""
    labels = sorted(set(groups))
    index = {g: i for i, g in enumerate(labels)}
    n_g = np.zeros(len(labels))
    c_g = np.zeros(len(labels))
    t_g = np.zeros(len(labels))
    for g, c, t in zip(groups, control, treatment):
        n_g[index[g]] += 1
        c_g[index[g]] += c
        t_g[index[g]] += t
    point = (t_g.sum() - c_g.sum()) / n_g.sum() * 100
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        pick = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        n = n_g[pick].sum(axis=1)
        draws[start:stop] = (t_g[pick].sum(axis=1) - c_g[pick].sum(axis=1)) / n * 100
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"point": round(float(point), 3), "low": round(float(low), 3),
            "high": round(float(high), 3), "groups": len(labels), "items": int(n_g.sum()),
            "resamples": resamples, "seed": seed}


def decide(schemas: Mapping[str, Mapping[str, Any]], guardrails: Mapping[str, bool]) -> dict:
    """Map frozen per-schema results and guardrails to exactly one outcome.

    ``schemas[name]`` holds ``accuracy`` (bootstrap dict) and ``answer_rate_gate``
    (bool). An answer-rate non-inferiority failure is a guardrail failure: accuracy is
    not interpreted when the treatment stops answering in the required format.
    """
    answer_ok = all(s["answer_rate_gate"] for s in schemas.values())
    points = [s["accuracy"]["point"] for s in schemas.values()]
    flip = min(points) < 0 < max(points)
    guard_ok = answer_ok and all(guardrails.values())
    if not guard_ok or any(s["accuracy"]["high"] < 0 for s in schemas.values()):
        outcome = "harm"
    elif all(s["accuracy"]["low"] > 0 for s in schemas.values()) and not flip:
        outcome = "promising"
    else:
        outcome = "inconclusive_power"
    return {"outcome": outcome, "schema_sign_flip": flip, "answer_rate_gates_pass": answer_ok,
            "guardrails": dict(guardrails), "guardrails_pass": guard_ok}


def frozen_evaluation_spec() -> dict:
    return {
        "design_version": DESIGN_VERSION,
        "labels": list(LABELS),
        "checkpoint": "final adapter of the fixed schedule only; no intermediate checkpoint "
                      "is ever evaluated on the gate",
        "looks": "exactly one gate look per arm; the analysis refuses to run twice",
        "arms_evaluated": ["control", "treatment"],
        "old_arm_A_evaluated": False,
        "primary_mediator": {
            "metric": "fixed-input exact-output accuracy (assert CALL == ANSWER passes on the "
                      "reference)",
            "schemas": SCHEMAS,
            "decoding": {"greedy": True, "max_new_tokens": {"prefill_assertion": 48,
                                                            "answer_schema": 96},
                         "batch": 8},
            "contrast": "treatment minus control, exactly paired items",
            "inference": {"method": "paired semantic-group cluster bootstrap", **BOOTSTRAP}},
        "answer_rate_gate": f"per schema, T answer rate >= C answer rate - "
                            f"{ANSWER_RATE_MARGIN_POINTS} points, checked before accuracy is "
                            "read; failure counts as a guardrail failure",
        "downstream": {
            "kill_at_8": "canonical production generation (successor settings: 8 candidates, "
                         "frozen temperature/top-p/seed) on the gate functions; reported with "
                         "a paired cluster bootstrap; not gating",
            "validity_guardrail": f"original guardrail: mean per-candidate reference validity "
                                  f"may fall by at most {VALIDITY_MARGIN_POINTS} points "
                                  "(point estimate, T - C)",
            "diversity_guardrail": f"mean exact-unique ratio may fall by at most "
                                   f"{int(DIVERSITY_MAX_RELATIVE_LOSS * 100)}% relative",
            "also_reported": ["exact-equality validity", "function-level P(at least one valid "
                              "killing candidate)", "nonanswer rate"]},
        "outcomes": {
            "promising": "answer-rate gates and guardrails pass, lower bound > 0 under both "
                         "schemas, no schema sign flip",
            "harm": "an upper bound < 0 under either schema, or any gate/guardrail fails",
            "inconclusive_power": "anything else; never rejects the intervention",
        },
        "confirmation_claims": "none: lower bound > +5 is not a gate of this screen",
    }


def spec_sha256(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def arm_command(python: str, arm: str) -> List[str]:
    """The durable launch command; the two arms may differ only in the arm token."""
    return [python, "scripts/gpu_run.py", "start", "--name", f"phase4_choice_b_{arm}", "--",
            python, "scripts/phase4_choice_b.py", "train", "--arm", arm]


def command_difference(control: Sequence[str], treatment: Sequence[str]) -> List[tuple]:
    if len(control) != len(treatment):
        return [("length", len(control), len(treatment))]
    return [(i, c, t) for i, (c, t) in enumerate(zip(control, treatment)) if c != t]


def gate_job(job) -> dict | None:
    """Picklable wrapper for parallel execution: ``(group_id, records)``."""
    return gate_function(*job)
