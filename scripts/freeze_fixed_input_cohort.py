"""Freeze the Phase 3A fixed-input cohort (read-only; no model, no GPU).

Deterministic and arm-blind: groups and records are ordered by a seeded hash,
inputs come from ``harness.fixed_input_probe.select_inputs``, and nothing reads
a model output.

Design (from the Phase 3 census):
* MBPP function records only, simple and moderate complexity tiers only: the
  unexposed lineage-disjoint train groups contain 6 complex records in total
  and 13 HumanEval records, so a complex or HumanEval exposure contrast is not
  possible;
* 120 unexposed groups, then 120 exposed groups (a record arm A was trained on)
  matched on the unexposed complexity-tier counts; one function per group;
* per function one discriminating upstream-test call and one discriminating
  novel perturbation not present in any test of any record in the group;
* an item is dropped if its call already appears in the A0 prompt text.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.candidate_policy import validate_function_assertion
from harness.fixed_input_probe import (
    LEVEL_DEFINITIONS, LEVELS, CONTROL_LEVEL, PROBE_SCHEMA_VERSION, SELECTION_SEED,
    build_prompt, select_inputs, stable_key, upstream_calls,
)
from harness.safe_execution import classify_assertions

VIEW = "data/corpus/v4_1_research_hardened_candidate/development_view"
CENSUS = "results/sft_root_cause_phase3_cohort_census.json"
PREFLIGHT = "results/v4_2_armA_successor_preflight.json"
COHORT = "results/sft_root_cause_phase3a_cohort.json"
PER_COHORT = 120
TIERS_ALLOWED = ("simple", "moderate")


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sft_targets(record) -> list[str]:
    """Arm A's literal supervision for a selected record: first 3 policy-valid,
    reference-valid, killing upstream tests (train_on_dataset.evaluate_pair)."""
    tests = [t["code"] for t in record["tests"]
             if validate_function_assertion(t["code"], record["entry_point"]).valid]
    rows = classify_assertions(tests, record["reference_code"], record["code_under_test"])
    return [row["test"] for row in rows if row["valid"] and row["killed"]][:3]


def choose(record, group_upstream):
    picked = select_inputs(record, group_upstream)
    if not (picked["upstream"] and picked["novel"]):
        return None
    items = []
    for kind in ("upstream", "novel"):
        chosen = picked[kind]
        try:
            for level in (*LEVELS, CONTROL_LEVEL):
                build_prompt(record, chosen["call"], chosen["expected_repr"], level)
        except ValueError:
            return None
        items.append({"input_kind": kind, "call": chosen["call"],
                      "expected_repr": chosen["expected_repr"],
                      "buggy_repr": chosen["buggy_repr"]})
    return items


def main() -> int:
    census = load(CENSUS)
    selected = set(load(PREFLIGHT)["selection"]["selected_record_ids"])
    if hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest() != \
            census["arm_a_exposure"]["selected_record_ids_sha256"]:
        raise SystemExit("REFUSED: exposure list differs from the census")
    tiers = {r["record_id"]: r["tier"] for r in load(f"{VIEW}/complexity_manifest.json")["records"]}
    train = load(f"{VIEW}/train.records.json")
    groups: dict[str, list] = {}
    for record in train:
        groups.setdefault(record["group_id"], []).append(record)
    exposed_groups = {r["group_id"] for r in train if r["id"] in selected}

    def eligible(record):
        return ((record.get("quality") or {}).get("execution_mode", "function") == "function"
                and record["source"]["upstream"] == "mbpp"
                and tiers.get(record["id"]) in TIERS_ALLOWED)

    def group_calls(group_id):
        calls = set()
        for r in groups[group_id]:
            calls.update(upstream_calls(r.get("tests") or [], r["entry_point"]))
        return calls

    functions, skipped = [], Counter()
    for cohort in ("unexposed", "exposed"):
        quota = (None if cohort == "unexposed" else
                 Counter(f["complexity_tier"] for f in functions if f["cohort"] == "unexposed"))
        pool = sorted((g for g in groups if (g in exposed_groups) == (cohort == "exposed")),
                      key=lambda g: stable_key("group", g))
        taken = 0
        for group_id in pool:
            if taken >= PER_COHORT:
                break
            records = [r for r in groups[group_id] if eligible(r)
                       and (cohort == "unexposed" or r["id"] in selected)]
            if quota is not None:
                records = [r for r in records if quota[tiers[r["id"]]] > 0]
            chosen = None
            calls = group_calls(group_id) if records else set()
            for record in sorted(records, key=lambda r: stable_key("record", r["id"])):
                items = choose(record, calls)
                if items:
                    chosen = (record, items)
                    break
            if chosen is None:
                skipped[f"{cohort}_group_without_usable_function"] += 1
                continue
            record, items = chosen
            targets = sft_targets(record) if cohort == "exposed" else []
            for item in items:
                item["call_in_arm_a_sft_target"] = any(f"assert {item['call']} " in t
                                                       or f"({item['call']})" in t
                                                       for t in targets)
            functions.append({"cohort": cohort, "record_id": record["id"],
                              "group_id": group_id, "entry_point": record["entry_point"],
                              "complexity_tier": tiers[record["id"]],
                              "bug_family": (record.get("provenance") or {}).get("mutation_type"),
                              "record_content_hash": record.get("content_hash"),
                              "items": items})
            if quota is not None:
                quota[tiers[record["id"]]] -= 1
            taken += 1
    items = [{**item, "item_id": f"{f['record_id']}::{item['input_kind']}",
              "cohort": f["cohort"], "record_id": f["record_id"], "group_id": f["group_id"]}
             for f in functions for item in f["items"]]
    body = {"schema_version": PROBE_SCHEMA_VERSION, "selection_seed": SELECTION_SEED,
            "levels": list(LEVELS), "control_level": CONTROL_LEVEL,
            "level_definitions": LEVEL_DEFINITIONS, "functions": functions}
    text = (json.dumps(body, indent=1, sort_keys=True) + "\n").encode("utf-8")
    publish_file_atomically(ROOT / COHORT, text)

    def balance(cohort):
        fs = [f for f in functions if f["cohort"] == cohort]
        return {"functions": len(fs), "semantic_groups": len({f["group_id"] for f in fs}),
                "complexity_tier": dict(Counter(f["complexity_tier"] for f in fs)),
                "bug_family": dict(Counter(f["bug_family"] for f in fs).most_common()),
                "items": 2 * len(fs),
                "upstream_calls_that_were_literal_sft_targets": sum(
                    i["call_in_arm_a_sft_target"] for f in fs for i in f["items"]
                    if i["input_kind"] == "upstream")}

    summary = {"cohort_path": COHORT, "cohort_sha256": hashlib.sha256(text).hexdigest(),
               "items": len(items), "unexposed": balance("unexposed"),
               "exposed": balance("exposed"), "skipped": dict(skipped),
               "inputs": {p: sha(p) for p in (CENSUS, PREFLIGHT, f"{VIEW}/train.records.json",
                                              f"{VIEW}/complexity_manifest.json")}}
    print(json.dumps(summary, indent=1))
    (ROOT / "results/sft_root_cause/phase3a_freeze_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
