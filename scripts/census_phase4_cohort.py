"""Read-only Phase 4 data preflight: can a fresh lineage-disjoint mediator-gate cohort
be frozen?  (No model, no GPU, no training; nothing is frozen here.)

Reads only the train shard, arm A's preflight selection, the complexity manifest
and the Phase 3A cohort.  Never reads val, ablation_dev, test, sealed-final,
reserved-confirmation or A-prime material (A-prime targets are GitHub
repositories outside the corpus, so they cannot overlap train lineages).

Pools, by semantic group:
* STRICT: train groups with no arm A training record, not in the Phase 3
  exposed cohort and not in the Phase 3 unexposed cohort;
* ARM-A-EXPOSED REMAINDER: arm A training groups not in either Phase 3 cohort.
  A Phase 4 control/treatment pair starting from the immutable base never saw
  these, so they are lineage-disjoint FOR THE PHASE 4 ARMS, but arm A itself
  cannot be compared fairly on them.

For each pool: records, groups, dataset/tier mix, and how many groups contain a
function with a verified discriminating upstream AND novel fixed input (the
Phase 3A construction), plus a power estimate scaled from Phase 3A.
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
from harness.fixed_input_probe import CONTROL_LEVEL, LEVELS, build_prompt, select_inputs, \
    stable_key, upstream_calls
from harness.parallel_execution import map_jobs

VIEW = "data/corpus/v4_1_research_hardened_candidate/development_view"
PREFLIGHT = "results/v4_2_armA_successor_preflight.json"
COHORT3 = "results/sft_root_cause_phase3a_cohort.json"
V2 = "results/sft_root_cause_phase3a_result_receipt_v2.json"
OUTPUT = "results/sft_root_cause_phase4_cohort_census.json"


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def feasible(job) -> dict:
    group_id, records = job
    calls = set()
    for r in records:
        calls.update(upstream_calls(r.get("tests") or [], r["entry_point"]))
    for record in sorted(records, key=lambda r: stable_key("record", r["id"])):
        if (record.get("quality") or {}).get("execution_mode", "function") != "function":
            continue
        picked = select_inputs(record, calls)
        if not (picked["upstream"] and picked["novel"]):
            continue
        try:
            for kind in ("upstream", "novel"):
                for level in (*LEVELS, CONTROL_LEVEL):
                    build_prompt(record, picked[kind]["call"], picked[kind]["expected_repr"],
                                 level)
        except ValueError:
            continue
        return {"group_id": group_id, "feasible": True, "record_id": record["id"]}
    return {"group_id": group_id, "feasible": False}


def main() -> int:
    selected = set(load(PREFLIGHT)["selection"]["selected_record_ids"])
    tiers = {r["record_id"]: r["tier"] for r in load(f"{VIEW}/complexity_manifest.json")["records"]}
    train = load(f"{VIEW}/train.records.json")
    groups: dict[str, list] = {}
    for record in train:
        groups.setdefault(record["group_id"], []).append(record)
    arm_a = {r["group_id"] for r in train if r["id"] in selected}
    phase3 = {c: {f["group_id"] for f in load(COHORT3)["functions"] if f["cohort"] == c}
              for c in ("exposed", "unexposed")}
    used3 = phase3["exposed"] | phase3["unexposed"]
    pools = {"strict": sorted(g for g in groups if g not in arm_a and g not in used3),
             "arm_a_exposed_remainder": sorted(g for g in groups if g in arm_a and g not in used3)}
    v2 = load(V2)
    ci = v2["strata"]["all"]["prefill"]["ci95_points"]
    se_240 = (ci[1] - ci[0]) / 3.92
    out = {}
    for name, pool in pools.items():
        recs = [r for g in pool for r in groups[g]]
        results = map_jobs(feasible, [(g, groups[g]) for g in pool])
        ok = [r for r in results if r["feasible"]]
        chosen = {r["record_id"] for r in ok}
        by_id = {r["id"]: r for r in recs}
        n = len(ok)
        se = se_240 * (240 / n) ** 0.5 if n else None
        out[name] = {
            "groups": len(pool), "records": len(recs),
            "execution_mode": dict(Counter((r.get("quality") or {}).get("execution_mode",
                                                                        "function")
                                           for r in recs)),
            "dataset": dict(Counter(r["source"]["upstream"] for r in recs).most_common()),
            "complexity_tier": dict(Counter(tiers.get(r["id"], "none") for r in recs)),
            "groups_with_a_feasible_fixed_input_function": n,
            "feasible_function_dataset": dict(Counter(by_id[i]["source"]["upstream"]
                                                      for i in chosen)),
            "feasible_function_tier": dict(Counter(tiers.get(i, "none") for i in chosen)),
            "group_ids_sha256": hashlib.sha256("\n".join(pool).encode()).hexdigest(),
            "feasible_group_ids_sha256": hashlib.sha256(
                "\n".join(sorted(r["group_id"] for r in ok)).encode()).hexdigest(),
            "power_estimate": {
                "basis": ("SE of a paired accuracy difference scaled from Phase 3A (240 groups, "
                          "2 items each) as 1/sqrt(groups); one function per group"),
                "se_points": round(se, 2) if se else None,
                "mde_80pct_power_points": round(2.8 * se, 1) if se else None}}
    census = {
        "schema_version": "oneiros_sft_root_cause_phase4_census_v1",
        "read_only": True, "model_calls": 0, "gpu_used": False, "training": False,
        "frozen_cohort": None,
        "not_read": ["val", "ablation_dev", "test", "sealed_final", "reserved_confirmation",
                     "A-prime confirmation (outside the corpus)"],
        "inputs": {p: sha(p) for p in (PREFLIGHT, COHORT3, V2, f"{VIEW}/train.records.json",
                                       f"{VIEW}/complexity_manifest.json")},
        "used_lineages": {"arm_a_training_groups": len(arm_a),
                          "phase3_exposed_cohort_groups": len(phase3["exposed"]),
                          "phase3_unexposed_cohort_groups": len(phase3["unexposed"]),
                          "train_groups_total": len(groups)},
        "pools": out,
        "phase3_unexposed_status": ("development evidence only: inspected and used to choose the "
                                    "Phase 4 intervention, so never the sole acceptance or "
                                    "confirmation panel"),
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(census, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({k: census[k] for k in ("used_lineages",)}, indent=1))
    for name, block in out.items():
        print(name, json.dumps(block))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
