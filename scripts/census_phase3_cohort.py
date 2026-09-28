"""Read-only membership census for the Phase 3 fixed-input probe (no GPU, no model).

Answers, with exact counts, whether a diagnostic cohort can contain:

* functions arm A was trained on (exposed semantic groups), split into inputs
  that were literal SFT targets and inputs that were not;
* train-split functions in semantic groups arm A never saw (lineage-disjoint,
  unexposed);
* a genuinely external, non-reserved cohort.

Arm A's exposure is the 2,400 records in its preflight selection (identical in
all three arm A preflight receipts and matching its training metadata).  Its
supervision targets were each selected record's first three policy-valid,
reference-valid, killing upstream tests (scripts/train_on_dataset.py
``evaluate_pair``; multi-mutant supervision unused).

Reads only the train shard, the arm A preflight receipts and metadata, and the
val/ablation_dev group identifiers (for disjointness).  Never reads test,
sealed-final or reserved-confirmation material.
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

VIEW = "data/corpus/v4_1_research_hardened_candidate/development_view"
PREFLIGHTS = ["results/v4_2_armA_clean_preflight.json",
              "results/v4_2_armA_successor_preflight.json",
              "results/v4_2_armA_successor_preflight_c1024.json"]
META = "checkpoints/local_sft_armA_baseline_successor_s42/sft_metadata.json"
EXCLUSIONS = f"{VIEW}/training_exclusions.json"
OUTPUT = "results/sft_root_cause_phase3_cohort_census.json"


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def id_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def mode(record) -> str:
    return (record.get("quality") or {}).get("execution_mode", "function")


TIERS: dict[str, str] = {}


def profile(records) -> dict:
    return {"records": len(records),
            "complexity_tier": dict(Counter(TIERS.get(r["id"], "unknown")
                                            for r in records).most_common()),
            "semantic_groups": len({r["group_id"] for r in records}),
            "max_records_per_group": max(Counter(r["group_id"] for r in records).values(),
                                         default=0),
            "dataset": dict(Counter(r["source"]["upstream"] for r in records).most_common()),
            "bug_family": dict(Counter((r.get("provenance") or {}).get("mutation_type")
                                       for r in records).most_common()),
            "group_ids_sha256": id_hash({r["group_id"] for r in records}),
            "record_ids_sha256": id_hash({r["id"] for r in records})}


def main() -> int:
    selections = [load(p)["selection"] for p in PREFLIGHTS]
    ids = [tuple(s["selected_record_ids"]) for s in selections]
    if len(set(ids)) != 1 or len({s["selection_sha256"] for s in selections}) != 1:
        raise SystemExit("REFUSED: the arm A preflight selections disagree")
    selected = set(ids[0])
    meta = load(META)
    stats = meta["bounded_selection_stats"]
    mm = meta["sampling_stats"]["multi_mutant_supervision"]["examples_used"]
    TIERS.update({r["record_id"]: r["tier"] for r in load(f"{VIEW}/complexity_manifest.json")
                  ["records"]})
    train = load(f"{VIEW}/train.records.json")
    by_id = {r["id"]: r for r in train}
    missing = sorted(selected - set(by_id))
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} selected IDs are not in the train shard")
    consistent = (len(selected) == stats["total_pairs"]
                  and sum(mode(by_id[i]) != "function" for i in selected)
                  == stats["repository_pairs"])
    exclusions = load(EXCLUSIONS)
    excluded = {e["record_id"] if isinstance(e, dict) else e for e in exclusions}
    functions = [r for r in train if mode(r) == "function"]
    exposed_groups = {by_id[i]["group_id"] for i in selected}
    exposed_selected = [r for r in functions if r["id"] in selected]
    exposed_sibling = [r for r in functions if r["id"] not in selected
                       and r["group_id"] in exposed_groups]
    unexposed = [r for r in functions if r["group_id"] not in exposed_groups]
    held = {split: {r["group_id"] for r in load(f"{VIEW}/{split}.records.json")}
            for split in ("val", "ablation_dev")}
    all_train_groups = {r["group_id"] for r in train}
    # Upstream-test inputs shared within a group: how often an unselected sibling's
    # tests were literal SFT targets through a selected sibling.
    target_tests: dict[str, set[str]] = {}
    for record in exposed_selected:
        target_tests.setdefault(record["group_id"], set()).update(
            t["code"] for t in record["tests"][:6])
    shared = sum(1 for r in exposed_sibling
                 if {t["code"] for t in r["tests"]} & target_tests.get(r["group_id"], set()))
    census = {
        "schema_version": "oneiros_sft_root_cause_phase3_census_v1",
        "read_only": True, "model_calls": 0, "gpu_used": False,
        "test_or_sealed_or_confirmation_read": False,
        "inputs": {p: sha(p) for p in [*PREFLIGHTS, META, f"{VIEW}/train.records.json",
                                       EXCLUSIONS, f"{VIEW}/val.records.json",
                                       f"{VIEW}/ablation_dev.records.json"]},
        "arm_a_exposure": {
            "selection_sha256": selections[0]["selection_sha256"],
            "selected_records": len(selected),
            "selected_record_ids_sha256": id_hash(selected),
            "consistent_with_training_metadata": consistent,
            "multi_mutant_examples_used": mm,
            "supervision_targets": ("first 3 policy-valid, reference-valid, killing upstream "
                                    "tests of each selected record (evaluate_pair)"),
            "training_exclusions_file_records": len(excluded)},
        "cohorts": {
            "exposed_selected_function_records": profile(exposed_selected),
            "exposed_group_unselected_siblings": {
                **profile(exposed_sibling),
                "siblings_sharing_a_test_with_a_selected_record": shared},
            "unexposed_lineage_disjoint_function_records": profile(unexposed)},
        "disjointness": {
            "exposed_vs_unexposed_groups": len(exposed_groups & {r["group_id"] for r in unexposed}),
            "train_vs_val_groups": len(all_train_groups & held["val"]),
            "train_vs_ablation_dev_groups": len(all_train_groups & held["ablation_dev"])},
        "external_cohort_options": [
            {"cohort": "val / ablation_dev", "lineage_disjoint": True,
             "usable": False,
             "why": ("spent; the protocol permits only descriptive analysis of retained "
                     "outputs, and they share the synthetic MBPP/HumanEval mutation "
                     "distribution, so they are unexposed but not external")},
            {"cohort": "repository-native A-prime admitted targets", "lineage_disjoint": True,
             "usable": False,
             "why": ("genuinely external, but native qualification (installed environments, "
                     "official tests on both revisions) has not been run, so no fixed input "
                     "can be executed on buggy and fixed revisions; 5 admitted targets in the "
                     "fresh pilot")},
            {"cohort": "reserved confirmation / consumed test / sealed final",
             "usable": False, "why": "forbidden"}],
    }
    census["h4_capability"] = {
        "exposure_effect_testable": bool(exposed_selected and unexposed),
        "distribution_shift_testable": False,
        "consequence": ("exposed-vs-unexposed comparison can support or weaken memorisation/"
                        "exposure failure and can weaken distribution shift; with no adequate "
                        "external cohort, distribution shift cannot be supported and H4's "
                        "distribution-shift arm stays open"),
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(census, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({k: census[k] for k in ("arm_a_exposure", "disjointness")}, indent=1))
    for name, block in census["cohorts"].items():
        print(name, {k: block[k] for k in block if k not in ("bug_family",)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
