"""Freeze the train-derived pilot panel for the execution-feedback experiment.

Search space: the hash-verified TRAIN shard of the development view only.  A
record qualifies when it is function-mode and its lineage is in none of:

* any training arm of the frozen control or later adapters (the 1,024-row
  control, the prompt-only arm, both trace arms, the dose treatment);
* the 100 pilot-development lineages, which host the 97-item mechanism panel
  and the 613-record retention panel;
* the 100 unopened confirmation lineages.

At most ``PER_LINEAGE_CAP`` records per lineage are kept, by stable rank.

The receipt is honest about what the panel is NOT.  Every train lineage,
these included, appears in the O1 oracle-dataset artifact: base-model
generations on them were produced and labelled for an earlier training-side
diagnostic.  The panel was never evaluated with the frozen control adapter
and is outside every training arm, but it is not untouched.  It is also
MBPP-only, simple/moderate-only, and has no repository records, because no
repository project or HumanEval lineage is free.  Results on it are an
exploratory pilot, never confirmatory.

Only the train shard is opened; the canonical records.json and every other
split stay closed.  CPU only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus_view import load_development_split
from harness.evaluation_admission import execution_mode_of, is_function_mode, scope_split
from harness.execution_supervision_sidecar import sha256_file, stable_rank
from harness.source_identity import canonical_sha256
from scripts.census_execution_dose_pool import CORPUS, corpus_disjointness_problems

SCHEMA = "oneiros_tool_assisted_panel_v1"
PANEL_NAME = "tool_assisted_pilot_train_derived"
PER_LINEAGE_CAP = 8
SOURCE_DIR = ROOT / "results" / "v4_3_execution_supervision_v1"
TRAINING_ARMS = (
    SOURCE_DIR / "arm_a.control.json",
    SOURCE_DIR / "arm_b.treatment.json",
    ROOT / "results" / "v4_3_execution_trace_v1" / "arm_unordered_events.json",
    ROOT / "results" / "v4_3_execution_trace_v1" / "arm_ordered_trace.json",
    ROOT / "results" / "v4_3_execution_dose_v1" / "arm_dose_treatment.json",
)
O1_CANDIDATES = ROOT / "results" / "v4_2_oracle_dataset_full" / "candidates.json"


def lineage_of(record: dict) -> str:
    return str(record.get("group_id") or record["id"])


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_tool_assisted_panel.json")
    args = parser.parse_args(argv)
    problems = corpus_disjointness_problems(CORPUS)
    if problems:
        raise SystemExit("REFUSED: " + "; ".join(problems))
    split = json.loads((SOURCE_DIR / "lineage_split.json").read_text(encoding="utf-8"))
    pilot = set(split["pilot_development_lineages"])
    confirm = set(split["unopened_confirmation_lineages"])
    trained: set[str] = set()
    arm_hashes = {}
    for path in TRAINING_ARMS:
        rows = json.loads(path.read_text(encoding="utf-8"))
        trained |= {str(row["function_lineage"]) for row in rows}
        arm_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    retention = json.loads((ROOT / "results" / "v4_3_execution_dose_retention_panel.json")
                           .read_text(encoding="utf-8"))
    mechanism = json.loads((SOURCE_DIR / "pilot_development.execution.json")
                           .read_text(encoding="utf-8"))
    excluded = trained | pilot | confirm

    records = load_development_split(CORPUS, "train")
    by_lineage: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if is_function_mode(execution_mode_of(record)) and lineage_of(record) not in excluded:
            by_lineage[lineage_of(record)].append(record)
    panel: list[dict] = []
    for lineage in sorted(by_lineage):
        ranked = sorted(by_lineage[lineage],
                        key=lambda r: stable_rank("tool_assisted", lineage, r["id"]))
        panel.extend(ranked[:PER_LINEAGE_CAP])
    panel.sort(key=lambda r: stable_rank("tool_assisted_order", r["id"]))
    ids = [str(r["id"]) for r in panel]
    lineages = {lineage_of(r) for r in panel}

    o1 = json.loads(O1_CANDIDATES.read_text(encoding="utf-8"))
    o1_lineages = {str(row["function_lineage"]) for row in o1}
    tiers = {}
    for row in o1:
        tiers.setdefault(str(row["record_id"]), row.get("complexity_tier"))
    from scripts.build_execution_supervision_dataset import _complexity_tier
    from utils.dataset_identity import dataset_name_from_source
    checks = {
        "no_training_arm_lineage": not (lineages & trained),
        "no_mechanism_or_retention_lineage": not (lineages & pilot),
        "no_retention_record": not (set(ids) & set(retention["record_ids"])),
        "no_mechanism_record": not (set(ids) & {str(r["record_id"]) for r in mechanism}),
        "no_confirmation_lineage": not (lineages & confirm),
        "ids_unique": len(ids) == len(set(ids)),
    }
    if not all(checks.values()):
        raise SystemExit(f"REFUSED: {checks}")
    scope = scope_split({PANEL_NAME: ids}, records, PANEL_NAME)
    if scope.target_count != len(ids):
        raise SystemExit("REFUSED: admission excluded panel records")
    receipt = {
        "schema_version": SCHEMA,
        "label": "train-derived exploratory pilot panel; not untouched, not confirmatory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "panel_name": PANEL_NAME,
        "selection_rule": (f"function-mode train records whose lineage is outside every "
                           f"training arm, the pilot-development and confirmation "
                           f"lineages; at most {PER_LINEAGE_CAP} per lineage by stable rank"),
        "records": len(ids), "lineages": len(lineages),
        "records_per_lineage_max": max(Counter(lineage_of(r) for r in panel).values()),
        "source_counts": dict(sorted(Counter(
            dataset_name_from_source(r.get("source") or {}) for r in panel).items())),
        "complexity_tier_counts": dict(sorted(Counter(
            tiers.get(str(r["id"])) or _complexity_tier(r) for r in panel).items())),
        "record_ids": ids, "record_ids_sha256": ids_sha256(ids),
        "record_lineages": {str(r["id"]): lineage_of(r) for r in panel},
        "admission_scope_sha256": scope.scope_sha256(),
        "checks": checks,
        "qualification": {
            "lineage_disjoint_from_training_arms_and_panels": True,
            "previously_inspected": True,
            "previous_inspection": ("base-model generations on every train lineage, these "
                                    "included, were labelled in the O1 oracle dataset "
                                    "(results/v4_2_oracle_dataset_full); the frozen control "
                                    "adapter has never been evaluated on them"),
            "lineages_in_o1_artifact": len(lineages & o1_lineages),
            "untouched_train_function_lineages_available": 0,
            "complex_tier_available": False,
            "humaneval_available": False,
            "repository_available": False,
            "repository_reason": "all 13 train repository projects are in the control arm",
            "evidence_class": "exploratory pilot; any positive result needs confirmation",
        },
        "inputs": {"lineage_split_sha256": sha256_file(SOURCE_DIR / "lineage_split.json"),
                   "training_arms_sha256": arm_hashes,
                   "retention_panel_sha256": sha256_file(
                       ROOT / "results" / "v4_3_execution_dose_retention_panel.json"),
                   "mechanism_panel_sha256": sha256_file(
                       SOURCE_DIR / "pilot_development.execution.json"),
                   "o1_candidates_sha256": sha256_file(O1_CANDIDATES),
                   "development_view_manifest_sha256": sha256_file(
                       CORPUS / "development_view" / "manifest.json")},
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False,
                    "canonical_records_json_opened": False},
        "source_sha256": canonical_sha256(Path(__file__)),
    }
    args.output.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: receipt[key] for key in (
        "records", "lineages", "records_per_lineage_max", "source_counts",
        "complexity_tier_counts", "record_ids_sha256", "checks")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
