"""Freeze the train-derived canonical test-generation retention panel.

The execution-dose experiment accepts an auxiliary-task gain only if canonical
test generation does not collapse.  That needs a Kill@8 panel that neither arm
trained on and that sits outside every protected split.  The 100
pilot-development lineages satisfy both: they were held out of the frozen
control and are held out of the dose treatment, and they come from the train
shard of the development view.

Selection is fixed before any model is evaluated on it: every function-mode
train record in a pilot-development lineage, at most ``PER_LINEAGE_CAP`` per
lineage in stable-rank order.  The panel is written as identifiers only.

Only the train shard is opened.  Validation, ablation_dev, test, sealed-final
and the unopened confirmation lineages are not read.  CPU only.
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
from harness.evaluation_admission import scope_split
from harness.execution_supervision_sidecar import sha256_file, stable_rank
from scripts.census_execution_dose_pool import CORPUS, SOURCE_DIR, corpus_disjointness_problems

SCHEMA = "oneiros_execution_dose_retention_panel_v1"
PANEL_NAME = "execution_dose_retention_train_derived"
PER_LINEAGE_CAP = 10


def lineage_of(record: dict) -> str:
    return str(record.get("group_id") or record["id"])


def select_panel(records: list[dict], pilot_lineages: set[str],
                 cap: int = PER_LINEAGE_CAP) -> list[dict]:
    """Function-mode records in pilot lineages, capped per lineage, stable order."""
    from harness.evaluation_admission import execution_mode_of, is_function_mode
    by_lineage: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        lineage = lineage_of(record)
        if lineage in pilot_lineages and is_function_mode(execution_mode_of(record)):
            by_lineage[lineage].append(record)
    panel: list[dict] = []
    for lineage in sorted(by_lineage):
        ranked = sorted(by_lineage[lineage],
                        key=lambda record: stable_rank("retention", lineage, record["id"]))
        panel.extend(ranked[:cap])
    panel.sort(key=lambda record: stable_rank("retention_order", record["id"]))
    return panel


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_execution_dose_retention_panel.json")
    args = parser.parse_args(argv)

    problems = corpus_disjointness_problems(CORPUS)
    source_manifest = json.loads((SOURCE_DIR / "manifest.json").read_text(encoding="utf-8"))
    split_path = SOURCE_DIR / "lineage_split.json"
    if sha256_file(split_path) != source_manifest["lineage_split_sha256"]:
        problems.append("frozen lineage split hash differs from its manifest")
    if problems:
        print("REFUSED: " + "; ".join(problems))
        return 2
    split = json.loads(split_path.read_text(encoding="utf-8"))
    pilot = set(split["pilot_development_lineages"])
    confirm = set(split["unopened_confirmation_lineages"])
    train_lineages = set(split["train_lineages"])

    records = load_development_split(CORPUS, "train")
    panel = select_panel(records, pilot)
    ids = [str(record["id"]) for record in panel]
    lineages = {lineage_of(record) for record in panel}
    control = json.loads((SOURCE_DIR / "arm_a.control.json").read_text(encoding="utf-8"))
    if sha256_file(SOURCE_DIR / "arm_a.control.json") != source_manifest["arm_a_sha256"]:
        print("REFUSED: frozen control arm hash mismatch")
        return 2
    trained_ids = {str(row["record_id"]) for row in control}
    trained_lineages = {str(row["function_lineage"]) for row in control}
    checks = {
        "all_lineages_are_pilot_development": lineages <= pilot,
        "no_confirmation_lineage": not (lineages & confirm),
        "no_train_lineage": not (lineages & train_lineages),
        "no_record_in_frozen_control": not (set(ids) & trained_ids),
        "no_lineage_in_frozen_control": not (lineages & trained_lineages),
        "ids_unique": len(ids) == len(set(ids)),
    }
    if not all(checks.values()):
        print(f"REFUSED: {checks}")
        return 2
    # Admission is the shared function-mode gate every Kill@8 run uses; the
    # panel must pass it with no exclusions.
    scope = scope_split({PANEL_NAME: ids}, records, PANEL_NAME)
    if scope.target_count != len(ids) or scope.excluded_repository or scope.excluded_unknown_mode:
        print("REFUSED: admission excluded panel records")
        return 2

    from utils.dataset_identity import dataset_name_from_source
    receipt = {
        "schema_version": SCHEMA,
        "label": ("train-derived canonical test-generation retention panel; "
                  "not validation, not generalisation, not a final-test panel"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "panel_name": PANEL_NAME,
        "selection_rule": (f"every function-mode train record in a pilot-development "
                           f"lineage, at most {PER_LINEAGE_CAP} per lineage by "
                           "stable_rank('retention', lineage, id)"),
        "records": len(ids),
        "lineages": len(lineages),
        "records_per_lineage_max": max(Counter(lineage_of(r) for r in panel).values()),
        "source_counts": dict(sorted(Counter(
            dataset_name_from_source(r.get("source") or {}) for r in panel).items())),
        "record_ids": ids,
        "record_ids_sha256": ids_sha256(ids),
        "record_lineages": {str(r["id"]): lineage_of(r) for r in panel},
        "admission_scope_sha256": scope.scope_sha256(),
        "checks": checks,
        "inputs": {
            "lineage_split_sha256": sha256_file(split_path),
            "control_arm_sha256": source_manifest["arm_a_sha256"],
            "development_view_manifest_sha256": sha256_file(
                CORPUS / "development_view" / "manifest.json"),
        },
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False},
        "source_sha256": sha256_file(Path(__file__)),
    }
    args.output.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: receipt[key] for key in (
        "records", "lineages", "records_per_lineage_max", "source_counts",
        "record_ids_sha256", "checks")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
