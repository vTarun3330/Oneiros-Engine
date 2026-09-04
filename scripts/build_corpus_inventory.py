"""Inventory the development corpus by origin group, defect family, and tier.

Produces the artifacts Parts 2, 3, and 5 of the research plan require:

* a synthetic-versus-repository inventory counted by UNIQUE SEMANTIC TARGET,
  not by raw mutation row - the two differ by roughly an order of magnitude on
  the synthetic side and a raw row count badly misstates the real balance;
* the real-world defect taxonomy with classification confidence;
* the repository complexity index.

Reads the development view only.  The sealed test split is never opened.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import sha256_file, write_json
from harness.corpus_annotation import (
    ANNOTATION_SCHEMA_VERSION,
    RecordAnnotation,
    annotate_split,
    annotations_sha256,
)
from harness.repository_complexity import (
    REPOSITORY_COMPLEXITY_POLICY_VERSION,
    REPOSITORY_COMPLEX_THRESHOLDS,
    REPOSITORY_MODERATE_THRESHOLDS,
)
from harness.repository_defect_taxonomy import DEFECT_FAMILIES, DEFECT_TAXONOMY_VERSION
from utils.reproducibility import source_tree_sha256


DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)
DEFAULT_OUTPUT = ROOT / "data" / "training_views" / "corpus_inventory_v1"
SPLITS = ("train", "ablation_dev", "val")


def _counter(values: Any) -> dict[str, int]:
    return dict(sorted(collections.Counter(values).items()))


def _unique_targets(annotations: list[RecordAnnotation], group: str) -> set[str]:
    return {
        item.function_lineage for item in annotations if item.origin_group == group
    }


def summarize_split(annotations: list[RecordAnnotation]) -> dict[str, Any]:
    repository = [item for item in annotations if item.origin_group == "real_repository"]
    synthetic = [item for item in annotations if item.origin_group == "synthetic_function"]
    manual = [item for item in annotations if item.origin_group == "manual_curated"]

    unique_synthetic = _unique_targets(annotations, "synthetic_function")
    unique_repository = _unique_targets(annotations, "real_repository")
    unique_total = len(unique_synthetic) + len(unique_repository)

    return {
        "records": {
            "total": len(annotations),
            "synthetic_function": len(synthetic),
            "real_repository": len(repository),
            "manual_curated": len(manual),
        },
        "unique_semantic_targets": {
            "synthetic_function": len(unique_synthetic),
            "real_repository": len(unique_repository),
            "manual_curated": len(_unique_targets(annotations, "manual_curated")),
            "total_excluding_manual": unique_total,
        },
        "effective_balance_by_unique_target": {
            "synthetic_fraction": round(
                len(unique_synthetic) / max(1, unique_total), 4
            ),
            "repository_fraction": round(
                len(unique_repository) / max(1, unique_total), 4
            ),
            "note": (
                "Counted by unique semantic target. Counted by raw rows the "
                "synthetic share would look far larger, because one benchmark "
                "function contributes many sibling mutation rows while one "
                "repository defect contributes exactly one."
            ),
        },
        "records_by_dataset": _counter(item.source_dataset for item in annotations),
        "synthetic_mutation_families": _counter(
            item.primary_bug_family for item in synthetic
        ),
        "repository_defect_families": _counter(
            item.primary_bug_family for item in repository
        ),
        "repository_defect_family_unique_targets": {
            family: len({
                item.function_lineage for item in repository
                if item.primary_bug_family == family
            })
            for family in sorted({item.primary_bug_family for item in repository})
        },
        "repository_classification_confidence": _counter(
            item.classification_confidence for item in repository
        ),
        "repository_classification_method": _counter(
            item.classification_method for item in repository
        ),
        "repository_projects": _counter(
            str(item.project or "unknown") for item in repository
        ),
        "repository_unique_projects": len({
            str(item.project or "unknown") for item in repository
        }),
        "test_frameworks": _counter(item.test_framework for item in annotations),
        "complexity_tiers": {
            "synthetic_function": _counter(item.complexity_tier for item in synthetic),
            "real_repository": _counter(item.complexity_tier for item in repository),
            "manual_curated": _counter(item.complexity_tier for item in manual),
        },
        "complex_fraction": {
            "synthetic_function": round(
                sum(item.complexity_tier == "complex" for item in synthetic)
                / max(1, len(synthetic)), 4,
            ),
            "real_repository": round(
                sum(item.complexity_tier == "complex" for item in repository)
                / max(1, len(repository)), 4,
            ),
        },
        "repository_parse_status": _counter(
            str(item.complexity_metrics.get("parse_status", "n/a"))
            for item in repository
        ),
    }


def build(view_dir: Path, output_dir: Path) -> dict[str, Any]:
    per_split: dict[str, list[RecordAnnotation]] = {}
    for split in SPLITS:
        path = view_dir / f"{split}.records.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        per_split[split] = annotate_split(records, split)

    all_annotations = [item for items in per_split.values() for item in items]

    # Project disjointness is a split-isolation guarantee, so it is verified
    # here rather than assumed from how the corpus was built.
    projects_by_split = {
        split: {
            str(item.project) for item in items
            if item.origin_group == "real_repository" and item.project
        }
        for split, items in per_split.items()
    }
    overlaps = {}
    split_names = list(projects_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            shared = sorted(projects_by_split[left] & projects_by_split[right])
            if shared:
                overlaps[f"{left}|{right}"] = shared

    inventory = {
        "schema_version": "oneiros_corpus_inventory_v1",
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "defect_taxonomy_version": DEFECT_TAXONOMY_VERSION,
        "repository_complexity_policy_version": REPOSITORY_COMPLEXITY_POLICY_VERSION,
        "source_view": str(view_dir.relative_to(ROOT)).replace("\\", "/"),
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_split_opened": False,
        "defect_family_vocabulary": list(DEFECT_FAMILIES),
        "repository_complexity_thresholds": {
            "complex": REPOSITORY_COMPLEX_THRESHOLDS,
            "moderate": REPOSITORY_MODERATE_THRESHOLDS,
            "rule": "any-of: a region reaching ANY threshold takes that tier",
        },
        "splits": {split: summarize_split(items) for split, items in per_split.items()},
        "repository_project_disjointness": {
            "projects_by_split": {
                split: sorted(values) for split, values in projects_by_split.items()
            },
            "overlaps": overlaps,
            "disjoint": not overlaps,
        },
        "annotations_sha256": annotations_sha256(all_annotations),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in per_split.items():
        write_json(
            output_dir / f"{split}.annotations.json",
            [item.to_dict() for item in items],
        )
    inventory["annotation_files"] = {
        split: {
            "filename": f"{split}.annotations.json",
            "record_count": len(items),
            "sha256": sha256_file(output_dir / f"{split}.annotations.json"),
        }
        for split, items in per_split.items()
    }
    write_json(output_dir / "inventory.json", inventory)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    inventory = build(arguments.view_dir, arguments.output_dir)
    print(json.dumps(
        {
            "splits": {
                split: {
                    "records": value["records"],
                    "unique_semantic_targets": value["unique_semantic_targets"],
                    "effective_balance": value["effective_balance_by_unique_target"],
                }
                for split, value in inventory["splits"].items()
            },
            "repository_project_disjoint": inventory[
                "repository_project_disjointness"
            ]["disjoint"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
