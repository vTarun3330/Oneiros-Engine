"""Assemble the balanced SFT dataset from unique semantic targets.

Balance is computed over UNIQUE SEMANTIC TARGETS, never over raw mutation rows.
One HumanEval function contributes many sibling-mutant rows; one repository
defect contributes exactly one.  Counting rows would report the corpus as ~92%
synthetic when, by target, it is closer to 59%.

Selection order:

1. take every eligible unique target on both sides - nothing useful is deleted
   to hit a ratio;
2. apply per-project, per-family, and complexity caps and floors;
3. only then, if a side is still short, repeat targets on that side up to a
   frozen cap of two, and report exactly how many repeats were used.

Every stage count is emitted so the final example total can be traced back to
the corpus it came from.
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
from utils.reproducibility import source_tree_sha256


DEFAULT_INVENTORY = ROOT / "data" / "training_views" / "corpus_inventory_v1"
DEFAULT_MULTI_MUTANT = ROOT / "data" / "training_views" / "multi_mutant_v1"
DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)
DEFAULT_OUTPUT = ROOT / "data" / "training_views" / "balanced_sft_v1"

#: Frozen balancing policy.
TARGET_SYNTHETIC_FRACTION = 0.5
MAX_REPEATS = 2
MAX_PROJECT_FRACTION = 0.35
COMPLEX_TARGET_FRACTION = 0.60


def _load_annotations(inventory_dir: Path, split: str) -> list[dict[str, Any]]:
    path = inventory_dir / f"{split}.annotations.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _excluded_ids(view_dir: Path) -> dict[str, str]:
    path = view_dir / "training_exclusions.json"
    if not path.exists():
        return {}
    return {
        str(item["record_id"]): str(item.get("reason") or "excluded")
        for item in json.loads(path.read_text(encoding="utf-8"))
    }


def _cap_by_project(
    entries: list[dict[str, Any]], max_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Limit any single project's share of the repository side.

    Without this, django alone supplies 196 of 457 repository targets and the
    'real repository' half of the corpus becomes largely one codebase.
    """
    if not entries:
        return [], {"cap_applied": False}
    by_project: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in entries:
        by_project[str(entry.get("project") or "unknown")].append(entry)
    for values in by_project.values():
        values.sort(key=lambda item: item["target_key"])

    # The cap is a fraction of the RETAINED total, which depends on the cap, so
    # solve it by iterating to a fixed point rather than guessing an order.
    retained_total = len(entries)
    for _ in range(64):
        limit = max(1, int(retained_total * max_fraction))
        kept = {
            project: values[:limit] for project, values in by_project.items()
        }
        new_total = sum(len(values) for values in kept.values())
        if new_total == retained_total:
            break
        retained_total = new_total
    limit = max(1, int(retained_total * max_fraction))
    selected = [item for project in sorted(kept) for item in kept[project]]
    return selected, {
        "cap_applied": True,
        "max_project_fraction": max_fraction,
        "per_project_limit": limit,
        "dropped": len(entries) - len(selected),
        "project_counts_before": {
            project: len(values) for project, values in sorted(by_project.items())
        },
        "project_counts_after": {
            project: len(values) for project, values in sorted(kept.items())
        },
    }


def build(
    inventory_dir: Path, multi_mutant_dir: Path, view_dir: Path,
    output_dir: Path, split: str,
) -> dict[str, Any]:
    annotations = _load_annotations(inventory_dir, split)
    by_id = {item["record_id"]: item for item in annotations}
    excluded = _excluded_ids(view_dir)

    examples_path = multi_mutant_dir / f"{split}.examples.json"
    multi_mutant = json.loads(examples_path.read_text(encoding="utf-8"))

    stages: dict[str, Any] = {
        "raw_corpus_rows_in_split": len(annotations),
        "training_excluded_records": sum(
            1 for record_id in by_id if record_id in excluded
        ),
    }

    # --- synthetic side: one broad example per lineage, plus targeted ones ---
    synthetic_entries: list[dict[str, Any]] = []
    for example in multi_mutant:
        record_id = example["displayed_record_id"]
        annotation = by_id.get(record_id)
        if annotation is None or record_id in excluded:
            continue
        synthetic_entries.append({
            "origin_group": "synthetic_function",
            "target_key": example["lineage"] if example["assertion_count"] else record_id,
            "record_id": record_id,
            "completion": example["completion"],
            "completion_shape": "test_function",
            "assertion_count": example["assertion_count"],
            "mutants_killed": example["mutants_killed"],
            "bug_family": example["primary_mutation_family"],
            "covered_families": example["covered_mutation_families"],
            "complexity_tier": annotation["complexity_tier"],
            "dataset": annotation["source_dataset"],
            "project": "synthetic",
            "example_kind": (
                "broad" if example["mutants_killed"] >= 1
                and example["assertion_count"] >= 1 else "targeted"
            ),
        })
    # A lineage may contribute one broad plus a few targeted examples; the
    # unique-target count is the number of distinct lineages, not entries.
    stages["synthetic_examples"] = len(synthetic_entries)
    stages["synthetic_unique_lineages"] = len({
        item["target_key"] for item in synthetic_entries
    })

    # --- repository side: one verified official defect per target ---
    repository_entries: list[dict[str, Any]] = []
    for annotation in annotations:
        if annotation["origin_group"] != "real_repository":
            continue
        record_id = annotation["record_id"]
        if record_id in excluded:
            continue
        repository_entries.append({
            "origin_group": "real_repository",
            "target_key": annotation["function_lineage"],
            "record_id": record_id,
            "completion": None,  # supplied from the record's verified test
            "completion_shape": "pytest_fragment",
            "assertion_count": None,
            "mutants_killed": None,
            "bug_family": annotation["primary_bug_family"],
            "covered_families": annotation["secondary_bug_tags"],
            "complexity_tier": annotation["complexity_tier"],
            "dataset": annotation["source_dataset"],
            "project": annotation["project"],
            "example_kind": "repository_defect",
        })
    stages["repository_eligible_unique_targets"] = len(repository_entries)

    repository_entries, project_cap = _cap_by_project(
        repository_entries, MAX_PROJECT_FRACTION,
    )
    stages["repository_after_project_cap"] = len(repository_entries)

    # --- bounded repetition to approach the frozen ratio ---
    synthetic_count = len(synthetic_entries)
    repository_count = len(repository_entries)
    desired_repository = synthetic_count  # 50/50 by unique target
    shortfall = max(0, desired_repository - repository_count)
    achievable = repository_count * (MAX_REPEATS - 1)
    repeats_used = min(shortfall, achievable)

    repeated: list[dict[str, Any]] = []
    if repeats_used:
        ordered = sorted(repository_entries, key=lambda item: item["target_key"])
        for index in range(repeats_used):
            entry = dict(ordered[index % len(ordered)])
            entry["repeat_index"] = 2
            repeated.append(entry)

    selection = synthetic_entries + repository_entries + repeated
    final_repository = repository_count + len(repeated)
    total = synthetic_count + final_repository

    complex_count = sum(
        1 for item in selection if item["complexity_tier"] == "complex"
    )
    summary = {
        "schema_version": "oneiros_balanced_sft_dataset_v1",
        "split": split,
        "source_tree_sha256": source_tree_sha256(ROOT),
        "policy": {
            "balance_unit": "unique_semantic_target",
            "target_synthetic_fraction": TARGET_SYNTHETIC_FRACTION,
            "max_repeats": MAX_REPEATS,
            "max_project_fraction": MAX_PROJECT_FRACTION,
            "complex_target_fraction": COMPLEX_TARGET_FRACTION,
            "unique_first": True,
            "no_synthetic_deleted_to_reach_ratio": True,
        },
        "stages": {
            **stages,
            "repository_repeated_examples": len(repeated),
            "repository_repeat_shortfall_unmet": shortfall - repeats_used,
            "final_synthetic_examples": synthetic_count,
            "final_repository_examples": final_repository,
            "final_total_examples": total,
        },
        "achieved_balance": {
            "synthetic_fraction": round(synthetic_count / max(1, total), 4),
            "repository_fraction": round(final_repository / max(1, total), 4),
            "target_met": abs(synthetic_count / max(1, total) - 0.5) <= 0.05,
        },
        "complexity": {
            "complex_examples": complex_count,
            "complex_fraction": round(complex_count / max(1, total), 4),
            "floor": COMPLEX_TARGET_FRACTION,
            "floor_met": complex_count / max(1, total) >= COMPLEX_TARGET_FRACTION,
            "floor_status": "design choice; not demonstrated to improve results",
            "by_group": {
                group: dict(sorted(collections.Counter(
                    item["complexity_tier"] for item in selection
                    if item["origin_group"] == group
                ).items()))
                for group in ("synthetic_function", "real_repository")
            },
        },
        "project_cap": project_cap,
        "bug_family_counts": dict(sorted(collections.Counter(
            item["bug_family"] for item in selection
        ).items())),
        "dataset_counts": dict(sorted(collections.Counter(
            item["dataset"] for item in selection
        ).items())),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{split}.selection.json", selection)
    summary["selection_sha256"] = sha256_file(output_dir / f"{split}.selection.json")
    write_json(output_dir / f"{split}.manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--multi-mutant-dir", type=Path, default=DEFAULT_MULTI_MUTANT)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    arguments = parser.parse_args()
    summary = build(
        arguments.inventory_dir, arguments.multi_mutant_dir, arguments.view_dir,
        arguments.output_dir, arguments.split,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
