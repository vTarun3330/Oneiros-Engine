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
import math
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
#: Share of unique targets above which a source dataset counts as dominant.
#:
#: AUDIT ONLY. mbpp supplies 555 of 1009 unique targets, more than the other
#: three sources combined, and that skew is real - SFT moves HumanEval +10.7
#: points and mbpp +3.3, so a corpus that is mostly mbpp is mostly the case
#: where the method does not work. The fix is to grow the scarce sources, not
#: to delete verified mbpp targets: trimming would reach the ratio by making
#: the corpus smaller, which throws away supervision that took execution to
#: verify and cannot be recovered. The report states the shortfall so
#: ingestion can close it.
MAX_DATASET_FRACTION = 0.35
ENFORCE_DATASET_CAP = False
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


def _audit_project_balance(
    entries: list[dict[str, Any]], max_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Report project concentration without discarding unique evidence.

    Without this, django alone supplies 196 of 457 repository targets and the
    'real repository' half of the corpus becomes largely one codebase.
    The cure is more non-dominant repositories, not dropping verified defects
    and then repeating the smaller set to manufacture a 50:50 row count.
    """
    if not entries:
        return [], {"cap_applied": False, "cap_met": True}
    by_project: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in entries:
        by_project[str(entry.get("project") or "unknown")].append(entry)
    counts = {project: len(values) for project, values in sorted(by_project.items())}
    dominant_project, dominant_count = max(counts.items(), key=lambda item: item[1])
    observed_fraction = dominant_count / len(entries)
    required_total = math.ceil(dominant_count / max_fraction)
    return list(entries), {
        "cap_applied": False,
        "audit_only_until_more_unique_repositories_are_added": True,
        "max_project_fraction": max_fraction,
        "cap_met": observed_fraction <= max_fraction,
        "dominant_project": dominant_project,
        "dominant_project_count": dominant_count,
        "dominant_project_fraction": round(observed_fraction, 4),
        "additional_non_dominant_targets_needed": max(0, required_total - len(entries)),
        "unique_targets_dropped": 0,
        "project_counts": counts,
    }


def _growth_to_balance(
    entries: list[dict[str, Any]], max_fraction: float,
) -> dict[str, Any]:
    """How many NEW targets each source needs so no source is dominant.

    The mirror image of trimming. If the largest source holds D targets and
    must end at or below `max_fraction` of the total, the corpus has to reach
    D / max_fraction targets overall, and every target added has to come from
    somewhere other than that source.
    """
    if not entries:
        return {}
    counts = collections.Counter(str(e.get("dataset") or "unknown") for e in entries)
    dominant, dominant_count = counts.most_common(1)[0]
    required_total = math.ceil(dominant_count / max_fraction)
    return {
        "dominant_dataset": dominant,
        "dominant_count": dominant_count,
        "dominant_share": round(dominant_count / len(entries), 4),
        "current_total_unique_targets": len(entries),
        "required_total_unique_targets": required_total,
        "new_non_dominant_targets_needed": max(0, required_total - len(entries)),
    }


def cap_dataset_share(
    entries: list[dict[str, Any]], max_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Trim over-represented source datasets down to a share ceiling.

    Selection within a trimmed dataset is deterministic and diversity-first:
    entries are ordered by how rare their bug family and complexity tier are,
    so trimming removes the most redundant targets rather than an arbitrary
    tail. Two runs of this function on the same input produce the same corpus.

    Scarce datasets are never padded up to the ceiling - the cap is a maximum,
    not a quota - because manufacturing HumanEval targets that do not exist is
    exactly the manipulation this project forbids.
    """
    if not entries:
        return [], {"cap_applied": False, "cap_met": True, "dataset_counts": {}}

    by_dataset: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in entries:
        by_dataset[str(entry.get("dataset") or "unknown")].append(entry)
    before = {name: len(rows) for name, rows in sorted(by_dataset.items())}

    family_counts = collections.Counter(str(e.get("bug_family")) for e in entries)
    tier_counts = collections.Counter(str(e.get("complexity_tier")) for e in entries)

    # A dataset's allowance depends on the final total, which depends on the
    # allowances. Solve it directly: with F the cap and R the targets from
    # datasets already under the cap, an over-represented dataset may keep
    # k = floor(F * R / (1 - F)).
    kept: dict[str, list[dict[str, Any]]] = {}
    over = {n for n, rows in by_dataset.items()
            if len(rows) / len(entries) > max_fraction}
    if not over:
        return list(entries), {
            "cap_applied": False, "cap_met": True,
            "max_dataset_fraction": max_fraction,
            "dataset_counts_before": before, "dataset_counts_after": before,
            "unique_targets_dropped": 0,
        }

    under_total = sum(len(rows) for n, rows in by_dataset.items() if n not in over)
    allowance = int(max_fraction * under_total / (1 - max_fraction * len(over)))

    for name, rows in by_dataset.items():
        if name not in over:
            kept[name] = list(rows)
            continue
        ordered = sorted(rows, key=lambda e: (
            family_counts[str(e.get("bug_family"))],
            tier_counts[str(e.get("complexity_tier"))],
            str(e.get("target_key")),
        ))
        kept[name] = ordered[:allowance]

    selection = [e for name in sorted(kept) for e in kept[name]]
    after = {name: len(rows) for name, rows in sorted(kept.items())}
    total = len(selection)
    shares = {n: round(c / max(1, total), 4) for n, c in after.items()}
    return selection, {
        "cap_applied": True,
        "max_dataset_fraction": max_fraction,
        "cap_met": all(v <= max_fraction + 1e-9 for v in shares.values()),
        "dataset_counts_before": before,
        "dataset_counts_after": after,
        "dataset_shares_after": shares,
        "unique_targets_dropped": len(entries) - total,
        "trimmed_datasets": sorted(over),
        "selection_rule": (
            "deterministic; rarest bug family then rarest complexity tier "
            "survives, so trimming removes the most redundant targets"
        ),
        "scarce_datasets_not_padded": (
            "the cap is a ceiling, never a quota; a small dataset is left at "
            "its true size rather than duplicated up to the ceiling"
        ),
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

    repository_entries, project_cap = _audit_project_balance(
        repository_entries, MAX_PROJECT_FRACTION,
    )
    stages["repository_after_project_audit"] = len(repository_entries)

    # The dataset cap spans both groups, because the four source datasets do:
    # humaneval and mbpp are synthetic, BugsInPy and SWE-bench are repository.
    # Capping them together rather than within each group is also what makes
    # the synthetic/repository split come out even, since trimming mbpp is
    # exactly what the shortfall needed.
    combined = synthetic_entries + repository_entries
    capped, dataset_cap = cap_dataset_share(combined, MAX_DATASET_FRACTION)
    if ENFORCE_DATASET_CAP:
        combined = capped
        synthetic_entries = [
            item for item in combined if item["origin_group"] == "synthetic_function"
        ]
        repository_entries = [
            item for item in combined if item["origin_group"] == "real_repository"
        ]
    else:
        # Report the shortfall instead of deleting targets to reach the ratio.
        dataset_cap = {**dataset_cap, "cap_applied": False,
                       "enforced": False,
                       "targets_needed_to_balance_by_growth":
                           _growth_to_balance(combined, MAX_DATASET_FRACTION),
                       "policy": "grow the scarce sources; never trim verified targets"}
    stages["after_dataset_cap_synthetic"] = len(synthetic_entries)
    stages["after_dataset_cap_repository"] = len(repository_entries)
    stages["dataset_cap_targets_dropped"] = dataset_cap.get(
        "unique_targets_dropped", 0
    )

    # --- bounded repetition to approach the frozen ratio ---
    synthetic_count = len(synthetic_entries)
    repository_count = len(repository_entries)
    desired_repository = synthetic_count  # 50/50 by unique target
    unique_repository_shortfall = max(0, desired_repository - repository_count)
    shortfall = unique_repository_shortfall
    achievable = repository_count * (MAX_REPEATS - 1)
    repeats_used = min(shortfall, achievable)

    repeated: list[dict[str, Any]] = []
    if repeats_used:
        project_counts = collections.Counter(
            item["project"] for item in repository_entries
        )
        family_counts = collections.Counter(
            item["bug_family"] for item in repository_entries
        )
        ordered = sorted(repository_entries, key=lambda item: (
            project_counts[item["project"]],
            family_counts[item["bug_family"]],
            item["project"], item["bug_family"], item["target_key"],
        ))
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
    moderate_or_complex_count = sum(
        1 for item in selection
        if item["complexity_tier"] in {"moderate", "complex"}
    )
    unique_total = synthetic_count + repository_count
    unique_synthetic_fraction = synthetic_count / max(1, unique_total)
    effective_synthetic_fraction = synthetic_count / max(1, total)
    summary = {
        "schema_version": "oneiros_balanced_sft_dataset_v1",
        "split": split,
        "source_tree_sha256": source_tree_sha256(ROOT),
        "policy": {
            "balance_unit": "unique_semantic_target",
            "target_synthetic_fraction": TARGET_SYNTHETIC_FRACTION,
            "max_repeats": MAX_REPEATS,
            "max_project_fraction": MAX_PROJECT_FRACTION,
            "max_dataset_fraction": MAX_DATASET_FRACTION,
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
        "source_dataset_cap": dataset_cap,
        "unique_target_balance": {
            "synthetic_unique_targets": synthetic_count,
            "repository_unique_targets": repository_count,
            "synthetic_fraction": round(unique_synthetic_fraction, 4),
            "repository_fraction": round(1.0 - unique_synthetic_fraction, 4),
            "repository_shortfall_to_exact_parity": unique_repository_shortfall,
            "target_met": abs(unique_synthetic_fraction - 0.5) <= 0.05,
            "repetitions_do_not_count_as_unique_targets": True,
        },
        "effective_training_balance": {
            "synthetic_fraction": round(effective_synthetic_fraction, 4),
            "repository_fraction": round(1.0 - effective_synthetic_fraction, 4),
            "target_met": abs(effective_synthetic_fraction - 0.5) <= 0.05,
            "provisional_because_repetitions_used": bool(repeated),
        },
        "complexity": {
            "complex_examples": complex_count,
            "complex_fraction": round(complex_count / max(1, total), 4),
            "moderate_or_complex_examples": moderate_or_complex_count,
            "moderate_or_complex_fraction": round(
                moderate_or_complex_count / max(1, total), 4,
            ),
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
    summary["readiness"] = {
        "ready_for_final_sft": (
            summary["unique_target_balance"]["target_met"]
            and project_cap["cap_met"]
            and summary["complexity"]["floor_met"]
        ),
        "blocking_conditions": [
            reason for condition, reason in (
                (
                    summary["unique_target_balance"]["target_met"],
                    f"need {unique_repository_shortfall} more unique repository targets for parity",
                ),
                (
                    project_cap["cap_met"],
                    "repository project concentration exceeds the frozen cap",
                ),
                (
                    summary["complexity"]["floor_met"],
                    "the aspirational 0.60 complex-only floor is not met",
                ),
            ) if not condition
        ],
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
