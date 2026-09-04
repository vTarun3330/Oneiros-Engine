"""Assemble the versioned SFT relearning dataset from development losers.

Reads one evaluation artifact for the frozen SFT model on a DEVELOPMENT split,
classifies the functions it failed, attaches verified supervision as the
correction, applies the balanced-replay caps, and writes a hashed dataset.

Refuses a validation or sealed-test artifact outright.  Never writes a model
output as a label.
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
from harness.relearning import (
    RELEARNING_SCHEMA_VERSION,
    assert_split_is_eligible,
    attach_corrections,
    balanced_replay,
    classify_loser,
    relearning_dataset_sha256,
)
from utils.reproducibility import source_tree_sha256


DEFAULT_INVENTORY = ROOT / "data" / "training_views" / "corpus_inventory_v1"
DEFAULT_MULTI_MUTANT = ROOT / "data" / "training_views" / "multi_mutant_v1"
DEFAULT_OUTPUT = ROOT / "data" / "training_views" / "relearning_v1"


def _load_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement"):
        raise SystemExit(f"Refusing {path}: sealed final-test measurement")
    split = str(payload.get("evaluation_split") or "")
    assert_split_is_eligible(split)
    if not isinstance(payload.get("function_results"), list):
        raise SystemExit(f"Refusing {path}: artifact has no function_results list")
    return payload


def _annotations(inventory_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    path = inventory_dir / f"{split}.annotations.json"
    if not path.exists():
        return {}
    return {
        str(item["record_id"]): item
        for item in json.loads(path.read_text(encoding="utf-8"))
    }


def _verified_completions(multi_mutant_dir: Path, split: str = "train") -> dict[str, str]:
    path = multi_mutant_dir / f"{split}.examples.json"
    if not path.exists():
        return {}
    completions: dict[str, str] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        if item.get("verified") and item.get("kills_displayed_target"):
            completions[str(item["displayed_record_id"])] = str(item["completion"])
    return completions


def build(
    evaluation: Path, base_evaluation: Path | None, inventory_dir: Path,
    multi_mutant_dir: Path, output_dir: Path,
    max_per_project: int, max_per_family: int, max_per_category: int,
) -> dict[str, Any]:
    payload = _load_evaluation(evaluation)
    split = str(payload.get("evaluation_split") or "")
    annotations = _annotations(inventory_dir, split)

    base_by_id: dict[str, dict[str, Any]] = {}
    if base_evaluation:
        base_payload = _load_evaluation(base_evaluation)
        base_by_id = {
            str(item.get("record_id")): item
            for item in base_payload.get("function_results", [])
        }

    losers = []
    for result in payload.get("function_results", []):
        record_id = str(result.get("record_id") or "")
        loser = classify_loser(
            result, split,
            base_result=base_by_id.get(record_id),
            model_run=str(evaluation.parent.name),
            checkpoint_step=payload.get("checkpoint_step"),
            seed=payload.get("seed"),
            prompt_version=str(payload.get("prompt_schema_version") or ""),
            annotation=annotations.get(record_id),
        )
        if loser is not None:
            losers.append(loser)

    verified = _verified_completions(multi_mutant_dir)
    corrections, attach_summary = attach_corrections(losers, verified)
    losers_by_id = {loser.record_id: loser for loser in losers}
    retained, replay_summary = balanced_replay(
        corrections, losers_by_id, max_per_project, max_per_family, max_per_category,
    )

    evaluated = len(payload.get("function_results", []))
    manifest = {
        "schema_version": RELEARNING_SCHEMA_VERSION,
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "round": 1,
        "source_evaluation": {
            "artifact": str(evaluation.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(evaluation),
            "split": split,
            "seed": payload.get("seed"),
            "checkpoint_step": payload.get("checkpoint_step"),
            "function_kill_rate": payload.get("function_kill_rate"),
            "functions_evaluated": evaluated,
        },
        "base_evaluation": (
            {
                "artifact": str(base_evaluation.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(base_evaluation),
            } if base_evaluation else None
        ),
        "split_isolation": {
            "eligible_splits": ["train", "ablation_dev"],
            "validation_or_sealed_test_used": False,
            "enforced_by": "harness.relearning.assert_split_is_eligible",
        },
        "supervision_policy": {
            "model_output_used_as_label": False,
            "correction_source": "verified multi-mutant completion",
            "verification": (
                "each correction was executed against the reference and every "
                "sibling mutant of its lineage when the dataset was built"
            ),
        },
        "losers": {
            "count": len(losers),
            "loser_rate": round(len(losers) / max(1, evaluated), 6),
            "by_dominant_category": dict(sorted(collections.Counter(
                loser.dominant_category for loser in losers
            ).items())),
            "by_origin_group": dict(sorted(collections.Counter(
                loser.origin_group for loser in losers
            ).items())),
            "by_complexity_tier": dict(sorted(collections.Counter(
                loser.complexity_tier for loser in losers
            ).items())),
            "worse_than_base": sum(1 for loser in losers if loser.worse_than_base),
        },
        "corrections": attach_summary,
        "balanced_replay": replay_summary,
        "dataset_sha256": relearning_dataset_sha256(retained),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "losers.json", [loser.to_dict() for loser in losers])
    write_json(
        output_dir / "corrections.json",
        [correction.to_dict() for correction in retained],
    )
    manifest["files"] = {
        "losers.json": sha256_file(output_dir / "losers.json"),
        "corrections.json": sha256_file(output_dir / "corrections.json"),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--base-evaluation", type=Path, default=None)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--multi-mutant-dir", type=Path, default=DEFAULT_MULTI_MUTANT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-project", type=int, default=40)
    parser.add_argument("--max-per-family", type=int, default=60)
    parser.add_argument("--max-per-category", type=int, default=120)
    arguments = parser.parse_args()

    manifest = build(
        arguments.evaluation, arguments.base_evaluation, arguments.inventory_dir,
        arguments.multi_mutant_dir, arguments.output_dir,
        arguments.max_per_project, arguments.max_per_family,
        arguments.max_per_category,
    )
    print(json.dumps({
        "losers": manifest["losers"],
        "corrections": manifest["corrections"],
        "balanced_replay": {
            key: value for key, value in manifest["balanced_replay"].items()
            if key in {"input_corrections", "retained", "dropped_by_cap"}
        },
        "dataset_sha256": manifest["dataset_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
