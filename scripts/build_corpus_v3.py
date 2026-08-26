"""Build immutable Oneiros V3 from V2 plus newly verified real BugsInPy records."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import (
    SCHEMA_VERSION, function_group_id, official_evidence_verifies_pair,
    record_content_hash, semantic_supervision_key,
    semantic_python, sha256_file, verify_corpus, write_json,
)

DATA_DIR = ROOT / "data"
V2_DIR = DATA_DIR / "corpus" / "v2"
V3_DIR = DATA_DIR / "corpus" / "v3"
INGESTION_DIR = DATA_DIR / "bugsinpy_v3_ingestion"
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _real_bug_split(project: str) -> str:
    """Keep every real task from a project in one deterministic split."""
    bucket = int(hashlib.sha256(project.lower().encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def _checkpoint(state_file: Path, input_fingerprint: str, phase: str, **details: Any) -> None:
    write_json(state_file, {
        "schema_version": 1,
        "builder": "build_corpus_v3",
        "input_fingerprint": input_fingerprint,
        "phase": phase,
        **details,
    })


def _build_input_fingerprint(
    v2_manifest: Dict[str, Any], ingestion_dir: Path = INGESTION_DIR,
) -> str:
    payload = {
        "v2_records": v2_manifest["files"]["records.json"]["sha256"],
        "v2_splits": v2_manifest["files"]["splits.json"]["sha256"],
        "staged_function_records": sha256_file(ingestion_dir / "materialized_records.json"),
        "staged_repository_records": sha256_file(ingestion_dir / "repository_fragment_records.json"),
        "staged_report": sha256_file(ingestion_dir / "ingestion_report.json"),
    }
    training_exclusions = ingestion_dir / "training_exclusions.json"
    if training_exclusions.exists():
        payload["training_exclusions"] = sha256_file(training_exclusions)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _semantic_repair(
    records: List[Dict[str, Any]], original_split: Dict[str, str],
) -> tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]]:
    """Normalize groups, remove equivalent supervision, and fail closed on conflicts.

    If an inherited semantic group crossed splits, the strictest original
    boundary wins (test, then validation, then train).  This never promotes a
    test-equivalent function into training.
    """
    normalized: List[Dict[str, Any]] = []
    for source_record in records:
        record = copy.deepcopy(source_record)
        mode = record.get("quality", {}).get("execution_mode", "function_assertion")
        if mode == "function_assertion":
            record["group_id"] = function_group_id(record["reference_code"], record["entry_point"])
        else:
            project = record.get("provenance", {}).get("project", "").lower()
            if not project:
                raise RuntimeError(f"Repository record {record.get('id')} lacks its project identity.")
            record["group_id"] = f"project:bugsinpy:{project}"
        record["content_hash"] = record_content_hash(record)
        normalized.append(record)

    duplicate_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    prompt_references: Dict[str, set[str]] = defaultdict(set)
    for record in normalized:
        prompt_key, supervision_key = semantic_supervision_key(record)
        reference_key = hashlib.sha256(
            semantic_python(record["reference_code"]).encode("utf-8")
        ).hexdigest()
        duplicate_buckets[supervision_key].append(record)
        prompt_references[prompt_key].add(reference_key)
    if any(len(references) > 1 for references in prompt_references.values()):
        raise RuntimeError("Semantic prompt conflict detected while repairing V3.")

    retained: List[Dict[str, Any]] = []
    duplicate_map: Dict[str, str] = {}
    for bucket in duplicate_buckets.values():
        bucket.sort(key=lambda item: (SPLIT_PRIORITY[original_split[item["id"]]], item["id"]))
        keeper = bucket[0]
        retained.append(keeper)
        for duplicate in bucket[1:]:
            duplicate_map[duplicate["id"]] = keeper["id"]

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in retained:
        groups[record["group_id"]].append(record)
    repaired_splits = {name: [] for name in ("train", "val", "test")}
    reassigned: Dict[str, Dict[str, str]] = {}
    for group_records in groups.values():
        destination = min(
            (original_split[record["id"]] for record in group_records),
            key=lambda split: SPLIT_PRIORITY[split],
        )
        for record in group_records:
            prior = original_split[record["id"]]
            repaired_splits[destination].append(record["id"])
            if prior != destination:
                reassigned[record["id"]] = {"from": prior, "to": destination}
    for values in repaired_splits.values():
        values.sort()
    retained.sort(key=lambda item: item["id"])
    return retained, repaired_splits, {
        "semantic_duplicate_records_removed": len(duplicate_map),
        "semantic_duplicate_map": dict(sorted(duplicate_map.items())),
        "semantic_split_reassignments": len(reassigned),
        "split_reassignment_map": dict(sorted(reassigned.items())),
    }


def build(
    output_dir: Path = V3_DIR,
    state_file: Optional[Path] = None,
    ingestion_dir: Path = INGESTION_DIR,
    corpus_id: str = "oneiros-corpus-v3",
) -> Dict[str, Any]:
    v2_manifest = verify_corpus(V2_DIR)
    state_file = state_file or output_dir.parent / f"{output_dir.name}_build_state.json"
    input_fingerprint = _build_input_fingerprint(v2_manifest, ingestion_dir)
    _checkpoint(state_file, input_fingerprint, "inputs_verified")
    records: List[Dict[str, Any]] = _load_json(V2_DIR / "records.json")
    splits: Dict[str, List[str]] = _load_json(V2_DIR / "splits.json")
    original_split = {
        record_id: split for split, record_ids in splits.items() for record_id in record_ids
    }
    external_index: List[Dict[str, Any]] = _load_json(V2_DIR / "external_eval_index.json")
    function_records = _load_json(ingestion_dir / "materialized_records.json")
    repository_records = _load_json(ingestion_dir / "repository_fragment_records.json")
    staged_records = [*function_records, *repository_records]
    report = _load_json(ingestion_dir / "ingestion_report.json")
    training_exclusions_path = ingestion_dir / "training_exclusions.json"
    training_exclusions = (
        _load_json(training_exclusions_path) if training_exclusions_path.exists() else []
    )

    existing = {record["id"]: record for record in records}
    added_records = []
    for record in staged_records:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported staged schema for {record.get('id')}")
        task_id = record.get("provenance", {}).get("official_task_id")
        project = record.get("provenance", {}).get("project")
        evidence = record.get("provenance", {}).get("official_test_evidence", {})
        if (
            not task_id or not project or record_content_hash(record) != record.get("content_hash")
            or not official_evidence_verifies_pair(evidence)
            or not record.get("quality", {}).get("official_targeted_test_fixed_pass_buggy_fail")
        ):
            raise RuntimeError(f"Refusing unverified V3 staged record: {record.get('id')}")
        prior = existing.get(record["id"])
        if prior is not None:
            if record_content_hash(prior) != record_content_hash(record):
                raise RuntimeError(f"V3 staging conflicts with immutable V2 record: {record['id']}")
            continue
        records.append(record)
        original_split[record["id"]] = _real_bug_split(project)
        existing[record["id"]] = record
        added_records.append(record)

    records, splits, semantic_repairs = _semantic_repair(records, original_split)
    retained_ids = {record["id"] for record in records}
    excluded_ids = {item.get("record_id") for item in training_exclusions}
    if None in excluded_ids or not excluded_ids.issubset(retained_ids):
        raise RuntimeError("Training exclusions reference missing canonical records.")
    added_records = [record for record in added_records if record["id"] in retained_ids]
    _checkpoint(
        state_file, input_fingerprint, "semantic_repair_complete",
        retained_records=len(records), **semantic_repairs,
    )
    accepted_by_task = defaultdict(list)
    for record in records:
        task_id = record.get("provenance", {}).get("official_task_id")
        if task_id:
            accepted_by_task[task_id].append(record["id"])
    verified_repository_only = {
        item["task_id"] for item in report
        if item.get("reason") in {
            "no_self_contained_assertion_pair",
            "no_self_contained_assertion_pair_or_repository_fragment",
        }
        and item.get("evidence", {}).get("fixed_returncode") == 0
        and item.get("evidence", {}).get("buggy_returncode", 0) != 0
    }
    verified_repository_only.update(
        record["provenance"]["official_task_id"] for record in staged_records
        if record["provenance"]["official_task_id"] not in accepted_by_task
    )
    for item in external_index:
        if item["id"] in accepted_by_task:
            item["status"] = "materialized_for_training"
            item["training_record_ids"] = sorted(accepted_by_task[item["id"]])
        elif item["id"] in verified_repository_only:
            item["status"] = "verified_repository_eval_only"
            item.pop("training_record_ids", None)
        else:
            item["status"] = "locked_external_eval_not_materialized"
            item.pop("training_record_ids", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "records.json", records)
    write_json(output_dir / "splits.json", splits)
    write_json(output_dir / "external_eval_index.json", external_index)
    if training_exclusions:
        write_json(output_dir / "training_exclusions.json", training_exclusions)
    records_by_task = Counter(record["task_type"] for record in records)
    all_real = [record for record in records if record["id"].startswith("official-repository::")]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "parent_corpus": {
            "corpus_id": v2_manifest["corpus_id"],
            "records_sha256": v2_manifest["files"]["records.json"]["sha256"],
            "splits_sha256": v2_manifest["files"]["splits.json"]["sha256"],
        },
        "quality_gate": {
            "all_training_records_verified": True,
            "reference_oracle_required": True,
            "group_disjoint_splits": True,
            "semantic_group_disjoint_splits": True,
            "semantic_supervision_deduplicated": True,
            "conflicting_supervision_rejected": True,
            "official_real_bugs_require_fixed_pass_buggy_fail": True,
            "repository_project_disjoint_splits": True,
            "canonical_overlong_records_explicitly_excluded": bool(training_exclusions),
            "locked_external_evaluation": True,
        },
        "training_records": len(records),
        "records_by_task": dict(sorted(records_by_task.items())),
        "splits": {
            name: {
                "record_count": len(record_ids),
                "group_count": len({record["group_id"] for record in records if record["id"] in record_ids}),
            }
            for name, record_ids in splits.items()
        },
        "official_real_bug_ingestion": {
            "total_materialized_record_count": len(all_real),
            "new_materialized_record_count": len(added_records),
            "new_materialized_task_count": len({record["provenance"]["official_task_id"] for record in added_records}),
            "verified_repository_eval_only_task_count": len(verified_repository_only),
            "ingestion_manifest": Path(os.path.relpath(
                ingestion_dir / "manifest.json", output_dir,
            )).as_posix(),
        },
        "semantic_repairs": semantic_repairs,
        "external_evaluation": {
            "locked_bugsinpy_repository_tasks": sum(
                item.get("status") not in {"materialized_for_training", "verified_repository_eval_only"}
                for item in external_index
            ),
            "materialized_for_training": sum(
                item.get("status") == "materialized_for_training" for item in external_index
            ),
            "verified_repository_eval_only": sum(
                item.get("status") == "verified_repository_eval_only" for item in external_index
            ),
        },
        "files": {
            filename: {"sha256": sha256_file(output_dir / filename)}
            for filename in ("records.json", "splits.json", "external_eval_index.json")
        },
    }
    if training_exclusions:
        manifest["files"]["training_exclusions.json"] = {
            "sha256": sha256_file(output_dir / "training_exclusions.json")
        }
        manifest["training_exclusions"] = {
            "record_count": len(training_exclusions),
            "reason": "canonical_retained_training_excluded",
        }
    write_json(output_dir / "manifest.json", manifest)
    _checkpoint(state_file, input_fingerprint, "candidate_written", output_dir=str(output_dir))
    verify_corpus(output_dir)
    _checkpoint(
        state_file, input_fingerprint, "complete", output_dir=str(output_dir),
        manifest_sha256=sha256_file(output_dir / "manifest.json"),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Oneiros corpus V3")
    parser.add_argument("--output-dir", type=Path, default=V3_DIR)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--ingestion-dir", type=Path, default=INGESTION_DIR)
    parser.add_argument("--corpus-id", default="oneiros-corpus-v3")
    args = parser.parse_args()
    print(json.dumps(build(
        args.output_dir, args.state_file, args.ingestion_dir, args.corpus_id,
    ), indent=2))


if __name__ == "__main__":
    main()
