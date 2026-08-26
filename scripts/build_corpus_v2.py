"""Build immutable Oneiros corpus v2 from v1 plus verified official real bugs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import SCHEMA_VERSION, sha256_file, verify_corpus, write_json

DATA_DIR = ROOT / "data"
V1_DIR = DATA_DIR / "corpus" / "v1"
V2_DIR = DATA_DIR / "corpus" / "v2"
INGESTION_DIR = DATA_DIR / "bugsinpy_v2_ingestion"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _real_bug_split(project: str) -> str:
    """Keep every real task from a project in exactly one deterministic split."""
    bucket = int(hashlib.sha256(project.lower().encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def build(allow_empty_real: bool = False) -> Dict[str, Any]:
    v1_manifest = verify_corpus(V1_DIR)
    records: List[Dict[str, Any]] = _load_json(V1_DIR / "records.json")
    splits: Dict[str, List[str]] = _load_json(V1_DIR / "splits.json")
    external_index: List[Dict[str, Any]] = _load_json(V1_DIR / "external_eval_index.json")
    function_records: List[Dict[str, Any]] = _load_json(INGESTION_DIR / "materialized_records.json") \
        if (INGESTION_DIR / "materialized_records.json").exists() else []
    repository_fragment_records: List[Dict[str, Any]] = _load_json(INGESTION_DIR / "repository_fragment_records.json") \
        if (INGESTION_DIR / "repository_fragment_records.json").exists() else []
    real_records = [*function_records, *repository_fragment_records]
    if not real_records and not allow_empty_real:
        raise RuntimeError(
            "No official real-bug records were materialized. Run ingest_bugsinpy_v2.py "
            "until at least one task passes the reproducibility gate."
        )
    ingestion_report = _load_json(INGESTION_DIR / "ingestion_report.json") \
        if (INGESTION_DIR / "ingestion_report.json").exists() else []
    verified_repository_only = {
        item["task_id"] for item in ingestion_report
        if item.get("reason") in {"no_self_contained_assertion_pair", "no_self_contained_assertion_pair_or_repository_fragment"}
        and item.get("evidence", {}).get("fixed_returncode") == 0
        and item.get("evidence", {}).get("buggy_returncode", 0) != 0
    }

    ids = {record["id"] for record in records}
    materialized_tasks = set()
    for record in real_records:
        if record.get("schema_version") != SCHEMA_VERSION or record["id"] in ids:
            raise RuntimeError("Materialized real-bug record has an unsupported schema or duplicate ID.")
        project = record.get("provenance", {}).get("project", "")
        if not project:
            raise RuntimeError("Materialized real-bug record lacks its official project provenance.")
        split = _real_bug_split(project)
        records.append(record)
        splits[split].append(record["id"])
        ids.add(record["id"])
        materialized_tasks.add(record["provenance"]["official_task_id"])

    for values in splits.values():
        values.sort()
    for item in external_index:
        if item["id"] in materialized_tasks:
            item["status"] = "materialized_for_training"
            item["training_record_ids"] = sorted(
                record["id"] for record in real_records
                if record["provenance"]["official_task_id"] == item["id"]
            )
        elif item["id"] in verified_repository_only:
            item["status"] = "verified_repository_eval_only"

    V2_DIR.mkdir(parents=True, exist_ok=True)
    write_json(V2_DIR / "records.json", sorted(records, key=lambda record: record["id"]))
    write_json(V2_DIR / "splits.json", splits)
    write_json(V2_DIR / "external_eval_index.json", external_index)
    group_counts = {
        split: len({record["group_id"] for record in records if record["id"] in record_ids})
        for split, record_ids in splits.items()
    }
    records_by_task = Counter(record["task_type"] for record in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "oneiros-corpus-v2",
        "parent_corpus": {
            "corpus_id": v1_manifest["corpus_id"],
            "records_sha256": v1_manifest["files"]["records.json"]["sha256"],
            "splits_sha256": v1_manifest["files"]["splits.json"]["sha256"],
        },
        "quality_gate": {
            "all_training_records_verified": True,
            "reference_oracle_required": True,
            "group_disjoint_splits": True,
            "official_real_bugs_require_fixed_pass_buggy_fail": True,
            "locked_external_evaluation": True,
        },
        "training_records": len(records),
        "records_by_task": dict(sorted(records_by_task.items())),
        "splits": {
            name: {"record_count": len(record_ids), "group_count": group_counts[name]}
            for name, record_ids in splits.items()
        },
        "official_real_bug_ingestion": {
            "materialized_task_count": len(materialized_tasks),
            "materialized_record_count": len(real_records),
            "materialized_function_record_count": len(function_records),
            "materialized_repository_fragment_record_count": len(repository_fragment_records),
            "verified_repository_eval_only_task_count": len(verified_repository_only),
            "ingestion_manifest": "../../bugsinpy_v2_ingestion/manifest.json",
        },
        "external_evaluation": {
            "locked_bugsinpy_repository_tasks": len(external_index) - len(materialized_tasks),
            "materialized_for_training": len(materialized_tasks),
            "verified_repository_eval_only": len(verified_repository_only),
        },
        "files": {
            filename: {"sha256": sha256_file(V2_DIR / filename)}
            for filename in ("records.json", "splits.json", "external_eval_index.json")
        },
    }
    write_json(V2_DIR / "manifest.json", manifest)
    # Verify only after its manifest exists; it is the same fail-closed gate
    # used by training.
    verify_corpus(V2_DIR)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Oneiros corpus v2")
    parser.add_argument("--allow-empty-real", action="store_true", help="Create a structural v2 snapshot without real records")
    args = parser.parse_args()
    print(json.dumps(build(allow_empty_real=args.allow_empty_real), indent=2))


if __name__ == "__main__":
    main()
