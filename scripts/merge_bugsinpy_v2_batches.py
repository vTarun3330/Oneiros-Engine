"""Merge isolated BugsInPy ingestion batches into the canonical v2 staging area.

Each project batch owns its repositories and JSON files while it runs.  This
merger is the sole writer of the shared staging report, preventing concurrent
workers from losing each other's verified outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.bugsinpy_v2 import discover_tasks, task_digest, write_json
from harness.corpus import record_content_hash

DATA_DIR = ROOT / "data"
OFFICIAL_REPOSITORY = DATA_DIR / "BugsInPy_repo"
STAGING_DIR = DATA_DIR / "bugsinpy_v2_ingestion"


def _load(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _training_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the immutable training-facing portion of a materialized record.

    Official test evidence retains useful diagnostics, but it deliberately
    contains per-run temporary paths, memory addresses, and timing text.  Two
    independent F2P executions of the *same* task can therefore have different
    content hashes despite having identical code, test fragment, task identity,
    and fixed-pass/buggy-fail evidence.  Those volatile diagnostics must not
    make a resumable merge reject an otherwise identical record.
    """
    payload = json.loads(json.dumps(record))
    payload.pop("content_hash", None)
    provenance = payload.get("provenance", {})
    provenance.pop("official_test_evidence", None)
    return payload


def _non_fragment_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return content that must stay immutable during a fragment refresh."""
    payload = _training_payload(record)
    payload.pop("tests", None)
    return payload


def _unique_records(
    groups: Iterable[List[Dict[str, Any]]], valid_task_ids: set[str],
    allow_refreshed_fragments: bool = False,
) -> List[Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        for record in group:
            task_id = record.get("provenance", {}).get("official_task_id")
            evidence = record.get("provenance", {}).get("official_test_evidence", {})
            if (
                not task_id or task_id not in valid_task_ids
                or record_content_hash(record) != record.get("content_hash")
                or evidence.get("fixed_returncode") != 0
                or evidence.get("buggy_returncode", 0) == 0
                or not record.get("quality", {}).get("official_targeted_test_fixed_pass_buggy_fail")
            ):
                raise RuntimeError(f"Refusing malformed or unverified materialized record: {record.get('id')}")
            existing = records.get(record["id"])
            if existing and existing["content_hash"] != record["content_hash"]:
                if _training_payload(existing) != _training_payload(record):
                    if not (
                        allow_refreshed_fragments
                        and _non_fragment_payload(existing) == _non_fragment_payload(record)
                    ):
                        raise RuntimeError(f"Conflicting content for materialized record: {record['id']}")
                    # Both copies independently passed the official F2P gate.
                    # With batch-first input, retain the refresh that changed
                    # only the supervised test fragment.
                    continue
                # Keep the first (canonical staging) copy.  Both copies were
                # independently verified above; only volatile diagnostics vary.
                continue
            records[record["id"]] = record
    return sorted(records.values(), key=lambda item: item["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent BugsInPy v2 ingestion batches")
    parser.add_argument("--batches-dir", default=str(STAGING_DIR / "batches"))
    parser.add_argument(
        "--output-dir", default=str(STAGING_DIR),
        help="Canonical staging directory to rebuild from the supplied batches.",
    )
    parser.add_argument(
        "--prefer-refreshed-fragments", action="store_true",
        help="Use batch-first test-fragment refreshes only when all non-test content is identical.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    tasks = discover_tasks(OFFICIAL_REPOSITORY)
    valid_task_ids = {task.id for task in tasks}
    batches_dir = Path(args.batches_dir)
    batch_dirs = sorted(path for path in batches_dir.iterdir() if path.is_dir()) if batches_dir.exists() else []

    input_dirs = (
        [*batch_dirs, output_dir]
        if args.prefer_refreshed_fragments else [output_dir, *batch_dirs]
    )
    function_records = _unique_records(
        [_load(path / "materialized_records.json", []) for path in input_dirs], valid_task_ids,
        allow_refreshed_fragments=args.prefer_refreshed_fragments,
    )
    repository_records = _unique_records(
        [_load(path / "repository_fragment_records.json", []) for path in input_dirs], valid_task_ids,
        allow_refreshed_fragments=args.prefer_refreshed_fragments,
    )
    all_records = [*function_records, *repository_records]
    records_by_task = {record["provenance"]["official_task_id"]: record for record in all_records}

    reports_by_task: Dict[str, Dict[str, Any]] = {}
    for directory in input_dirs:
        for item in _load(directory / "ingestion_report.json", []):
            task_id = item.get("task_id")
            if task_id not in valid_task_ids:
                raise RuntimeError(f"Batch report includes unknown official task: {task_id}")
            # Batch outcomes are newer retry evidence than the seeded staging
            # report, so later batch directories intentionally replace it.
            reports_by_task[task_id] = item
    for task_id, record in records_by_task.items():
        reports_by_task[task_id] = {
            "task_id": task_id,
            "status": "accepted",
            "record_id": record["id"],
            "record_mode": record.get("quality", {}).get("execution_mode", "function_assertion"),
        }
    report = [reports_by_task[task_id] for task_id in sorted(reports_by_task)]

    write_json(output_dir / "materialized_records.json", function_records)
    write_json(output_dir / "repository_fragment_records.json", repository_records)
    write_json(output_dir / "ingestion_report.json", report)
    summary = {
        "official_task_count": len(tasks),
        "official_task_digest": task_digest(tasks),
        "processed_task_count": len(report),
        "ingestion_complete": len(report) == len(tasks),
        "materialized_record_count": len(function_records),
        "repository_fragment_record_count": len(repository_records),
        "outcomes": dict(sorted(Counter(item.get("reason", item["status"]) for item in report).items())),
        "batches_merged": [path.name for path in batch_dirs],
    }
    write_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
