"""Materialize reproducible official BugsInPy records for corpus v2.

This is intentionally resumable. It never converts a patch fragment directly
to a training item: failed tasks are recorded with their exclusion reason.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.bugsinpy_v2 import (
    RepositoryCache, _repository_record, discover_tasks, materialize_task, task_digest, write_json,
)

DATA_DIR = ROOT / "data"
OFFICIAL_REPOSITORY = DATA_DIR / "BugsInPy_repo"
OUTPUT_DIR = DATA_DIR / "bugsinpy_v2_ingestion"


def load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducibly ingest official BugsInPy tasks for Oneiros v2")
    parser.add_argument("--projects", default="", help="Comma-separated official project names")
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Independent ingestion state directory; use one per parallel project batch",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum unprocessed tasks (0 means all selected)")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds allowed for each fixed or buggy test")
    parser.add_argument(
        "--prepare-environment", action="store_true",
        help="Create an isolated environment and install compatible official requirements",
    )
    parser.add_argument("--runner-python", default=sys.executable, help="Python interpreter used to create isolated environments")
    parser.add_argument(
        "--runtimes-dir", default=str(OUTPUT_DIR / "runtimes"),
        help="Directory containing project-local python3.6/python3.7/python3.8 runtimes",
    )
    parser.add_argument(
        "--repository-cache-dir", default="",
        help="Optional shared read/write Git cache; allows a new staging area to reuse fetched histories.",
    )
    parser.add_argument("--retry-excluded", action="store_true", help="Retry tasks previously excluded")
    parser.add_argument(
        "--retry-run-id", default="",
        help="Stable orchestration run ID. Completed exclusions in this run are skipped on resume.",
    )
    parser.add_argument(
        "--refresh-only", action="store_true",
        help="Reconcile persisted verified records and rewrite the manifest without running tests",
    )
    parser.add_argument(
        "--refresh-fragments", action="store_true",
        help="Re-extract persisted repository test fragments from their recorded official evidence without rerunning tests",
    )
    parser.add_argument(
        "--materialize-verified", action="store_true",
        help="Materialize newly supported repository test fragments from persisted official F2P evidence without rerunning tests",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    tasks = discover_tasks(OFFICIAL_REPOSITORY)
    selected_projects = {project.strip().lower() for project in args.projects.split(",") if project.strip()}
    if selected_projects:
        tasks = [task for task in tasks if task.project.lower() in selected_projects]
    if not tasks:
        raise RuntimeError("No official BugsInPy tasks matched the requested selection.")

    report_path = output_dir / "ingestion_report.json"
    records_path = output_dir / "materialized_records.json"
    repository_records_path = output_dir / "repository_fragment_records.json"
    retry_checkpoint_path = output_dir / "retry_checkpoint.json"
    report = load_json(report_path, [])
    records = load_json(records_path, [])
    repository_records = load_json(repository_records_path, [])
    # A machine or terminal interruption can happen after an immutable record
    # is written but before the human-readable report/summary is refreshed.
    # Treat the record (which contains the original official F2P evidence) as
    # authoritative so a retry never needlessly reruns that task.
    materialized_by_task = {}
    for record in [*records, *repository_records]:
        provenance = record.get("provenance", {})
        task_id = provenance.get("official_task_id")
        if task_id:
            materialized_by_task[task_id] = record
    reconciled_report = []
    seen_report_tasks = set()
    for item in report:
        task_id = item.get("task_id")
        record = materialized_by_task.get(task_id)
        if record:
            item = {
                "task_id": task_id,
                "status": "accepted",
                "record_id": record["id"],
                "record_mode": record.get("quality", {}).get("execution_mode", "function_assertion"),
            }
        reconciled_report.append(item)
        seen_report_tasks.add(task_id)
    for task_id, record in materialized_by_task.items():
        if task_id not in seen_report_tasks:
            reconciled_report.append({
                "task_id": task_id,
                "status": "accepted",
                "record_id": record["id"],
                "record_mode": record.get("quality", {}).get("execution_mode", "function_assertion"),
            })
    report = sorted(reconciled_report, key=lambda item: item["task_id"])
    write_json(report_path, report)
    retry_checkpoint = load_json(retry_checkpoint_path, {"schema_version": 1, "runs": {}})
    if retry_checkpoint.get("schema_version") != 1 or not isinstance(retry_checkpoint.get("runs"), dict):
        raise RuntimeError("Ingestion retry checkpoint is malformed.")
    retry_run = None
    retry_completed = set()
    if args.retry_run_id:
        signature = {
            "official_task_digest": task_digest(tasks),
            "projects": sorted(selected_projects),
            "timeout_seconds": args.timeout,
            "prepare_environment": args.prepare_environment,
        }
        retry_run = retry_checkpoint["runs"].get(args.retry_run_id)
        if retry_run is not None and retry_run.get("signature") != signature:
            raise RuntimeError(
                f"Retry run {args.retry_run_id!r} was created with a different task/configuration signature."
            )
        if retry_run is None:
            retry_run = {
                "signature": signature,
                "status": "in_progress",
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "completed_task_ids": [],
            }
            retry_checkpoint["runs"][args.retry_run_id] = retry_run
            write_json(retry_checkpoint_path, retry_checkpoint)
        retry_completed = set(retry_run.get("completed_task_ids", []))
    processed = {
        item["task_id"] for item in report
        if (
            item.get("status") == "accepted" or not args.retry_excluded
            or item["task_id"] in retry_completed
        )
    }
    selected = [] if (args.refresh_only or args.refresh_fragments or args.materialize_verified) else [task for task in tasks if task.id not in processed]
    if args.limit:
        selected = selected[:args.limit]

    cache = RepositoryCache(
        Path(args.repository_cache_dir) if args.repository_cache_dir else output_dir / "repositories"
    )
    runtimes_dir = Path(args.runtimes_dir)

    if args.refresh_fragments:
        tasks_by_id = {task.id: task for task in tasks}
        refreshed = []
        for old_record in repository_records:
            task_id = old_record.get("provenance", {}).get("official_task_id")
            task = tasks_by_id.get(task_id)
            if task is None:
                raise RuntimeError(f"Cannot refresh repository fragment with unknown task: {task_id}")
            repository = cache.ensure_commit(task, task.fixed_commit)
            refreshed_record = _repository_record(
                task, cache, repository,
                old_record.get("provenance", {}).get("official_test_evidence", {}),
            )
            if refreshed_record is None or refreshed_record["id"] != old_record.get("id"):
                raise RuntimeError(f"Could not safely refresh repository fragment for {task_id}")
            refreshed.append(refreshed_record)
        repository_records = sorted(refreshed, key=lambda item: item["id"])
        write_json(repository_records_path, repository_records)

    materialized_from_persisted_evidence = 0
    if args.materialize_verified:
        tasks_by_id = {task.id: task for task in tasks}
        by_record_id = {record["id"]: record for record in repository_records}
        refreshed_report = []
        for item in report:
            evidence = item.get("evidence", {})
            task = tasks_by_id.get(item.get("task_id"))
            if (
                task is None or item.get("status") == "accepted"
                or evidence.get("fixed_returncode") != 0
                or evidence.get("buggy_returncode", 0) == 0
            ):
                refreshed_report.append(item)
                continue
            repository = cache.ensure_commit(task, task.fixed_commit)
            repository_record = _repository_record(task, cache, repository, evidence)
            if repository_record is None:
                refreshed_report.append(item)
                continue
            existing = by_record_id.get(repository_record["id"])
            if existing and existing.get("content_hash") != repository_record.get("content_hash"):
                raise RuntimeError(f"Conflicting persisted materialization for {task.id}")
            by_record_id[repository_record["id"]] = repository_record
            refreshed_report.append({
                "task_id": task.id,
                "status": "accepted",
                "record_id": repository_record["id"],
                "record_mode": repository_record["quality"]["execution_mode"],
                "materialized_from_persisted_official_evidence": True,
            })
            materialized_from_persisted_evidence += 1
        repository_records = sorted(by_record_id.values(), key=lambda item: item["id"])
        report = sorted(refreshed_report, key=lambda item: item["task_id"])
        write_json(repository_records_path, repository_records)
        write_json(report_path, report)

    def runner_for(task):
        version_parts = task.python_version.split(".")
        if len(version_parts) >= 2:
            executable = runtimes_dir / f"python{version_parts[0]}.{version_parts[1]}" / "python.exe"
            if executable.exists():
                return str(executable)
        return args.runner_python

    accepted_ids = {record["id"] for record in records}
    repository_accepted_ids = {record["id"] for record in repository_records}
    for index, task in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {task.id}", flush=True)
        record, repository_record, outcome = materialize_task(
            task, cache, timeout=args.timeout, prepare_environment=args.prepare_environment
            , runner_python=runner_for(task)
        )
        report = [item for item in report if item.get("task_id") != task.id]
        report.append(outcome)
        if record and record["id"] not in accepted_ids:
            records.append(record)
            accepted_ids.add(record["id"])
        if repository_record and repository_record["id"] not in repository_accepted_ids:
            repository_records.append(repository_record)
            repository_accepted_ids.add(repository_record["id"])
        report.sort(key=lambda item: item["task_id"])
        records.sort(key=lambda item: item["id"])
        repository_records.sort(key=lambda item: item["id"])
        write_json(report_path, report)
        write_json(records_path, records)
        write_json(repository_records_path, repository_records)
        if retry_run is not None:
            # Publish this cursor only after all authoritative result files.
            # A crash before this write safely reruns the task; a crash after
            # it resumes at the next task without discarding any outcome.
            retry_completed.add(task.id)
            retry_run["completed_task_ids"] = sorted(retry_completed)
            retry_run["completed_task_count"] = len(retry_completed)
            retry_run["last_completed_task_id"] = task.id
            retry_run["last_completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(retry_checkpoint_path, retry_checkpoint)

    if retry_run is not None:
        accepted_tasks = {
            item["task_id"] for item in report if item.get("status") == "accepted"
        }
        retry_run["status"] = "complete" if all(
            task.id in accepted_tasks or task.id in retry_completed for task in tasks
        ) else "in_progress"
        retry_run["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(retry_checkpoint_path, retry_checkpoint)

    summary = {
        "official_task_count": len(tasks),
        "official_task_digest": task_digest(tasks),
        "processed_task_count": len(report),
        "materialized_record_count": len(records),
        "repository_fragment_record_count": len(repository_records),
        "outcomes": dict(sorted(Counter(item.get("reason", item["status"]) for item in report).items())),
        "selection": {
            "projects": sorted(selected_projects), "timeout_seconds": args.timeout,
            "prepare_environment": args.prepare_environment,
            "refresh_only": args.refresh_only,
            "refresh_fragments": args.refresh_fragments,
            "materialize_verified": args.materialize_verified,
            "materialized_from_persisted_evidence": materialized_from_persisted_evidence,
            "runner_python": args.runner_python,
            "runtimes_dir": str(runtimes_dir),
            "repository_cache_dir": str(cache.root),
            "retry_run_id": args.retry_run_id or None,
            "retry_checkpoint": str(retry_checkpoint_path) if args.retry_run_id else None,
        },
    }
    write_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
