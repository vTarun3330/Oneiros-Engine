"""Run isolated BugsInPy project batches with bounded local concurrency.

The script is deliberately resumable: each project writes only inside its own
batch directory, then the merger atomically rebuilds shared staging metadata.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.bugsinpy_v2 import discover_tasks, task_digest, write_json

DATA_DIR = ROOT / "data"
OFFICIAL_REPOSITORY = DATA_DIR / "BugsInPy_repo"
STAGING_DIR = DATA_DIR / "bugsinpy_v2_ingestion"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible BugsInPy v2 project ingestion batches")
    parser.add_argument("--projects", default="", help="Optional comma-separated project subset")
    parser.add_argument(
        "--output-dir", default=str(STAGING_DIR),
        help="Independent canonical staging directory for this orchestration run.",
    )
    parser.add_argument("--workers", type=int, default=3, help="Maximum simultaneous project batches")
    parser.add_argument("--timeout", type=int, default=120, help="Per fixed or buggy official test timeout")
    parser.add_argument("--runner-python", default=sys.executable, help="Interpreter for batch coordinator and 3.8 tasks")
    parser.add_argument(
        "--repository-cache-dir", default="",
        help="Optional shared Git cache used by all isolated project batches.",
    )
    parser.add_argument(
        "--runtimes-dir", default="",
        help="Optional directory of provisioned historical Python runtimes.",
    )
    parser.add_argument("--retry-excluded", action="store_true", help="Retry tasks whose prior batch outcome was exclusion")
    parser.add_argument("--run-id", default="", help="Stable run ID for exact task-level resume")
    parser.add_argument(
        "--new-run", action="store_true",
        help="Start a new retry generation even if an incomplete coordinator checkpoint exists",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(OFFICIAL_REPOSITORY)
    counts = Counter(task.project for task in tasks)
    requested = {item.strip().lower() for item in args.projects.split(",") if item.strip()}
    projects = [project for project in counts if not requested or project.lower() in requested]
    if not projects:
        raise RuntimeError("No official projects matched the requested batch selection.")
    # Largest projects start first, reducing idle workers near the end.
    projects.sort(key=lambda project: (-counts[project], project.lower()))

    state_path = output_dir / "orchestrator_checkpoint.json"
    signature = {
        "official_task_digest": task_digest(tasks),
        "projects": projects,
        "retry_excluded": args.retry_excluded,
        "timeout_seconds": args.timeout,
        "workers": args.workers,
        "runner_python": args.runner_python,
        "repository_cache_dir": args.repository_cache_dir,
        "runtimes_dir": args.runtimes_dir,
    }
    prior_state = None
    if state_path.exists():
        prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        if prior_state.get("schema_version") != 1:
            raise RuntimeError("BugsInPy coordinator checkpoint has an unsupported schema.")
    requested_run_id = args.run_id.strip()
    can_resume = (
        prior_state is not None and prior_state.get("status") in {"in_progress", "failed"}
        and prior_state.get("signature") == signature and not args.new_run
        and (not requested_run_id or prior_state.get("run_id") == requested_run_id)
    )
    if can_resume:
        state = prior_state
        run_id = state["run_id"]
        for project, status in list(state.get("projects", {}).items()):
            if status == "running":
                state["projects"][project] = "queued"
    else:
        if prior_state and prior_state.get("status") in {"in_progress", "failed"} and not args.new_run:
            raise RuntimeError(
                "An incomplete coordinator checkpoint has a different configuration. "
                "Resume with the original arguments or explicitly use --new-run."
            )
        run_id = requested_run_id or (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "in_progress",
            "signature": signature,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "projects": {project: "queued" for project in projects},
        }
    state["status"] = "in_progress"
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(state_path, state)
    print(f"coordinator run_id={run_id}", flush=True)

    batches_dir = output_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    running: Dict[str, Tuple[subprocess.Popen, object]] = {}
    remaining = [project for project in projects if state["projects"].get(project) != "complete"]
    completed: List[Tuple[str, int]] = []
    ingestion_script = ROOT / "scripts" / "ingest_bugsinpy_v2.py"

    while remaining or running:
        while remaining and len(running) < args.workers:
            project = remaining.pop(0)
            project_output_dir = batches_dir / project
            project_output_dir.mkdir(parents=True, exist_ok=True)
            # Seed just the project report so --retry-excluded reruns the
            # previously excluded tasks while retaining completed acceptances.
            batch_report = project_output_dir / "ingestion_report.json"
            if not batch_report.exists():
                root_report = Path(args.output_dir) / "ingestion_report.json"
                if root_report.exists():
                    existing = json.loads(root_report.read_text(encoding="utf-8"))
                    write_json(batch_report, [
                        item for item in existing
                        if item.get("task_id", "").split("::")[1:2] == [project]
                    ])
            command = [
                args.runner_python, str(ingestion_script), "--projects", project,
                "--output-dir", str(project_output_dir), "--prepare-environment",
                "--timeout", str(args.timeout),
                "--runtimes-dir", args.runtimes_dir or str(STAGING_DIR / "runtimes"),
                "--retry-run-id", run_id,
            ]
            if args.retry_excluded:
                command.append("--retry-excluded")
            if args.repository_cache_dir:
                command.extend(["--repository-cache-dir", args.repository_cache_dir])
            handle = open(project_output_dir / "worker.log", "a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
            running[project] = (process, handle)
            state["projects"][project] = "running"
            state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(state_path, state)
            print(f"started {project} ({counts[project]} tasks)", flush=True)

        time.sleep(2)
        for project, (process, handle) in list(running.items()):
            result = process.poll()
            if result is None:
                continue
            handle.close()
            del running[project]
            completed.append((project, result))
            state["projects"][project] = "complete" if result == 0 else "failed"
            state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(state_path, state)
            print(f"finished {project}: exit={result}; {len(completed)}/{len(projects)} projects complete", flush=True)

    merge = subprocess.run(
        [
            args.runner_python, str(ROOT / "scripts" / "merge_bugsinpy_v2_batches.py"),
            "--batches-dir", str(batches_dir), "--output-dir", str(Path(args.output_dir)),
        ],
        cwd=str(ROOT), text=True, check=False,
    )
    failed = [project for project, status in completed if status]
    if merge.returncode or failed:
        state["status"] = "failed"
        state["failed_projects"] = failed
        state["merge_exit"] = merge.returncode
        state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(state_path, state)
        raise RuntimeError(f"Batch ingestion completed with failures: projects={failed}, merge_exit={merge.returncode}")
    state["status"] = "complete"
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    state["updated_at"] = state["completed_at"]
    write_json(state_path, state)


if __name__ == "__main__":
    main()
