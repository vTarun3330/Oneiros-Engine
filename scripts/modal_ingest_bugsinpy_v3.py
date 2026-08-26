"""Checkpointed Linux retry of excluded BugsInPy tasks on Modal.

Each task is an independent durable function call and atomically writes one
result to a persistent Modal Volume.  A local submission ledger is updated
after every spawn, so loss of power/Wi-Fi can be resumed without discarding
completed fixed-pass/buggy-fail evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.bugsinpy_v2 import discover_tasks
from harness.corpus import write_json

APP_NAME = "oneiros-v3-linux-ingestion"
VOLUME_NAME = "oneiros-v3-ingestion-volume"
REMOTE_ROOT = Path("/oneiros-state/bugsinpy-linux")
LOCAL_ROOT = ROOT / "data" / "bugsinpy_v3_linux_ingestion"
OFFICIAL_REPOSITORY = ROOT / "data" / "BugsInPy_repo"
PILOT_AUDIT = ROOT / "results" / "v3_linux_pilot_audit.json"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    # Modal's function runtime itself uses Python 3.10, while audited Conda
    # interpreters below reproduce every Python minor version declared by the
    # BugsInPy inventory (3.6, 3.7, and 3.8). The exact runner patch version is
    # persisted in every task's official-test evidence.
    modal.Image.micromamba(python_version="3.10")
    .apt_install(
        "git", "build-essential", "gfortran", "pkg-config", "libffi-dev",
        "libssl-dev", "libxml2-dev", "libxslt1-dev", "zlib1g-dev",
        "libjpeg-dev", "libfreetype6-dev",
    )
    .run_commands(
        "micromamba create -y -p /opt/oneiros-runtimes/python3.6 -c conda-forge python=3.6 pip",
        "micromamba create -y -p /opt/oneiros-runtimes/python3.7 -c conda-forge python=3.7 pip",
        "micromamba create -y -p /opt/oneiros-runtimes/python3.8 -c conda-forge python=3.8 pip",
    )
    .pip_install("pytest<8", "virtualenv")
    # Modal imports the serialized function module before the function body
    # runs, so the bundled Oneiros packages must already be importable at
    # container start (not only after ingest_task inserts sys.path).
    .env({"PYTHONPATH": "/root/oneiros"})
    .add_local_dir("baseline", remote_path="/root/oneiros/baseline")
    .add_local_dir("config", remote_path="/root/oneiros/config")
    .add_local_dir("harness", remote_path="/root/oneiros/harness")
    .add_local_dir("data/BugsInPy_repo", remote_path="/root/oneiros/data/BugsInPy_repo")
)


def _slug(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", task_id)


def _select_task_ids(
    action: str,
    ledger: Dict[str, Any],
    eligible: List[str],
    requested_task_ids: str,
    known_task_ids: set[str],
) -> List[str]:
    """Return the exact durable selection for submit/resume.

    An explicit selection is part of the checkpoint contract. Resuming such a
    run without repeating ``--task-ids`` must reuse the checkpoint selection
    instead of widening back to every historically excluded task.
    """
    if requested_task_ids:
        selected = list(dict.fromkeys(
            value.strip() for value in requested_task_ids.split(",") if value.strip()
        ))
    elif action == "resume":
        prior = (ledger.get("selection") or {}).get("explicit_task_ids") or []
        selected = list(dict.fromkeys(prior)) if prior else list(eligible)
    else:
        selected = list(eligible)

    invalid = sorted(set(selected) - known_task_ids)
    if invalid:
        raise ValueError(f"Selected task IDs are not in the BugsInPy inventory: {invalid}")
    return selected


def _verification_evidence(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the persisted fixed-pass/buggy-fail evidence, when present."""
    record = payload.get("function_record") or payload.get("repository_record") or {}
    provenance = record.get("provenance") or {}
    return provenance.get("official_test_evidence") or provenance.get("verification_evidence")


def _validate_promotable_payload(payload: Dict[str, Any], task_id: str) -> None:
    """Reject promotion unless the saved result is an evidenced acceptance."""
    summary = payload.get("summary") or {}
    if summary.get("task_id") != task_id or summary.get("status") != "accepted":
        raise RuntimeError(f"Pilot result is not an accepted match for {task_id}")
    record = payload.get("function_record") or payload.get("repository_record")
    if not isinstance(record, dict):
        raise RuntimeError(f"Pilot result has no training record for {task_id}")
    quality = record.get("quality") or {}
    if not quality.get("pair_behaviorally_verified"):
        raise RuntimeError(f"Pilot result lacks behavioral verification for {task_id}")
    evidence = _verification_evidence(payload) or {}
    fixed_returncode = evidence.get("fixed_returncode")
    buggy_returncode = evidence.get("buggy_returncode")
    if fixed_returncode != 0 or buggy_returncode in {None, 0}:
        raise RuntimeError(
            f"Pilot result lacks fixed-pass/buggy-fail evidence for {task_id}: "
            f"fixed={fixed_returncode!r}, buggy={buggy_returncode!r}"
        )


@app.function(
    image=image,
    volumes={"/oneiros-state": volume},
    timeout=1800,
    cpu=2,
    memory=4096,
    max_containers=8,
)
def ingest_task(task_id: str, run_id: str, test_timeout: int = 300) -> Dict[str, Any]:
    """Execute and persist one idempotent Linux fixed/buggy verification."""
    print(
        json.dumps({
            "event": "task_started",
            "run_id": run_id,
            "task_id": task_id,
            "test_timeout_seconds": test_timeout,
        }),
        flush=True,
    )
    sys.path.insert(0, "/root/oneiros")
    from harness.bugsinpy_v2 import RepositoryCache, discover_tasks, materialize_task, write_json

    repository_root = Path("/root/oneiros/data/BugsInPy_repo")
    result_dir = REMOTE_ROOT / run_id / "results"
    result_path = result_dir / f"{_slug(task_id)}.json"
    volume.reload()
    if result_path.exists():
        persisted = json.loads(result_path.read_text(encoding="utf-8"))
        print(
            json.dumps({
                "event": "task_reused",
                "run_id": run_id,
                **persisted["summary"],
            }),
            flush=True,
        )
        return persisted["summary"]

    tasks = {task.id: task for task in discover_tasks(repository_root)}
    task = tasks.get(task_id)
    if task is None:
        raise RuntimeError(f"Unknown BugsInPy task: {task_id}")
    requested_minor = ".".join(task.python_version.split(".")[:2])
    historical_python = Path(
        f"/opt/oneiros-runtimes/python{requested_minor}/bin/python"
    )
    if not historical_python.exists():
        outcome = {
            "task_id": task_id,
            "status": "excluded",
            "reason": "historical_runtime_unavailable",
            "requested_python_version": task.python_version,
        }
        payload = {
            "schema_version": 1,
            "summary": {**outcome, "record_id": None, "run_id": run_id},
            "outcome": outcome,
            "function_record": None,
            "repository_record": None,
        }
        write_json(result_path, payload)
        volume.commit()
        print(json.dumps({"event": "task_completed", **payload["summary"]}), flush=True)
        return payload["summary"]
    cache = RepositoryCache(Path("/tmp/oneiros-bugsinpy-repositories"))
    record, repository_record, outcome = materialize_task(
        task,
        cache,
        timeout=test_timeout,
        prepare_environment=True,
        runner_python=str(historical_python),
    )
    summary = {
        "task_id": task_id,
        "status": outcome.get("status", "excluded"),
        "reason": outcome.get("reason"),
        "record_id": (record or repository_record or {}).get("id"),
        "run_id": run_id,
    }
    payload = {
        "schema_version": 1,
        "summary": summary,
        "outcome": outcome,
        "function_record": record,
        "repository_record": repository_record,
    }
    write_json(result_path, payload)
    volume.commit()
    print(json.dumps({"event": "task_completed", **summary}), flush=True)
    return summary


@app.function(image=image, volumes={"/oneiros-state": volume}, timeout=300)
def run_status(run_id: str) -> Dict[str, Any]:
    result_dir = REMOTE_ROOT / run_id / "results"
    volume.reload()
    summaries = []
    if result_dir.exists():
        for path in sorted(result_dir.glob("*.json")):
            summaries.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    accepted = sum(item.get("status") == "accepted" for item in summaries)
    return {
        "run_id": run_id,
        "completed": len(summaries),
        "accepted": accepted,
        "excluded": len(summaries) - accepted,
        "summaries": summaries,
    }


@app.function(image=image, volumes={"/oneiros-state": volume}, timeout=300)
def promote_results(source_run_id: str, target_run_id: str, task_ids: List[str]) -> Dict[str, Any]:
    """Atomically seed audited accepted results into a durable full run."""
    if source_run_id == target_run_id:
        raise ValueError("source and target run IDs must differ")
    source_dir = REMOTE_ROOT / source_run_id / "results"
    target_dir = REMOTE_ROOT / target_run_id / "results"
    volume.reload()
    promoted = []
    already_present = []
    for task_id in task_ids:
        source_path = source_dir / f"{_slug(task_id)}.json"
        target_path = target_dir / f"{_slug(task_id)}.json"
        if not source_path.exists():
            raise RuntimeError(f"Missing source result for promotion: {task_id}")
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        _validate_promotable_payload(source_payload, task_id)
        if target_path.exists():
            target_payload = json.loads(target_path.read_text(encoding="utf-8"))
            _validate_promotable_payload(target_payload, task_id)
            if target_payload.get("summary", {}).get("record_id") != source_payload.get("summary", {}).get("record_id"):
                raise RuntimeError(f"Conflicting target result already exists for {task_id}")
            already_present.append(task_id)
            continue
        promoted_payload = dict(source_payload)
        promoted_payload["summary"] = {
            **source_payload["summary"],
            "run_id": target_run_id,
        }
        promoted_payload["promotion"] = {
            "source_run_id": source_run_id,
            "source_record_id": source_payload["summary"].get("record_id"),
            "audit": "results/v3_linux_pilot_audit.json",
        }
        write_json(target_path, promoted_payload)
        promoted.append(task_id)
    if promoted:
        volume.commit()
    return {
        "source_run_id": source_run_id,
        "target_run_id": target_run_id,
        "requested": len(task_ids),
        "promoted": promoted,
        "already_present": already_present,
    }


def _excluded_task_ids(projects: str = "") -> List[str]:
    report_path = ROOT / "data" / "bugsinpy_v3_ingestion" / "ingestion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    allowed = {value.strip().lower() for value in projects.split(",") if value.strip()}
    tasks_by_project: Dict[str, List[str]] = defaultdict(list)
    for item in report:
        task_id = item.get("task_id", "")
        parts = task_id.split("::")
        if item.get("status") == "accepted" or len(parts) < 3:
            continue
        if allowed and parts[1].lower() not in allowed:
            continue
        tasks_by_project[parts[1]].append(task_id)
    # Deterministic round-robin prevents a bounded pilot or interrupted full
    # submission from being dominated by the alphabetically first project.
    for values in tasks_by_project.values():
        values.sort()
    task_ids = []
    projects_in_order = sorted(tasks_by_project, key=str.lower)
    depth = 0
    while True:
        added = False
        for project in projects_in_order:
            values = tasks_by_project[project]
            if depth < len(values):
                task_ids.append(values[depth])
                added = True
        if not added:
            break
        depth += 1
    return task_ids


def _ledger_path(run_id: str) -> Path:
    return LOCAL_ROOT / run_id / "submission_checkpoint.json"


def _load_ledger(run_id: str) -> Dict[str, Any]:
    path = _ledger_path(run_id)
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("schema_version") != 1 or ledger.get("run_id") != run_id:
            raise RuntimeError("Modal ingestion submission checkpoint is malformed.")
        return ledger
    return {"schema_version": 1, "run_id": run_id, "submitted": {}}


def _audited_pilot_task_ids(source_run_id: str) -> List[str]:
    """Load and locally revalidate the exact pilot records approved by audit."""
    audit = json.loads(PILOT_AUDIT.read_text(encoding="utf-8"))
    gates = audit.get("quality_gates") or {}
    required_true = {
        "all_selected_results_accepted",
        "record_schema_and_hash_valid",
        "python_and_tests_compile",
        "fixed_pass_buggy_fail",
        "selected_test_observed_buggy_fail",
        "all_completions_fit_sft_context",
        "semantic_supervision_unique",
        "project_group_split_safe",
    }
    if (
        audit.get("ready") is not True
        or any(gates.get(name) is not True for name in required_true)
        or gates.get("semantic_prompt_conflicts") != 0
    ):
        raise RuntimeError("The V3 Linux pilot audit is not fully ready")
    records = audit.get("records") or []
    task_ids = sorted({item.get("task_id") for item in records if item.get("task_id")})
    if len(task_ids) != audit.get("accepted_records") or len(task_ids) != audit.get("result_files_checked"):
        raise RuntimeError("Pilot audit task counts are inconsistent")
    for task_id in task_ids:
        path = LOCAL_ROOT / source_run_id / "remote_results" / "results" / f"{_slug(task_id)}.json"
        if not path.exists():
            raise RuntimeError(f"Missing locally audited pilot payload: {path}")
        _validate_promotable_payload(json.loads(path.read_text(encoding="utf-8")), task_id)
    return task_ids


def _sync_volume(run_id: str) -> None:
    destination = LOCAL_ROOT / run_id / "remote_results"
    # ``modal volume get`` interprets a nonexistent destination as a file on
    # Windows, concatenating a remote directory into one unusable stream.
    # Create the directory explicitly and retain any prior malformed stream
    # for auditability instead of deleting it.
    if destination.exists() and not destination.is_dir():
        invalid = destination.with_name(f"{destination.name}.invalid")
        os.replace(destination, invalid)
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        # Modal's CLI emits Unicode progress/success symbols. Force UTF-8 so
        # Windows' legacy console code page cannot abort an otherwise valid
        # volume download with a ``charmap`` encoding error.
        sys.executable, "-X", "utf8", "-m", "modal", "volume", "get", "--force",
        VOLUME_NAME, f"bugsinpy-linux/{run_id}/results", str(destination),
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


def _manage(
    action: str = "status",
    run_id: str = "v3-linux-expansion-1",
    source_run_id: str = "v3-linux-pilot-final-1",
    projects: str = "",
    limit: int = 0,
    test_timeout: int = 300,
    task_ids: str = "",
) -> None:
    """Submit, inspect, sync, or cancel calls on the deployed worker app.

    Looking up the deployed functions is essential: calls spawned against an
    ephemeral ``modal run`` app are cancelled when its local entrypoint exits.
    Calls spawned against this deployed app keep running across local process,
    Wi-Fi, and power interruptions, while the local ledger remains resumable.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, and hyphen")
    deployed_ingest = modal.Function.from_name(APP_NAME, "ingest_task")
    deployed_status = modal.Function.from_name(APP_NAME, "run_status")
    ledger = _load_ledger(run_id)

    if action == "promote":
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", source_run_id):
            raise ValueError("source_run_id contains invalid characters")
        task_ids = _audited_pilot_task_ids(source_run_id)
        deployed_promote = modal.Function.from_name(APP_NAME, "promote_results")
        result = deployed_promote.remote(source_run_id, run_id, task_ids)
        status = deployed_status.remote(run_id)
        remote_tasks = {item["task_id"] for item in status["summaries"]}
        if not set(task_ids).issubset(remote_tasks):
            raise RuntimeError("Promoted pilot records are missing from the target run")
        ledger["promoted"] = {
            task_id: {"source_run_id": source_run_id, "status": "accepted"}
            for task_id in task_ids
        }
        ledger["status"] = "pilot_promoted"
        write_json(_ledger_path(run_id), ledger)
        write_json(LOCAL_ROOT / run_id / "status_checkpoint.json", status)
        print(json.dumps({
            **result,
            "target_completed": status["completed"],
            "target_accepted": status["accepted"],
            "checkpoint": str(_ledger_path(run_id)),
        }, indent=2))
        return

    if action == "status":
        status = deployed_status.remote(run_id)
        write_json(LOCAL_ROOT / run_id / "status_checkpoint.json", status)
        print(json.dumps({key: value for key, value in status.items() if key != "summaries"}, indent=2))
        return
    if action == "sync":
        status = deployed_status.remote(run_id)
        write_json(LOCAL_ROOT / run_id / "status_checkpoint.json", status)
        _sync_volume(run_id)
        print(json.dumps({key: value for key, value in status.items() if key != "summaries"}, indent=2))
        return
    if action == "cancel":
        cancelled = 0
        for item in ledger["submitted"].values():
            call_id = item.get("function_call_id")
            if not call_id:
                continue
            try:
                modal.FunctionCall.from_id(call_id).cancel()
                cancelled += 1
            except Exception as exc:
                item["cancel_error"] = repr(exc)
        ledger["status"] = "cancelled"
        write_json(_ledger_path(run_id), ledger)
        print(json.dumps({"run_id": run_id, "cancelled_calls": cancelled}, indent=2))
        return
    if action == "reconcile":
        if not task_ids:
            raise ValueError("reconcile requires the exact approved --task-ids selection")
        known_task_ids = {task.id for task in discover_tasks(OFFICIAL_REPOSITORY)}
        selected = _select_task_ids("submit", ledger, [], task_ids, known_task_ids)
        selected_set = set(selected)
        retained: Dict[str, Any] = {}
        removed: Dict[str, Any] = {}
        cancelled = 0
        for submitted_task_id, item in ledger["submitted"].items():
            if submitted_task_id in selected_set:
                retained[submitted_task_id] = item
                continue
            cancellation = {**item, "reason": "outside_reconciled_explicit_selection"}
            call_id = item.get("function_call_id")
            if call_id:
                try:
                    modal.FunctionCall.from_id(call_id).cancel()
                    cancellation["cancelled"] = True
                    cancelled += 1
                except Exception as exc:
                    cancellation["cancelled"] = False
                    cancellation["cancel_error"] = repr(exc)
            removed[submitted_task_id] = cancellation
        prior_removed = ledger.get("cancelled_unselected") or {}
        ledger["cancelled_unselected"] = {**prior_removed, **removed}
        ledger["submitted"] = retained
        ledger["selection"] = {
            "projects": "",
            "limit": 0,
            "test_timeout": (ledger.get("selection") or {}).get("test_timeout", test_timeout),
            "explicit_task_ids": selected,
            "task_count": len(selected),
        }
        ledger["status"] = "submitted"
        write_json(_ledger_path(run_id), ledger)
        print(json.dumps({
            "run_id": run_id,
            "retained_calls": len(retained),
            "removed_calls": len(removed),
            "cancelled_calls": cancelled,
            "checkpoint": str(_ledger_path(run_id)),
        }, indent=2))
        return
    if action not in {"submit", "resume"}:
        raise ValueError(
            "action must be submit, resume, promote, status, sync, cancel, or reconcile"
        )

    remote = deployed_status.remote(run_id)
    completed = {item["task_id"] for item in remote["summaries"]}
    eligible = _excluded_task_ids(projects)
    # Explicit IDs are also used for audited regeneration of previously
    # accepted records (for example, after improving safe AST compaction).
    # Validate them against the immutable upstream inventory rather than
    # limiting them to the earlier exclusion report.
    known_task_ids = {task.id for task in discover_tasks(OFFICIAL_REPOSITORY)}
    selected = _select_task_ids(
        action, ledger, eligible, task_ids, known_task_ids,
    )
    if limit:
        selected = selected[:limit]
    ledger["selection"] = {
        "projects": projects,
        "limit": limit,
        "test_timeout": test_timeout,
        "explicit_task_ids": selected if task_ids else [],
        "task_count": len(selected),
    }
    ledger["status"] = "submitting"
    write_json(_ledger_path(run_id), ledger)

    submitted_now = 0
    for task_id in selected:
        if task_id in completed:
            continue
        previous = ledger["submitted"].get(task_id)
        if previous and previous.get("function_call_id"):
            try:
                modal.FunctionCall.from_id(previous["function_call_id"]).get(timeout=0)
                completed.add(task_id)
                continue
            except TimeoutError:
                continue
            except Exception as exc:
                previous["prior_call_error"] = repr(exc)
        call = deployed_ingest.spawn(task_id, run_id, test_timeout)
        ledger["submitted"][task_id] = {
            "function_call_id": call.object_id,
            "dashboard_url": call.get_dashboard_url(),
        }
        submitted_now += 1
        write_json(_ledger_path(run_id), ledger)

    ledger["status"] = "submitted"
    write_json(_ledger_path(run_id), ledger)
    print(json.dumps({
        "run_id": run_id,
        "selected": len(selected),
        "already_completed": len(completed & set(selected)),
        "submitted_now": submitted_now,
        "checkpoint": str(_ledger_path(run_id)),
    }, indent=2))


@app.local_entrypoint()
def main(
    action: str = "status",
    run_id: str = "v3-linux-expansion-1",
    source_run_id: str = "v3-linux-pilot-final-1",
    projects: str = "",
    limit: int = 0,
    test_timeout: int = 300,
    task_ids: str = "",
) -> None:
    _manage(action, run_id, source_run_id, projects, limit, test_timeout, task_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("submit", "resume", "promote", "status", "sync", "cancel", "reconcile"),
        default="status",
    )
    parser.add_argument("--run-id", default="v3-linux-expansion-1")
    parser.add_argument("--source-run-id", default="v3-linux-pilot-final-1")
    parser.add_argument("--projects", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--test-timeout", type=int, default=300)
    parser.add_argument("--task-ids", default="")
    arguments = parser.parse_args()
    _manage(
        arguments.action,
        arguments.run_id,
        arguments.source_run_id,
        arguments.projects,
        arguments.limit,
        arguments.test_timeout,
        arguments.task_ids,
    )
