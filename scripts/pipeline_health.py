"""One command that answers: is the pipeline actually working right now?

Every stall in this project looked identical to progress from the outside, and
each was found late by reading a log by hand:

* a queue waited forever on a completion marker that only success writes, and
  the card idled 2.7 hours while the log's last line still read "starting";
* a corpus build crashed writing a manifest field after 667 lineages of real
  work, and "no new output" was indistinguishable from "slow";
* an evaluation was refused in seconds for a missing adapter, and the queue
  behind it never started;
* a monitor grepped only for success markers, so a process that died before
  printing looked exactly like one still running.

The common shape is that silence is ambiguous. This resolves it by checking
what SHOULD be true when work is happening, and reporting a specific failure
rather than an absence:

  claimed   - a queue log says a job started and never reported a result
  observed  - GPU utilisation, live processes, run status, artifact freshness

A job claimed to be running with a cold GPU and no advancing progress line is
STALLED. A run whose status is terminal but whose queue never reported it is
STRANDED. Both are named, not merely absent.

Exit code is 0 when healthy, 1 when something needs attention, so it can drive
a watchdog without parsing prose.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TERMINAL_STATES = {"completed", "failed", "killed", "crashed", "timeout"}
#: A training run's first minutes are legitimately CPU-only: supervision is
#: verified by executing candidates in sandboxed subprocesses before the model
#: is ever loaded. Calling that a stall would cry wolf on every launch.
GPU_GRACE_SECONDS = 600
#: gpu_queue polls every 30s, so between a run reaching a terminal state and
#: its queue writing the result line there is a normal window where the last
#: job looks finished-but-unreported and the card is briefly idle. Reporting
#: that is a false alarm on EVERY successful handoff, which is precisely how a
#: watchdog becomes noise nobody reads. Three poll intervals of margin.
HANDOFF_GRACE_SECONDS = 120


def _gpu_utilisation() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def _live_queue_count() -> int | None:
    """How many gpu_queue processes are actually alive.

    A finished queue leaves its log behind forever. queue_memorisation_arm.log
    was killed after its last job started and before the job's result line was
    written, so from the text alone it claims a job has been running since
    2026-09-06. Age cannot tell that apart from a real stall - the 2.7 hour
    idle looked identical - so liveness is observed instead of inferred.

    The probe must not count itself: its own command line contains the string
    it searches for, so an unfiltered query never returns zero and every dead
    queue's log reads as a live stranded one.

    None means the question could not be answered, which is reported as unknown
    rather than resolved either way.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "@(Get-CimInstance Win32_Process | Where-Object { "
             "$_.Name -like 'python*' -and "
             "$_.CommandLine -like '*gpu_queue.py*' }).Count"],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _run_for_job(job: str) -> dict[str, Any]:
    """The most recent run belonging to a named job, not merely the newest run.

    The stranded check originally compared against whatever run was newest,
    so a queue stuck on job A went unnoticed the moment any run B started.
    """
    runs = ROOT / "runs"
    if not runs.is_dir():
        return {}
    matches = sorted(
        (d for d in runs.iterdir() if d.is_dir() and d.name.endswith(f"-{job}")),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    if not matches:
        return {}
    try:
        status = json.loads((matches[0] / "status.json").read_text(encoding="utf-8"))
    except Exception:
        return {"run_id": matches[0].name, "state": "unreadable"}
    status_file = matches[0] / "status.json"
    return {
        "run_id": status.get("run_id", matches[0].name),
        "state": status.get("state"),
        "detail": str((status.get("termination") or {}).get("detail") or "")[:200],
        # When the run reached its terminal state. A live queue polls, so if it
        # has written nothing since this moment it is no longer there.
        "settled_at": status_file.stat().st_mtime,
    }


def _queue_state(log: Path) -> dict[str, Any]:
    """What the queue claims: the last job started, and whether it reported."""
    if not log.exists():
        return {"log": log.name, "exists": False}
    text = log.read_text(encoding="utf-8", errors="ignore")
    started = re.findall(r"\[QUEUE\] starting (\S+)", text)
    reported = set(re.findall(r"\[QUEUE\] (\S+) -> ", text))
    pending = [name for name in started if name not in reported]
    return {
        "log": log.name,
        "exists": True,
        "jobs_started": len(started),
        "jobs_reported": len(reported),
        "claimed_running": pending[-1] if pending else None,
        "mtime_age_seconds": round(time.time() - log.stat().st_mtime),
        "mtime": log.stat().st_mtime,
    }


def _latest_run() -> dict[str, Any]:
    runs = ROOT / "runs"
    if not runs.is_dir():
        return {}
    directories = sorted(
        (d for d in runs.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    if not directories:
        return {}
    latest = directories[0]
    try:
        status = json.loads((latest / "status.json").read_text(encoding="utf-8"))
    except Exception:
        return {"run_id": latest.name, "state": "unreadable"}
    progress = status.get("progress") or {}
    # `name` is only written when a run finishes, so a LIVE job has none. The
    # run_id is "<timestamp>-<name>", and without deriving it the stranded-job
    # check can never match anything that is still running - which is exactly
    # when it matters.
    name = status.get("name")
    if not name:
        run_id = str(status.get("run_id") or latest.name)
        name = run_id.split("-", 2)[-1] if "-" in run_id else run_id
    stdout = latest / "stdout.log"
    settled_at = (
        (latest / "status.json").stat().st_mtime
        if status.get("state") in TERMINAL_STATES else None
    )
    return {
        "settled_at": settled_at,
        "run_id": status.get("run_id", latest.name),
        "name": name,
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "termination": (status.get("termination") or {}).get("reason"),
        "detail": str((status.get("termination") or {}).get("detail") or "")[:200],
        "started_utc": status.get("start_utc"),
        "stdout_age_seconds": (
            round(time.time() - stdout.stat().st_mtime) if stdout.exists() else None
        ),
        "last_log_line": str(progress.get("last_log_line") or "")[:160],
    }


def check(queue_logs: list[Path], live_queues: int | None = None) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []

    gpu = _gpu_utilisation()
    if gpu is None:
        warnings.append("nvidia-smi did not answer; GPU state unknown")
    if live_queues is None:
        live_queues = _live_queue_count()

    queues = [_queue_state(path) for path in queue_logs]
    claiming = [q for q in queues if q.get("claimed_running")]
    run = _latest_run()

    active: list[dict[str, Any]] = []
    for queue in claiming:
        job = queue["claimed_running"]
        job_run = _run_for_job(job)
        queue["claimed_job_run_state"] = job_run.get("state")
        age = queue.get("mtime_age_seconds", 0)
        log_mtime = queue.get("mtime") or 0

        # A queue whose claimed job has already reached a terminal state is not
        # running anything. Whether that is a stranded LIVE queue or the log of
        # a dead one is decided by observing processes, never by age.
        if job_run.get("state") in TERMINAL_STATES:
            # Whether this PARTICULAR queue is alive cannot be answered by
            # counting gpu_queue processes: any other queue running elsewhere
            # makes every dead queue's log read as stranded. A live queue polls
            # and reports within its poll interval, so a queue that has written
            # nothing since its job settled - long ago - is gone.
            settled = job_run.get("settled_at") or 0
            written_since = log_mtime >= settled
            settled_for = time.time() - settled
            queue["wrote_after_job_settled"] = written_since
            queue["job_settled_seconds_ago"] = round(settled_for)
            if settled_for <= HANDOFF_GRACE_SECONDS:
                # Normal handoff in progress; the queue has not had a poll yet.
                queue["verdict"] = "handoff"
                active.append(queue)
                continue
            if not written_since and settled_for > GPU_GRACE_SECONDS:
                queue["verdict"] = "finished_queue_log"
                warnings.append(
                    f"{queue['log']} never recorded a result for '{job}', "
                    f"but that run reached {job_run['state']} "
                    f"{round(settled_for)}s ago and the queue has written "
                    "nothing since; this is a finished queue's log, not a stall"
                )
            else:
                queue["verdict"] = "stranded"
                problems.append(
                    f"STRANDED: run '{job}' is {job_run['state']} but its "
                    f"queue never reported it; the queue is waiting on a "
                    f"marker that will not appear "
                    f"({job_run.get('detail') or 'no detail'})"
                )
            continue

        if live_queues == 0:
            queue["verdict"] = "abandoned"
            problems.append(
                f"ABANDONED: {queue['log']} claims '{job}' is running, no run "
                "for it reached a terminal state, and no gpu_queue process is "
                "alive; the work was lost rather than finished"
            )
            continue

        queue["verdict"] = "active"
        active.append(queue)
        if gpu is not None and gpu < 5 and age > GPU_GRACE_SECONDS:
            problems.append(
                f"STALLED: {queue['log']} claims '{job}' is running, GPU at "
                f"{gpu}% and the log has not moved for {age}s"
            )

    if run.get("state") == "failed":
        problems.append(
            f"FAILED RUN: {run.get('name')} exit={run.get('exit_code')} "
            f"reason={run.get('termination')} :: {run.get('detail')}"
        )
    if run.get("state") == "running":
        age = run.get("stdout_age_seconds")
        if age is not None and age > GPU_GRACE_SECONDS and (gpu or 0) < 5:
            problems.append(
                f"SILENT RUN: {run.get('name')} is marked running but has "
                f"written nothing for {age}s with the GPU at {gpu}%"
            )

    if not active and run.get("state") == "running":
        warnings.append(
            f"run '{run.get('name')}' is live but no queue claims it; it was "
            "probably launched directly, so nothing will start after it"
        )
    # An idle card is the failure this project keeps paying for, so it is a
    # problem rather than a warning. It was previously reported as a warning,
    # and warnings were not printed at all when any problem existed - so a
    # false alarm from a dead queue's log hid two finished seeds and an idle
    # GPU behind it.
    settled_for = (
        time.time() - run["settled_at"] if run.get("settled_at") else None
    )
    within_handoff = (
        settled_for is not None and settled_for <= HANDOFF_GRACE_SECONDS
    )
    if (not active and run.get("state") in TERMINAL_STATES
            and (gpu or 0) < 5 and not within_handoff):
        problems.append(
            f"IDLE: nothing is running. Last run '{run.get('name')}' is "
            f"{run.get('state')}, no queue has outstanding work, GPU at {gpu}%"
        )

    return {
        "schema_version": "oneiros_pipeline_health_v2",
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_utilisation_percent": gpu,
        "live_gpu_queue_processes": live_queues,
        "queues": queues,
        "latest_run": run,
        "problems": problems,
        "warnings": warnings,
        "healthy": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-log", type=Path, action="append", default=None)
    parser.add_argument("--quiet", action="store_true",
                        help="print only when something needs attention")
    arguments = parser.parse_args()

    logs = arguments.queue_log or sorted((ROOT / "results").glob("queue_*.log"))
    report = check(logs)

    if report["problems"]:
        for line in report["problems"]:
            print(f"[UNHEALTHY] {line}", flush=True)
        # Printed even here: suppressing warnings behind a problem is how a
        # false alarm once hid an idle GPU and two finished evaluations.
        for line in report["warnings"]:
            print(f"[NOTE] {line}", flush=True)
        return 1
    if not arguments.quiet:
        run = report["latest_run"]
        print(json.dumps({
            "healthy": True,
            "gpu": report["gpu_utilisation_percent"],
            "run": run.get("name"),
            "state": run.get("state"),
            "last": run.get("last_log_line"),
            "warnings": report["warnings"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
