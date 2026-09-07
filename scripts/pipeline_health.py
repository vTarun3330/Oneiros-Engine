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
    return {
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


def check(queue_logs: list[Path]) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []

    gpu = _gpu_utilisation()
    if gpu is None:
        warnings.append("nvidia-smi did not answer; GPU state unknown")

    queues = [_queue_state(path) for path in queue_logs]
    active = [q for q in queues if q.get("claimed_running")]
    run = _latest_run()

    for queue in active:
        job = queue["claimed_running"]
        age = queue.get("mtime_age_seconds", 0)
        # A queue claiming a job while the card is cold and nothing has been
        # written for a long time is the exact signature of every stall here.
        if gpu is not None and gpu < 5 and age > GPU_GRACE_SECONDS:
            problems.append(
                f"STALLED: {queue['log']} claims '{job}' is running, GPU at "
                f"{gpu}% and the log has not moved for {age}s"
            )
        if run.get("state") in TERMINAL_STATES and run.get("name") == job:
            problems.append(
                f"STRANDED: run '{job}' is {run['state']} but its queue never "
                f"reported it; the queue is waiting on a marker that will not "
                f"appear ({run.get('detail') or 'no detail'})"
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
    if not active and run.get("state") in TERMINAL_STATES:
        warnings.append("no queue has work outstanding; the GPU is idle")

    return {
        "schema_version": "oneiros_pipeline_health_v1",
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_utilisation_percent": gpu,
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
