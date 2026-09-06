"""Run a queue of GPU jobs back to back, so the card never waits on a human.

Each job is a durable gpu_run invocation. The queue launches one, waits for it
to finish, and starts the next immediately. The idle windows this removes are
not the training itself but the gaps between jobs - the minutes or hours a
finished run sits complete while nobody has noticed and launched its successor.

A failed job does not silently sink the rest of the queue: the failure is
reported and, unless --stop-on-failure is given, the queue continues, because a
seed that crashes should not cost the other two.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv-gpu" / "Scripts" / "python.exe")


def _status(run_id: str) -> dict[str, Any]:
    path = ROOT / "runs" / run_id / "status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def launch(name: str, command: list[str]) -> str | None:
    started = subprocess.run(
        [PYTHON, "scripts/gpu_run.py", "start", "--name", name, "--"] + command,
        cwd=ROOT, capture_output=True, text=True,
    )
    try:
        return json.loads(started.stdout)["run_id"]
    except Exception:
        print(f"[QUEUE] could not launch {name}: {started.stdout}{started.stderr}",
              flush=True)
        return None


def wait(run_id: str, poll_seconds: int = 30) -> dict[str, Any]:
    marker = ROOT / "runs" / run_id / ".complete"
    while not marker.exists():
        time.sleep(poll_seconds)
    return _status(run_id)


def process_is_alive(pid: int) -> bool:
    """True while `pid` is still running.

    Used to chain one queue behind another. --after waits on a single run_id,
    which cannot express "after those ten jobs" because the last job's run_id
    does not exist until the ninth finishes. Waiting on the earlier queue's own
    process does express it, and needs nothing to be known in advance.
    """
    if os.name == "nt":
        finished = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in finished.stdout
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def wait_for_pid(pid: int, poll_seconds: int = 30) -> None:
    while process_is_alive(pid):
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job", action="append", required=True,
        help="name=<shell-free argv joined by spaces>, run in order",
    )
    parser.add_argument(
        "--after", default=None,
        help=(
            "Wait for this run_id to complete before starting the queue. "
            "Without it a queue launched while a run is in flight is refused "
            "by the collision guard, which protects the artifacts but wastes "
            "the queue."
        ),
    )
    parser.add_argument(
        "--after-pid", type=int, default=None,
        help=(
            "wait for this process to exit before starting the queue. Use it "
            "to chain behind an earlier gpu_queue whose final run_id is not "
            "yet known, so the card does not idle between experiments."
        ),
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    arguments = parser.parse_args()

    if arguments.after_pid:
        print(f"[QUEUE] waiting for pid {arguments.after_pid} to exit", flush=True)
        wait_for_pid(arguments.after_pid, arguments.poll_seconds)
        print(f"[QUEUE] pid {arguments.after_pid} exited", flush=True)

    if arguments.after:
        print(f"[QUEUE] waiting for {arguments.after}", flush=True)
        previous = wait(arguments.after, arguments.poll_seconds)
        state = previous.get("state")
        print(f"[QUEUE] {arguments.after} -> {state}", flush=True)
        if state != "completed":
            # Every queued job here consumes what the prerequisite produced. A
            # queue launched while the trainer was still running once tried to
            # evaluate an adapter that did not exist yet; the trainer refused
            # it, but the queue should not have asked.
            print(
                f"[QUEUE] abort: {arguments.after} did not complete, so the "
                "queued jobs have nothing to consume",
                flush=True,
            )
            return 1

    results = []
    for entry in arguments.job:
        name, _, command_text = entry.partition("=")
        command = command_text.split()
        print(f"[QUEUE] starting {name}", flush=True)
        run_id = launch(name, command)
        if run_id is None:
            results.append({"name": name, "state": "launch_failed"})
            if arguments.stop_on_failure:
                break
            continue
        status = wait(run_id, arguments.poll_seconds)
        state = status.get("state")
        detail = (status.get("termination") or {}).get("detail")
        print(f"[QUEUE] {name} -> {state} ({detail})", flush=True)
        results.append({"name": name, "run_id": run_id, "state": state,
                        "detail": detail})
        if state != "completed" and arguments.stop_on_failure:
            print("[QUEUE] stopping: --stop-on-failure", flush=True)
            break

    print(json.dumps(results, indent=2), flush=True)
    return 0 if all(row.get("state") == "completed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
