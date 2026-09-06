"""A failed job must advance the queue, not hang it.

gpu_run writes the .complete marker only on success. Waiting on that marker
alone means one refused job stops the queue forever: the card idles, the queue
chained behind it never starts, and the log's last line still reads "starting",
so silence is indistinguishable from progress.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.gpu_queue as queue


def _run_dir(tmp_path: Path, run_id: str, state: str, marker: bool) -> None:
    directory = tmp_path / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "status.json").write_text(
        json.dumps({"run_id": run_id, "state": state}), encoding="utf-8"
    )
    if marker:
        (directory / ".complete").write_text("", encoding="utf-8")


def test_a_failed_run_without_a_marker_still_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    _run_dir(tmp_path, "r1", "failed", marker=False)

    status = queue.wait("r1", poll_seconds=0)
    assert status["state"] == "failed"


def test_a_completed_run_with_a_marker_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    _run_dir(tmp_path, "r2", "completed", marker=True)

    assert queue.wait("r2", poll_seconds=0)["state"] == "completed"


def test_every_terminal_state_ends_the_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    for index, state in enumerate(sorted(queue.TERMINAL_STATES)):
        run_id = f"terminal{index}"
        _run_dir(tmp_path, run_id, state, marker=False)
        assert queue.wait(run_id, poll_seconds=0)["state"] == state


def test_a_running_job_keeps_the_queue_waiting(tmp_path, monkeypatch):
    """The fix must not make the queue skip ahead of work still in flight."""
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    _run_dir(tmp_path, "r3", "running", marker=False)

    done: list[dict] = []
    thread = threading.Thread(
        target=lambda: done.append(queue.wait("r3", poll_seconds=0)), daemon=True
    )
    thread.start()
    time.sleep(0.2)
    assert not done, "wait returned while the run was still going"

    _run_dir(tmp_path, "r3", "completed", marker=True)
    thread.join(timeout=5)
    assert done and done[0]["state"] == "completed"


def test_a_missing_status_file_does_not_end_the_wait(tmp_path, monkeypatch):
    """An unwritten status file means "not started yet", not "finished"."""
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path / "runs" / "r4").mkdir(parents=True)

    done: list[dict] = []
    thread = threading.Thread(
        target=lambda: done.append(queue.wait("r4", poll_seconds=0)), daemon=True
    )
    thread.start()
    time.sleep(0.2)
    assert not done
    _run_dir(tmp_path, "r4", "failed", marker=False)
    thread.join(timeout=5)
    assert done and done[0]["state"] == "failed"
