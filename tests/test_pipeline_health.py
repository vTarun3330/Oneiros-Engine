"""The health check must detect failures, not merely report health.

A checker that only ever says "healthy" is worse than none: it converts an
unknown into a reassurance. Each case here is a stall this project actually
had, and each must be NAMED rather than appear as an absence.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.pipeline_health as health


def _log(tmp_path: Path, text: str, age_seconds: int = 0) -> Path:
    path = tmp_path / "queue_x.log"
    path.write_text(text, encoding="utf-8")
    if age_seconds:
        old = time.time() - age_seconds
        import os
        os.utime(path, (old, old))
    return path


def test_a_cold_gpu_under_a_claimed_job_is_reported_as_stalled(tmp_path, monkeypatch):
    """The 2.7-hour idle: queue says running, card is cold, log frozen."""
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {"state": "completed", "name": "other"})
    log = _log(tmp_path, "[QUEUE] starting jobA\n", age_seconds=3600)

    report = health.check([log])
    assert report["healthy"] is False
    assert any("STALLED" in p and "jobA" in p for p in report["problems"])


def test_a_terminal_run_the_queue_never_reported_is_stranded(tmp_path, monkeypatch):
    """The refused evaluation: run failed, queue waits on a marker forever."""
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 50)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "failed", "name": "jobA", "exit_code": 1,
        "termination": "python_exception", "detail": "no adapter",
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n")

    report = health.check([log])
    assert report["healthy"] is False
    assert any("STRANDED" in p for p in report["problems"])


def test_a_failed_run_is_always_a_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 50)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "failed", "name": "jobZ", "exit_code": 1,
        "termination": "cuda_oom", "detail": "out of memory",
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n[QUEUE] jobA -> completed\n")

    report = health.check([log])
    assert report["healthy"] is False
    assert any("FAILED RUN" in p and "cuda_oom" in p for p in report["problems"])


def test_a_silent_running_job_with_a_cold_gpu_is_a_problem(tmp_path, monkeypatch):
    """The crashed build that looked slow: marked running, writing nothing."""
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "running", "name": "jobA", "stdout_age_seconds": 3600,
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n[QUEUE] jobA -> completed\n")

    report = health.check([log])
    assert report["healthy"] is False
    assert any("SILENT RUN" in p for p in report["problems"])


def test_a_healthy_pipeline_reports_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 70)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "running", "name": "jobA", "stdout_age_seconds": 20,
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n")

    report = health.check([log])
    assert report["healthy"] is True
    assert report["problems"] == []


def test_the_cpu_bound_startup_window_is_not_called_a_stall(tmp_path, monkeypatch):
    """Every training run begins CPU-only while supervision is verified.

    Flagging that would make the watchdog fire on every launch, and a watchdog
    that cries wolf is one nobody reads.
    """
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "running", "name": "jobA", "stdout_age_seconds": 60,
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n", age_seconds=60)

    report = health.check([log])
    assert report["healthy"] is True


def test_an_idle_gpu_with_no_outstanding_work_is_a_warning_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {"state": "completed", "name": "jobA"})
    log = _log(tmp_path, "[QUEUE] starting jobA\n[QUEUE] jobA -> completed\n")

    report = health.check([log])
    assert report["healthy"] is True
    assert any("idle" in w for w in report["warnings"])
