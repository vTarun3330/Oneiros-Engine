"""The health check must detect failures, not merely report health.

A checker that only ever says "healthy" is worse than none: it converts an
unknown into a reassurance. Each case here is a stall this project actually
had, and each must be NAMED rather than appear as an absence.
"""
from __future__ import annotations

import json
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
    """The 2.7-hour idle: queue says running, card is cold, log frozen.

    live_queues is passed explicitly here and everywhere below. Left to
    default, check() probes the real machine, and this test passed or failed
    depending on whether an unrelated gpu_queue happened to be running at the
    time - which is how it passed at 15:00 and failed at 15:40 with no code
    change between.
    """
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {"state": "completed", "name": "other"})
    log = _log(tmp_path, "[QUEUE] starting jobA\n", age_seconds=3600)

    report = health.check([log], live_queues=1)
    assert report["healthy"] is False
    assert any("STALLED" in p and "jobA" in p for p in report["problems"])


def test_a_terminal_run_the_queue_never_reported_is_stranded(tmp_path, monkeypatch):
    """The refused evaluation: run failed, queue waits on a marker forever.

    Stranding is now resolved against the run belonging to the CLAIMED job
    rather than whatever run happens to be newest, so a queue stuck on job A
    is still caught once an unrelated job B has started.
    """
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 50)
    monkeypatch.setattr(health, "_run_for_job", lambda job: {
        "run_id": "20260906-000000-jobA", "state": "failed",
        # Comfortably outside HANDOFF_GRACE_SECONDS, so this is a real
        # stranding rather than a queue that has not polled yet.
        "detail": "no adapter", "settled_at": time.time() - 3_000,
    })
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "failed", "name": "jobA", "exit_code": 1,
        "termination": "python_exception", "detail": "no adapter",
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n")

    report = health.check([log], live_queues=1)
    assert report["healthy"] is False
    assert any("STRANDED" in p for p in report["problems"])


def test_a_failed_run_is_always_a_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 50)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "failed", "name": "jobZ", "exit_code": 1,
        "termination": "cuda_oom", "detail": "out of memory",
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n[QUEUE] jobA -> completed\n")

    report = health.check([log], live_queues=1)
    assert report["healthy"] is False
    assert any("FAILED RUN" in p and "cuda_oom" in p for p in report["problems"])


def test_a_silent_running_job_with_a_cold_gpu_is_a_problem(tmp_path, monkeypatch):
    """The crashed build that looked slow: marked running, writing nothing."""
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "running", "name": "jobA", "stdout_age_seconds": 3600,
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n[QUEUE] jobA -> completed\n")

    report = health.check([log], live_queues=1)
    assert report["healthy"] is False
    assert any("SILENT RUN" in p for p in report["problems"])


def test_a_healthy_pipeline_reports_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 70)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "running", "name": "jobA", "stdout_age_seconds": 20,
    })
    log = _log(tmp_path, "[QUEUE] starting jobA\n")

    report = health.check([log], live_queues=1)
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

    report = health.check([log], live_queues=1)
    assert report["healthy"] is True


def test_an_idle_gpu_with_no_outstanding_work_is_a_failure(tmp_path, monkeypatch):
    """Deliberately reversed. This test previously asserted the opposite.

    Treating an idle card as a warning was wrong twice over: the standing
    instruction on this project is that the GPU must not sit idle, and
    warnings were not printed at all while any problem existed. On 2026-09-08
    a false STALLED alarm from a dead queue's log therefore hid both a
    finished pair of seed evaluations and an idle GPU behind it.

    The previous expectation stays in git history rather than being deleted,
    and the reversal is recorded here so it reads as a decision, not a drift.
    """
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run",
                        lambda: {"state": "completed", "name": "jobA"})
    log = _log(tmp_path, "[QUEUE] starting jobA\n[QUEUE] jobA -> completed\n")

    report = health.check([log], live_queues=0)
    assert report["healthy"] is False
    assert any(problem.startswith("IDLE") for problem in report["problems"])


# --- per-queue liveness, added after the watchdog cried wolf --------------
#
# queue_memorisation_arm.log has claimed 'reg_val_s42' is running since
# 2026-09-06. The job actually failed and the queue was killed before writing
# its result line, so the log claims a running job forever. The first fix
# counted gpu_queue processes globally, which was worse: any OTHER queue
# running made every dead queue's log read as stranded again, and the false
# alarm hid a genuinely idle GPU behind it.

def _settled_run(tmp_path, job, state, age_seconds):
    import os
    import time
    run = tmp_path / "runs" / f"20260906-000000-{job}"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({
        "run_id": run.name, "state": state,
        "termination": {"reason": "error", "detail": "adapter missing"},
    }), encoding="utf-8")
    when = time.time() - age_seconds
    os.utime(run / "status.json", (when, when))
    return run


def _queue_log(tmp_path, name, job, age_seconds):
    import os
    import time
    log = tmp_path / name
    log.write_text(f"[QUEUE] starting {job}\n", encoding="utf-8")
    when = time.time() - age_seconds
    os.utime(log, (when, when))
    return log


def test_a_finished_queues_log_is_a_note_not_a_stall(tmp_path, monkeypatch):
    """The exact false alarm: dead queue, job long settled, nothing since."""
    import scripts.pipeline_health as health

    _settled_run(tmp_path, "reg_val_s42", "failed", age_seconds=100_000)
    log = _queue_log(tmp_path, "queue_old.log", "reg_val_s42", age_seconds=110_000)
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 60)

    report = health.check([log], live_queues=1)
    assert not any("STRANDED" in p for p in report["problems"]), (
        "a queue that has written nothing since its job settled long ago is "
        "gone, not waiting"
    )
    assert any("finished queue" in w for w in report["warnings"])


def test_a_live_queue_waiting_on_a_settled_job_is_still_stranded(tmp_path, monkeypatch):
    """The real stranding must survive the fix for the false alarm."""
    import os
    import time

    import scripts.pipeline_health as health

    _settled_run(tmp_path, "reg_val_s42", "failed", age_seconds=3_000)
    log = _queue_log(tmp_path, "queue_live.log", "reg_val_s42", age_seconds=10)
    now = time.time()
    os.utime(log, (now, now))
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 60)

    report = health.check([log], live_queues=1)
    assert any("STRANDED" in p for p in report["problems"])


def test_an_idle_gpu_with_nothing_queued_is_a_problem(tmp_path, monkeypatch):
    """Reported as a warning before, and warnings were hidden behind problems."""
    import scripts.pipeline_health as health

    run = _settled_run(tmp_path, "v42_val_s44", "completed", age_seconds=3_000)
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)

    report = health.check([], live_queues=0)
    assert any(p.startswith("IDLE") for p in report["problems"]), (
        "an idle card with no outstanding work is the failure this project "
        "keeps paying for; it must not be a warning"
    )
    assert run.exists()


def test_warnings_are_printed_even_when_a_problem_exists(capsys, tmp_path, monkeypatch):
    import scripts.pipeline_health as health

    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: None)
    monkeypatch.setattr(health, "check", lambda *a, **k: {
        "problems": ["IDLE: nothing is running"],
        "warnings": ["a warning that must not be swallowed"],
        "latest_run": {}, "gpu_utilisation_percent": None,
    })
    monkeypatch.setattr(sys, "argv", ["pipeline_health.py"])

    assert health.main() == 1
    printed = capsys.readouterr().out
    assert "[UNHEALTHY] IDLE" in printed
    assert "must not be swallowed" in printed


# --- the handoff race, caught by the watchdog firing on a healthy queue ----
#
# gpu_queue polls every 30s. Between a run reaching a terminal state and its
# queue writing the result line, the last job looks finished-but-unreported
# and the card is briefly idle. On 2026-09-08 that produced STRANDED and IDLE
# alerts for base_val_s52 while the queue was working perfectly and started
# relearn_val_s52 moments later. A watchdog that fires on every successful
# handoff is noise, and noise is what let the original 2.7 hour stall survive
# a monitor.

def test_a_job_that_settled_moments_ago_is_a_handoff_not_a_stranding(monkeypatch, tmp_path):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_run_for_job", lambda job: {
        "run_id": "20260908-040121-jobA", "state": "completed",
        "detail": "child exited 0", "settled_at": time.time() - 5,
    })
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "completed", "name": "jobA", "settled_at": time.time() - 5,
    })
    log = _log(tmp_path, "[QUEUE] starting jobA" + chr(92) + "n")

    report = health.check([log], live_queues=1)
    assert report["problems"] == [], (
        "a queue mid-handoff is working, not stranded; alerting here fires on "
        "every successful job transition"
    )


def test_a_job_settled_long_ago_and_unreported_is_still_stranded(monkeypatch, tmp_path):
    """The grace must not swallow the real stranding it sits next to."""
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 50)
    monkeypatch.setattr(health, "_run_for_job", lambda job: {
        "run_id": "20260906-000000-jobA", "state": "failed",
        "detail": "no adapter", "settled_at": time.time() - 3_000,
    })
    monkeypatch.setattr(health, "_latest_run", lambda: {"state": "failed", "name": "jobA"})
    log = _log(tmp_path, "[QUEUE] starting jobA" + chr(92) + "n")

    report = health.check([log], live_queues=1)
    assert any("STRANDED" in problem for problem in report["problems"])


def test_an_idle_card_between_two_jobs_is_not_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "completed", "name": "jobA", "settled_at": time.time() - 10,
    })

    report = health.check([], live_queues=1)
    assert not any(problem.startswith("IDLE") for problem in report["problems"])


def test_an_idle_card_long_after_the_last_job_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(health, "_gpu_utilisation", lambda: 0)
    monkeypatch.setattr(health, "_latest_run", lambda: {
        "state": "completed", "name": "jobA", "settled_at": time.time() - 3_000,
    })

    report = health.check([], live_queues=0)
    assert any(problem.startswith("IDLE") for problem in report["problems"])
