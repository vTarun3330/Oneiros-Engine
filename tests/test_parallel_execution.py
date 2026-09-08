"""The pool may change speed and nothing else.

A parallel harness that reorders results, or that loses the rest of the work
when one job fails, would make every artifact depend on scheduling. That is
worse than being slow, so these tests are about equivalence rather than
throughput.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.parallel_execution import default_workers, map_jobs, progress_map


def test_results_come_back_in_input_order():
    """Out-of-order completion must not reorder the output."""
    def slow_for_small(value: int) -> int:
        # Later jobs finish FIRST, so a naive as-completed collection would
        # scramble the order.
        time.sleep(0.02 if value < 3 else 0.0)
        return value * 10

    assert map_jobs(slow_for_small, [0, 1, 2, 3, 4], workers=5) == [0, 10, 20, 30, 40]


def test_one_worker_and_many_workers_agree():
    """The number of workers is a scheduling choice, not a semantic one."""
    jobs = list(range(50))
    serial = map_jobs(lambda v: v * v, jobs, workers=1)
    parallel = map_jobs(lambda v: v * v, jobs, workers=16)
    assert serial == parallel


def test_a_failing_job_does_not_discard_the_others():
    def sometimes_raises(value: int) -> str:
        if value == 3:
            raise RuntimeError("boom")
        return f"ok{value}"

    results = map_jobs(
        sometimes_raises, [1, 2, 3, 4], workers=4,
        on_error=lambda job, exc: f"failed{job}:{type(exc).__name__}",
    )
    assert results == ["ok1", "ok2", "failed3:RuntimeError", "ok4"]


def test_without_a_handler_the_exception_is_raised():
    """A caller that would rather stop than report partial data can."""
    with pytest.raises(RuntimeError):
        map_jobs(lambda v: (_ for _ in ()).throw(RuntimeError("x")), [1], workers=2)


def test_an_empty_job_list_is_not_an_error():
    assert map_jobs(lambda v: v, [], workers=8) == []


def test_workers_never_exceed_the_job_count():
    assert map_jobs(lambda v: v, [1, 2], workers=64) == [1, 2]


def test_some_cores_are_left_for_the_machine():
    import os
    assert default_workers() < (os.cpu_count() or 4) or (os.cpu_count() or 4) <= 2
    assert default_workers() >= 1


def test_progress_is_reported_so_a_long_run_is_not_silent(capsys):
    progress_map(lambda v: v, list(range(10)), "probing", every=5, workers=4)
    printed = capsys.readouterr().out
    assert "probing" in printed
    assert "10/10" in printed


def test_sandboxed_execution_still_goes_through_the_worker():
    """Isolation must be unchanged: the pool schedules subprocesses only."""
    from harness.safe_execution import execute_code

    def job(value: int):
        return execute_code(
            "def f(x):\n    return x + 1\n", f"result = f({value})", 5.0)

    results = map_jobs(job, [1, 2, 3], workers=3)
    assert [ok for ok, _result, _error in results] == [True, True, True]

    forbidden = execute_code(
        "def f(x):\n    return x\n", "import os", 5.0)
    assert not forbidden[0], "the sandbox must still refuse imports under the pool"
