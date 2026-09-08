"""Run independent sandboxed executions across cores, changing only scheduling.

Every analysis in this project executes one sandboxed subprocess at a time on
a 32-core machine. The work is embarrassingly parallel - the Atheris rerun is
4542 completely independent fuzz runs, and each candidate execution is
independent of every other - so the serial loop was costing hours for no
reason.

What this deliberately does NOT change:

* isolation. Each job still goes through the same worker subprocess with the
  same restricted builtins, the same source policy and the same timeout. The
  pool schedules those subprocesses; it does not run anything in-process.
* determinism. Results come back in input order, always, so a report built
  from them is byte-identical whether it ran on one worker or thirty. A
  parallel harness that reorders results would make every artifact depend on
  scheduling, which is worse than being slow.
* failure handling. A job that raises is captured and returned as a failed
  result rather than killing the pool, so one pathological target cannot
  discard the other four thousand.

Threads rather than processes: each job's real work already happens in a
child process, so the parent is only waiting on I/O. Threads avoid pickling
the payloads and avoid the process-pool spawn cost on Windows.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

#: Leave a couple of cores for the OS and for whatever else is running. The
#: aim is to finish sooner, not to make the machine unusable while it does.
RESERVED_CORES = 2


def default_workers() -> int:
    cores = os.cpu_count() or 4
    return max(1, cores - RESERVED_CORES)


def map_jobs(
    function: Callable[[T], R],
    jobs: Sequence[T],
    workers: int | None = None,
    on_error: Callable[[T, BaseException], R] | None = None,
) -> list[R]:
    """Apply `function` to every job, in parallel, returning input order.

    `on_error` converts an exception into a result so one failing job does not
    discard the rest. Without it the exception is re-raised, which is the right
    default for callers that would rather stop than report partial data.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    count = workers if workers is not None else default_workers()
    count = max(1, min(count, len(jobs)))
    if count == 1:
        results = []
        for job in jobs:
            try:
                results.append(function(job))
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                if on_error is None:
                    raise
                results.append(on_error(job, exc))
        return results

    ordered: list[Any] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {
            pool.submit(function, job): index for index, job in enumerate(jobs)
        }
        for future, index in futures.items():
            try:
                ordered[index] = future.result()
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                if on_error is None:
                    raise
                ordered[index] = on_error(jobs[index], exc)
    return ordered


def progress_map(
    function: Callable[[T], R],
    jobs: Sequence[T],
    label: str,
    every: int = 200,
    workers: int | None = None,
    on_error: Callable[[T, BaseException], R] | None = None,
) -> list[R]:
    """map_jobs with a periodic line, so a long run is not silent.

    Silence is what let a 2.7 hour stall look like progress on this project,
    so a parallel run that prints nothing for forty minutes is not an
    improvement worth having.
    """
    jobs = list(jobs)
    total = len(jobs)
    done = 0

    def wrapped(job: T) -> R:
        nonlocal done
        result = function(job)
        done += 1
        if done % every == 0 or done == total:
            print(f"{label} {done}/{total}", flush=True)
        return result

    return map_jobs(wrapped, jobs, workers=workers, on_error=on_error)
