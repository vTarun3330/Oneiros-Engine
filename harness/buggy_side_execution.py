"""Run one candidate against the code under test ONLY, and say where it failed.

This is the observation the execution-feedback repair loop is allowed to make.
It reuses the restricted-execution primitives of :mod:`harness.safe_execution`
(fresh isolated interpreter, restricted built-ins and imports, tracing
deadline, hard parent timeout, POSIX limits where available) and adds one fact
the existing worker does not report: **which frame raised**.

That fact is what makes the repair rules enforceable:

* an exception whose innermost frame is the *candidate* (``NameError`` for a
  name the candidate invented, ``TypeError`` from calling the function with
  the wrong arity) is a malformed artifact and may be repaired;
* an exception whose innermost frame is the *code under test* is behaviour of
  the buggy implementation - possibly the very defect - and is retained
  untouched, never "repaired" away;
* an ``AssertionError`` is an observed assertion failure: retained, never
  repaired, and never reported to the model as right or wrong.

STRUCTURAL LEAKAGE BOUNDARY: the payload has no field for a reference or fixed
implementation, gold tests or a mutation diff.  There is nothing hidden in
scope to leak.  Each call runs in a fresh process and a fresh namespace, and
the candidate source is re-executed from text, so no mutable object survives
from one execution to the next.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

if __name__ == "__main__":  # worker process: -I leaves the project off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.safe_execution import (
    MAX_PAYLOAD_BYTES,
    _BoundedTextIO,
    _ExecutionDeadline,
    _apply_posix_limits,
    _restricted_builtins,
    _run_with_deadline,
    _source_policy_error,
)

EXECUTOR_VERSION = "oneiros_buggy_side_executor_v1"
TARGET_FILENAME = "<code_under_test>"
CANDIDATE_FILENAME = "<candidate>"
#: Observations this executor can return.  Closed set.
STATUSES = ("pass", "assertion_error", "exception", "timeout", "infrastructure_error")
ORIGINS = ("candidate", "code_under_test", "none", "harness")


def _innermost_origin(tb) -> str:
    """The innermost frame that belongs to the candidate or the code under test.

    Harness frames (the deadline tracer, the restricted importer) are skipped,
    so a timeout is attributed to the loop that was running when it fired.
    """
    origin = "harness"
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if filename == TARGET_FILENAME:
            origin = "code_under_test"
        elif filename == CANDIDATE_FILENAME:
            origin = "candidate"
        tb = tb.tb_next
    return origin


def _execute(code_under_test: str, candidate: str, timeout: float) -> Dict[str, Any]:
    namespace = {"__builtins__": _restricted_builtins(), "__name__": "__oneiros_candidate__"}
    stdout, stderr = _BoundedTextIO(), _BoundedTextIO()
    started = time.perf_counter()
    try:
        import random
        random.seed(0)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                _run_with_deadline(code_under_test, namespace, TARGET_FILENAME, timeout)
            except BaseException as exc:  # the shown code itself would not load
                return {"status": "infrastructure_error", "origin": "harness",
                        "exception_type": type(exc).__name__,
                        "message": "code under test failed to load",
                        "elapsed_seconds": time.perf_counter() - started}
            _run_with_deadline(candidate, namespace, CANDIDATE_FILENAME, timeout)
        return {"status": "pass", "origin": "none", "exception_type": "",
                "message": "", "elapsed_seconds": time.perf_counter() - started}
    except AssertionError as exc:
        return {"status": "assertion_error", "origin": _innermost_origin(exc.__traceback__),
                "exception_type": "AssertionError", "message": "",
                "elapsed_seconds": time.perf_counter() - started}
    except _ExecutionDeadline as exc:
        return {"status": "timeout", "origin": _innermost_origin(exc.__traceback__),
                "exception_type": "Timeout", "message": "",
                "elapsed_seconds": time.perf_counter() - started}
    except BaseException as exc:
        return {"status": "exception", "origin": _innermost_origin(exc.__traceback__),
                "exception_type": type(exc).__name__, "message": str(exc)[:200],
                "elapsed_seconds": time.perf_counter() - started}


def _worker_main() -> int:
    try:
        payload = json.load(sys.stdin)
        timeout = float(payload["timeout_seconds"])
        _apply_posix_limits(timeout * 2 + 1)
        result = _execute(str(payload["code_under_test"]), str(payload["candidate"]), timeout)
        json.dump(result, sys.stdout, separators=(",", ":"))
        return 0
    except BaseException as exc:
        json.dump({"status": "infrastructure_error", "origin": "harness",
                   "exception_type": type(exc).__name__, "message": "worker failure",
                   "elapsed_seconds": 0.0}, sys.stdout)
        return 2


def execute_on_code_under_test(code_under_test: str, candidate: str,
                               timeout_seconds: float) -> Dict[str, Any]:
    """Execute ``candidate`` against ``code_under_test`` in a fresh process.

    Signature is the boundary: there is no parameter through which a reference
    implementation could arrive.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    policy_error = _source_policy_error(code_under_test)
    if policy_error:
        return {"status": "infrastructure_error", "origin": "harness",
                "exception_type": "SourcePolicy", "message": "code under test refused",
                "elapsed_seconds": 0.0, "executor_version": EXECUTOR_VERSION}
    payload = json.dumps({"code_under_test": code_under_test, "candidate": candidate,
                          "timeout_seconds": timeout_seconds})
    if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("execution payload exceeds safety limit")
    environment = {"PATH": os.environ.get("PATH", ""),
                   "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                   "WINDIR": os.environ.get("WINDIR", ""),
                   "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="oneiros-buggy-") as working_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                input=payload, text=True, encoding="utf-8", errors="replace",
                capture_output=True, cwd=working_dir, env=environment,
                timeout=timeout_seconds * 2 + 2.0, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "origin": "harness", "exception_type": "Timeout",
                    "message": "", "elapsed_seconds": timeout_seconds * 2 + 2.0,
                    "executor_version": EXECUTOR_VERSION}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"status": "infrastructure_error", "origin": "harness",
                  "exception_type": "InvalidWorkerResponse", "message": "",
                  "elapsed_seconds": 0.0}
    if result.get("status") not in STATUSES or result.get("origin") not in ORIGINS:
        result = {"status": "infrastructure_error", "origin": "harness",
                  "exception_type": "InvalidWorkerResponse", "message": "",
                  "elapsed_seconds": 0.0}
    result["executor_version"] = EXECUTOR_VERSION
    return result


if __name__ == "__main__" and "--worker" in sys.argv:
    raise SystemExit(_worker_main())
