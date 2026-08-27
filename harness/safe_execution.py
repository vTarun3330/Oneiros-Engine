"""Restricted, separate-process execution for function-level assertions.

This is a defence-in-depth harness, not a claim of perfect OS sandboxing.  It
combines a fresh isolated Python process, a temporary working directory,
restricted built-ins/imports, per-execution tracing deadlines, a hard parent
timeout, and POSIX resource limits when available.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_TIMEOUT_SECONDS = 0.5
MAX_PAYLOAD_BYTES = 2_000_000
_ALLOWED_IMPORT_ROOTS = {
    "__future__", "base64", "bisect", "calendar", "collections", "copy", "dataclasses",
    "datetime", "decimal", "enum", "fractions", "functools", "hashlib",
    "heapq", "itertools", "json", "math", "operator", "random", "re",
    "statistics", "string", "typing",
}
_DANGEROUS_SOURCE_NAMES = {
    "__import__", "breakpoint", "compile", "ctypes", "delattr", "eval", "exec",
    "getattr", "globals", "input", "locals", "multiprocessing", "open", "os",
    "pathlib", "pickle", "requests", "resource", "setattr", "shutil", "signal",
    "socket", "subprocess", "sys", "tempfile", "threading", "vars",
}


class _BoundedTextIO(io.StringIO):
    """Capture diagnostic output without allowing unbounded worker memory use."""

    def __init__(self, limit: int = 8_000):
        super().__init__()
        self.limit = limit

    def write(self, text: str) -> int:
        original_length = len(text)
        remaining = max(0, self.limit - self.tell())
        if remaining:
            super().write(text[:remaining])
        return original_length


def _source_policy_error(source: str) -> str:
    """Reject source capable of reaching process/filesystem/network primitives."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return f"source_syntax_error:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            else:
                names = [node.module.split(".", 1)[0]] if node.module else []
            forbidden = [name for name in names if name not in _ALLOWED_IMPORT_ROOTS]
            if forbidden:
                return f"source_import_not_allowed:{forbidden[0]}"
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_SOURCE_NAMES:
            return f"source_name_not_allowed:{node.id}"
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _DANGEROUS_SOURCE_NAMES
        ):
            return f"source_attribute_not_allowed:{node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_SOURCE_NAMES:
                return f"source_call_not_allowed:{node.func.id}"
    return ""


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _restricted_builtins() -> Dict[str, Any]:
    allowed_names = {
        "__build_class__", "abs", "all", "any", "ascii", "bin", "bool",
        "bytearray", "bytes", "callable", "chr", "classmethod", "complex",
        "dict", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "hash", "hex", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "object",
        "oct", "ord", "pow", "print", "property", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "staticmethod", "str", "sum",
        "super", "tuple", "type", "zip",
    }
    namespace = {name: getattr(builtins, name) for name in allowed_names}
    for name, value in vars(builtins).items():
        if isinstance(value, type) and issubclass(value, BaseException):
            namespace[name] = value
    namespace["__import__"] = _safe_import
    return namespace


class _ExecutionDeadline(Exception):
    pass


def _run_with_deadline(code: str, namespace: Dict[str, Any], filename: str, timeout: float):
    deadline = time.perf_counter() + timeout

    def trace(frame, event, arg):
        if time.perf_counter() > deadline:
            raise _ExecutionDeadline("execution deadline exceeded")
        return trace

    old_trace = sys.gettrace()
    old_recursion = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(200)
        sys.settrace(trace)
        exec(compile(code, filename, "exec"), namespace)
    finally:
        sys.settrace(old_trace)
        sys.setrecursionlimit(old_recursion)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return {"repr": repr(value)[:2_000], "type": type(value).__name__}


def _execute(function_code: str, test_code: str, timeout: float) -> Dict[str, Any]:
    namespace = {
        "__builtins__": _restricted_builtins(),
        "__name__": "__oneiros_candidate__",
    }
    stdout = _BoundedTextIO()
    stderr = _BoundedTextIO()
    started = time.perf_counter()
    try:
        random.seed(0)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            _run_with_deadline(function_code, namespace, "<function>", timeout)
            _run_with_deadline(test_code, namespace, "<candidate>", timeout)
        return {
            "status": "pass",
            "ok": True,
            "result": _json_safe(namespace.get("result", namespace.get("output"))),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "error": "",
            "elapsed_seconds": time.perf_counter() - started,
        }
    except AssertionError as exc:
        status, error = "assertion_error", f"AssertionError: {exc}"
    except _ExecutionDeadline:
        status, error = "timeout", "TIMEOUT"
    except SystemExit:
        status, error = "system_exit", "SystemExit"
    except KeyboardInterrupt:
        status, error = "keyboard_interrupt", "KeyboardInterrupt"
    except BaseException as exc:
        status, error = "error", f"{type(exc).__name__}: {str(exc)[:500]}"
    return {
        "status": status,
        "ok": False,
        "result": None,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "error": error,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _apply_posix_limits(total_timeout: float) -> None:
    try:
        import resource
    except ImportError:
        return
    limits = (
        (resource.RLIMIT_CPU, max(1, math.ceil(total_timeout)), max(1, math.ceil(total_timeout) + 1)),
        (resource.RLIMIT_AS, 512 * 1024 * 1024, 512 * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 1024 * 1024, 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64, 64),
    )
    for key, soft, hard in limits:
        try:
            resource.setrlimit(key, (soft, hard))
        except (OSError, ValueError):
            pass


def _worker_main() -> int:
    try:
        payload = json.load(sys.stdin)
        timeout = float(payload.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        tests = list(payload.get("tests", []))
        total_timeout = max(timeout, timeout * max(1, len(tests)) * 2)
        _apply_posix_limits(total_timeout)
        golden = str(payload["golden_code"])
        mutant = payload.get("mutant_code")
        for label, source in (("golden", golden), ("mutant", mutant)):
            if source is None:
                continue
            policy_error = _source_policy_error(str(source))
            if policy_error:
                raise ValueError(f"{label}:{policy_error}")
        rows = []
        for test in tests:
            golden_result = _execute(golden, str(test), timeout)
            mutant_result = None
            if mutant is not None and golden_result["ok"]:
                mutant_result = _execute(str(mutant), str(test), timeout)
            rows.append({"test": str(test), "golden": golden_result, "mutant": mutant_result})
        json.dump({"rows": rows}, sys.stdout, separators=(",", ":"))
        return 0
    except BaseException as exc:
        json.dump({"worker_error": f"{type(exc).__name__}: {str(exc)[:500]}"}, sys.stdout)
        return 2


def _invoke_worker(
    golden_code: str,
    tests: Iterable[str],
    mutant_code: str | None,
    timeout_seconds: float,
) -> List[Dict[str, Any]]:
    tests = list(tests)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    source_errors = [
        ("golden", _source_policy_error(golden_code)),
        ("mutant", _source_policy_error(mutant_code)) if mutant_code is not None else ("mutant", ""),
    ]
    for label, error in source_errors:
        if error:
            return [{
                "test": test,
                "golden": {
                    "status": "source_policy_error",
                    "ok": False,
                    "result": None,
                    "error": f"{label}:{error}",
                },
                "mutant": None,
            } for test in tests]
    payload = json.dumps({
        "golden_code": golden_code,
        "mutant_code": mutant_code,
        "tests": tests,
        "timeout_seconds": timeout_seconds,
    })
    if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("execution payload exceeds safety limit")
    total_timeout = max(1.0, timeout_seconds * max(1, len(tests)) * (2 if mutant_code is not None else 1) + 0.75)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
    }
    with tempfile.TemporaryDirectory(prefix="oneiros-exec-") as working_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                input=payload,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=working_dir,
                env=environment,
                timeout=total_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [{
                "test": test,
                "golden": {"status": "timeout", "ok": False, "result": None, "error": "WORKER_TIMEOUT"},
                "mutant": None,
            } for test in tests]
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        response = {"worker_error": f"invalid worker response: {completed.stderr[:500]}"}
    if completed.returncode != 0 or response.get("worker_error"):
        error = response.get("worker_error", f"worker exited {completed.returncode}")
        return [{
            "test": test,
            "golden": {"status": "worker_error", "ok": False, "result": None, "error": error},
            "mutant": None,
        } for test in tests]
    return list(response.get("rows", []))


def execute_code(
    function_code: str,
    test_code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Tuple[bool, Any, str]:
    """Execute one test in a separate process using the restricted worker."""
    row = _invoke_worker(function_code, [test_code], None, timeout_seconds)[0]
    result = row["golden"]
    return bool(result["ok"]), result.get("result"), str(result.get("error", ""))


def execute_assertions(
    tests: Iterable[str],
    function_code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """Execute assertions against only the visible code under test.

    Unlike :func:`classify_assertions`, this helper has no reference/mutant
    comparison.  It is therefore safe to use for iterative inference feedback:
    the policy can see compilation/runtime information from the supplied code,
    but it cannot receive hidden oracle information from the fixed program.
    """
    rows = _invoke_worker(function_code, tests, None, timeout_seconds)
    return [
        {
            "test": str(row.get("test", "")),
            **dict(row.get("golden") or {}),
        }
        for row in rows
    ]


def classify_assertions(
    tests: Iterable[str],
    golden_code: str,
    mutant_code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """Return reference/mutant outcomes for each assertion in one worker."""
    rows = _invoke_worker(golden_code, tests, mutant_code, timeout_seconds)
    for row in rows:
        golden_ok = bool(row["golden"].get("ok"))
        mutant_result = row.get("mutant") or {}
        row["valid"] = golden_ok
        row["killed"] = golden_ok and not bool(mutant_result.get("ok"))
    return rows


if __name__ == "__main__" and "--worker" in sys.argv:
    raise SystemExit(_worker_main())
