"""Leakage-resistant execution supervision for code shown to the model.

This module is intentionally separate from the historical evaluator.  It
executes one literal-input call against *only* the supplied shown code, records
the shown-code line events, and serializes the return value into a small,
typed, Python-literal-safe object.  It does not compare two implementations
and its prompt builder has no channel for reference code, gold assertions, or
patch information.

This is defence in depth rather than a claim of a perfect OS sandbox.  As in
``harness.safe_execution``, untrusted Python runs in a fresh isolated process,
in a temporary directory, with restricted builtins/imports, a tracing
deadline, a hard parent deadline, a small environment, and POSIX resource
limits where available.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
from dataclasses import dataclass
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
from typing import Any, Mapping


DEFAULT_TIMEOUT_SECONDS = 0.5
DEFAULT_TRACE_EVENT_LIMIT = 128
DEFAULT_DETERMINISM_RUNS = 2

OUTPUT_PREDICTION_TASK_KIND = "execution_output_prediction"
OUTPUT_PREDICTION_PROMPT_SCHEMA = "oneiros_execution_output_prediction_v1"

MAX_PAYLOAD_BYTES = 256_000
MAX_SOURCE_BYTES = 64_000
MAX_SOURCE_LINE_BYTES = 2_048
MAX_SOURCE_AST_NODES = 4_000
MAX_SOURCE_AST_DEPTH = 80
MAX_CALL_BYTES = 4_096
MAX_CALL_AST_NODES = 256
MAX_TRACE_EVENT_LIMIT = 128
MAX_DISTINCT_TRACE_TRANSITIONS = 64
MAX_TRACE_PAYLOAD_BYTES = 4_096
MAX_DETERMINISM_RUNS = 3
MAX_CAPTURE_CHARS = 8_000

MAX_VALUE_DEPTH = 6
MAX_VALUE_ITEMS = 128
MAX_VALUE_LITERAL_BYTES = 8_192
MAX_VALUE_STRING_BYTES = 2_048
MAX_VALUE_INTEGER_ABS = 10**12

_SHOWN_FILENAME = "<shown_code>"
_CALL_FILENAME = "<shown_call>"

# Kept aligned with the established safe-execution allowlist.  Imports remain
# subject to source-policy checks, and private/dunder attribute access is
# rejected even on an allowed module.
_ALLOWED_IMPORT_ROOTS = {
    "__future__", "base64", "bisect", "calendar", "collections", "copy",
    "dataclasses", "datetime", "decimal", "enum", "fractions", "functools",
    "hashlib", "heapq", "itertools", "json", "math", "operator", "random",
    "re", "statistics", "string", "typing",
}
_DANGEROUS_NAMES = {
    "__import__", "breakpoint", "compile", "ctypes", "delattr", "eval",
    "exec", "getattr", "globals", "input", "locals", "memoryview",
    "multiprocessing", "open", "os", "pathlib", "pickle", "requests",
    "resource", "setattr", "shutil", "signal", "socket", "subprocess",
    "sys", "tempfile", "threading", "vars",
}


class ExecutionSupervisionRejected(ValueError):
    """The proposed example is unsafe, unreliable, or not representable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail[:500]
        message = reason if not self.detail else f"{reason}: {self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class ExecutionEvidence:
    """Accepted evidence from independent, agreeing worker processes."""

    trace: tuple[Mapping[str, Any], ...]
    actual: Mapping[str, Any]
    determinism_runs: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace": [dict(event) for event in self.trace],
            "actual": dict(self.actual),
            "determinism_runs": self.determinism_runs,
        }


class _BoundedTextIO(io.StringIO):
    def write(self, text: str) -> int:
        original_length = len(text)
        remaining = max(0, MAX_CAPTURE_CHARS - self.tell())
        if remaining:
            super().write(text[:remaining])
        return original_length


class _ExecutionDeadline(Exception):
    pass


class _TraceCapExceeded(Exception):
    pass


class _UnsupportedValue(Exception):
    pass


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_ast_depth(child) for child in children)


def _source_policy_error(source: str, entry_point: str) -> str:
    if not isinstance(source, str) or not source.strip():
        return "empty_shown_code"
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return "shown_code_too_large"
    if not entry_point or not entry_point.isidentifier() or entry_point in _DANGEROUS_NAMES:
        return "invalid_entry_point"
    lines = source.splitlines()
    if any(len(line.encode("utf-8")) > MAX_SOURCE_LINE_BYTES for line in lines):
        return "shown_code_line_too_large"
    try:
        tree = ast.parse(source, filename=_SHOWN_FILENAME, mode="exec")
    except SyntaxError as exc:
        return f"shown_code_syntax_error:{exc.msg}"
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_SOURCE_AST_NODES:
        return "shown_code_ast_too_large"
    if _ast_depth(tree) > MAX_SOURCE_AST_DEPTH:
        return "shown_code_ast_too_deep"
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        and node in tree.body
        for node in nodes
    ):
        return "entry_point_not_defined_at_module_scope"
    if any(isinstance(node, ast.AsyncFunctionDef) and node.name == entry_point for node in nodes):
        return "async_entry_point_not_supported"
    for node in nodes:
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".", 1)[0] for alias in node.names]
            forbidden = [name for name in roots if name not in _ALLOWED_IMPORT_ROOTS]
            if forbidden:
                return f"shown_code_import_not_allowed:{forbidden[0]}"
        if isinstance(node, ast.ImportFrom):
            root = node.module.split(".", 1)[0] if node.module else ""
            if node.level or root not in _ALLOWED_IMPORT_ROOTS:
                return f"shown_code_import_not_allowed:{root or 'relative'}"
        if isinstance(node, ast.Name) and (
            node.id in _DANGEROUS_NAMES or node.id.startswith("__")
        ):
            return f"shown_code_name_not_allowed:{node.id}"
        if isinstance(node, ast.Attribute) and (
            node.attr in _DANGEROUS_NAMES or node.attr.startswith("_")
        ):
            return f"shown_code_attribute_not_allowed:{node.attr}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_NAMES:
                return f"shown_code_call_not_allowed:{node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _DANGEROUS_NAMES:
                return f"shown_code_call_not_allowed:{node.func.attr}"
    return ""


def _call_policy_error(call_expression: str, entry_point: str) -> str:
    if not isinstance(call_expression, str) or not call_expression.strip():
        return "empty_call_expression"
    if len(call_expression.encode("utf-8")) > MAX_CALL_BYTES:
        return "call_expression_too_large"
    try:
        tree = ast.parse(call_expression, filename=_CALL_FILENAME, mode="eval")
    except SyntaxError as exc:
        return f"call_syntax_error:{exc.msg}"
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_CALL_AST_NODES:
        return "call_ast_too_large"
    call = tree.body
    if not isinstance(call, ast.Call):
        return "call_must_be_direct_entry_point_call"
    if not isinstance(call.func, ast.Name) or call.func.id != entry_point:
        return "call_must_be_direct_entry_point_call"
    if any(keyword.arg is None for keyword in call.keywords):
        return "starred_call_arguments_not_allowed"
    # ast.literal_eval is the authoritative definition of an allowed input:
    # no helper calls, attribute reads, comprehensions, or ambient names.
    for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
        if isinstance(argument, ast.Starred):
            return "starred_call_arguments_not_allowed"
        try:
            value = ast.literal_eval(argument)
            _typed_literal(value)
        except (ValueError, TypeError, MemoryError, RecursionError, _UnsupportedValue):
            return "call_arguments_must_be_supported_literals"
    return ""


def execution_supervision_call_policy_error(
    call_expression: str, entry_point: str,
) -> str:
    """Public, side-effect-free call-policy check for corpus prefiltering."""
    return _call_policy_error(call_expression, entry_point)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _restricted_builtins() -> dict[str, Any]:
    allowed_names = {
        "__build_class__", "abs", "all", "any", "ascii", "bin", "bool",
        "bytearray", "bytes", "callable", "chr", "classmethod", "complex",
        "dict", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "hash", "hex", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "object",
        "oct", "ord", "pow", "print", "property", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "staticmethod",
        "str", "sum", "super", "tuple", "type", "zip",
    }
    namespace = {name: getattr(builtins, name) for name in allowed_names}
    for name, value in vars(builtins).items():
        if isinstance(value, type) and issubclass(value, BaseException):
            namespace[name] = value
    namespace["__import__"] = _safe_import
    return namespace


def _literal_text(value: Any, *, depth: int = 0, counter: list[int] | None = None) -> str:
    """Return a canonical literal without invoking arbitrary object reprs."""
    if counter is None:
        counter = [0]
    if depth > MAX_VALUE_DEPTH:
        raise _UnsupportedValue("value nesting exceeds limit")
    counter[0] += 1
    if counter[0] > MAX_VALUE_ITEMS:
        raise _UnsupportedValue("value item count exceeds limit")

    if value is None:
        return "None"
    if type(value) is bool:
        return "True" if value else "False"
    if type(value) is int:
        if abs(value) > MAX_VALUE_INTEGER_ABS:
            raise _UnsupportedValue("integer exceeds limit")
        return repr(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise _UnsupportedValue("non-finite float")
        return repr(value)
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise _UnsupportedValue("non-finite complex")
        return repr(value)
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_VALUE_STRING_BYTES:
            raise _UnsupportedValue("string exceeds limit")
        return repr(value)
    if type(value) is bytes:
        if len(value) > MAX_VALUE_STRING_BYTES:
            raise _UnsupportedValue("bytes exceeds limit")
        return repr(value)
    if type(value) is list:
        parts = [_literal_text(item, depth=depth + 1, counter=counter) for item in value]
        return "[" + ", ".join(parts) + "]"
    if type(value) is tuple:
        parts = [_literal_text(item, depth=depth + 1, counter=counter) for item in value]
        suffix = "," if len(parts) == 1 else ""
        return "(" + ", ".join(parts) + suffix + ")"
    if type(value) is dict:
        pairs: list[tuple[str, str]] = []
        for key, item in value.items():
            if type(key) not in (type(None), bool, int, float, complex, str, bytes, tuple):
                raise _UnsupportedValue("unsupported dictionary key")
            key_text = _literal_text(key, depth=depth + 1, counter=counter)
            item_text = _literal_text(item, depth=depth + 1, counter=counter)
            pairs.append((key_text, item_text))
        pairs.sort(key=lambda pair: pair[0])
        return "{" + ", ".join(f"{key}: {item}" for key, item in pairs) + "}"
    raise _UnsupportedValue(f"unsupported value type: {type(value).__name__}")


def _typed_literal(value: Any) -> dict[str, str]:
    literal = _literal_text(value)
    if len(literal.encode("utf-8")) > MAX_VALUE_LITERAL_BYTES:
        raise _UnsupportedValue("literal exceeds limit")
    try:
        recovered = ast.literal_eval(literal)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise _UnsupportedValue("literal does not round-trip") from exc
    if type(recovered) is not type(value) or recovered != value:
        raise _UnsupportedValue("typed literal does not round-trip")
    return {"type": type(value).__name__, "literal": literal}


def _validate_typed_literal_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != {"type", "literal"}:
        return False
    if not isinstance(payload["type"], str) or not isinstance(payload["literal"], str):
        return False
    if len(payload["literal"].encode("utf-8")) > MAX_VALUE_LITERAL_BYTES:
        return False
    try:
        recovered = ast.literal_eval(payload["literal"])
        return _typed_literal(recovered) == payload
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError, _UnsupportedValue):
        return False


def _validate_trace_payload(trace: Any, source: str, event_limit: int) -> bool:
    if not isinstance(trace, list) or len(trace) > event_limit:
        return False
    source_lines = source.splitlines()
    transitions: set[tuple[int, int]] = set()
    previous: int | None = None
    for event in trace:
        if not isinstance(event, dict) or set(event) != {"line", "source"}:
            return False
        line_number = event["line"]
        if not isinstance(line_number, int) or isinstance(line_number, bool):
            return False
        if not 1 <= line_number <= len(source_lines):
            return False
        if event["source"] != source_lines[line_number - 1]:
            return False
        if previous is not None:
            transitions.add((previous, line_number))
            if len(transitions) > MAX_DISTINCT_TRACE_TRANSITIONS:
                return False
        previous = line_number
    serialized = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    return len(serialized.encode("utf-8")) <= MAX_TRACE_PAYLOAD_BYTES


def _apply_posix_limits(total_timeout: float) -> None:
    try:
        import resource
    except ImportError:
        return
    limits = (
        (resource.RLIMIT_CPU, max(1, math.ceil(total_timeout)), max(2, math.ceil(total_timeout) + 1)),
        (resource.RLIMIT_AS, 512 * 1024 * 1024, 512 * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 1024 * 1024, 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64, 64),
    )
    for key, soft, hard in limits:
        try:
            resource.setrlimit(key, (soft, hard))
        except (OSError, ValueError):
            pass


def _execute_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = str(payload["shown_code"])
    call_expression = str(payload["call_expression"])
    entry_point = str(payload["entry_point"])
    timeout = float(payload["timeout_seconds"])
    trace_limit = int(payload["trace_event_limit"])
    source_error = _source_policy_error(source, entry_point)
    call_error = _call_policy_error(call_expression, entry_point)
    if source_error or call_error:
        return {
            "status": "harness_failure",
            "error": source_error or call_error,
        }

    deadline = time.perf_counter() + timeout
    source_lines = source.splitlines()
    trace_events: list[dict[str, Any]] = []
    trace_transitions: set[tuple[int, int]] = set()
    collect = False

    def tracer(frame, event, arg):
        del arg
        if time.perf_counter() > deadline:
            raise _ExecutionDeadline("execution deadline exceeded")
        if collect and event == "line" and frame.f_code.co_filename == _SHOWN_FILENAME:
            if len(trace_events) >= trace_limit:
                raise _TraceCapExceeded("trace event cap exceeded")
            line_number = frame.f_lineno
            if not 1 <= line_number <= len(source_lines):
                raise RuntimeError("trace line outside shown source")
            if trace_events:
                trace_transitions.add((int(trace_events[-1]["line"]), line_number))
                if len(trace_transitions) > MAX_DISTINCT_TRACE_TRANSITIONS:
                    raise _TraceCapExceeded("distinct trace transition cap exceeded")
            event_payload = {
                "line": line_number,
                "source": source_lines[line_number - 1],
            }
            trace_events.append(event_payload)
            serialized_trace = json.dumps(
                trace_events, ensure_ascii=False, separators=(",", ":")
            )
            if len(serialized_trace.encode("utf-8")) > MAX_TRACE_PAYLOAD_BYTES:
                raise _TraceCapExceeded("serialized trace payload cap exceeded")
        return tracer

    namespace = {
        "__builtins__": _restricted_builtins(),
        "__name__": "__oneiros_execution_supervision__",
    }
    stdout = _BoundedTextIO()
    stderr = _BoundedTextIO()
    old_trace = sys.gettrace()
    old_recursion = sys.getrecursionlimit()
    try:
        random.seed(0)
        sys.setrecursionlimit(200)
        sys.settrace(tracer)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(source, _SHOWN_FILENAME, "exec"), namespace)
            collect = True
            actual_value = eval(
                compile(call_expression, _CALL_FILENAME, "eval"), namespace
            )
            collect = False
        actual = _typed_literal(actual_value)
        return {"status": "ok", "trace": trace_events, "actual": actual}
    except _TraceCapExceeded as exc:
        return {"status": "trace_cap_overflow", "error": str(exc)[:500]}
    except _ExecutionDeadline:
        return {"status": "timeout", "error": "execution deadline exceeded"}
    except _UnsupportedValue as exc:
        return {"status": "unsupported_value", "error": str(exc)[:500]}
    except (SystemExit, KeyboardInterrupt) as exc:
        return {"status": "execution_error", "error": type(exc).__name__}
    except BaseException as exc:
        return {
            "status": "execution_error",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }
    finally:
        sys.settrace(old_trace)
        sys.setrecursionlimit(old_recursion)


def _worker_main() -> int:
    try:
        payload = json.load(sys.stdin)
        timeout = float(payload.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        _apply_posix_limits(max(1.0, timeout * 2))
        response = _execute_worker(payload)
        json.dump(response, sys.stdout, separators=(",", ":"), ensure_ascii=False)
        return 0
    except BaseException as exc:
        json.dump(
            {"status": "harness_failure", "error": f"{type(exc).__name__}: {str(exc)[:500]}"},
            sys.stdout,
            separators=(",", ":"),
        )
        return 2


def _invoke_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if len(serialized.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ExecutionSupervisionRejected("payload_too_large")
    timeout = float(payload["timeout_seconds"])
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
    }
    with tempfile.TemporaryDirectory(prefix="oneiros-supervision-") as working_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                input=serialized,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=working_dir,
                env=environment,
                timeout=max(1.0, timeout + 0.75),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionSupervisionRejected("timeout", "worker hard deadline exceeded") from exc
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutionSupervisionRejected(
            "harness_failure", f"invalid worker response: {completed.stderr[:300]}"
        ) from exc
    if completed.returncode != 0 or not isinstance(response, dict):
        detail = response.get("error", "") if isinstance(response, dict) else ""
        raise ExecutionSupervisionRejected(
            "harness_failure", detail or f"worker exited {completed.returncode}"
        )
    return response


def collect_execution_evidence(
    *,
    shown_code: str,
    entry_point: str,
    call_expression: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    trace_event_limit: int = DEFAULT_TRACE_EVENT_LIMIT,
    determinism_runs: int = DEFAULT_DETERMINISM_RUNS,
) -> ExecutionEvidence:
    """Execute a literal-input call and retain evidence only if it is reliable.

    Each determinism check uses a new interpreter and temporary directory.
    Trace and typed actual output must agree exactly across every run.
    """
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 < timeout_seconds <= 10
    ):
        raise ExecutionSupervisionRejected("invalid_timeout")
    if not isinstance(trace_event_limit, int) or isinstance(trace_event_limit, bool):
        raise ExecutionSupervisionRejected("invalid_trace_event_limit")
    if not 1 <= trace_event_limit <= MAX_TRACE_EVENT_LIMIT:
        raise ExecutionSupervisionRejected("invalid_trace_event_limit")
    if not isinstance(determinism_runs, int) or isinstance(determinism_runs, bool):
        raise ExecutionSupervisionRejected("invalid_determinism_runs")
    if not 2 <= determinism_runs <= MAX_DETERMINISM_RUNS:
        raise ExecutionSupervisionRejected("invalid_determinism_runs")
    source_error = _source_policy_error(shown_code, entry_point)
    if source_error:
        raise ExecutionSupervisionRejected("source_policy_error", source_error)
    call_error = _call_policy_error(call_expression, entry_point)
    if call_error:
        raise ExecutionSupervisionRejected("call_policy_error", call_error)

    payload = {
        "shown_code": shown_code,
        "entry_point": entry_point,
        "call_expression": call_expression,
        "timeout_seconds": float(timeout_seconds),
        "trace_event_limit": trace_event_limit,
    }
    observations: list[dict[str, Any]] = []
    for _ in range(determinism_runs):
        response = _invoke_worker(payload)
        status = response.get("status")
        if status != "ok":
            reason = status if status in {
                "timeout", "trace_cap_overflow", "unsupported_value",
                "execution_error", "harness_failure",
            } else "harness_failure"
            raise ExecutionSupervisionRejected(reason, str(response.get("error", "")))
        observation = {
            "trace": response.get("trace"),
            "actual": response.get("actual"),
        }
        if (
            not _validate_trace_payload(
                observation["trace"], shown_code, trace_event_limit
            )
            or not _validate_typed_literal_payload(observation["actual"])
        ):
            raise ExecutionSupervisionRejected("harness_failure", "worker response schema mismatch")
        observations.append(observation)
    if any(observation != observations[0] for observation in observations[1:]):
        raise ExecutionSupervisionRejected(
            "nondeterminism", "trace or actual output differed across fresh workers"
        )
    first = observations[0]
    return ExecutionEvidence(
        trace=tuple(dict(event) for event in first["trace"]),
        actual=dict(first["actual"]),
        determinism_runs=determinism_runs,
    )


def build_execution_supervision_prompt(
    *, specification: str, shown_code: str, entry_point: str, call_expression: str,
) -> str:
    """Build a prompt from shown inputs only.

    The deliberately closed signature is part of the leakage boundary: it has
    no ``**kwargs`` and no parameter for another implementation, assertions,
    expected values, or patches.
    """
    if not isinstance(specification, str) or not specification.strip():
        raise ExecutionSupervisionRejected("empty_specification")
    if len(specification.encode("utf-8")) > 32_000:
        raise ExecutionSupervisionRejected("specification_too_large")
    source_error = _source_policy_error(shown_code, entry_point)
    if source_error:
        raise ExecutionSupervisionRejected("source_policy_error", source_error)
    call_error = _call_policy_error(call_expression, entry_point)
    if call_error:
        raise ExecutionSupervisionRejected("call_policy_error", call_error)
    shown_input = {
        "specification": specification,
        "shown_code": shown_code,
        "entry_point": entry_point,
        "call_expression": call_expression,
    }
    schema = {
        "trace": [{"line": "integer", "source": "shown source line"}],
        "actual": {"type": "Python type name", "literal": "safe Python literal"},
        "intended": {"type": "Python type name", "literal": "safe Python literal"},
        "differs": "boolean",
    }
    return (
        "Predict the execution of the supplied Python code for the supplied call. "
        "Use only the inputs in SHOWN_INPUT. Return exactly one compact JSON object "
        "matching OUTPUT_SCHEMA, with no Markdown or commentary.\n"
        f"SHOWN_INPUT={json.dumps(shown_input, ensure_ascii=False, separators=(',', ':'))}\n"
        f"OUTPUT_SCHEMA={json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_output_prediction_prompt(
    *, specification: str, shown_code: str, entry_point: str,
    call_expression: str,
) -> str:
    """Build the focused call-to-oracle prompt used by the first A/B pilot.

    Like :func:`build_execution_supervision_prompt`, the signature is closed:
    reference code, expected values, patches, and gold tests cannot be passed
    to the renderer.  The shown function may be defective.  The requested
    completion has the same one-assertion shape as canonical Oneiros SFT, so
    the controlled pilot can hold completion bytes and loss-token mass exactly
    constant while changing only the prompt task.
    """
    if not isinstance(specification, str) or not specification.strip():
        raise ExecutionSupervisionRejected("empty_specification")
    if len(specification.encode("utf-8")) > 32_000:
        raise ExecutionSupervisionRejected("specification_too_large")
    source_error = _source_policy_error(shown_code, entry_point)
    if source_error:
        raise ExecutionSupervisionRejected("source_policy_error", source_error)
    call_error = _call_policy_error(call_expression, entry_point)
    if call_error:
        raise ExecutionSupervisionRejected("call_policy_error", call_error)
    return f"""### ASSERTION EXECUTION-SUPERVISION TASK

You are given a Python function, its intended public behaviour, and one
literal-input call. The function shown may be defective. Write the single
Python `assert` statement that states what the call must evaluate to for a
correct implementation.

Return only that assert statement. Do not return prose, Markdown, a test
function, imports, a repair, or additional assertions.

### Intended behaviour

{specification.strip()}

### Function (may be defective)

{shown_code.strip()}

### Call

{call_expression.strip()}

### Assertion
"""


def build_output_prediction_completion(
    *, call_expression: str, shown_evidence: ExecutionEvidence,
    intended_evidence: ExecutionEvidence,
) -> tuple[str, dict[str, Any]]:
    """Return an execution-verified assertion plus non-model-visible evidence.

    Both outcomes must already have survived independent determinism runs.
    The intended literal is never inserted into the prompt; it appears only in
    the completion and in the private provenance record.
    """
    if not isinstance(shown_evidence, ExecutionEvidence) or not isinstance(
        intended_evidence, ExecutionEvidence
    ):
        raise ExecutionSupervisionRejected("invalid_execution_evidence")
    # Validate the call independently of any prompt construction.  The entry
    # point is the direct callee named by the already-policy-checked call.
    try:
        call_tree = ast.parse(call_expression, mode="eval")
    except SyntaxError as exc:
        raise ExecutionSupervisionRejected("call_policy_error", exc.msg) from exc
    if not isinstance(call_tree.body, ast.Call) or not isinstance(
        call_tree.body.func, ast.Name
    ):
        raise ExecutionSupervisionRejected("call_policy_error")
    entry_point = call_tree.body.func.id
    call_error = _call_policy_error(call_expression, entry_point)
    if call_error:
        raise ExecutionSupervisionRejected("call_policy_error", call_error)
    if not _validate_typed_literal_payload(dict(shown_evidence.actual)) or not (
        _validate_typed_literal_payload(dict(intended_evidence.actual))
    ):
        raise ExecutionSupervisionRejected("invalid_execution_evidence")
    intended_literal = str(intended_evidence.actual["literal"])
    completion = f"assert {call_expression.strip()} == {intended_literal}"
    evidence = {
        "actual": dict(shown_evidence.actual),
        "intended": dict(intended_evidence.actual),
        "differs": dict(shown_evidence.actual) != dict(intended_evidence.actual),
        "shown_determinism_runs": shown_evidence.determinism_runs,
        "intended_determinism_runs": intended_evidence.determinism_runs,
    }
    return completion, evidence


def format_output_prediction_chat_prompt(tokenizer, user_prompt: str) -> str:
    """Render exactly the user-only chat contract used by the TAP diagnostic."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_execution_supervision_completion(
    evidence: ExecutionEvidence, *, intended_output: Any,
) -> str:
    """Build the fixed completion; ``differs`` is computed, never supplied."""
    if not isinstance(evidence, ExecutionEvidence):
        raise ExecutionSupervisionRejected("invalid_execution_evidence")
    trace = [dict(event) for event in evidence.trace]
    if (
        not _validate_typed_literal_payload(dict(evidence.actual))
        or not isinstance(evidence.determinism_runs, int)
        or not 2 <= evidence.determinism_runs <= MAX_DETERMINISM_RUNS
        or len(trace) > MAX_TRACE_EVENT_LIMIT
    ):
        raise ExecutionSupervisionRejected("invalid_execution_evidence")
    # The source itself is unavailable at this boundary, so enforce the parts
    # of the trace schema that do not require matching back to source lines.
    if any(
        not isinstance(event, dict)
        or set(event) != {"line", "source"}
        or not isinstance(event["line"], int)
        or isinstance(event["line"], bool)
        or event["line"] < 1
        or not isinstance(event["source"], str)
        for event in trace
    ):
        raise ExecutionSupervisionRejected("invalid_execution_evidence")
    transitions = {
        (int(first["line"]), int(second["line"]))
        for first, second in zip(trace, trace[1:])
    }
    serialized_trace = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    if (
        len(transitions) > MAX_DISTINCT_TRACE_TRANSITIONS
        or len(serialized_trace.encode("utf-8")) > MAX_TRACE_PAYLOAD_BYTES
    ):
        raise ExecutionSupervisionRejected("invalid_execution_evidence")
    try:
        intended = _typed_literal(intended_output)
    except _UnsupportedValue as exc:
        raise ExecutionSupervisionRejected("unsupported_intended_value", str(exc)) from exc
    payload = {
        "trace": trace,
        "actual": dict(evidence.actual),
        "intended": intended,
        # Type-sensitive comparison is intentional: bool/int or int/float
        # mismatches are useful execution-supervision signals.
        "differs": dict(evidence.actual) != intended,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_execution_supervision_example(
    *,
    specification: str,
    shown_code: str,
    entry_point: str,
    call_expression: str,
    intended_output: Any,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    trace_event_limit: int = DEFAULT_TRACE_EVENT_LIMIT,
) -> dict[str, str]:
    """Return one prompt/completion pair after every execution gate passes."""
    prompt = build_execution_supervision_prompt(
        specification=specification,
        shown_code=shown_code,
        entry_point=entry_point,
        call_expression=call_expression,
    )
    evidence = collect_execution_evidence(
        shown_code=shown_code,
        entry_point=entry_point,
        call_expression=call_expression,
        timeout_seconds=timeout_seconds,
        trace_event_limit=trace_event_limit,
    )
    completion = build_execution_supervision_completion(
        evidence, intended_output=intended_output
    )
    return {"prompt": prompt, "completion": completion}


if __name__ == "__main__" and "--worker" in sys.argv:
    raise SystemExit(_worker_main())
