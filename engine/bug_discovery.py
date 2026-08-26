"""Evidence-backed bug discovery for user-supplied Python functions.

The module combines static checks, type-guided boundary probes, lightweight
coverage feedback, explicit contract tests, optional differential testing, and
optional LLM test proposals.  It never labels an LLM assertion as a confirmed
bug without an independent oracle (a user contract or reference function).
"""
from __future__ import annotations

import ast
import builtins as python_builtins
import itertools
import multiprocessing as mp
import os
import re
import sys
import time
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List

from harness.candidate_policy import validate_function_assertion


SAFE_IMPORTS = {
    "__future__", "base64", "bisect", "collections", "copy", "dataclasses",
    "datetime", "decimal", "fractions", "functools", "heapq", "itertools",
    "json", "math", "operator", "random", "re", "statistics", "string", "typing",
}
FORBIDDEN_IMPORTS = {
    "builtins", "ctypes", "multiprocessing", "os", "pathlib", "pickle",
    "requests", "shutil", "socket", "subprocess", "sys", "threading",
}
FORBIDDEN_CALLS = {
    "breakpoint", "compile", "eval", "exec", "exit", "getattr", "globals",
    "input", "locals", "open", "setattr", "__import__",
}
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "iter", "len", "list", "map", "max",
    "min", "next", "range", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError", "AssertionError", "ZeroDivisionError", "OverflowError",
}


@dataclass
class ExecutionOutcome:
    success: bool
    value_type: str = ""
    value_repr: str = ""
    error_type: str = ""
    error_message: str = ""
    elapsed_ms: float = 0.0
    lines: List[int] = field(default_factory=list)
    branches: List[tuple[int, int]] = field(default_factory=list)


@dataclass
class BugFinding:
    category: str
    severity: str
    confidence: str
    status: str
    message: str
    reproduction: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class BugAnalysisReport:
    entry_point: str
    safe_to_execute: bool
    findings: List[BugFinding]
    suggested_tests: List[str]
    executed_probes: int
    coverage_lines: int
    coverage_branches: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_source(source: str) -> ast.Module:
    return ast.parse(source)


def _entry_function(source: str, entry_point: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = _parse_source(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    raise ValueError(f"Entry point '{entry_point}' is not a top-level function.")


def execution_safety_violations(source: str) -> List[str]:
    """Reject code that should not be executed by the local analysis sandbox."""
    try:
        tree = _parse_source(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    violations: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            for name in names:
                if name in FORBIDDEN_IMPORTS or name not in SAFE_IMPORTS:
                    violations.append(f"import '{name}' is not allowed in the execution sandbox")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            violations.append(f"call to '{node.func.id}' is not allowed in the execution sandbox")
        if isinstance(node, ast.Attribute):
            base_name = node.value.id if isinstance(node.value, ast.Name) else ""
            if (
                base_name in FORBIDDEN_IMPORTS
                or node.attr.startswith("_")
                or node.attr in FORBIDDEN_IMPORTS
                or node.attr in FORBIDDEN_CALLS
            ):
                violations.append(
                    f"attribute access '{node.attr}' is not allowed in the execution sandbox"
                )
    return sorted(set(violations))


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level or root not in SAFE_IMPORTS:
        raise ImportError(f"Import of '{name}' is disabled in the analysis sandbox")
    return __import__(name, globals, locals, fromlist, level)


def _sandbox_worker(source: str, payload: str, mode: str, output: mp.Queue) -> None:
    """Run one probe in a child process with restricted builtins and tracing."""
    lines: set[int] = set()
    branches: set[tuple[int, int]] = set()
    previous_line: int | None = None

    def trace(frame, event, arg):
        nonlocal previous_line
        if frame.f_code.co_filename == "<oneiros-target>" and event == "line":
            line = frame.f_lineno
            lines.add(line)
            if previous_line is not None:
                branches.add((previous_line, line))
            previous_line = line
        return trace

    builtins = {name: getattr(python_builtins, name) for name in SAFE_BUILTINS}
    builtins["__import__"] = _safe_import
    namespace = {"__builtins__": builtins, "__name__": "__oneiros_target__"}
    started = time.perf_counter()
    try:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except (ImportError, OSError, ValueError):
            pass
        with tempfile.TemporaryDirectory(prefix="oneiros-analysis-") as working_dir:
            os.chdir(working_dir)
            exec(compile(source, "<oneiros-target>", "exec"), namespace)
            sys.settrace(trace)
            if mode == "call":
                value = eval(compile(payload, "<oneiros-probe>", "eval"), namespace)
            elif mode == "assert":
                exec(compile(payload, "<oneiros-contract>", "exec"), namespace)
                value = None
            else:
                raise ValueError(f"Unknown sandbox mode: {mode}")
        output.put(ExecutionOutcome(
            success=True,
            value_type=type(value).__name__,
            value_repr=repr(value)[:1000],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            lines=sorted(lines),
            branches=sorted(branches),
        ))
    except BaseException as exc:
        output.put(ExecutionOutcome(
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc)[:1000],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            lines=sorted(lines),
            branches=sorted(branches),
        ))
    finally:
        sys.settrace(None)


def run_in_sandbox(source: str, payload: str, mode: str, timeout_seconds: float = 1.0) -> ExecutionOutcome:
    """Execute a single call/assertion in a killable subprocess."""
    violations = execution_safety_violations(source)
    if violations:
        return ExecutionOutcome(
            success=False,
            error_type="SandboxPolicyError",
            error_message="; ".join(violations),
        )
    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(target=_sandbox_worker, args=(source, payload, mode, output))
    started = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        return ExecutionOutcome(
            success=False,
            error_type="Timeout",
            error_message=f"Probe exceeded {timeout_seconds:.1f}s sandbox limit",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    try:
        return output.get_nowait()
    except Exception:
        return ExecutionOutcome(
            success=False,
            error_type="SandboxFailure",
            error_message="Sandbox exited without returning an outcome",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def _annotation_kind(annotation: ast.expr | None) -> str:
    if annotation is None:
        return "any"
    text = ast.unparse(annotation).lower()
    if "list" in text or "sequence" in text or "iterable" in text:
        return "list"
    if "dict" in text or "mapping" in text:
        return "dict"
    if "tuple" in text:
        return "tuple"
    if "str" in text:
        return "str"
    if "float" in text:
        return "float"
    if "bool" in text:
        return "bool"
    if "int" in text:
        return "int"
    return "any"


def _values_for_kind(kind: str) -> List[Any]:
    values = {
        "int": [0, 1, -1, 2, -2, 10, -10],
        "float": [0.0, 1.0, -1.0, 0.5, 1e-9, 1e9],
        "str": ["", "a", "abc", " ", "Aa0!"],
        "list": [[], [0], [1], [-1, 0, 1], [1, 1, 2]],
        "dict": [{}, {"a": 1}, {"": 0, "a": -1}],
        "tuple": [(), (0,), (1, 2)],
        "bool": [False, True],
        "any": [None, 0, 1, "", [], {}],
    }
    return values[kind]


def generate_boundary_calls(source: str, entry_point: str, limit: int = 48) -> List[str]:
    """Build deterministic, literal-only calls from the signature's type hints."""
    function = _entry_function(source, entry_point)
    arguments = [arg for arg in function.args.posonlyargs + function.args.args if arg.arg not in {"self", "cls"}]
    required_count = len(arguments) - len(function.args.defaults)
    values = [_values_for_kind(_annotation_kind(arg.annotation)) for arg in arguments]
    if not arguments:
        return [f"{entry_point}()"]
    calls: List[str] = []
    for count in range(required_count, len(arguments) + 1):
        for combo in itertools.islice(itertools.product(*values[:count]), limit):
            calls.append(f"{entry_point}({', '.join(repr(value) for value in combo)})")
            if len(calls) >= limit:
                return calls
    return calls


def _static_findings(source: str, entry_point: str) -> List[BugFinding]:
    findings: List[BugFinding] = []
    try:
        function = _entry_function(source, entry_point)
    except (SyntaxError, ValueError) as exc:
        return [BugFinding("syntax", "high", "high", "confirmed", str(exc))]
    for default in function.args.defaults + function.args.kw_defaults:
        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
            findings.append(BugFinding(
                "mutable_default_argument", "medium", "high", "potential",
                "Mutable default argument retains state between calls.",
                evidence={"line": default.lineno},
            ))
    for node in ast.walk(function):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append(BugFinding(
                "bare_except", "medium", "high", "potential",
                "Bare except catches system and programming errors, hiding failures.",
                evidence={"line": node.lineno},
            ))
        if isinstance(node, ast.ExceptHandler) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            findings.append(BugFinding(
                "swallowed_exception", "medium", "high", "potential",
                "Exception handler silently discards an error.",
                evidence={"line": node.lineno},
            ))
        if isinstance(node, ast.Compare) and any(isinstance(item, ast.Constant) and item.value is None for item in node.comparators):
            if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                findings.append(BugFinding(
                    "none_comparison", "low", "medium", "potential",
                    "Use `is None` / `is not None` instead of equality for None.",
                    evidence={"line": node.lineno},
                ))
    return findings


def _safe_contract_assertions(tests: Iterable[str], entry_point: str) -> List[str]:
    """Accept only standalone assertions that call the submitted entry point."""
    safe: List[str] = []
    for text in tests:
        policy = validate_function_assertion(text, entry_point)
        if policy.valid:
            safe.append(ast.unparse(ast.parse(text).body[0]))
    return list(dict.fromkeys(safe))


def _same_outcome(left: ExecutionOutcome, right: ExecutionOutcome) -> bool:
    if left.success != right.success:
        return False
    if not left.success:
        return left.error_type == right.error_type
    return (left.value_type, left.value_repr) == (right.value_type, right.value_repr)


def analyze_function(
    source: str,
    entry_point: str,
    *,
    contract_tests: Iterable[str] = (),
    reference_source: str | None = None,
    suggested_tests: Iterable[str] = (),
    max_probes: int = 48,
    timeout_seconds: float = 1.0,
) -> BugAnalysisReport:
    """Analyze one user function and return only reproducible findings.

    Contract failures and reference differences are confirmed. Unspecified
    exceptions are surfaced as low-confidence observations, not bug claims.
    """
    started = time.perf_counter()
    findings = _static_findings(source, entry_point)
    violations = execution_safety_violations(source)
    if reference_source:
        violations.extend(f"reference: {item}" for item in execution_safety_violations(reference_source))
    if violations:
        findings.extend(BugFinding("sandbox_policy", "high", "high", "confirmed", violation) for violation in sorted(set(violations)))
        return BugAnalysisReport(
            entry_point, False, findings, [], 0, 0, 0, time.perf_counter() - started
        )

    coverage_lines: set[int] = set()
    coverage_branches: set[tuple[int, int]] = set()
    executed = 0
    for assertion in _safe_contract_assertions(contract_tests, entry_point):
        outcome = run_in_sandbox(source, assertion, "assert", timeout_seconds)
        executed += 1
        coverage_lines.update(outcome.lines)
        coverage_branches.update(outcome.branches)
        if not outcome.success:
            findings.append(BugFinding(
                "contract_violation", "high", "high", "confirmed",
                "The submitted function violates a user-supplied contract assertion.",
                assertion,
                {"error_type": outcome.error_type, "error_message": outcome.error_message},
            ))

    seen_calls = set()
    for call in generate_boundary_calls(source, entry_point, max_probes):
        if call in seen_calls:
            continue
        seen_calls.add(call)
        target = run_in_sandbox(source, call, "call", timeout_seconds)
        executed += 1
        coverage_lines.update(target.lines)
        coverage_branches.update(target.branches)
        if reference_source:
            reference = run_in_sandbox(reference_source, call, "call", timeout_seconds)
            if not _same_outcome(target, reference):
                findings.append(BugFinding(
                    "differential_behavior", "high", "high", "confirmed",
                    "Submitted implementation differs from the supplied reference implementation.",
                    f"assert {call} == {reference.value_repr}" if reference.success else call,
                    {"target": asdict(target), "reference": asdict(reference)},
                ))
        elif not target.success:
            findings.append(BugFinding(
                "unexpected_exception", "low", "low", "observation",
                "Boundary probe raised an exception; provide a contract or reference to classify it as a bug.",
                call,
                {"error_type": target.error_type, "error_message": target.error_message},
            ))

    safe_suggestions = _safe_contract_assertions(suggested_tests, entry_point)
    return BugAnalysisReport(
        entry_point=entry_point,
        safe_to_execute=True,
        findings=findings,
        suggested_tests=safe_suggestions,
        executed_probes=executed,
        coverage_lines=len(coverage_lines),
        coverage_branches=len(coverage_branches),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )


def read_contract_tests(path: Path) -> List[str]:
    """Read standalone assertions from a Python test file or a JSON string list."""
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json
        value = json.loads(content)
        return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    return [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Assert)]


if __name__ == "__main__":
    buggy = """def reciprocal(value: int) -> float:\n    return 1 / value\n"""
    reference = """def reciprocal(value: int) -> float:\n    if value == 0:\n        return 0.0\n    return 1 / value\n"""
    import json
    print(json.dumps(analyze_function(buggy, "reciprocal", reference_source=reference).to_dict(), indent=2))
