"""Actual-Atheris differential fuzzing harness for function-level targets.

This is the real ``atheris`` package driving libFuzzer, not the in-repo
simulated coverage fuzzer in :mod:`baseline.coverage_fuzzer`.  The two must
never be reported under the same name.

Fairness contract
-----------------
Atheris finds crashes.  The Oneiros task is to distinguish a buggy
implementation from a correct one, and most of these defects return a wrong
value rather than raising.  Scoring Atheris only on crashes would understate it
badly, so this harness runs a DIFFERENTIAL oracle: each fuzzed input is applied
to both the buggy and the reference implementation, and a divergence counts as
a kill.

That deliberately gives Atheris information Oneiros never receives - Oneiros
never sees the reference implementation.  The comparison is therefore generous
to Atheris by construction, and any Oneiros advantage is not an artifact of
withholding the oracle.

Both outcomes are recorded separately:
  * ``crash_kill``    - buggy raises where reference does not;
  * ``semantic_kill`` - both return, but the values differ.

Run under Python 3.11: atheris 2.3.0 has no wheel for 3.12 and its native
extension does not compile there.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


HARNESS_VERSION = "oneiros_atheris_differential_harness_v1"

_SUPPORTED_KINDS = ("int", "float", "str", "bool", "list_int", "list_float", "list_str", "any")


def _annotation_kind(annotation: ast.expr | None) -> str:
    """Map a parameter annotation to a generator kind."""
    if annotation is None:
        return ""
    text = ast.unparse(annotation) if hasattr(ast, "unparse") else ""
    text = text.replace(" ", "")
    mapping = {
        "int": "int", "float": "float", "str": "str", "bool": "bool",
        "List[int]": "list_int", "list[int]": "list_int",
        "List[float]": "list_float", "list[float]": "list_float",
        "List[str]": "list_str", "list[str]": "list_str",
    }
    return mapping.get(text, "")


def _kind_of_value(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)) and value:
        inner = {_kind_of_value(item) for item in value}
        if inner == {"int"}:
            return "list_int"
        if inner <= {"int", "float"}:
            return "list_float"
        if inner == {"str"}:
            return "list_str"
    return "any"


def infer_parameter_kinds(
    function_code: str, entry_point: str, example_assertions: list[str],
) -> list[str]:
    """Infer one generator kind per parameter.

    Annotations are used first.  Where they are absent - which is the norm for
    MBPP - the concrete arguments in the benchmark's own reference assertions
    are parsed and their runtime types used instead.  This is what makes the
    input strategy structured rather than raw bytes, and it is the strongest
    reasonable setup available offline.
    """
    tree = ast.parse(function_code)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            target = node
            break
    if target is None:
        raise ValueError(f"entry point {entry_point!r} not found")
    parameters = list(target.args.posonlyargs) + list(target.args.args)
    kinds = [_annotation_kind(parameter.annotation) for parameter in parameters]

    if not all(kinds):
        observed: dict[int, list[str]] = {}
        for assertion in example_assertions:
            try:
                parsed = ast.parse(assertion)
            except SyntaxError:
                continue
            for node in ast.walk(parsed):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name) or node.func.id != entry_point:
                    continue
                for index, argument in enumerate(node.args):
                    try:
                        value = ast.literal_eval(argument)
                    except (ValueError, SyntaxError):
                        continue
                    observed.setdefault(index, []).append(_kind_of_value(value))
        for index in range(len(kinds)):
            if kinds[index]:
                continue
            seen = observed.get(index) or []
            kinds[index] = max(set(seen), key=seen.count) if seen else "any"
    return [kind or "any" for kind in kinds]


def _consume(provider: Any, kind: str) -> Any:
    if kind == "int":
        return provider.ConsumeIntInRange(-1000, 1000)
    if kind == "float":
        value = provider.ConsumeFloatInRange(-1000.0, 1000.0)
        return 0.0 if math.isnan(value) else value
    if kind == "bool":
        return provider.ConsumeBool()
    if kind == "str":
        return provider.ConsumeUnicodeNoSurrogates(24)
    if kind == "list_int":
        return [provider.ConsumeIntInRange(-100, 100) for _ in range(provider.ConsumeIntInRange(0, 8))]
    if kind == "list_float":
        return [provider.ConsumeFloatInRange(-100.0, 100.0) for _ in range(provider.ConsumeIntInRange(0, 8))]
    if kind == "list_str":
        return [provider.ConsumeUnicodeNoSurrogates(8) for _ in range(provider.ConsumeIntInRange(0, 5))]
    return provider.ConsumeIntInRange(-100, 100)


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            if math.isnan(float(left)) and math.isnan(float(right)):
                return True
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return left == right
    try:
        return bool(left == right)
    except Exception:
        return False


def fuzz_one_target(
    task: dict[str, Any], max_runs: int, time_budget_seconds: float,
    persist: Any = None, unit_timeout_seconds: float = 5.0, seed: int = 42,
) -> dict[str, Any]:
    """Fuzz one buggy/reference pair and report the first distinguishing input.

    ``persist`` is called with the current verdict whenever it changes.
    libFuzzer terminates the process itself once ``-runs`` is exhausted, and
    that exit path runs no Python teardown - no ``finally``, no ``atexit``.
    Anything written only after ``atheris.Fuzz()`` returns is therefore lost,
    so the verdict is checkpointed as soon as it is known.
    """
    import atheris

    entry_point = task["entry_point"]
    buggy_namespace: dict[str, Any] = {}
    reference_namespace: dict[str, Any] = {}
    exec(task["buggy_code"], buggy_namespace)          # noqa: S102 - fuzz target
    exec(task["reference_code"], reference_namespace)  # noqa: S102 - fuzz oracle
    buggy = buggy_namespace[entry_point]
    reference = reference_namespace[entry_point]

    kinds = infer_parameter_kinds(
        task["reference_code"], entry_point, task.get("example_assertions") or [],
    )
    # "incomplete" is the honest starting value. libFuzzer can terminate this
    # process itself - on its per-unit timeout, or when -runs is exhausted -
    # and the checkpoint on disk must never claim a verdict that was not
    # reached. It becomes "survived" only when Fuzz() returns normally.
    state: dict[str, Any] = {
        "outcome": "incomplete", "runs": 0, "witness": None,
        "kill_kind": None, "started": time.time(),
    }

    def snapshot() -> dict[str, Any]:
        return {
            "task_id": task.get("task_id"),
            "entry_point": entry_point,
            "parameter_kinds": kinds,
            "outcome": state["outcome"],
            "kill_kind": state["kill_kind"],
            "runs": state["runs"],
            "elapsed_seconds": round(time.time() - state["started"], 3),
            "witness": state["witness"],
            "harness_error": state.get("harness_error"),
        }

    def checkpoint() -> None:
        if persist is not None:
            persist(snapshot())

    checkpoint()

    def target(data: bytes) -> None:
        # Never raise out of the fuzz target. libFuzzer treats an escaping
        # exception as a crash, writes an artifact, and terminates the process
        # before results can be recorded - which is how the first version of
        # this harness lost every result it produced. Once the verdict is
        # known, remaining iterations become cheap no-ops instead.
        if state["outcome"] not in ("incomplete", "survived"):
            return
        if time.time() - state["started"] > time_budget_seconds:
            state["outcome"] = "time_budget_exhausted"
            checkpoint()
            return
        state["runs"] += 1
        provider = atheris.FuzzedDataProvider(data)
        try:
            arguments = [_consume(provider, kind) for kind in kinds]
        except Exception:
            return
        try:
            expected = reference(*arguments)
        except Exception:
            # The reference itself rejects this input, so it is out of contract
            # and any buggy-side behaviour on it proves nothing.
            return
        try:
            actual = buggy(*arguments)
        except Exception as exc:
            state["outcome"] = "killed"
            state["kill_kind"] = "crash_kill"
            state["witness"] = {
                "arguments": repr(arguments)[:400],
                "reference_result": repr(expected)[:200],
                "buggy_exception": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
            checkpoint()
            return
        if not _equal(actual, expected):
            state["outcome"] = "killed"
            state["kill_kind"] = "semantic_kill"
            state["witness"] = {
                "arguments": repr(arguments)[:400],
                "reference_result": repr(expected)[:200],
                "buggy_result": repr(actual)[:200],
            }
            checkpoint()
            return

    # -timeout bounds a SINGLE input. Without it a fuzzed argument that sends
    # the target into a near-infinite loop hangs the whole process: the wall
    # clock check below only runs between inputs, never inside one.
    atheris.Setup(
        [
            sys.argv[0], f"-runs={max_runs}", "-max_len=256",
            f"-seed={seed}",
            f"-timeout={max(1, int(unit_timeout_seconds))}",
            "-print_final_stats=0",
        ],
        target, enable_python_coverage=True,
    )
    try:
        atheris.Fuzz()
        if state["outcome"] == "incomplete":
            state["outcome"] = "survived"
    except SystemExit:
        if state["outcome"] == "incomplete":
            state["outcome"] = "survived"
    except Exception as exc:  # pragma: no cover - libFuzzer teardown paths
        state.setdefault("harness_error", f"{type(exc).__name__}: {str(exc)[:200]}")

    checkpoint()
    return snapshot()


def finalize_result(
    output_path: Path, process_returncode: int, wall_limit: str,
    runner_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Turn the runner's process status into a durable terminal verdict.

    libFuzzer may call ``exit(0)`` when ``-runs`` is exhausted, bypassing all
    Python teardown.  The fuzz callback checkpoints an ``incomplete`` result
    before entering libFuzzer; this separate process converts that checkpoint
    to ``survived`` only after the shell has observed a clean return code.
    Outer ``timeout`` terminations and harness failures remain distinct.
    """
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = payload.get("results") or []
    for row in rows:
        row["process_returncode"] = process_returncode
        if row.get("outcome") != "incomplete":
            continue
        if process_returncode == 0:
            row["outcome"] = "survived"
            row["runs"] = max(
                int(row.get("runs") or 0), int(payload.get("max_runs") or 0),
            )
        elif process_returncode == 70:
            row["outcome"] = "unit_timeout"
            row["harness_error"] = (
                "libFuzzer terminated a single input at the configured "
                "per-unit timeout"
            )
        elif process_returncode in {124, 137}:
            row["outcome"] = "outer_wall_timeout"
            row["harness_error"] = (
                f"runner exceeded outer wall limit {wall_limit} "
                f"(return code {process_returncode})"
            )
        else:
            row["outcome"] = "harness_failure"
            row["harness_error"] = (
                f"atheris process exited with return code {process_returncode}"
            )
        if runner_elapsed_seconds is not None:
            row["elapsed_seconds"] = max(
                float(row.get("elapsed_seconds") or 0.0), runner_elapsed_seconds,
            )
    payload["runner_finalized"] = True
    payload["runner_process_returncode"] = process_returncode
    payload["runner_wall_limit"] = wall_limit
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", help="JSON file of fuzz tasks")
    parser.add_argument("--output", help="JSON results file")
    parser.add_argument("--max-runs", type=int, default=20000)
    parser.add_argument("--time-budget", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unit-timeout", type=float, default=5.0,
        help="libFuzzer per-input timeout; bounds one pathological argument.",
    )
    parser.add_argument(
        "--task-index", type=int,
        help=(
            "Index of the single task to fuzz. atheris.Setup() may be called "
            "only once per process, so each target needs its own process; the "
            "driver loops over indexes."
        ),
    )
    parser.add_argument(
        "--finalize-output",
        help="Finalize a checkpoint after the separate fuzz process exits.",
    )
    parser.add_argument("--process-returncode", type=int)
    parser.add_argument("--wall-limit", default="unknown")
    parser.add_argument("--runner-elapsed-seconds", type=float)
    arguments = parser.parse_args()

    if arguments.finalize_output:
        if arguments.process_returncode is None:
            parser.error("--finalize-output requires --process-returncode")
        finalize_result(
            Path(arguments.finalize_output),
            arguments.process_returncode,
            arguments.wall_limit,
            arguments.runner_elapsed_seconds,
        )
        return 0
    if not arguments.tasks or not arguments.output or arguments.task_index is None:
        parser.error("fuzzing requires --tasks, --output, and --task-index")

    with open(arguments.tasks, encoding="utf-8") as handle:
        tasks = json.load(handle)
    if not 0 <= arguments.task_index < len(tasks):
        raise SystemExit(f"task index {arguments.task_index} out of range")
    task = tasks[arguments.task_index]

    def write(result: dict[str, Any]) -> None:
        with open(arguments.output, "w", encoding="utf-8") as handle:
            json.dump({
                "harness_version": HARNESS_VERSION,
                "python_version": sys.version.split()[0],
                "max_runs": arguments.max_runs,
                "time_budget_seconds": arguments.time_budget,
                "seed": arguments.seed,
                "results": [result],
            }, handle, indent=2)

    try:
        results = [fuzz_one_target(
            task, arguments.max_runs, arguments.time_budget, persist=write,
            unit_timeout_seconds=arguments.unit_timeout,
            seed=arguments.seed,
        )]
    except Exception as exc:
        results = [{
            "task_id": task.get("task_id"),
            "entry_point": task.get("entry_point"),
            "outcome": "unsupported_target",
            "kill_kind": None,
            "runs": 0,
            "harness_error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "traceback": traceback.format_exc()[-800:],
        }]
        write(results[0])
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump({
            "harness_version": HARNESS_VERSION,
            "python_version": sys.version.split()[0],
            "max_runs": arguments.max_runs,
            "time_budget_seconds": arguments.time_budget,
            "seed": arguments.seed,
            "results": results,
        }, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
