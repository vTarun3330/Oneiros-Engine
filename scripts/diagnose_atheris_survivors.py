"""Why did each surviving target survive?

A survival rate is not a diagnosis. A mutant can survive because nothing could
ever kill it, because the fuzzer's input adapter cannot construct the shape of
input that would, or because the search simply did not find it in budget.
Those demand opposite responses - the first means the target should not be
counted, the second is a harness defect, only the third is a fair loss - and
the aggregate number cannot tell them apart.

For each survivor this asks two executable questions:

1. Is the mutant killable at all? The benchmark's own reference assertions are
   run against reference and mutant. If one distinguishes them, a killing
   input demonstrably exists.
2. Could the adapter have produced it? The inferred parameter kinds are
   compared against the types the killing call actually uses. `_consume` falls
   back to an int for any kind it does not recognise, so a tuple, dict or set
   parameter is fed an integer, the reference rejects it, and the run learns
   nothing.

Nothing here fuzzes or regenerates; it re-executes finished artifacts. The
gold assertions are used only as a DIAGNOSTIC oracle, never fed to a model.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.parallel_execution import progress_map
from harness.safe_execution import execute_code

#: What `_consume` in the harness can actually build.
REPRESENTABLE = {
    "int": (int,), "float": (float, int), "bool": (bool,), "str": (str,),
    "list_int": (list,), "list_float": (list,), "list_str": (list,),
}


def _argument_types(call_source: str) -> list[str] | None:
    """Runtime types of a call's literal arguments, or None if not literal."""
    try:
        node = ast.parse(call_source, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    types: list[str] = []
    for argument in node.args:
        try:
            value = ast.literal_eval(argument)
        except (ValueError, SyntaxError):
            return None
        types.append(type(value).__name__)
    return types


def _calls_in(assertion: str, entry_point: str) -> str | None:
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == entry_point:
            try:
                return ast.unparse(node)
            except Exception:
                return None
    return None


def _runs(code: str, statement: str, timeout: float) -> tuple[bool, str]:
    ok, result, error = execute_code(code, statement, timeout)
    return ok, str(error)


def diagnose_one(job: dict[str, Any]) -> dict[str, Any]:
    task = job["task"]
    kinds = job["parameter_kinds"]
    timeout = job["timeout"]
    entry = str(task.get("entry_point") or "")
    reference = str(task.get("reference_code") or "")
    mutant = str(task.get("buggy_code") or "")
    assertions = list(task.get("example_assertions") or [])

    killing_call = None
    for assertion in assertions:
        reference_ok, _ = _runs(reference, assertion, timeout)
        if not reference_ok:
            continue
        mutant_ok, _ = _runs(mutant, assertion, timeout)
        if not mutant_ok:
            killing_call = _calls_in(assertion, entry)
            if killing_call:
                break

    if killing_call is None:
        return {
            "task_id": task.get("task_id"),
            "verdict": "no_known_killing_input",
            "detail": (
                "no reference assertion distinguishes reference from mutant, "
                "so this may be an equivalent mutant or simply outside the "
                "assertions the benchmark ships"
            ),
        }

    types = _argument_types(killing_call)
    if types is None:
        return {"task_id": task.get("task_id"),
                "verdict": "killing_input_not_literal",
                "detail": "the killing call's arguments are not literals"}

    if len(types) != len(kinds):
        return {"task_id": task.get("task_id"), "verdict": "arity_mismatch",
                "detail": f"adapter inferred {len(kinds)} kinds for {len(types)} arguments"}

    unrepresentable = [
        f"{kind}!={actual}" for kind, actual in zip(kinds, types)
        if type(None) is not type and actual not in
        {t.__name__ for t in REPRESENTABLE.get(kind, ())}
    ]
    if unrepresentable:
        return {
            "task_id": task.get("task_id"),
            "verdict": "adapter_cannot_represent_the_killing_input",
            "detail": ", ".join(unrepresentable[:4]),
            "killing_call": killing_call[:120],
        }
    return {
        "task_id": task.get("task_id"),
        "verdict": "reachable_but_not_found",
        "detail": "the adapter could build this input; the search did not find it",
        "killing_call": killing_call[:120],
    }


def run(directory: Path, tasks_path: Path, sample: int,
        timeout: float) -> dict[str, Any]:
    payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    task_list = payload["tasks"] if isinstance(payload, dict) and "tasks" in payload else payload
    tasks = {str(task["task_id"]): task for task in task_list}

    survivors: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(directory / "*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in (data.get("results") or [data]):
            if row.get("outcome") != "survived":
                continue
            task = tasks.get(str(row.get("task_id")))
            if task is None:
                continue
            survivors.append({
                "task": task,
                "parameter_kinds": list(row.get("parameter_kinds") or []),
                "timeout": timeout,
            })

    # Deterministic sample: sorted by task id and taken in order, so the same
    # targets are examined every time rather than a fresh random draw that
    # could be re-rolled until it said something convenient.
    survivors.sort(key=lambda job: str(job["task"]["task_id"]))
    if sample and sample < len(survivors):
        stride = len(survivors) / sample
        survivors = [survivors[int(i * stride)] for i in range(sample)]

    findings = progress_map(diagnose_one, survivors, "survivor diagnosis", every=100)
    verdicts = Counter(f["verdict"] for f in findings)

    examples: dict[str, Any] = {}
    for finding in findings:
        examples.setdefault(finding["verdict"], finding)

    return {
        "schema_version": "oneiros_atheris_survivor_diagnosis_v1",
        "set": directory.name,
        "sealed_final_test_accessed": False,
        "survivors_examined": len(findings),
        "sampling": "deterministic even stride over task ids, not random",
        "verdicts": dict(verdicts.most_common()),
        "verdict_shares": {
            k: round(v / len(findings), 4) for k, v in verdicts.most_common()
        } if findings else {},
        "one_example_per_verdict": examples,
        "gold_assertion_use": (
            "reference assertions are executed here as a diagnostic oracle "
            "only. They are never placed in a model prompt and never enter "
            "training."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tasks", type=Path,
                        default=ROOT / "results" / "v4_2_atheris_tasks_val.json")
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = run(arguments.directory, arguments.tasks, arguments.sample,
                 arguments.timeout)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "one_example_per_verdict"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
