"""How much of the mbpp ceiling is wrong VALUES rather than wrong INPUTS?

The dominant mbpp failure is wrong_expected_value: the candidate asserts
behaviour the CORRECT reference does not exhibit, so it fails before it can
ever distinguish the mutant. That has two very different causes, and the
distinction decides what to build next:

* the model probed an input where reference and mutant agree, and also got the
  value wrong. Fixing value prediction changes nothing - the test could never
  have killed anything.
* the model probed an input where reference and mutant DIFFER, and only the
  expected value was wrong. That candidate is one correct value away from a
  kill, and what the specification gap costs is recoverable.

This measures which, by re-running each failed candidate's own call against
the reference and against the mutant and comparing their behaviour. It never
consults the model and never regenerates.

The number it produces is an UPPER BOUND and must be reported as one: it
assumes perfect value prediction on inputs the model already chose, which no
intervention will reach. Its use is the other direction - if the bound sits
close to the current kill rate, value prediction is not the lever at all and
the model is probing the wrong inputs.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.safe_execution import execute_code
from metrics.research_evaluation import classify_candidate_failure


def call_expression(candidate: str, entry_point: str) -> str | None:
    """The candidate's own call to the function under test, as source.

    Taking the model's chosen input rather than inventing one is the whole
    point: the question is what the model was already probing, not what a
    better prober would probe.
    """
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.id if isinstance(function, ast.Name)
            else function.attr if isinstance(function, ast.Attribute)
            else None
        )
        if name == entry_point:
            try:
                return ast.unparse(node)
            except Exception:
                return None
    return None


def _behaviour(code: str, call: str, timeout: float) -> tuple[str, str]:
    """(status, repr of the value) of one call against one implementation.

    The worker returns whatever the candidate leaves in `result`, not stdout,
    so the probe assigns rather than prints. repr() is used so that 1 and True
    and "1" are three different behaviours rather than one.
    """
    probe = "result = repr(" + call + ")"
    ok, result, error = execute_code(code, probe, timeout)
    if not ok:
        return ("error", str(error).split(":", 1)[0].strip())
    return ("ok", str(result if result is not None else "").strip())


def analyse(artifact: Path, corpus_dir: Path, benchmark: str,
            timeout: float, limit: int | None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    distinguishing = agreeing = unparsed = errored = 0
    recoverable_functions = 0
    considered_functions = 0

    for result in payload.get("function_results") or []:
        if str(result.get("dataset_name") or "") != benchmark:
            continue
        if result.get("killed"):
            continue
        record = by_id.get(str(result.get("record_id")))
        if record is None:
            continue
        reference = str(record.get("reference_code") or "")
        mutant = str(record.get("code_under_test") or "")
        entry = str(record.get("entry_point") or "")
        support = str(record.get("support_context") or "")
        if not (reference and mutant and entry):
            continue
        considered_functions += 1

        family = str(result.get("bug_family") or "unknown")
        recoverable = False
        for outcome in result.get("candidate_outcomes") or []:
            if classify_candidate_failure(outcome, family) != "wrong_expected_value":
                continue
            if limit is not None and (distinguishing + agreeing + errored) >= limit:
                break
            call = call_expression(str(outcome.get("code") or ""), entry)
            if call is None:
                unparsed += 1
                continue
            reference_behaviour = _behaviour(support + "\n" + reference, call, timeout)
            mutant_behaviour = _behaviour(support + "\n" + mutant, call, timeout)
            if reference_behaviour[0] == "error" and mutant_behaviour[0] == "error" \
                    and reference_behaviour == mutant_behaviour:
                errored += 1
                continue
            if reference_behaviour != mutant_behaviour:
                distinguishing += 1
                recoverable = True
            else:
                agreeing += 1
        if recoverable:
            recoverable_functions += 1

    total = distinguishing + agreeing + errored
    rows = [r for r in payload.get("function_results") or []
            if str(r.get("dataset_name") or "") == benchmark]
    functions = len(rows)
    killed = sum(1 for r in rows if r.get("killed"))

    return {
        "schema_version": "oneiros_value_prediction_headroom_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "benchmark": benchmark,
        "functions": functions,
        "functions_killed": killed,
        "kill_rate": round(killed / functions, 6) if functions else None,
        "unkilled_functions_examined": considered_functions,
        "wrong_expected_value_candidates_examined": total,
        "candidates_probing_a_distinguishing_input": distinguishing,
        "candidates_probing_an_agreeing_input": agreeing,
        "candidates_erroring_on_both": errored,
        "candidates_without_a_parseable_call": unparsed,
        "distinguishing_share": round(distinguishing / total, 6) if total else None,
        "functions_recoverable_by_correct_values": recoverable_functions,
        "upper_bound_kill_rate_with_perfect_values": round(
            (killed + recoverable_functions) / functions, 6) if functions else None,
        "interpretation": (
            "an upper bound, not a forecast: it assumes every wrong expected "
            "value on an already-distinguishing input becomes correct, which "
            "no intervention will achieve. If this bound is close to the "
            "current kill rate, value prediction is not the lever and the "
            "model is probing the wrong inputs."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--benchmark", default="mbpp")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap candidates examined, for a fast smoke run")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = analyse(arguments.artifact, arguments.corpus,
                     arguments.benchmark, arguments.timeout, arguments.limit)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
