"""Are humaneval kills earned, or copied out of the prompt?

46% of ablation_dev humaneval records state, in the prompt, an input and the
correct output for it where the mutant produces something else. Asserting that
stated pair kills the mutant with no reasoning at all.

That a shortcut EXISTS does not mean it is taken. This checks whether the
candidates that actually killed are the stated pair, by comparing each killing
candidate against the examples parsed out of that record's own specification.

A candidate counts as copied when it calls the function with the same
arguments as a stated example AND asserts a value equal to that example's
stated output. Same input with a different asserted value is not copying, and
neither is a different input, so the count is a floor rather than a ceiling.
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
from scripts.audit_native_example_leakage import examples_in


def _normalise(source: str) -> str | None:
    """Canonical source for an expression, so spacing cannot hide a match."""
    try:
        return ast.unparse(ast.parse(source, mode="eval").body)
    except SyntaxError:
        return None


def _assertion_parts(candidate: str, entry_point: str) -> tuple[str, str] | None:
    """(call, asserted value) for a simple `assert call == value` candidate."""
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left, right = test.left, test.comparators[0]
        for call_side, value_side in ((left, right), (right, left)):
            if isinstance(call_side, ast.Call):
                function = call_side.func
                name = (
                    function.id if isinstance(function, ast.Name)
                    else function.attr if isinstance(function, ast.Attribute)
                    else None
                )
                if name == entry_point:
                    try:
                        return (ast.unparse(call_side), ast.unparse(value_side))
                    except Exception:
                        return None
    return None


def measure(artifact: Path, corpus_dir: Path, benchmark: str) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    killed = copied = killed_with_examples = 0
    unparsed = 0
    for result in payload.get("function_results") or []:
        if str(result.get("dataset_name") or "") != benchmark:
            continue
        if not result.get("killed"):
            continue
        killed += 1
        record = by_id.get(str(result.get("record_id")))
        if record is None:
            continue
        entry = str(record.get("entry_point") or "")
        stated = examples_in(str(record.get("specification") or ""), entry)
        if not stated:
            continue
        killed_with_examples += 1
        pairs = {
            (_normalise(example["call"]), _normalise(example["output"]))
            for example in stated
        }
        for outcome in result.get("candidate_outcomes") or []:
            if not outcome.get("killed"):
                continue
            parts = _assertion_parts(str(outcome.get("code") or ""), entry)
            if parts is None:
                unparsed += 1
                continue
            if (_normalise(parts[0]), _normalise(parts[1])) in pairs:
                copied += 1
                break

    return {
        "schema_version": "oneiros_example_copying_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "benchmark": benchmark,
        "functions_killed": killed,
        "killed_functions_whose_prompt_states_an_example": killed_with_examples,
        "killed_by_asserting_a_stated_example": copied,
        "share_of_kills": round(copied / killed, 4) if killed else None,
        "share_of_kills_where_an_example_was_available": round(
            copied / killed_with_examples, 4) if killed_with_examples else None,
        "killing_candidates_not_a_simple_equality": unparsed,
        "interpretation": (
            "a floor, not a ceiling: only `assert call == value` candidates "
            "matching a stated pair exactly are counted, so paraphrases, "
            "reordered arguments and equivalent literals are missed. A kill "
            "counted here required no determination of the function's "
            "behaviour beyond reading the prompt."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--benchmark", default="humaneval")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = measure(arguments.artifact, arguments.corpus, arguments.benchmark)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
