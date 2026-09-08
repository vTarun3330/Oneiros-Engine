"""Give mbpp specifications the worked examples they have never had.

Measured across train, ablation_dev and val: 0.0% of mbpp specifications carry
an input/output example, against 61-88% of humaneval's, and the median mbpp
specification is one sentence of 78 characters. The consequence is measured
too - the model probes an input where the bug shows in 76% of its failed mbpp
candidates and simply asserts the wrong value, because nothing in the prompt
says what the function returns.

This synthesises examples the same way humaneval's authors supplied them: as a
property of the FUNCTION, fixed for every mutant of it.

Three rules make that defensible rather than leakage:

* inputs are sampled from the signature. The mutant is never executed, never
  parsed and never consulted, so no input can be selected for distinguishing
  it. The gold tests are never read either - an input taken from a gold test
  would be an input chosen by the oracle to be discriminating, which is the
  same leak wearing a different hat.
* outputs come from executing the REFERENCE, which is what a specification
  states and what humaneval's docstrings already state.
* the share of synthesised examples that happen to distinguish a mutant is
  measured and reported rather than assumed to be zero. humaneval's own
  docstring examples have exactly this property; the number has to be on the
  record so the comparison is honest.

Nothing here decides to use the result. It writes examples and a disclosure
report; adopting them into a corpus is a separate, reviewable step.
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

#: Candidate argument values, tried in order. Deliberately small and boring:
#: an example exists to show the shape of the answer, not to stress the
#: function. Anything derived from the gold tests is excluded by construction
#: because this list is fixed and written by hand.
VALUE_LADDER: tuple[str, ...] = (
    "0", "1", "2", "5", "-1", "10", "3.5",
    "[]", "[1]", "[1, 2, 3]", "[3, 1, 2]", "[[1, 2], [3, 4]]",
    "''", "'abc'", "'hello world'", "'a b c'",
    "()", "(1, 2)", "((1, 2), (3, 4))",
    "{}", "{'a': 1}", "{1, 2, 3}",
    "True", "None",
)

#: Names carry intent in mbpp references, and trying the plausible shape first
#: turns a combinatorial search into a couple of attempts for most functions.
NAME_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("s", "str", "string", "text", "word", "sentence", "chars"),
     ("'abc'", "'hello world'", "'a b c'", "''")),
    (("l", "lst", "list", "arr", "array", "nums", "numbers", "items", "seq"),
     ("[1, 2, 3]", "[3, 1, 2]", "[1]", "[]")),
    (("t", "tup", "tuple", "test_tup", "test_tuple"),
     ("(1, 2)", "((1, 2), (3, 4))", "()")),
    (("d", "dict", "mapping"), ("{'a': 1}", "{}")),
    (("n", "num", "number", "x", "k", "size", "count", "index", "i"),
     ("1", "2", "5", "10", "0")),
)


def _hinted_values(parameter: str) -> tuple[str, ...]:
    lowered = parameter.lower()
    for names, values in NAME_HINTS:
        if lowered in names:
            return values
    return ()


def parameters_of(reference_code: str, entry_point: str) -> list[str] | None:
    """Parameter names of the function under test, or None if not found."""
    try:
        tree = ast.parse(reference_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == entry_point:
            arguments = node.args
            if arguments.vararg or arguments.kwarg:
                return None
            return [a.arg for a in (arguments.posonlyargs + arguments.args)]
    return None


def _argument_candidates(parameter: str) -> list[str]:
    hinted = list(_hinted_values(parameter))
    return hinted + [v for v in VALUE_LADDER if v not in hinted]


def _call(entry_point: str, arguments: list[str]) -> str:
    return entry_point + "(" + ", ".join(arguments) + ")"


def _evaluate(reference_code: str, support: str, call: str,
              timeout: float) -> tuple[bool, str]:
    probe = "result = repr(" + call + ")"
    ok, result, _error = execute_code(support + "\n" + reference_code, probe, timeout)
    if not ok:
        return (False, "")
    return (True, str(result if result is not None else "").strip())


def synthesise_for_record(record: dict[str, Any], wanted: int, budget: int,
                          timeout: float) -> list[dict[str, str]]:
    """Worked examples for one lineage, derived from the reference alone."""
    entry = str(record.get("entry_point") or "")
    reference = str(record.get("reference_code") or "")
    support = str(record.get("support_context") or "")
    parameters = parameters_of(reference, entry)
    if not entry or not reference or parameters is None or not parameters:
        return []

    ladders = [_argument_candidates(name) for name in parameters]
    examples: list[dict[str, str]] = []
    seen_calls: set[str] = set()
    attempts = 0

    # One argument varies at a time against a fixed first choice for the rest.
    # A full product is unaffordable and unnecessary: examples should differ in
    # one visible way, which is also what makes them readable.
    for position, ladder in enumerate(ladders):
        for value in ladder:
            if len(examples) >= wanted or attempts >= budget:
                break
            arguments = [ladders[i][0] for i in range(len(ladders))]
            arguments[position] = value
            call = _call(entry, arguments)
            if call in seen_calls:
                continue
            seen_calls.add(call)
            attempts += 1
            ok, output = _evaluate(reference, support, call, timeout)
            if not ok or not output:
                continue
            if len(output) > 120:
                continue
            examples.append({"call": call, "output": output})
        if len(examples) >= wanted or attempts >= budget:
            break
    return examples


def distinguishes(record: dict[str, Any], example: dict[str, str],
                  timeout: float) -> bool:
    """Whether this example would fail on the mutant.

    Called only for the DISCLOSURE report, after the examples are fixed. The
    mutant cannot influence which examples exist, only the count reported here.
    """
    mutant = str(record.get("code_under_test") or "")
    support = str(record.get("support_context") or "")
    if not mutant:
        return False
    ok, output = _evaluate(mutant, support, example["call"], timeout)
    return (not ok) or output != example["output"]


def run(corpus_dir: Path, split: str, benchmark: str, wanted: int,
        budget: int, timeout: float, limit: int | None) -> dict[str, Any]:
    if split == "test":
        raise SystemExit("refusing to synthesise against the sealed test split")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    wanted_ids = set(splits.get(split) or [])

    by_group: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record["id"])
        if record_id not in wanted_ids or benchmark not in record_id.lower():
            continue
        by_group.setdefault(str(record["group_id"]), record)

    groups = sorted(by_group)
    if limit is not None:
        groups = groups[:limit]

    examples: dict[str, list[dict[str, str]]] = {}
    covered = distinguishing = total_examples = 0
    for group in groups:
        record = by_group[group]
        found = synthesise_for_record(record, wanted, budget, timeout)
        if not found:
            continue
        examples[group] = found
        covered += 1
        total_examples += len(found)
        distinguishing += sum(
            1 for example in found if distinguishes(record, example, timeout))

    return {
        "schema_version": "oneiros_worked_examples_v1",
        "corpus_dir": corpus_dir.name,
        "split": split,
        "benchmark": benchmark,
        "sealed_final_test_accessed": False,
        "lineages_considered": len(groups),
        "lineages_with_examples": covered,
        "coverage": round(covered / len(groups), 4) if groups else None,
        "examples_total": total_examples,
        "examples_per_covered_lineage": round(total_examples / covered, 2)
        if covered else None,
        "disclosure": {
            "examples_that_distinguish_their_mutant": distinguishing,
            "share": round(distinguishing / total_examples, 4)
            if total_examples else None,
            "why_this_is_reported": (
                "humaneval's native docstring examples have the same property: "
                "authored for the function, they sometimes fail on a mutant. "
                "Inputs here are sampled from the signature and never from the "
                "mutant or the gold tests, so no example is selected for "
                "distinguishing anything - but the rate must be on the record "
                "for the humaneval comparison to be honest."
            ),
        },
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--split", default="ablation_dev")
    parser.add_argument("--benchmark", default="mbpp")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--attempt-budget", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = run(arguments.corpus, arguments.split, arguments.benchmark,
                 arguments.examples, arguments.attempt_budget,
                 arguments.timeout, arguments.limit)
    write_json(arguments.output, report)
    summary = {k: v for k, v in report.items() if k != "examples"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
