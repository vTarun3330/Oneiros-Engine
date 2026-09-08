"""Do humaneval's own docstring examples hand the model a killing assertion?

Synthesising worked examples for mbpp raised an obvious objection: an example
whose output differs on the mutant is a killing test written into the prompt,
and a model that copies it has not generated a test.

That objection is only decisive if humaneval is not already like this. 97% of
humaneval specifications in this corpus carry a worked example, those examples
were authored for the function with no knowledge of our mutants, and the model
scores 0.84 on humaneval against 0.60 on mbpp. So the rate has to be measured
on the corpus we already report, not assumed.

This parses the examples out of humaneval specifications, executes each one
against the mutant that record actually contains, and reports how often the
prompt already states a value the mutant does not produce.

It reads specifications and mutants that are already in the corpus, changes
nothing, and refuses the sealed test split.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.safe_execution import execute_code

#: ">>> call" followed by its result on the next line, and the inline arrow
#: forms humaneval prose uses. Both state an input and its expected output.
DOCTEST = re.compile(r"^\s*>>>\s*(?P<call>.+?)\s*$\n\s*(?P<output>\S.*?)\s*$",
                     re.MULTILINE)
ARROW = re.compile(r"^\s*(?P<call>[A-Za-z_]\w*\(.*?\))\s*(?:=>|==>|->|==)\s*"
                   r"(?P<output>.+?)\s*$", re.MULTILINE)


def examples_in(specification: str, entry_point: str) -> list[dict[str, str]]:
    """Input/output pairs the specification states, as executable source."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern in (DOCTEST, ARROW):
        for match in pattern.finditer(specification):
            call = match.group("call").strip()
            output = match.group("output").strip()
            if entry_point not in call or not output:
                continue
            if output.startswith(">>>"):
                continue
            try:
                ast.parse(call, mode="eval")
                ast.parse(output, mode="eval")
            except SyntaxError:
                continue
            if call in seen:
                continue
            seen.add(call)
            found.append({"call": call, "output": output})
    return found


def stated_matches_reference(stated: str, reference_repr: str) -> bool:
    """Does the value the prompt STATES equal what the reference produces?

    Omitting this check was a real defect: a record whose docstring states an
    output the reference does not produce was counted as handing the model a
    killing assertion. Copying that stated pair FAILS on correct code, so it
    kills nothing - it is a wrong example, not a giveaway.

    Compared as values rather than text, so "0b11" and '0b11' are the same
    answer written two ways. Falls back to normalised text when either side is
    not a literal.
    """
    stated_text = (stated or "").strip()
    if not stated_text:
        return False
    try:
        return ast.literal_eval(stated_text) == ast.literal_eval(reference_repr)
    except (ValueError, SyntaxError):
        return stated_text.strip("\"'") == (reference_repr or "").strip("\"'")


def _value(code: str, support: str, call: str, timeout: float) -> tuple[bool, str]:
    ok, result, _error = execute_code(
        support + "\n" + code, "result = repr(" + call + ")", timeout)
    if not ok:
        return (False, "")
    return (True, str(result if result is not None else "").strip())


def audit(corpus_dir: Path, split: str, benchmark: str, timeout: float,
          limit: int | None) -> dict[str, Any]:
    if split == "test":
        raise SystemExit("refusing to audit the sealed test split")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    wanted = set(splits.get(split) or [])

    rows = [r for r in records
            if str(r["id"]) in wanted and benchmark in str(r["id"]).lower()]
    if limit is not None:
        rows = rows[:limit]

    with_examples = 0
    total_examples = 0
    stated_and_differing = 0
    stated_value_wrong = 0
    records_with_a_giveaway = 0
    unverifiable = 0

    for record in rows:
        entry = str(record.get("entry_point") or "")
        specification = str(record.get("specification") or "")
        reference = str(record.get("reference_code") or "")
        mutant = str(record.get("code_under_test") or "")
        support = str(record.get("support_context") or "")
        found = examples_in(specification, entry)
        if not found:
            continue
        with_examples += 1
        giveaway = False
        for example in found:
            total_examples += 1
            reference_ok, reference_value = _value(
                reference, support, example["call"], timeout)
            mutant_ok, mutant_value = _value(
                mutant, support, example["call"], timeout)
            if not reference_ok:
                # The stated example does not run on the reference, so it
                # cannot be treated as a statement of correct behaviour.
                unverifiable += 1
                continue
            if not stated_matches_reference(example["output"], reference_value):
                # The prompt states a value the reference does not produce.
                # Asserting it fails on correct code, so it is a wrong example
                # rather than a handed-over killing assertion.
                stated_value_wrong += 1
                continue
            if (not mutant_ok) or mutant_value != reference_value:
                stated_and_differing += 1
                giveaway = True
        if giveaway:
            records_with_a_giveaway += 1

    return {
        "schema_version": "oneiros_native_example_leakage_v1",
        "corpus_dir": corpus_dir.name,
        "split": split,
        "benchmark": benchmark,
        "sealed_final_test_accessed": False,
        "records_examined": len(rows),
        "records_with_a_parsed_example": with_examples,
        "examples_parsed": total_examples,
        "examples_unverifiable_on_the_reference": unverifiable,
        "examples_stating_a_value_the_reference_does_not_produce": stated_value_wrong,
        "examples_whose_value_the_mutant_does_not_produce": stated_and_differing,
        "share_of_examples": round(stated_and_differing / total_examples, 4)
        if total_examples else None,
        "records_whose_prompt_states_a_killing_value": records_with_a_giveaway,
        "share_of_records_with_examples": round(
            records_with_a_giveaway / with_examples, 4) if with_examples else None,
        "interpretation": (
            "a record counted here has, in its own prompt, an input and the "
            "correct output for it, where the mutant produces something else. "
            "Asserting that stated pair kills the mutant without the model "
            "determining anything. This is a property of the upstream "
            "benchmark, not of a change made here, and it is measured so that "
            "humaneval and mbpp numbers are compared knowing it."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--split", default="ablation_dev")
    parser.add_argument("--benchmark", default="humaneval")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = audit(arguments.corpus, arguments.split, arguments.benchmark,
                   arguments.timeout, arguments.limit)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
