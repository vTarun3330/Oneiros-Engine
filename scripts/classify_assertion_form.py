"""Did the model actually write relations, or just write better literal oracles?

A prompt change that raises kill@8 has not thereby demonstrated its mechanism.
The metamorphic instruction could plausibly help for a reason that has nothing
to do with relations: it is longer, it slows the model down, it repeats the
word "specification". Without this classifier a gain would be attributed to the
mechanism on the strength of the intention behind it, which is exactly the
reasoning that produced two retracted claims earlier in this project.

Every candidate is sorted by the FORM of its assertion:

* ``literal_oracle``  - a call compared for EQUALITY against a constant. The
  only form the oracle problem fully applies to, and what the model produced
  by default.
* ``relational``      - the entry point is called more than once, so the
  assertion relates two behaviours rather than naming a value.
* ``property``        - the call is wrapped in ``len``/``type``/``isinstance``
  and compared, so the assertion constrains a property of the result.
* ``bounded``         - a call compared against a constant by an INEQUALITY or
  membership (``> 0``, ``!= None``, ``in {...}``). Added after review: this is
  value-free in the sense that matters, since asserting a result is positive
  never requires knowing which positive number it is. Folding it into
  ``literal_oracle`` on the strength of "it mentions a constant" would have
  undercounted exactly the behaviour being looked for.
* ``other``           - anything else, reported rather than silently bucketed.

The classifier reads recorded candidate text. It never executes and never sees
the reference.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.evaluation_protocol import protocol_of

FORMS = ("literal_oracle", "relational", "property", "bounded", "other")
#: Only equality against a constant pins an exact value. Everything else
#: constrains the result without naming it.
_EXACT_VALUE_OPERATORS = (ast.Eq,)
_PROPERTY_WRAPPERS = {"len", "type", "isinstance", "sorted", "set", "str",
                      "list", "tuple", "abs", "sum", "max", "min"}


def _calls_to(node: ast.AST, entry_point: str) -> int:
    count = 0
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        name = (
            function.id if isinstance(function, ast.Name)
            else function.attr if isinstance(function, ast.Attribute)
            else None
        )
        if name == entry_point:
            count += 1
    return count


def _is_constant(node: ast.AST) -> bool:
    """A literal, including a container built only from literals."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Call, ast.Name, ast.Attribute,
                              ast.Subscript)):
            return False
    return True


def classify(candidate: str, entry_point: str) -> str:
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return "other"
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    if not asserts:
        return "other"

    forms: list[str] = []
    for statement in asserts:
        test = statement.test
        calls = _calls_to(test, entry_point)
        if calls >= 2:
            # Two calls to the function under test in one assertion is the
            # defining shape of a metamorphic relation.
            forms.append("relational")
            continue
        if isinstance(test, ast.Compare) and len(test.comparators) == 1:
            left, right = test.left, test.comparators[0]
            wrapped = any(
                isinstance(side, ast.Call)
                and isinstance(side.func, ast.Name)
                and side.func.id in _PROPERTY_WRAPPERS
                and _calls_to(side, entry_point) >= 1
                for side in (left, right)
            )
            if wrapped:
                forms.append("property")
                continue
            if calls == 1 and (_is_constant(right) or _is_constant(left)):
                exact = isinstance(test.ops[0], _EXACT_VALUE_OPERATORS)
                forms.append("literal_oracle" if exact else "bounded")
                continue
        forms.append("other")

    # One output, one form: the most informative assertion it contains.
    for preferred in ("relational", "property", "bounded", "literal_oracle"):
        if preferred in forms:
            return preferred
    return "other"


def build(artifact: Path, corpus_dir: Path) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    forms: Counter = Counter()
    killing_forms: Counter = Counter()
    per_dataset: dict[str, Counter] = {}
    per_dataset_killing: dict[str, Counter] = {}
    examples: dict[str, list[str]] = {form: [] for form in FORMS}

    for result in payload.get("function_results") or []:
        record = by_id.get(str(result.get("record_id")))
        entry = str((record or {}).get("entry_point") or "")
        dataset = str(result.get("dataset_name") or "unknown")
        if not entry:
            continue
        for outcome in result.get("candidate_outcomes") or []:
            code = str(outcome.get("code") or "")
            if not code:
                continue
            form = classify(code, entry)
            forms[form] += 1
            per_dataset.setdefault(dataset, Counter())[form] += 1
            if outcome.get("killed"):
                killing_forms[form] += 1
                per_dataset_killing.setdefault(dataset, Counter())[form] += 1
            if len(examples[form]) < 6 and len(code) < 160:
                examples[form].append(code.strip())

    total = sum(forms.values()) or 1
    killing_total = sum(killing_forms.values()) or 1
    return {
        "schema_version": "oneiros_assertion_form_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "evaluation_protocol": protocol_of(payload),
        "output_instruction_variant": (
            payload.get("evaluation_profile") or {}
        ).get("output_instruction_variant"),
        "sealed_final_test_accessed": False,
        "candidates_classified": sum(forms.values()),
        "form_counts": dict(forms.most_common()),
        "form_shares": {
            form: round(count / total, 6) for form, count in forms.most_common()},
        "killing_form_counts": dict(killing_forms.most_common()),
        "killing_form_shares": {
            form: round(count / killing_total, 6)
            for form, count in killing_forms.most_common()},
        # The headline for this measurement: how much of the output does not
        # depend on predicting an exact value.
        "value_free_share": round(
            sum(forms.get(f, 0) for f in ("relational", "property", "bounded"))
            / total, 6),
        "value_free_share_of_kills": round(
            sum(killing_forms.get(f, 0)
                for f in ("relational", "property", "bounded"))
            / killing_total, 6),
        "per_benchmark_form_counts": {
            dataset: dict(counter.most_common())
            for dataset, counter in sorted(per_dataset.items())},
        "per_benchmark_killing_form_counts": {
            dataset: dict(counter.most_common())
            for dataset, counter in sorted(per_dataset_killing.items())},
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    reports = [build(path, arguments.corpus) for path in arguments.artifacts]
    write_json(arguments.output, {"reports": reports})
    for report in reports:
        print("")
        print(report["artifact"])
        print("  instruction = " + str(report["output_instruction_variant"]))
        print("  all candidates : " + json.dumps(report["form_shares"]))
        print("  killing only   : " + json.dumps(report["killing_form_shares"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
