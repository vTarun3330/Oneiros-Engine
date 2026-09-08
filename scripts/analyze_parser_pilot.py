"""Score one set of generations under both parser interpretations.

The pilot generates each candidate exactly once with the raw output retained.
This then reads that same retained text and evaluates it twice: once as the
frozen protocol would (first line beginning ``assert ``, everything else
discarded) and once as the successor does (the whole output judged by the
widened policy).

Because both interpretations read identical text, the difference between them
carries no generation randomness. Any change is attributable to the parser
alone, which is what makes this a controlled comparison rather than two runs
that happen to differ.

The question it exists to answer: does the trained model actually emit useful
multi-assertion tests, or has it been emitting single assertions all along and
the collapse cost nothing?

ablation_dev only. Locked validation is not used to debug or redesign a
protocol, so a val artifact is refused outright.
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import count_assertions, validate_generated_test
from harness.corpus import write_json
from metrics.research_evaluation import evaluate_candidate_slots


def first_assertion_of(raw: str) -> str | None:
    """Exactly what the frozen parser keeps: the first ``assert `` line."""
    for line in (raw or "").split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert "):
            return stripped
    return None


def whole_output_of(raw: str) -> str:
    """The successor's view: the output intact, fences unwrapped."""
    code = (raw or "").strip()
    if code.startswith("```"):
        body = code.split("```")
        if len(body) >= 2:
            candidate = body[1]
            first, _, rest = candidate.partition("\n")
            code = (rest if first.strip().isalpha() else candidate).strip()
    return code


def _shape_of(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "unparseable"
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        return "test_function"
    if any(isinstance(node, ast.Assert) for node in tree.body):
        return "bare_assertion"
    return "other"


def _incomplete_ast(code: str) -> bool:
    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True


def analyse(artifact: Path, corpus_dir: Path, timeout: float) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    split = str(payload.get("evaluation_split") or "")
    if payload.get("final_test_measurement") or split == "test":
        raise SystemExit(f"{artifact} is sealed final-test data; refusing")
    if split == "val":
        raise SystemExit(
            f"{artifact} is locked validation. P0 diagnosis runs on "
            "ablation_dev only; locked validation is not used to debug or "
            "redesign the protocol."
        )

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    assertion_histogram: Counter[int] = Counter()
    shapes: Counter[str] = Counter()
    stats = {
        "raw_outputs": 0,
        "raw_outputs_retained": 0,
        "with_one_assertion": 0,
        "with_multiple_assertions": 0,
        "with_no_assertion": 0,
        "whole_output_parse_valid": 0,
        "whole_output_policy_valid": 0,
        "legacy_parse_valid": 0,
        "valid_legacy_invalid_whole": 0,
        "valid_whole_invalid_legacy": 0,
        "incomplete_ast": 0,
        "duplicate_raw_outputs": 0,
    }
    completion_chars: list[int] = []
    duplicate_seen: Counter[str] = Counter()

    legacy_slots: dict[str, list[dict[str, Any]]] = {}
    whole_slots: dict[str, list[dict[str, Any]]] = {}
    record_meta: dict[str, dict[str, str]] = {}

    for result in payload.get("function_results") or []:
        record_id = str(result.get("record_id"))
        record = by_id.get(record_id)
        if record is None:
            continue
        entry = str(record.get("entry_point") or "")
        record_meta[record_id] = {
            "entry_point": entry,
            "reference_code": str(record.get("reference_code") or ""),
            "mutant_code": str(record.get("code_under_test") or ""),
            "support_context": str(record.get("support_context") or ""),
        }
        for outcome in result.get("candidate_outcomes") or []:
            stats["raw_outputs"] += 1
            raw = outcome.get("raw_output")
            if not isinstance(raw, str):
                continue
            stats["raw_outputs_retained"] += 1
            completion_chars.append(len(raw))
            duplicate_seen[raw.strip()] += 1

            whole = whole_output_of(raw)
            legacy = first_assertion_of(raw)
            count = count_assertions(whole)
            assertion_histogram[count] += 1
            if count == 0:
                stats["with_no_assertion"] += 1
            elif count == 1:
                stats["with_one_assertion"] += 1
            else:
                stats["with_multiple_assertions"] += 1
            shapes[_shape_of(whole)] += 1
            if _incomplete_ast(whole):
                stats["incomplete_ast"] += 1

            whole_policy = validate_generated_test(whole, entry, allow_test_function=True)
            legacy_policy = (
                validate_generated_test(legacy, entry, allow_test_function=False)
                if legacy else None
            )
            if whole_policy.valid:
                stats["whole_output_policy_valid"] += 1
            if legacy is not None:
                stats["legacy_parse_valid"] += 1
            if legacy_policy is not None and legacy_policy.valid and not whole_policy.valid:
                stats["valid_legacy_invalid_whole"] += 1
            if whole_policy.valid and (legacy_policy is None or not legacy_policy.valid):
                stats["valid_whole_invalid_legacy"] += 1

            legacy_slots.setdefault(record_id, []).append(
                {"code": legacy, "parse_valid": legacy is not None})
            whole_slots.setdefault(record_id, []).append(
                {"code": whole, "parse_valid": bool(whole)})

    stats["duplicate_raw_outputs"] = sum(
        count - 1 for count in duplicate_seen.values() if count > 1)

    interpretations: dict[str, Any] = {}
    killed_only_by_later_assertions = 0
    for label, slots, allow in (
        ("legacy_first_assertion", legacy_slots, False),
        ("successor_whole_output", whole_slots, True),
    ):
        killed = valid = killing = requested = 0
        per_record: dict[str, bool] = {}
        for record_id, candidate_slots in slots.items():
            meta = record_meta[record_id]
            outcomes = evaluate_candidate_slots(
                candidate_slots, meta["support_context"] + "\n" + meta["reference_code"],
                meta["support_context"] + "\n" + meta["mutant_code"],
                meta["entry_point"], allow_test_function=allow,
            )
            requested += len(outcomes)
            valid += sum(1 for o in outcomes if o.get("reference_valid"))
            killing += sum(1 for o in outcomes if o.get("killed"))
            was_killed = any(o.get("killed") for o in outcomes)
            per_record[record_id] = was_killed
            killed += 1 if was_killed else 0
        interpretations[label] = {
            "functions": len(slots),
            "functions_killed": killed,
            "function_kill_rate": round(killed / max(len(slots), 1), 6),
            "requested_candidates": requested,
            "reference_valid_candidates": valid,
            "reference_valid_rate_per_requested": round(valid / max(requested, 1), 6),
            "killing_candidates": killing,
            "candidate_kill_rate_per_valid": round(killing / max(valid, 1), 6),
            "_per_record": per_record,
        }

    legacy_records = interpretations["legacy_first_assertion"].pop("_per_record")
    whole_records = interpretations["successor_whole_output"].pop("_per_record")
    killed_only_by_later_assertions = sum(
        1 for record_id, killed in whole_records.items()
        if killed and not legacy_records.get(record_id)
    )
    lost_under_successor = sum(
        1 for record_id, killed in legacy_records.items()
        if killed and not whole_records.get(record_id)
    )

    return {
        "schema_version": "oneiros_parser_pilot_v1",
        "protocol": "oneiros_whole_output_successor",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": split,
        "sealed_final_test_accessed": False,
        "paired": (
            "both interpretations read the SAME retained raw outputs, so the "
            "difference carries no generation randomness"
        ),
        "raw_output_stats": stats,
        "assertion_count_histogram": dict(sorted(assertion_histogram.items())),
        "output_shapes": dict(shapes.most_common()),
        "completion_length_chars": {
            "median": int(statistics.median(completion_chars)) if completion_chars else None,
            "p90": int(statistics.quantiles(completion_chars, n=10)[8])
            if len(completion_chars) > 10 else None,
            "max": max(completion_chars) if completion_chars else None,
        },
        "interpretations": interpretations,
        "functions_killed_only_under_successor": killed_only_by_later_assertions,
        "functions_killed_only_under_legacy": lost_under_successor,
        "interpretation_note": (
            "functions_killed_only_under_successor counts targets that die "
            "when assertions beyond the first are executed. If it is near "
            "zero the collapse cost nothing and the model was emitting single "
            "assertions regardless."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = analyse(arguments.artifact, arguments.corpus, arguments.timeout)
    write_json(arguments.output, report)
    summary = {k: v for k, v in report.items() if k != "assertion_count_histogram"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
