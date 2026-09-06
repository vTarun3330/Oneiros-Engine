"""Measure what the first-assertion parser costs, on identical generations.

The frozen evaluation parses each model output by keeping the first line that
starts with ``assert `` and discarding the rest. A run with --retain-raw-output
stores what the model actually emitted, so the two parse rules can be compared
without regenerating anything: same model, same seed, same prompts, same
sampled outputs, only the rule that turns an output into a candidate differs.

That is the only way to attribute a difference to the rule rather than to
sampling noise, and it costs no GPU time.

The comparison is one-directional by construction. Keeping every assertion can
only add chances to detect the mutant - the first assertion is still there -
so the whole-output rule cannot score lower unless a longer test fails on the
reference implementation, which is a real and reportable outcome rather than
noise. Both directions are counted.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import executable_candidate, validate_generated_test
from harness.corpus import write_json
from harness.safe_execution import execute_code
from metrics.research_evaluation import wilson_interval
from utils.reproducibility import source_tree_sha256

DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)
EXECUTION_TIMEOUT_SECONDS = 5.0


def first_assertion(output: str) -> str:
    """Reproduce the frozen parser exactly: first 'assert ' line, nothing else."""
    for line in output.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert "):
            return stripped
    return ""


def whole_output(output: str) -> str:
    """Keep the output intact, unwrapping a fenced block if present."""
    code = output.strip()
    if code.startswith("```"):
        parts = code.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            first, _, rest = candidate.partition("\n")
            code = (rest if first.strip().isalpha() else candidate).strip()
    return code


def kills(code: str, entry: str, reference: str, mutant: str) -> tuple[bool, bool]:
    """Return (reference_valid, kills_mutant) under the two-sided rule.

    A test function must be INVOKED, not merely defined. ``def test_x(): ...``
    executed on its own defines a name and runs no assertion, so it passes on
    the reference and on the mutant alike and every candidate looks like a
    survivor. The evaluator appends the call through executable_candidate, and
    a rescore that omits it does not reproduce the frozen pipeline - it
    measures nothing at all.
    """
    if not code.strip():
        return False, False
    policy = validate_generated_test(code, entry, allow_test_function=True)
    if not policy.valid:
        return False, False
    runnable = executable_candidate(code, policy.shape)
    reference_ok, _, _ = execute_code(reference, runnable, EXECUTION_TIMEOUT_SECONDS)
    if not reference_ok:
        return False, False
    mutant_ok, _, _ = execute_code(mutant, runnable, EXECUTION_TIMEOUT_SECONDS)
    return True, not mutant_ok


def rescore(artifact: Path, view_dir: Path) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement"):
        raise SystemExit(f"Refusing {artifact}: sealed final-test measurement")
    split = str(payload.get("evaluation_split") or "")
    if split in {"test", "final", "sealed"}:
        raise SystemExit(f"Refusing sealed split {split!r}")

    records = {
        str(record["id"]): record
        for record in json.loads(
            (view_dir / f"{split}.records.json").read_text(encoding="utf-8")
        )
    }

    first_killed = whole_killed = 0
    functions = 0
    missing_raw = 0
    assertion_counts: list[int] = []
    only_whole: list[str] = []
    only_first: list[str] = []

    for row in payload.get("function_results") or []:
        record = records.get(str(row.get("record_id") or ""))
        if record is None:
            continue
        reference = record.get("reference_code") or ""
        mutant = record.get("code_under_test") or ""
        entry = record.get("entry_point") or ""
        if not (reference and mutant and entry):
            continue
        functions += 1

        first_hit = whole_hit = False
        for outcome in row.get("candidate_outcomes") or []:
            raw = str(outcome.get("raw_output") or "")
            if not raw:
                missing_raw += 1
                continue
            if not first_hit:
                _, killed = kills(first_assertion(raw), entry, reference, mutant)
                first_hit = first_hit or killed
            if not whole_hit:
                code = whole_output(raw)
                if code.lstrip().startswith("def test"):
                    assertion_counts.append(code.count("assert "))
                _, killed = kills(code, entry, reference, mutant)
                whole_hit = whole_hit or killed
            if first_hit and whole_hit:
                break

        first_killed += int(first_hit)
        whole_killed += int(whole_hit)
        if whole_hit and not first_hit:
            only_whole.append(str(row.get("record_id")))
        if first_hit and not whole_hit:
            only_first.append(str(row.get("record_id")))

    total = max(1, functions)
    return {
        "schema_version": "oneiros_parse_rule_rescore_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "split": split,
        "seed": payload.get("seed"),
        "functions": functions,
        "candidates_without_retained_raw_output": missing_raw,
        "frozen_first_assertion": {
            "killed": first_killed,
            "kill_rate": round(first_killed / total, 6),
            "kill_rate_wilson_95": wilson_interval(first_killed, total),
        },
        "whole_output": {
            "killed": whole_killed,
            "kill_rate": round(whole_killed / total, 6),
            "kill_rate_wilson_95": wilson_interval(whole_killed, total),
        },
        "delta_kill_rate": round((whole_killed - first_killed) / total, 6),
        "functions_killed_only_by_whole_output": len(only_whole),
        "functions_killed_only_by_first_assertion": len(only_first),
        "assertions_per_test_function": {
            "mean": round(statistics.mean(assertion_counts), 3)
            if assertion_counts else None,
            "median": statistics.median(assertion_counts)
            if assertion_counts else None,
            "max": max(assertion_counts) if assertion_counts else None,
            "test_functions_seen": len(assertion_counts),
        },
        "interpretation": (
            "generations are held fixed and only the parse rule varies, so the "
            "delta is attributable to the rule. Keeping every assertion can "
            "only add detection chances, because the first assertion is still "
            "present; a function killed only by the first-assertion rule means "
            "the longer test failed on the reference implementation, which is "
            "a real regression and is counted separately."
        ),
        "sample_functions_recovered": sorted(only_whole)[:20],
        "sample_functions_lost": sorted(only_first)[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_parse_rule_rescore.json",
    )
    arguments = parser.parse_args()
    report = rescore(arguments.artifact, arguments.view_dir)
    write_json(arguments.output, report)
    print(json.dumps({
        k: report[k] for k in (
            "functions", "frozen_first_assertion", "whole_output",
            "delta_kill_rate", "functions_killed_only_by_whole_output",
            "functions_killed_only_by_first_assertion",
            "assertions_per_test_function",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
