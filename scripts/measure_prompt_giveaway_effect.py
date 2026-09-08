"""Split humaneval by whether its prompt gives the answer away.

46% of ablation_dev humaneval records state, in their own prompt, an input and
the correct output for it that the mutant does not produce. Asserting that
stated pair kills the mutant with no reasoning at all, and 44-46% of humaneval
kills are exactly that assertion.

The question this answers is what humaneval looks like WITHOUT those records.
It stratifies the same evaluation - no regeneration, no new run - into records
whose prompt states a killing value and records whose prompt does not, and
reports the kill rate of each. The mbpp rate from the same artifact is carried
alongside, because mbpp has no worked examples at all and is therefore the
natural comparison for the clean stratum.

Nothing here is a correction applied to a reported number. It is a measurement
of how much of the humaneval/mbpp gap is a property of the prompts rather than
of the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from scripts.audit_native_example_leakage import (
    _value, examples_in, stated_matches_reference,
)


def _states_a_killing_value(record: dict[str, Any], timeout: float) -> bool | None:
    """Whether this record's own prompt states a value its mutant fails.

    None when no example in the prompt could be verified against the
    reference, which is different from "no giveaway" and is counted apart.
    """
    entry = str(record.get("entry_point") or "")
    support = str(record.get("support_context") or "")
    reference = str(record.get("reference_code") or "")
    mutant = str(record.get("code_under_test") or "")
    examples = examples_in(str(record.get("specification") or ""), entry)
    if not examples:
        return False
    verified_any = False
    for example in examples:
        reference_ok, reference_value = _value(
            reference, support, example["call"], timeout)
        if not reference_ok:
            continue
        verified_any = True
        if not stated_matches_reference(example["output"], reference_value):
            # The prompt states a value the reference does not produce.
            # Copying it fails on correct code, so it is a wrong example, not
            # a handed-over killing assertion.
            continue
        mutant_ok, mutant_value = _value(mutant, support, example["call"], timeout)
        if (not mutant_ok) or mutant_value != reference_value:
            return True
    return False if verified_any else None


def stratify(artifact: Path, corpus_dir: Path, timeout: float) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    strata = {
        "humaneval_prompt_states_a_killing_value": {"functions": 0, "killed": 0},
        "humaneval_no_giveaway": {"functions": 0, "killed": 0},
        "humaneval_unverifiable_example": {"functions": 0, "killed": 0},
        "mbpp": {"functions": 0, "killed": 0},
    }

    for result in payload.get("function_results") or []:
        benchmark = str(result.get("dataset_name") or "")
        killed = 1 if result.get("killed") else 0
        if benchmark == "mbpp":
            strata["mbpp"]["functions"] += 1
            strata["mbpp"]["killed"] += killed
            continue
        if benchmark != "humaneval":
            continue
        record = by_id.get(str(result.get("record_id")))
        if record is None:
            continue
        verdict = _states_a_killing_value(record, timeout)
        key = (
            "humaneval_unverifiable_example" if verdict is None
            else "humaneval_prompt_states_a_killing_value" if verdict
            else "humaneval_no_giveaway"
        )
        strata[key]["functions"] += 1
        strata[key]["killed"] += killed

    for bucket in strata.values():
        bucket["kill_rate"] = (
            round(bucket["killed"] / bucket["functions"], 6)
            if bucket["functions"] else None
        )

    giveaway = strata["humaneval_prompt_states_a_killing_value"]
    clean = strata["humaneval_no_giveaway"]
    mbpp = strata["mbpp"]
    return {
        "schema_version": "oneiros_prompt_giveaway_effect_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "strata": strata,
        "giveaway_advantage_points": (
            round((giveaway["kill_rate"] - clean["kill_rate"]) * 100, 2)
            if giveaway["kill_rate"] is not None and clean["kill_rate"] is not None
            else None
        ),
        "clean_humaneval_minus_mbpp_points": (
            round((clean["kill_rate"] - mbpp["kill_rate"]) * 100, 2)
            if clean["kill_rate"] is not None and mbpp["kill_rate"] is not None
            else None
        ),
        "interpretation": (
            "the same evaluation, stratified rather than re-run. If clean "
            "humaneval sits near mbpp, the humaneval/mbpp gap is a property "
            "of the prompts rather than of the model or the benchmark's "
            "difficulty."
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

    report = stratify(arguments.artifact, arguments.corpus, arguments.timeout)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
