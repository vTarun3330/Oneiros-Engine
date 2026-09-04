"""Rescore an existing evaluation under the widened candidate shape policy.

An arm trained on multi-mutant supervision emits ``def test_*()`` candidates
and must be scored under the widened shape policy, or it is judged by a rule
that rejects what it was taught to produce.  Comparing it against a baseline
scored under the narrower assertion-only rule would then be a protocol
difference wearing the costume of a result.

This closes that gap without regenerating anything.  Evaluation artifacts store
the raw text of every candidate, so a candidate the frozen policy rejected
outright can be re-validated under the widened rule and, if it now passes,
executed against the reference and the mutant exactly as the model arms are.
Nothing is re-sampled, so the comparison holds the generations fixed and varies
only the scoring rule - which is the only honest way to isolate it.

The rescore is monotone by construction: it can only ever admit candidates that
were previously discarded, so a kill rate can rise but never fall.  A drop
means a bug, and the script says so rather than reporting it.
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

from harness.candidate_policy import validate_generated_test
from harness.corpus import write_json
from harness.safe_execution import execute_code
from metrics.research_evaluation import wilson_interval
from utils.reproducibility import source_tree_sha256

DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)
EXECUTION_TIMEOUT_SECONDS = 5.0


def _load_records(view_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    path = view_dir / f"{split}.records.json"
    return {
        str(record["id"]): record
        for record in json.loads(path.read_text(encoding="utf-8"))
    }


def rescore(
    artifact: Path, view_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement"):
        raise SystemExit(f"Refusing {artifact}: sealed final-test measurement")
    split = str(payload.get("evaluation_split") or "")
    if split in {"test", "final", "sealed"}:
        raise SystemExit(f"Refusing sealed split {split!r}")
    records = _load_records(view_dir, split)

    original_killed = 0
    rescored_killed = 0
    newly_admitted = 0
    newly_killing = 0
    rejected_without_stored_text = 0
    functions_flipped: list[str] = []
    examined = 0

    for result in payload.get("function_results", []):
        record_id = str(result.get("record_id") or "")
        was_killed = bool(result.get("killed"))
        original_killed += int(was_killed)
        now_killed = was_killed
        record = records.get(record_id)

        if not was_killed and record is not None:
            golden = record.get("reference_code") or ""
            mutant = record.get("code_under_test") or ""
            entry = record.get("entry_point") or ""
            for outcome in result.get("candidate_outcomes") or []:
                # Only candidates the frozen policy rejected outright can change.
                if outcome.get("policy_valid"):
                    continue
                code = str(outcome.get("code") or "")
                if not code.strip():
                    # The artifact stores no text for a candidate that failed
                    # generation outright. That is an empty output, not a
                    # rejected shape, so the widened rule cannot reach it - and
                    # saying so is the difference between "nothing qualified"
                    # and "nothing was there to qualify".
                    rejected_without_stored_text += 1
                    continue
                examined += 1
                if not validate_generated_test(
                    code, entry, allow_test_function=True
                ).valid:
                    continue
                newly_admitted += 1
                reference_ok, _, _ = execute_code(
                    golden, code, EXECUTION_TIMEOUT_SECONDS
                )
                if not reference_ok:
                    continue
                mutant_ok, _, _ = execute_code(
                    mutant, code, EXECUTION_TIMEOUT_SECONDS
                )
                if not mutant_ok:
                    newly_killing += 1
                    now_killed = True
                    functions_flipped.append(record_id)
                    break

        rescored_killed += int(now_killed)

    total = len(payload.get("function_results", []))
    monotone = rescored_killed >= original_killed
    return {
        "artifact": str(artifact).replace("\\", "/"),
        "split": split,
        "seed": payload.get("seed"),
        "functions": total,
        "frozen_assertion_only": {
            "killed": original_killed,
            "kill_rate": round(original_killed / max(1, total), 6),
            "kill_rate_wilson_95": wilson_interval(original_killed, max(1, total)),
        },
        "widened_test_function_allowed": {
            "killed": rescored_killed,
            "kill_rate": round(rescored_killed / max(1, total), 6),
            "kill_rate_wilson_95": wilson_interval(rescored_killed, max(1, total)),
        },
        "delta_kill_rate": round(
            (rescored_killed - original_killed) / max(1, total), 6
        ),
        "rejected_candidates_examined": examined,
        "rejected_candidates_without_stored_text": rejected_without_stored_text,
        "interpretation": (
            "no rejected candidate carried stored text, so every frozen-policy "
            "rejection here was an empty or ungenerated output rather than a "
            "test-function shape the widened rule would admit. The widened "
            "policy therefore gives this arm no advantage at all, which is what "
            "makes it fair to compare against an arm scored under it."
            if rejected_without_stored_text and not examined else
            "rejected candidates carried text and were re-validated under the "
            "widened rule; see newly_admitted_candidates for how many qualified"
        ),
        "newly_admitted_candidates": newly_admitted,
        "newly_killing_candidates": newly_killing,
        "functions_flipped_to_killed": sorted(set(functions_flipped)),
        "monotone_as_expected": monotone,
        "monotonicity_note": (
            "the widened rule only admits candidates the frozen rule rejected, "
            "so the kill count can rise but never fall"
        ) if monotone else (
            "BUG: the rescore lost kills, which the widening cannot cause"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_shape_policy_rescore.json",
    )
    arguments = parser.parse_args()

    rows = [rescore(path, arguments.view_dir) for path in arguments.artifact]
    report = {
        "schema_version": "oneiros_shape_policy_rescore_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "purpose": (
            "hold the generations fixed and vary only the candidate shape rule, "
            "so an arm scored under the widened policy can be compared against "
            "a baseline scored the same way"
        ),
        "rescored": rows,
        "all_monotone": all(row["monotone_as_expected"] for row in rows),
    }
    write_json(arguments.output, report)
    print(json.dumps([
        {
            "artifact": Path(row["artifact"]).name,
            "seed": row["seed"],
            "frozen": row["frozen_assertion_only"]["kill_rate"],
            "widened": row["widened_test_function_allowed"]["kill_rate"],
            "delta": row["delta_kill_rate"],
            "newly_admitted": row["newly_admitted_candidates"],
        }
        for row in rows
    ], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
