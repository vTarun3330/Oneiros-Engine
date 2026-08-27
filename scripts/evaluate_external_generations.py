"""Evaluate validation candidates exported by a modern external LLM baseline.

Input is JSONL with one object per validation function::

    {"record_id": "...", "model": "provider/model", "seed": 42,
     "candidates": ["assert target(...) == ...", ...]}

The script intentionally supports only the canonical validation split.  It
cannot open the sealed final test split, and it uses the same ordered candidate
metrics as native Phi-3/SFT/DPO evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.corpus import sha256_file, verify_corpus
from metrics.research_evaluation import (
    evaluate_candidate_slots,
    evaluation_profile_sha256,
    function_result,
    summarise_function_results,
)
from scripts.train_on_dataset import (
    FUNCTION_EXECUTION_MODE,
    load_phase3_pairs,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("External generation file is empty")
    return rows


def evaluate_external_generations(
    corpus_dir: Path,
    input_path: Path,
    *,
    candidate_budget: int = 8,
    allow_partial_smoke: bool = False,
) -> Dict[str, Any]:
    manifest = verify_corpus(corpus_dir)
    pairs = [
        pair for pair in load_phase3_pairs(corpus_dir, "val")
        if pair.get("execution_mode", FUNCTION_EXECUTION_MODE) == FUNCTION_EXECUTION_MODE
    ]
    by_id = {pair["id"]: pair for pair in pairs}
    rows = _read_jsonl(input_path)
    models = {str(row.get("model", "")).strip() for row in rows}
    seeds = {row.get("seed") for row in rows}
    if len(models) != 1 or not next(iter(models)):
        raise ValueError("Every row must declare the same non-empty model identity")
    if len(seeds) != 1 or not isinstance(next(iter(seeds)), int):
        raise ValueError("Every row must declare the same integer seed")

    supplied: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if record_id not in by_id:
            raise ValueError(f"Unknown or non-function validation record: {record_id!r}")
        if record_id in supplied:
            raise ValueError(f"Duplicate external generation record: {record_id}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > candidate_budget:
            raise ValueError(
                f"{record_id} must contain a candidates list of at most {candidate_budget} items"
            )
        if any(not isinstance(candidate, str) for candidate in candidates):
            raise ValueError(f"{record_id} contains a non-string candidate")
        supplied[record_id] = row

    expected_ids = set(by_id)
    if not allow_partial_smoke and set(supplied) != expected_ids:
        missing = len(expected_ids - set(supplied))
        raise ValueError(
            f"Full external baseline requires all {len(expected_ids)} validation functions; "
            f"{missing} are missing"
        )

    function_results = []
    for record_id in [pair["id"] for pair in pairs if pair["id"] in supplied]:
        pair = by_id[record_id]
        candidates = supplied[record_id]["candidates"]
        slots = []
        for rank in range(1, candidate_budget + 1):
            code = candidates[rank - 1] if rank <= len(candidates) else None
            slots.append({
                "rank": rank,
                "parse_valid": code is not None,
                "code": code,
                "raw_output_sha256": (
                    hashlib.sha256(code.encode("utf-8")).hexdigest() if code else None
                ),
            })
        outcomes = evaluate_candidate_slots(
            slots, pair["golden_code"], pair["mutant_code"], pair["entry_point"]
        )
        function_results.append(function_result(
            record_id,
            str(pair.get("bug_family", "unknown")),
            pair["entry_point"],
            outcomes,
            source_name=str(pair.get("source_name", "unknown")),
            project=str(pair.get("project", "unknown")),
        ))

    scope_ids = [item["record_id"] for item in function_results]
    profile = {
        "source": "external_generation_jsonl",
        "candidate_budget": candidate_budget,
        "partial_smoke": allow_partial_smoke,
        "k_values": [1, 2, 4, 8],
    }
    return {
        "mode": "external_llm_validation_only",
        "model": next(iter(models)),
        "seed": next(iter(seeds)),
        "evaluation_split": "val",
        "final_test_measurement": False,
        "corpus_id": manifest["corpus_id"],
        "tests_per_function": candidate_budget,
        "model_artifact_sha256": sha256_file(input_path),
        "evaluation_scope_sha256": hashlib.sha256(
            json.dumps(scope_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "evaluation_profile": profile,
        "evaluation_profile_sha256": evaluation_profile_sha256(profile),
        **summarise_function_results(function_results),
        "function_results": function_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=8, choices=range(1, 9))
    parser.add_argument(
        "--allow-partial-smoke", action="store_true",
        help="Allow a labelled validation subset for pipeline smoke testing",
    )
    args = parser.parse_args()
    result = evaluate_external_generations(
        args.corpus_dir,
        args.input,
        candidate_budget=args.candidate_budget,
        allow_partial_smoke=args.allow_partial_smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "model": result["model"],
        "seed": result["seed"],
        "functions": result["function_validation_records"],
        "function_kill_rate": result["function_kill_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
