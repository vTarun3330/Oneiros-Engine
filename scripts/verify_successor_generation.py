"""Prove a successor generation is complete and honest before it is mined.

The oracle dataset will be built from exactly one artifact, so that artifact
has to be checked rather than trusted. Three questions decide it:

1. IS IT THE RUN WE ASKED FOR? The immutable run contract is compared field by
   field against what was required, so a run that quietly used a different
   sampling temperature, parser mode or model revision is rejected by name
   rather than absorbed.

2. ARE THE RAW OUTPUTS ACTUALLY THERE? ``retain_raw_output`` being true is a
   claim about intent. This checks every candidate actually carries raw text,
   and that the text hashes to the recorded ``raw_output_sha256`` - a retained
   output that does not match its own hash is worse than a missing one.

3. WAS ANYTHING TRUNCATED? Two kinds. Prompt truncation is refused by the
   pipeline and shows up as ``prompt_budget_failure``; that count must be zero.
   Completion truncation is silent, so it is measured: an output whose token
   count equals the completion limit was cut off mid-sentence, and an output
   that does not parse may be the same thing showing its other face.

A dataset built on truncated outputs would carry corrections for tests the
model never finished writing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

REQUIRED_CONTRACT = {
    "candidate_parse_mode": "whole_output",
    "retain_raw_output": True,
    "allow_test_function_candidates": True,
    "max_new_tokens": 1024,
    "prompt_token_limit": 1024,
    "candidates_per_function": 8,
    "base_model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "temperature": 0.7,
    "top_p": 0.9,
}

RAW_FIELDS = ("raw_output", "raw_text", "output_text")

#: PREDECLARED, before the run produced a number. An output whose token count
#: reaches the completion limit was cut off mid-sentence: the model had more to
#: say and the budget ended it. A few such outputs are tolerable noise; many
#: mean the 1024-token budget is itself too small for this panel, and a dataset
#: mined from them would carry corrections for tests the model never finished.
#:
#: Set at 2%. The measured precedent is the 128-token budget, where raising it
#: to 1024 cut incomplete ASTs from 481 to 49 of 4336 candidates - 1.1%. A rate
#: above 2% at 1024 would mean this budget is failing worse than the old one
#: did after its fix, which is a blocker rather than a footnote.
MAX_COMPLETION_LIMIT_HIT_RATE = 0.02

#: Candidates in these states may never supply an oracle-correction label. A
#: truncated or unparseable output is not evidence of what the model would have
#: written, so a "correction" built against it corrects nothing.
INELIGIBLE_FOR_ORACLE_LABELS = (
    "suspected_completion_truncation",
    "unparseable_raw_output",
)


def _raw_of(outcome: dict[str, Any]) -> str | None:
    for field in RAW_FIELDS:
        value = outcome.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def verify(artifact: Path, model_name: str, revision: str) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    problems: list[str] = []

    if payload.get("evaluation_split") != "train":
        problems.append(
            f"evaluation_split is {payload.get('evaluation_split')!r}; the "
            "oracle dataset may be mined only from train")

    contract = payload.get("run_contract") or {}
    if not contract:
        problems.append(
            "no run_contract in the artifact: this run predates the immutable "
            "contract and its generation settings cannot be verified")
    for key, expected in REQUIRED_CONTRACT.items():
        actual = contract.get(key)
        if actual != expected:
            problems.append(f"run_contract.{key} is {actual!r}, required {expected!r}")

    # Raw-output completeness.
    results = payload.get("function_results") or []
    candidates = 0
    with_raw = 0
    hash_mismatch = 0
    at_completion_limit = 0
    unparseable = 0
    missing_hash = 0
    ineligible_ids: set[tuple[str, int]] = set()
    token_limit = int(contract.get("max_new_tokens") or 0)

    tokenizer = None
    if token_limit:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        except Exception as exc:  # pragma: no cover - environment dependent
            problems.append(f"could not load tokenizer to measure truncation: {exc}")

    for result in results:
        for outcome in result.get("candidate_outcomes") or []:
            candidates += 1
            raw = _raw_of(outcome)
            recorded_hash = outcome.get("raw_output_sha256")
            if raw is None:
                continue
            with_raw += 1
            if not recorded_hash:
                missing_hash += 1
            elif hashlib.sha256(raw.encode("utf-8")).hexdigest() != recorded_hash:
                hash_mismatch += 1
            truncated = False
            if tokenizer is not None:
                if len(tokenizer(raw, add_special_tokens=False)["input_ids"]) >= token_limit:
                    at_completion_limit += 1
                    truncated = True
                    ineligible_ids.add(
                        (str(result.get("record_id")), int(outcome.get("rank") or 0)))
            try:
                ast.parse(raw)
            except SyntaxError:
                unparseable += 1
                if not truncated:
                    ineligible_ids.add(
                        (str(result.get("record_id")), int(outcome.get("rank") or 0)))

    if candidates and with_raw == 0:
        problems.append(
            "no candidate carries raw output text despite retain_raw_output "
            "being declared; the artifact cannot source oracle corrections")
    elif candidates and with_raw < candidates:
        problems.append(
            f"only {with_raw} of {candidates} candidates carry raw output")
    if hash_mismatch:
        problems.append(
            f"{hash_mismatch} retained outputs do not match their recorded "
            "raw_output_sha256")

    prompt_failures = int(payload.get("prompt_budget_failed_functions") or 0)
    if prompt_failures:
        problems.append(
            f"{prompt_failures} functions failed the prompt budget: prompts "
            "were refused rather than sliced, so those records generated "
            "nothing")

    limit_hit_rate = (at_completion_limit / candidates) if candidates else 0.0
    if limit_hit_rate > MAX_COMPLETION_LIMIT_HIT_RATE:
        problems.append(
            f"{at_completion_limit} of {candidates} outputs "
            f"({limit_hit_rate:.2%}) reached the {token_limit}-token completion "
            f"limit, above the predeclared {MAX_COMPLETION_LIMIT_HIT_RATE:.0%} "
            "threshold: the completion budget is too small for this panel and "
            "the dataset build is blocked")

    completed = int(payload.get("function_validation_records") or 0)
    results_present = len(results)
    if completed and results_present != completed:
        problems.append(
            f"{results_present} function results present but the run declared "
            f"{completed}; the artifact is incomplete")

    return {
        "schema_version": "oneiros_successor_generation_verification_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "sealed_final_test_accessed": False,
        "evaluation_split": payload.get("evaluation_split"),
        "run_contract": contract,
        "run_contract_sha256": payload.get("run_contract_sha256"),
        "function_results": results_present,
        "candidates": candidates,
        "candidates_with_raw_output": with_raw,
        "raw_output_hash_mismatches": hash_mismatch,
        "candidates_missing_recorded_hash": missing_hash,
        "outputs_reaching_completion_limit": at_completion_limit,
        "suspected_completion_truncation": at_completion_limit,
        "outputs_reaching_completion_limit_share": round(limit_hit_rate, 6),
        "completion_limit_hit_threshold": MAX_COMPLETION_LIMIT_HIT_RATE,
        "completion_limit_threshold_exceeded": (
            limit_hit_rate > MAX_COMPLETION_LIMIT_HIT_RATE),
        "unparseable_outputs": unparseable,
        "unparseable_share": round(unparseable / candidates, 6) if candidates else None,
        "candidates_ineligible_for_oracle_labels": len(ineligible_ids),
        "ineligible_reasons": list(INELIGIBLE_FOR_ORACLE_LABELS),
        "ineligible_candidate_keys": sorted(
            [list(k) for k in ineligible_ids])[:200],
        "prompt_budget_failed_functions": prompt_failures,
        "no_runtime_token_truncation": prompt_failures == 0,
        "function_kill_rate": payload.get("function_kill_rate"),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "evaluation_scope_sha256": payload.get("evaluation_scope_sha256"),
        "problems": problems,
        "verified": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = verify(arguments.artifact, arguments.model_name,
                    arguments.model_revision)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "run_contract"},
                     indent=2))
    if not report["verified"]:
        print("\nVERIFICATION FAILED - the dataset build must not start.")
        return 1
    print("\nverified: artifact may source the oracle-structured dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
