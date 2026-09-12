"""Prove every prompt fits alongside its completion BEFORE a model is loaded.

A 1024-token completion budget is only honest if the prompt it sits beside
leaves room for it. When it does not, there are two ways to fail: the run can
silently shorten the output, which quietly changes what is being measured, or
it can slice the prompt, which quietly changes what the model was asked. This
project refuses both - `compact_unified_user_prompt` fails closed rather than
slicing required sections - but the failure then arrives one record at a time,
mid-run, after the model is in memory.

So this checks the whole split up front, using the real tokenizer and the real
chat template, and reports the distribution rather than a pass/fail alone. It
loads the tokenizer only, never the model: a gate that costs a model load is a
gate people skip.

The gate fails if ANY record cannot fit its prompt and the full completion
budget inside the sequence limit. Failing closed here costs seconds; failing
closed at record 4,000 costs hours.
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

from engine.prompt_budget import PromptBudgetError, compact_unified_user_prompt
from engine.test_generation_prompt import (
    build_unified_user_prompt, format_chat_prompt,
)
from harness.corpus import write_json
from harness.corpus_view import load_development_split

SEALED_SPLIT = "test"


FUNCTION_MODE = "function_assertion"


def execution_mode_of(record: dict[str, Any]) -> str:
    quality = record.get("quality")
    if isinstance(quality, dict) and quality.get("execution_mode"):
        return str(quality["execution_mode"])
    return FUNCTION_MODE


def _record_prompt(record: dict[str, Any], information_variant: str,
                   instruction_variant: str) -> str:
    return build_unified_user_prompt(
        code_under_test=(record.get("prompt_code_under_test")
                         or record.get("code_under_test") or ""),
        execution_mode=execution_mode_of(record),
        specification=record.get("specification", ""),
        support_context=record.get("support_context", ""),
        target_symbols=record.get("target_symbols"),
        entry_point=record.get("entry_point", ""),
        information_variant=information_variant,
        output_instruction_variant=instruction_variant,
    )


def run(split: str, corpus_dir: Path, model_name: str, revision: str,
        prompt_budget: int, completion_budget: int, sequence_limit: int,
        information_variant: str, instruction_variant: str) -> dict[str, Any]:
    if split == SEALED_SPLIT:
        raise SystemExit("refusing to render prompts for the sealed final test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    # The hash-verified development shard for THIS split only. The canonical
    # records.json holds all four splits including the sealed final test, so a
    # train-only gate that opened it would have the sealed records in memory
    # for no reason - an avoidable exposure, and one nothing in the output
    # would reveal. load_development_split verifies the shard's hashes and the
    # sealed split is never materialised into the view at all.
    records = load_development_split(corpus_dir, split, include_excluded=True)
    by_id = {str(r["id"]): r for r in records}
    ids = [str(r["id"]) for r in records]

    lengths: list[int] = []
    failures: list[dict[str, Any]] = []
    compacted = 0
    skipped = 0
    # The generation panel is function-mode records only; repository records
    # are held separately and are not generated on, so gating the run on
    # whether THEY fit would fail the gate on records the run never touches.
    # Measured: the train split is 5595 function + 457 repository, and 5595 is
    # exactly the panel size the pipeline reports.
    held_repository = 0

    for record_id in ids:
        record = by_id.get(record_id)
        if record is None:
            skipped += 1
            continue
        if execution_mode_of(record) != FUNCTION_MODE:
            held_repository += 1
            continue
        try:
            prompt = _record_prompt(record, information_variant,
                                    instruction_variant)
        except ValueError:
            skipped += 1
            continue
        try:
            result = compact_unified_user_prompt(
                tokenizer, prompt, prompt_budget, format_chat_prompt)
        except (PromptBudgetError, ValueError) as exc:
            failures.append({"record_id": record_id, "reason": str(exc)[:200]})
            continue
        tokens = int(result.final_token_count)
        lengths.append(tokens)
        if getattr(result, "compacted", False):
            compacted += 1
        if tokens + completion_budget > sequence_limit:
            failures.append({
                "record_id": record_id,
                "reason": (f"rendered prompt {tokens} + completion "
                           f"{completion_budget} exceeds sequence limit "
                           f"{sequence_limit}")})

    lengths.sort()

    def pct(p: float) -> int | None:
        if not lengths:
            return None
        return lengths[min(int(p * len(lengths)), len(lengths) - 1)]

    return {
        "schema_version": "oneiros_sequence_fit_gate_v2",
        "split": split,
        "sealed_final_test_accessed": False,
        "record_source": "hash-verified development view shard for this split only",
        "canonical_records_json_opened": False,
        "model_name": model_name,
        "model_revision": revision,
        "prompt_token_limit": prompt_budget,
        "completion_token_limit": completion_budget,
        "sequence_limit": sequence_limit,
        "max_rendered_prompt_plus_completion": (
            (lengths[-1] + completion_budget) if lengths else None),
        "records_considered": len(ids),
        "repository_records_held_not_generated_on": held_repository,
        "generation_panel_size": len(lengths) + len(failures),
        "records_measured": len(lengths),
        "records_skipped": skipped,
        "records_compacted": compacted,
        "prompt_token_distribution": {
            "min": lengths[0] if lengths else None,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": lengths[-1] if lengths else None,
            "mean": round(statistics.mean(lengths), 1) if lengths else None,
        },
        "failures": failures[:25],
        "failure_count": len(failures),
        "gate_passed": not failures,
        "policy": (
            "the gate fails if ANY record cannot fit its prompt plus the FULL "
            "completion budget. Output length is never silently reduced."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--prompt-budget", type=int, default=1024)
    parser.add_argument("--completion-budget", type=int, default=1024)
    parser.add_argument("--sequence-limit", type=int, default=2048)
    parser.add_argument("--prompt-information-variant", default="full")
    parser.add_argument("--output-instruction-variant", default="self_contained")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = run(arguments.split, arguments.corpus, arguments.model_name,
                 arguments.model_revision, arguments.prompt_budget,
                 arguments.completion_budget, arguments.sequence_limit,
                 arguments.prompt_information_variant,
                 arguments.output_instruction_variant)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"},
                     indent=2))
    if not report["gate_passed"]:
        print("\nSEQUENCE FIT GATE FAILED - aborting before any model load.")
        for failure in report["failures"][:10]:
            print("  " + failure["record_id"] + ": " + failure["reason"])
        return 1
    print("\ngate passed: max rendered prompt + completion = "
          + str(report["max_rendered_prompt_plus_completion"])
          + " <= " + str(report["sequence_limit"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
