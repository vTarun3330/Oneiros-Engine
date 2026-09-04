"""Audit section-aware prompt compaction across the development splits.

Part 8 of the research plan.  Compaction here is not truncation: it removes
whole support-context units and whole AST units, never a fragment of a
statement, and it FAILS CLOSED when the required sections - instructions,
behavioural specification, the complete target unit, required imports - cannot
fit the budget.  A prompt that cannot fit is rejected and counted, not silently
cut down.

Reports, per execution mode:

* how many prompts fit untouched, how many needed compaction, how many were
  rejected outright for budget;
* how many support units and code units were removed;
* maximum and percentile token counts.

Reads the development view only.  The sealed test split is never opened.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.prompt_budget import PromptBudgetError, compact_unified_user_prompt
from engine.test_generation_prompt import build_unified_user_prompt, format_chat_prompt
from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256


DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)
SPLITS = ("train", "ablation_dev", "val")

#: Frozen mode budgets. The function budget of 1024 is the selected
#: configuration; the repository budget matches the corpus manifest.
MODE_BUDGETS = {"function": 1024, "repository": 1024}


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _mode(record: dict[str, Any]) -> str:
    return "repository" if record.get("task_mode") == "repository" else "function"


def audit_split(records: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODE_BUDGETS:
        by_mode[mode] = {
            "budget_tokens": MODE_BUDGETS[mode],
            "prompts": 0,
            "fit_without_compaction": 0,
            "compacted": 0,
            "rejected_for_budget": 0,
            "malformed": 0,
            "support_units_removed": 0,
            "code_units_removed": 0,
            "original_tokens": [],
            "final_tokens": [],
            "rejection_reasons": collections.Counter(),
        }

    for record in records:
        mode = _mode(record)
        bucket = by_mode[mode]
        bucket["prompts"] += 1
        try:
            prompt = build_unified_user_prompt(
                code_under_test=(
                    record.get("prompt_code_under_test")
                    or record.get("code_under_test") or ""
                ),
                execution_mode=str(
                    (record.get("quality") or {}).get("execution_mode")
                    or "function_assertion"
                ),
                specification=record.get("specification") or "",
                support_context=record.get("support_context") or "",
                target_symbols=record.get("target_symbols") or None,
                entry_point=record.get("entry_point") or "",
            )
        except Exception as exc:
            bucket["malformed"] += 1
            bucket["rejection_reasons"][f"prompt_build:{type(exc).__name__}"] += 1
            continue

        try:
            result = compact_unified_user_prompt(
                tokenizer, prompt, MODE_BUDGETS[mode], format_chat_prompt,
            )
        except PromptBudgetError as exc:
            # Fail-closed: the required sections do not fit. The record yields no
            # prompt at all rather than a mutilated one.
            bucket["rejected_for_budget"] += 1
            bucket["rejection_reasons"][str(exc)[:90]] += 1
            continue

        bucket["original_tokens"].append(int(result.original_token_count))
        bucket["final_tokens"].append(int(result.final_token_count))
        if result.compacted:
            bucket["compacted"] += 1
            bucket["support_units_removed"] += int(result.support_units_dropped)
            bucket["code_units_removed"] += int(result.code_units_dropped)
        else:
            bucket["fit_without_compaction"] += 1

    summary: dict[str, Any] = {}
    for mode, bucket in by_mode.items():
        original = bucket.pop("original_tokens")
        final = bucket.pop("final_tokens")
        reasons = bucket.pop("rejection_reasons")
        prompts = max(1, bucket["prompts"])
        summary[mode] = {
            **bucket,
            "compaction_rate": round(bucket["compacted"] / prompts, 6),
            "rejection_rate": round(bucket["rejected_for_budget"] / prompts, 6),
            "original_tokens": {
                "max": max(original) if original else 0,
                "p50": _percentile(original, 0.50),
                "p90": _percentile(original, 0.90),
                "p99": _percentile(original, 0.99),
                "mean": round(statistics.fmean(original), 2) if original else 0,
            },
            "final_tokens": {
                "max": max(final) if final else 0,
                "p50": _percentile(final, 0.50),
                "p90": _percentile(final, 0.90),
                "p99": _percentile(final, 0.99),
            },
            "rejection_reasons": dict(reasons.most_common(8)),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_prompt_compaction_audit.json",
    )
    arguments = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)

    report: dict[str, Any] = {
        "schema_version": "oneiros_prompt_compaction_audit_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "tokenizer": arguments.tokenizer,
        "sealed_final_test_accessed": False,
        "policy": {
            "strategy": "section_aware_ast_units_before_chat_v4_1",
            "fail_closed": True,
            "never_slices_a_statement_or_ast_node": True,
            "preserved_sections": [
                "task instructions", "behavioural specification",
                "the complete target function or AST unit",
                "required imports and declarations",
            ],
            "removable_sections": ["optional support-context units, whole units only"],
            "metric_naming": (
                "prompt_compacted_examples; prompt_truncated_examples is retained "
                "as a deprecated alias because the behaviour is compaction, not "
                "truncation"
            ),
        },
        "mode_budgets": MODE_BUDGETS,
        "splits": {},
    }
    for split in SPLITS:
        path = arguments.view_dir / f"{split}.records.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        report["splits"][split] = audit_split(records, tokenizer)
        print(f"  {split}: {len(records)} records audited", flush=True)

    write_json(arguments.output, report)
    print(json.dumps({
        split: {
            mode: {
                key: value for key, value in bucket.items()
                if key in {
                    "prompts", "fit_without_compaction", "compacted",
                    "rejected_for_budget", "compaction_rate", "rejection_rate",
                    "support_units_removed",
                }
            }
            for mode, bucket in modes.items()
        }
        for split, modes in report["splits"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
