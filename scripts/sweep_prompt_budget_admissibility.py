"""Measure which function prompt budgets can actually prompt each panel.

Promptability is a hard prerequisite, not a performance result: a record whose
required sections (task instructions, target symbols, specification, localized
target code) cannot fit the budget is refused by fail-closed section-aware
compaction, and would silently drop out of the evaluation panel.

Token counts depend on the tokenizer, so this sweep must be re-derived for every
base model. It loads no model weights - only the tokenizer - and touches only
the training-partition and locked-validation development shards, never the
sealed test.

Usage:
    python scripts/sweep_prompt_budget_admissibility.py
    python scripts/sweep_prompt_budget_admissibility.py \\
        --base-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct

Writes results/v4_1_prompt_budget_sweep_<model_slug>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.train_on_dataset as tod  # noqa: E402
from config import CANONICAL_CORPUS_VERSION, model_config  # noqa: E402
from engine.prompt_budget import (  # noqa: E402
    PromptBudgetError,
    compact_unified_user_prompt,
)
from engine.test_generation_prompt import format_chat_prompt  # noqa: E402
from harness.corpus import verify_corpus  # noqa: E402
from harness.corpus_view import verify_development_view  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_BUDGETS = (512, 768, 1024, 1280)
DEFAULT_PANELS = ("ablation_dev", "val")


def sweep(
    base_model_name: str | None,
    base_model_revision: str | None,
    budgets: tuple[int, ...],
    panels: tuple[str, ...],
    corpus_version: str,
    local_files_only: bool,
) -> dict:
    from transformers import AutoTokenizer

    resolved_name = base_model_name or model_config.model_name
    resolved_revision = (
        base_model_revision
        if base_model_revision is not None
        else (
            model_config.model_revision
            if resolved_name == model_config.model_name
            else "main"
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_name,
        revision=resolved_revision,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tod.REQUIRE_SPLIT_ISOLATION = True
    tod.CORPUS_VERSION = corpus_version
    tod.PROMPT_INFORMATION_VARIANT = "full"
    tod.OUTPUT_INSTRUCTION_VARIANT = "self_contained"

    corpus_dir = ROOT / "data" / "corpus" / corpus_version
    verify_corpus(corpus_dir)
    verify_development_view(corpus_dir, ["train", *panels])

    started = time.time()
    panel_results: dict[str, dict] = {}
    for panel in panels:
        pairs = [
            pair
            for pair in tod.load_phase3_pairs(corpus_dir, panel)
            if pair.get("execution_mode", tod.FUNCTION_EXECUTION_MODE)
            == tod.FUNCTION_EXECUTION_MODE
        ]
        # Render every prompt once; only the budget check varies per budget.
        prompts = [(pair["id"], tod.build_pair_prompt(pair)) for pair in pairs]
        per_budget: dict[str, dict] = {}
        for budget in budgets:
            unpromptable: list[str] = []
            retained_tokens: list[int] = []
            for record_id, prompt in prompts:
                try:
                    compaction = compact_unified_user_prompt(
                        tokenizer, prompt, budget, format_chat_prompt
                    )
                except (PromptBudgetError, ValueError):
                    unpromptable.append(record_id)
                    continue
                retained_tokens.append(len(compaction.token_ids))
            per_budget[str(budget)] = {
                "function_records": len(prompts),
                "promptable": len(prompts) - len(unpromptable),
                "unpromptable": len(unpromptable),
                "unpromptable_fraction": (
                    round(len(unpromptable) / len(prompts), 6) if prompts else 0.0
                ),
                "max_retained_prompt_tokens": max(retained_tokens, default=0),
                "unpromptable_record_ids": unpromptable[:25],
            }
            print(
                f"[{panel}] budget={budget}: "
                f"{len(prompts) - len(unpromptable)}/{len(prompts)} promptable, "
                f"{len(unpromptable)} unpromptable",
                flush=True,
            )
        panel_results[panel] = {
            "function_records": len(prompts),
            "by_budget": per_budget,
        }

    admissible = [
        budget
        for budget in budgets
        if all(
            panel_results[panel]["by_budget"][str(budget)]["unpromptable"] == 0
            for panel in panels
        )
    ]
    return {
        "schema_version": "oneiros_v4_1_prompt_budget_sweep_v1",
        "final_test_measurement": False,
        "corpus_version": corpus_version,
        "base_model_name": resolved_name,
        "base_model_revision": resolved_revision,
        "panels": list(panels),
        "budgets_tested": list(budgets),
        "panel_results": panel_results,
        "admissible_budgets": admissible,
        "smallest_admissible_budget": min(admissible) if admissible else None,
        "note": (
            "Promptability is a prerequisite, not a performance result. A budget "
            "with any unpromptable record would silently change the evaluation "
            "panel and is not permitted. Token counts are tokenizer-specific, so "
            "this sweep is only valid for the base model named above."
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--base-model-revision", default=None)
    parser.add_argument("--corpus-version", default=CANONICAL_CORPUS_VERSION)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS)
    )
    parser.add_argument(
        "--panels", nargs="+", default=list(DEFAULT_PANELS),
        choices=["ablation_dev", "val"],
    )
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = sweep(
        args.base_model_name,
        args.base_model_revision,
        tuple(args.budgets),
        tuple(args.panels),
        args.corpus_version,
        local_files_only=not args.allow_tokenizer_download,
    )
    slug = report["base_model_name"].replace("/", "_").replace(".", "_")
    output = args.output or RESULTS_DIR / f"v4_1_prompt_budget_sweep_{slug}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "base_model_name": report["base_model_name"],
        "admissible_budgets": report["admissible_budgets"],
        "smallest_admissible_budget": report["smallest_admissible_budget"],
        "output": str(output),
    }, indent=2))
    return 0 if report["admissible_budgets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
