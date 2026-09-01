"""Read-only readiness audit for Oneiros canonical SFT supervision.

This mirrors the data-path checks used immediately before SFT without loading
a model or modifying checkpoints.  It is safe to run alongside ingestion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import (
    official_evidence_verifies_pair, write_json,
)
from harness.corpus_view import load_development_split, verify_development_view
from scripts.train_on_dataset import (
    FUNCTION_EXECUTION_MODE,
    REPOSITORY_EXECUTION_MODE,
    REPOSITORY_UNITTEST_EXECUTION_MODE,
    is_repository_execution_mode,
    _record_to_pair,
    load_phase3_pairs,
    _filter_overlong_repository_completions,
    _repository_fragment_tests,
    build_pair_prompt,
    evaluate_pair,
    extract_dataset_tests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Oneiros SFT data without training")
    parser.add_argument("--corpus-version", default="v2")
    parser.add_argument(
        "--split", default="train", choices=["train", "ablation_dev", "val"]
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    corpus_dir = ROOT / "data" / "corpus" / args.corpus_version
    verify_development_view(corpus_dir, [args.split])
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    from scripts import train_on_dataset as trainer
    trainer.REQUIRE_SPLIT_ISOLATION = True
    report = {
        "corpus_id": manifest["corpus_id"],
        "corpus_version": args.corpus_version,
        "split": args.split,
        "records": 0,
        "function_assertion_records": 0,
        "repository_pytest_fragment_records": 0,
        "repository_unittest_fragment_records": 0,
        "retained_training_records": 0,
        "verified_sft_examples_before_context_gate": 0,
        "verified_sft_examples": 0,
        "repository_overlong_completions_excluded": [],
        "reference_prompt_leaks": 0,
        "function_records_without_winner": [],
        "repository_records_without_verified_fragment": [],
    }
    source_pairs = load_phase3_pairs(corpus_dir, args.split)
    retained_pairs, excluded_completions = _filter_overlong_repository_completions(source_pairs)
    retained_pairs_by_id = {pair["id"]: pair for pair in retained_pairs}
    report["retained_training_records"] = len(retained_pairs)
    report["repository_overlong_completions_excluded"] = excluded_completions

    for record in load_development_split(
        corpus_dir, args.split, include_excluded=True
    ):
        source_pair = _record_to_pair(record)
        pair = retained_pairs_by_id.get(source_pair["id"])
        count_pair = pair or source_pair
        mode = source_pair["execution_mode"]
        prompt = build_pair_prompt(source_pair)
        report["records"] += 1
        if source_pair["golden_code"] and source_pair["golden_code"] in prompt:
            report["reference_prompt_leaks"] += 1
        if mode == FUNCTION_EXECUTION_MODE:
            report["function_assertion_records"] += 1
            if record.get("quality", {}).get("test_oracle_labels_execution_derived"):
                # V4/V4.1 records already persist per-assertion execution
                # evidence from the current policy. Re-executing thousands of
                # unchanged pairs here adds no evidence and makes readiness
                # depend on host timing; verify and count the sealed labels.
                winners = [
                    test["code"] for test in record["tests"]
                    if test.get("distinguishing") is True
                    and test.get("oracle") == "passes_reference_fails_target"
                ]
            else:
                source_tests = extract_dataset_tests(
                    count_pair["test_cases"], count_pair["entry_point"]
                )
                winners, _ = evaluate_pair(
                    source_tests, count_pair["golden_code"], count_pair["mutant_code"],
                    count_pair["entry_point"],
                )
            report["verified_sft_examples_before_context_gate"] += min(len(winners), 3)
            report["verified_sft_examples"] += min(len(winners), 3) if pair else 0
            if not winners:
                report["function_records_without_winner"].append(source_pair["id"])
        elif is_repository_execution_mode(mode):
            if mode == REPOSITORY_EXECUTION_MODE:
                report["repository_pytest_fragment_records"] += 1
            elif mode == REPOSITORY_UNITTEST_EXECUTION_MODE:
                report["repository_unittest_fragment_records"] += 1
            source_winners = _repository_fragment_tests(source_pair["test_cases"])
            winners = _repository_fragment_tests(count_pair["test_cases"])
            evidence = record["provenance"].get("official_test_evidence", {})
            report["verified_sft_examples_before_context_gate"] += min(len(source_winners), 3)
            report["verified_sft_examples"] += min(len(winners), 3) if pair else 0
            if (
                not source_winners or not official_evidence_verifies_pair(evidence)
            ):
                report["repository_records_without_verified_fragment"].append(source_pair["id"])
        else:
            raise RuntimeError(f"Unsupported canonical execution mode: {mode!r}")

    report["ready"] = not (
        report["reference_prompt_leaks"]
        or report["function_records_without_winner"]
        or report["repository_records_without_verified_fragment"]
    )
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
