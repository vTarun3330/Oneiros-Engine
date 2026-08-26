"""Build an exact, GPU-free audit of the opt-in V3 SFT contingency sampler."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CANONICAL_CORPUS_VERSION
from engine.sft_trainer import SFTDataPoint
from harness.corpus import verify_corpus
from scripts.train_on_dataset import (
    FUNCTION_EXECUTION_MODE,
    _filter_overlong_repository_completions,
    _repository_fragment_tests,
    balanced_repeat_examples,
    build_prompt,
    deduplicate_sft_examples,
    evaluate_pair,
    extract_dataset_tests,
    is_repository_execution_mode,
    load_phase3_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _counts(items, attribute):
    return dict(sorted(Counter(getattr(item, attribute) for item in items).items()))


def build_audit(
    corpus_version: str,
    real_target_fraction: float,
    max_real_repeats: int,
    synthetic_balance_fraction: float,
    max_synthetic_repeats: int,
):
    corpus_dir = PROJECT_ROOT / "data" / "corpus" / corpus_version
    manifest = verify_corpus(corpus_dir)
    readiness_path = PROJECT_ROOT / "results" / f"{corpus_version}_train_readiness.json"
    if not readiness_path.exists():
        raise RuntimeError(f"Missing locked context-gate audit: {readiness_path}")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not readiness.get("ready"):
        raise RuntimeError("The locked train readiness report is not ready=true")
    train_pairs, overlong_completions = _filter_overlong_repository_completions(
        load_phase3_pairs(corpus_dir, "train")
    )
    if len(overlong_completions) != len(
        readiness.get("repository_overlong_completions_excluded", [])
    ):
        raise RuntimeError("Live tokenizer gate disagrees with the locked readiness audit")

    synthetic = []
    repository = []
    for pair_index, pair in enumerate(train_pairs, start=1):
        mode = pair.get("execution_mode", FUNCTION_EXECUTION_MODE)
        if is_repository_execution_mode(mode):
            winners = _repository_fragment_tests(pair.get("test_cases", []))
        else:
            source_tests = extract_dataset_tests(
                pair.get("test_cases", []), pair["entry_point"]
            )
            winners, _ = evaluate_pair(
                source_tests, pair["golden_code"], pair["mutant_code"],
                pair["entry_point"],
            )
        prompt = build_prompt(
            pair["mutant_code"],
            pair["entry_point"],
            pair.get("specification", ""),
            mode,
        )
        target = repository if is_repository_execution_mode(mode) else synthetic
        for completion in winners[:3]:
            target.append(
                SFTDataPoint(
                    prompt=prompt,
                    completion=completion,
                    function_id=pair["id"],
                    project=pair.get("project", "synthetic"),
                    bug_family=pair.get("bug_family", "unknown"),
                )
            )
        if pair_index % 500 == 0 or pair_index == len(train_pairs):
            print(
                f"audited {pair_index}/{len(train_pairs)} records: "
                f"synthetic={len(synthetic)} repository={len(repository)}",
                flush=True,
            )

    verified_raw_synthetic = len(synthetic)
    verified_raw_repository = len(repository)
    exact_supervision = Counter(
        (item.prompt, item.completion.strip()) for item in [*synthetic, *repository]
    )
    synthetic, synthetic_dedup = deduplicate_sft_examples(synthetic)
    repository, repository_dedup = deduplicate_sft_examples(repository)

    effective_synthetic = list(synthetic)
    synthetic_stats = None
    if synthetic_balance_fraction:
        effective_synthetic, synthetic_stats = balanced_repeat_examples(
            synthetic,
            math.ceil(len(synthetic) * (1.0 + synthetic_balance_fraction)),
            max_synthetic_repeats,
            "bug_family",
        )
    desired_real = math.ceil(
        len(effective_synthetic)
        * real_target_fraction
        / (1.0 - real_target_fraction)
    )
    balanced_repository, repository_stats = balanced_repeat_examples(
        repository, desired_real, max_real_repeats, "project"
    )
    current_uniform_desired_real = math.ceil(
        verified_raw_synthetic
        * real_target_fraction
        / (1.0 - real_target_fraction)
    )
    old_uniform_repeats = min(
        max_real_repeats,
        max(1, math.ceil(current_uniform_desired_real / max(verified_raw_repository, 1))),
    )
    old_uniform_real = verified_raw_repository * old_uniform_repeats
    contingency_total = len(effective_synthetic) + len(balanced_repository)

    return {
        "corpus_version": corpus_version,
        "corpus_id": manifest["corpus_id"],
        "corpus_records": manifest["training_records"],
        "train_records": readiness["records"],
        "context_eligible_train_records": readiness["retained_training_records"],
        "overlong_repository_completions_excluded": len(overlong_completions),
        "verified_raw_synthetic_examples": verified_raw_synthetic,
        "verified_raw_repository_examples": verified_raw_repository,
        "exact_duplicate_supervision_instances": sum(
            count - 1 for count in exact_supervision.values()
        ),
        "contingency_deduplication": {
            "synthetic": synthetic_dedup,
            "repository": repository_dedup,
        },
        "raw_repository_project_counts": _counts(repository, "project"),
        "contingency_synthetic_bug_family_counts_after_dedup": _counts(
            synthetic, "bug_family"
        ),
        "current_uniform_sampler": {
            "real_repeats": old_uniform_repeats,
            "effective_repository_examples": old_uniform_real,
            "effective_total_examples": verified_raw_synthetic + old_uniform_real,
            "effective_real_fraction": round(
                old_uniform_real / (verified_raw_synthetic + old_uniform_real), 6
            ),
        },
        "contingency_balanced_sampler": {
            "synthetic_balance_fraction": synthetic_balance_fraction,
            "effective_synthetic_examples": len(effective_synthetic),
            "desired_repository_examples": desired_real,
            "effective_repository_examples": len(balanced_repository),
            "effective_total_examples": contingency_total,
            "effective_real_fraction": round(
                len(balanced_repository) / contingency_total, 6
            ),
            "repository": repository_stats,
            "synthetic": synthetic_stats,
        },
        "recommended_first_fallback": {
            "learning_rate": 0.0001,
            "epochs": 2,
            "balanced_sampling": True,
            "synthetic_balance_fraction": synthetic_balance_fraction,
            "max_synthetic_repeats": max_synthetic_repeats,
            "real_target_fraction": real_target_fraction,
            "max_real_repeats": max_real_repeats,
            "validation_functions": 500,
            "validation_patience": 5,
            "best_validation_adapter_selection": True,
            "final_test_split_used": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-version", default=CANONICAL_CORPUS_VERSION)
    parser.add_argument("--real-target-fraction", type=float, default=0.20)
    parser.add_argument("--max-real-repeats", type=int, default=8)
    parser.add_argument("--synthetic-balance-fraction", type=float, default=0.0)
    parser.add_argument("--max-synthetic-repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "v3_sft_contingency_audit.json",
    )
    args = parser.parse_args()
    audit = build_audit(
        args.corpus_version,
        args.real_target_fraction,
        args.max_real_repeats,
        args.synthetic_balance_fraction,
        args.max_synthetic_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
