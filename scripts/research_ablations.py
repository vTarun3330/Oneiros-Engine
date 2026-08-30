"""Plan, smoke-test, and analyse Oneiros research ablations.

This script never opens the sealed test split.  ``smoke`` uses tiny synthetic
functions and deterministic candidate assertions, so it validates execution,
Kill@k/Pass@k, diversity, aggregation, and paired-policy diagnostics on a local
CPU without downloading Phi-3 or requiring Modal.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CANONICAL_CORPUS_VERSION
from metrics.research_evaluation import (
    aggregate_seed_results,
    compare_policy_results,
    evaluate_candidate_slots,
    evaluation_profile_sha256,
    function_result,
    prioritise_diverse_slots,
    sanitise_family_name,
    summarise_function_results,
)


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_HOLDOUT_FAMILIES = (
    "index",
    "negate removal",
    "logical",
    "off by one",
    "boundary",
    "arithmetic",
)


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _modal_command(
    *,
    phase: str,
    run_name: str,
    corpus_version: str,
    seed: int,
    max_validation_functions: int = 0,
    feedback_rounds: int = 0,
    diversity_mode: str = "none",
    holdout_family: str = "",
    evaluation_split: str = "ablation_dev",
    fresh: bool = False,
    max_pairs: int = 0,
    prompt_information_variant: str = "full",
    output_instruction_variant: str = "self_contained",
    real_target_fraction: float | None = None,
    balanced_sampling: bool | None = None,
    synthetic_balance_fraction: float | None = None,
    synthetic_balance_mode: str = "none",
    execution_mode: str = "",
    prompt_token_limit: int = 0,
) -> str:
    parts = [
        "py", "-3.12", "scripts/modal_train.py",
        "--phase", phase,
        "--run-name", run_name,
        "--corpus-version", corpus_version,
        "--seed", str(seed),
        "--evaluation-split", evaluation_split,
    ]
    if max_validation_functions:
        parts.extend(["--max-validation-functions", str(max_validation_functions)])
    if max_pairs:
        parts.extend(["--max-pairs", str(max_pairs)])
    if feedback_rounds:
        parts.extend(["--eval-feedback-rounds", str(feedback_rounds)])
    if diversity_mode != "none":
        parts.extend(["--eval-diversity-mode", diversity_mode])
    if holdout_family:
        parts.extend(["--holdout-bug-family", holdout_family])
    if prompt_information_variant != "full":
        parts.extend(["--prompt-information-variant", prompt_information_variant])
    if output_instruction_variant != "self_contained":
        parts.extend(["--output-instruction-variant", output_instruction_variant])
    if real_target_fraction is not None:
        parts.extend(["--sft-real-target-fraction", str(real_target_fraction)])
    if balanced_sampling is not None:
        parts.append("--sft-balanced-sampling" if balanced_sampling else "--no-sft-balanced-sampling")
    if synthetic_balance_fraction is not None:
        parts.extend(["--sft-synthetic-balance-fraction", str(synthetic_balance_fraction)])
    if synthetic_balance_mode != "none":
        parts.extend(["--sft-synthetic-balance-mode", synthetic_balance_mode])
    if execution_mode:
        parts.extend(["--execution-mode", execution_mode])
    if prompt_token_limit:
        parts.extend(["--sft-prompt-token-limit", str(prompt_token_limit)])
    if fresh:
        parts.append("--fresh")
    return " ".join(_quote(part) for part in parts)


def _preflight_command(
    *,
    run_name: str,
    corpus_version: str,
    max_pairs: int,
    prompt_information_variant: str = "full",
    output_instruction_variant: str = "self_contained",
    real_target_fraction: float | None = None,
    balanced_sampling: bool | None = None,
    synthetic_balance_fraction: float | None = None,
    synthetic_balance_mode: str = "none",
    execution_mode: str = "",
    prompt_token_limit: int = 0,
) -> str:
    parts = [
        "py", "-3.12", "scripts/preflight_sft_run.py",
        "--corpus-version", corpus_version,
        "--max-pairs", str(max_pairs),
        "--epochs", "1",
        "--batch-size", "1",
        "--learning-rate", "0.00005",
        "--lr-scheduler-type", "constant_with_warmup",
        "--repository-completion-token-limit", "1024",
        "--min-function-kill-rate", "0.58",
        "--output", f"results/{run_name}_preflight.json",
    ]
    if prompt_information_variant != "full":
        parts.extend(["--prompt-information-variant", prompt_information_variant])
    if output_instruction_variant != "self_contained":
        parts.extend(["--output-instruction-variant", output_instruction_variant])
    if real_target_fraction is not None:
        parts.extend(["--real-target-fraction", str(real_target_fraction)])
    if balanced_sampling is not None:
        parts.append("--balanced-sampling" if balanced_sampling else "--no-balanced-sampling")
    if synthetic_balance_fraction is not None:
        parts.extend(["--synthetic-balance-fraction", str(synthetic_balance_fraction)])
    if synthetic_balance_mode != "none":
        parts.extend(["--synthetic-balance-mode", synthetic_balance_mode])
    if execution_mode:
        parts.extend(["--execution-mode", execution_mode])
    if prompt_token_limit:
        parts.extend(["--prompt-token-limit", str(prompt_token_limit)])
    return " ".join(_quote(part) for part in parts)


def build_ablation_plan(
    run_name: str,
    corpus_version: str = CANONICAL_CORPUS_VERSION,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    holdout_families: Sequence[str] = DEFAULT_HOLDOUT_FAMILIES,
    smoke_functions: int = 32,
) -> Dict[str, Any]:
    """Build predeclared training-only ablation-dev commands for Modal execution."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("Run name contains unsupported characters")
    if len(set(seeds)) < 3 or any(int(seed) < 0 for seed in seeds):
        raise ValueError("Research plan requires at least three distinct non-negative seeds")
    families = [sanitise_family_name(family) for family in holdout_families]
    families = [family for family in families if family]

    experiments: List[Dict[str, Any]] = []

    def add(experiment_id: str, axis: str, commands: Iterable[str], **metadata: Any) -> None:
        experiments.append({
            "id": experiment_id,
            "axis": axis,
            "status": "planned_modal_not_run",
            "uses_final_test": False,
            "commands": list(commands),
            **metadata,
        })

    def controlled_sft_experiment(
        experiment_id: str,
        axis: str,
        run_suffix: str,
        *,
        max_pairs: int = 800,
        prompt_information_variant: str = "full",
        output_instruction_variant: str = "self_contained",
        real_target_fraction: float | None = None,
        balanced_sampling: bool | None = None,
        synthetic_balance_fraction: float | None = None,
        synthetic_balance_mode: str = "none",
        training_execution_mode: str = "",
        prompt_token_limit: int = 0,
        **metadata: Any,
    ) -> None:
        treatment_run = f"{run_name}_{run_suffix}"
        common = {
            "run_name": treatment_run,
            "corpus_version": corpus_version,
            "prompt_information_variant": prompt_information_variant,
            "output_instruction_variant": output_instruction_variant,
            "real_target_fraction": real_target_fraction,
            "balanced_sampling": balanced_sampling,
            "synthetic_balance_fraction": synthetic_balance_fraction,
            "synthetic_balance_mode": synthetic_balance_mode,
            "prompt_token_limit": prompt_token_limit,
        }
        train = _modal_command(
            phase="sft", seed=seeds[0], max_pairs=max_pairs, fresh=True,
            execution_mode=training_execution_mode, **common
        )
        train += (
            " --sft-epochs 1 --sft-learning-rate 0.00005"
            " --sft-lr-scheduler-type constant_with_warmup --sft-batch-size 1"
            " --sft-repository-completion-token-limit 1024"
            " --sft-monitor-validation-functions 100 --sft-monitor-patience 3"
            " --sft-monitor-min-function-kill-rate 0.58"
        )
        evaluations = [
            _modal_command(phase="sft_eval", seed=seed, **common)
            for seed in seeds
        ]
        add(
            experiment_id,
            axis,
            [train, *evaluations],
            run_name=treatment_run,
            preflight_command=(
                _preflight_command(
                    run_name=treatment_run,
                    corpus_version=corpus_version,
                    max_pairs=max_pairs,
                    prompt_information_variant=prompt_information_variant,
                    output_instruction_variant=output_instruction_variant,
                    real_target_fraction=real_target_fraction,
                    balanced_sampling=balanced_sampling,
                    synthetic_balance_fraction=synthetic_balance_fraction,
                    synthetic_balance_mode=synthetic_balance_mode,
                    execution_mode=training_execution_mode,
                    prompt_token_limit=prompt_token_limit,
                )
                if max_pairs else None
            ),
            training_size=max_pairs or "full_eligible_train",
            training_command_index=0,
            evaluation_command_indexes=list(range(1, len(evaluations) + 1)),
            seeds=list(seeds),
            prompt_information_variant=prompt_information_variant,
            output_instruction_variant=output_instruction_variant,
            real_target_fraction=real_target_fraction,
            balanced_sampling=balanced_sampling,
            synthetic_balance_fraction=synthetic_balance_fraction,
            synthetic_balance_mode=synthetic_balance_mode,
            training_execution_mode=training_execution_mode or None,
            prompt_token_limit=prompt_token_limit or None,
            **metadata,
        )

    # Primary V4.1 ablation matrix. Every runnable treatment trains only on
    # train and selects only on the fixed training-derived ablation_dev split.
    for variant, label in (
        ("code_only", "A0"),
        ("code_specification", "A1"),
        ("full", "A2"),
    ):
        controlled_sft_experiment(
            f"A_prompt_information_{label}",
            "prompt_information",
            f"{label.lower()}_{variant}",
            prompt_information_variant=variant,
            hypothesis="Measure the incremental value of specification and legitimate support context.",
        )

    for variant, label in (
        ("legacy_exactly_one", "C0"),
        ("self_contained", "C1"),
    ):
        controlled_sft_experiment(
            f"C_output_instruction_{label}",
            "output_instruction",
            f"{label.lower()}_{variant}",
            output_instruction_variant=variant,
            hypothesis="Measure whether revised one-test-case wording improves candidate validity.",
        )

    for fraction in (0.0, 0.10, 0.20, 0.30):
        label = int(round(fraction * 100))
        controlled_sft_experiment(
            f"F_repository_weight_{label}",
            "training_composition",
            f"f_repo_{label}",
            real_target_fraction=fraction,
            training_execution_mode=("function_assertion" if fraction == 0.0 else ""),
            hypothesis="Measure repository-supervision transfer without changing the evaluator.",
        )

    for label, balanced, fraction, balance_mode in (
        ("G0_proportional", False, 0.0, "none"),
        ("G1_dataset_balanced", True, 1.0, "dataset"),
        ("G2_dataset_family_balanced", True, 1.0, "dataset_family"),
    ):
        controlled_sft_experiment(
            label,
            "sampling_balance",
            label.lower(),
            balanced_sampling=balanced,
            real_target_fraction=0.0 if not balanced else 0.20,
            synthetic_balance_fraction=fraction,
            synthetic_balance_mode=balance_mode,
            hypothesis="Measure whether deterministic balancing reduces source/family dominance.",
        )

    for size in (800, 2000, 4000, 0):
        label = str(size) if size else "full"
        controlled_sft_experiment(
            f"I_training_scale_{label}",
            "training_scale",
            f"i_scale_{label}",
            max_pairs=size,
            hypothesis="Measure whether the current policy is undertrained.",
        )

    # Group J screens the declared function prompt budget.  At 512 tokens the
    # fail-closed compaction cannot render a prompt for most of the corpus, so
    # this is a prerequisite for every other treatment rather than a tuning knob.
    for budget in (512, 768, 1024):
        controlled_sft_experiment(
            f"J_prompt_budget_{budget}",
            "prompt_budget",
            f"j_budget_{budget}",
            prompt_token_limit=budget,
            hypothesis=(
                "Measure whether a function prompt budget that actually fits the "
                "required sections improves reference-validity and Kill@k. The "
                "sequence budget is 2048 and function completions are capped at "
                "128, so 768 and 1024 both fit without changing the evaluator."
            ),
            note=(
                "Changing the budget changes the evaluation scope hash. Results "
                "are not comparable with V4 numbers produced at 512 unless the "
                "V4 adapter is re-evaluated under the same budget."
            ),
        )

    add(
        "B_legacy_vs_unified",
        "prompt_schema",
        [],
        status="blocked_clean_control_unavailable",
        decision="INCONCLUSIVE",
        reason=(
            "The historical dataset-specific repository prompt is not a clean control because "
            "it can reintroduce gold-derived context. No GPU command is emitted until a leak-safe "
            "legacy renderer has independent tests."
        ),
    )
    add(
        "D_head_tail_vs_section_compaction",
        "prompt_compaction",
        [],
        status="local_safety_only",
        decision="INCONCLUSIVE",
        reason=(
            "Both helpers exist, but production remains fail-closed on section-aware compaction; "
            "head/tail is not emitted as a GPU treatment until malformed-prompt accounting is wired."
        ),
    )
    add(
        "E_localization_assumption",
        "localization_source",
        [],
        status="blocked_paired_scope_unavailable",
        decision="INCONCLUSIVE",
        reason=(
            "A paired public-vs-oracle localization panel is not yet materialized. Oracle-derived "
            "localization cannot become the headline condition."
        ),
    )
    add(
        "H_supervision_policy",
        "candidate_supervision_quality",
        [],
        status="strict_treatment_ready_control_rejected",
        decision="INCONCLUSIVE",
        reason=(
            "V4.1 strict reference-pass/buggy-fail supervision is active. The older corpus is not "
            "a clean control because data and leakage policy changed together."
        ),
    )

    add(
        "base_vs_sft_base",
        "model_stage",
        (
            _modal_command(
                phase="base_eval", run_name=run_name, corpus_version=corpus_version,
                seed=seed,
            )
            for seed in seeds
        ),
        model="pinned_base_phi3",
        seeds=list(seeds),
    )
    add(
        "base_vs_sft_sft",
        "model_stage",
        (
            _modal_command(
                phase="sft_eval", run_name=run_name, corpus_version=corpus_version,
                seed=seed,
            )
            for seed in seeds
        ),
        model="frozen_sft_adapter",
        seeds=list(seeds),
    )

    for feedback_rounds in (1, 2):
        add(
            f"sft_feedback_{feedback_rounds}",
            "execution_feedback",
            (
                _modal_command(
                    phase="sft_eval", run_name=run_name,
                    corpus_version=corpus_version, seed=seed,
                    feedback_rounds=feedback_rounds,
                )
                for seed in seeds
            ),
            feedback_rounds=feedback_rounds,
            fixed_candidate_budget=8,
            seeds=list(seeds),
        )

    for diversity_mode in ("ast", "input_shape"):
        add(
            f"sft_diversity_{diversity_mode}",
            "candidate_diversity",
            (
                _modal_command(
                    phase="sft_eval", run_name=run_name,
                    corpus_version=corpus_version, seed=seed,
                    diversity_mode=diversity_mode,
                )
                for seed in seeds
            ),
            diversity_mode=diversity_mode,
            fixed_candidate_budget=8,
            seeds=list(seeds),
        )

    add(
        "sft_feedback_1_plus_ast_diversity",
        "combined_feedback_diversity",
        (
            _modal_command(
                phase="sft_eval", run_name=run_name, corpus_version=corpus_version,
                seed=seed, feedback_rounds=1, diversity_mode="ast",
            )
            for seed in seeds
        ),
        feedback_rounds=1,
        diversity_mode="ast",
        fixed_candidate_budget=8,
        seeds=list(seeds),
    )

    for family in families:
        family_slug = re.sub(r"[^a-z0-9]+", "-", family).strip("-")
        holdout_run = f"{run_name}_lofo_{family_slug}"
        train_command = _modal_command(
            phase="sft", run_name=holdout_run, corpus_version=corpus_version,
            seed=seeds[0], holdout_family=family, fresh=True,
        )
        evaluation_commands = [
            _modal_command(
                phase="sft_eval", run_name=holdout_run,
                corpus_version=corpus_version, seed=seed,
                holdout_family=family,
            )
            for seed in seeds
        ]
        add(
            f"leave_one_family_out_{family_slug}",
            "mutation_family_generalisation",
            [train_command, *evaluation_commands],
            holdout_bug_family=family,
            training_command_index=0,
            evaluation_command_indexes=list(range(1, len(evaluation_commands) + 1)),
            seeds=list(seeds),
        )

    smoke_commands = []
    for phase, feedback, diversity in (
        ("base_eval", 0, "none"),
        ("sft_eval", 0, "none"),
        ("sft_eval", 1, "none"),
        ("sft_eval", 0, "ast"),
    ):
        smoke_commands.append(_modal_command(
            phase=phase,
            run_name=run_name,
            corpus_version=corpus_version,
            seed=seeds[0],
            max_validation_functions=smoke_functions,
            feedback_rounds=feedback,
            diversity_mode=diversity,
        ))

    plan_identity = {
        "run_name": run_name,
        "corpus_version": corpus_version,
        "seeds": list(seeds),
        "holdout_families": families,
        "candidate_budget": 8,
        "split": "ablation_dev",
        "primary_groups": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    }
    matrix_identity = [
        {
            "id": item["id"],
            "axis": item["axis"],
            "commands": item["commands"],
            "preflight_command": item.get("preflight_command"),
            "status": item["status"],
        }
        for item in experiments
    ]
    return {
        "schema_version": 2,
        "objective": "execution-verified bug-finding generalisation",
        "final_test_sealed": True,
        "plan_identity": plan_identity,
        "plan_sha256": evaluation_profile_sha256(plan_identity),
        "experiment_matrix_sha256": evaluation_profile_sha256({"experiments": matrix_identity}),
        "selection_rule": {
            "priority": [
                "reference_validity",
                "execution_validity",
                "Kill@8",
                "Kill@1_and_Kill@4",
                "equal_weight_dataset_macro",
                "mutation_family_robustness",
            ],
            "locked_validation_gate_is_separate": True,
            "negative_results_are_retained": True,
        },
        "local_cpu_smoke_command": "py -3.12 scripts/research_ablations.py smoke",
        "modal_smoke_commands": smoke_commands,
        "experiments": experiments,
        "deferred_external_systems": [
            {
                "id": "modern_llm_baselines",
                "reason": "requires generated candidate JSONL from explicitly selected external models",
                "ready_component": "scripts/evaluate_external_generations.py",
            },
            {
                "id": "native_repository_real_bug_evaluation",
                "reason": "requires isolated project environments and generated repository tests",
                "ready_component": "BugsInPy/SWE-bench ingestion and locked external task index",
            },
        ],
    }


def _slot(rank: int, code: str | None) -> Dict[str, Any]:
    return {
        "rank": rank,
        "parse_valid": code is not None,
        "code": code,
        "raw_output_sha256": f"smoke-{rank}",
    }


def _smoke_policy_results(diversity_mode: str = "none") -> Dict[str, Any]:
    examples = [
        {
            "id": "smoke_add",
            "family": "arithmetic",
            "entry": "add",
            "golden": "def add(a, b):\n    return a + b\n",
            "mutant": "def add(a, b):\n    return a - b\n",
            "slots": [
                _slot(1, None),
                _slot(2, "assert add(0, 0) == 0"),
                _slot(3, "assert add(2, 3) == 5"),
                _slot(4, "assert add(-2, 1) == -1"),
                _slot(5, "assert add(2, 3) == 5"),
                _slot(6, "assert missing(1) == 1"),
                _slot(7, "assert add(10, 1) == 11"),
                _slot(8, "assert add(1, -4) == -3"),
            ],
        },
        {
            "id": "smoke_boundary",
            "family": "boundary",
            "entry": "is_adult",
            "golden": "def is_adult(age):\n    return age >= 18\n",
            "mutant": "def is_adult(age):\n    return age > 18\n",
            "slots": [
                _slot(1, "assert is_adult(21) is True"),
                _slot(2, "assert is_adult(17) is False"),
                _slot(3, "assert is_adult(18) is True"),
                _slot(4, None),
                _slot(5, "assert is_adult(0) is False"),
                _slot(6, "assert is_adult(100) is True"),
                _slot(7, "assert is_adult(-1) is False"),
                _slot(8, "assert is_adult(19) is True"),
            ],
        },
        {
            "id": "smoke_survivor",
            "family": "logical",
            "entry": "positive",
            "golden": "def positive(x):\n    return x > 0\n",
            "mutant": "def positive(x):\n    return x >= 0\n",
            "slots": [
                _slot(1, "assert positive(1) is True"),
                _slot(2, "assert positive(-1) is False"),
                _slot(3, None),
                _slot(4, "assert positive(2) is True"),
                _slot(5, "assert positive(-2) is False"),
                _slot(6, "assert positive(3) is True"),
                _slot(7, "assert positive(-3) is False"),
                _slot(8, "assert positive(4) is True"),
            ],
        },
    ]
    results = []
    for example in examples:
        slots = prioritise_diverse_slots(
            example["slots"], example["entry"], diversity_mode
        )
        outcomes = evaluate_candidate_slots(
            slots, example["golden"], example["mutant"], example["entry"]
        )
        results.append(function_result(
            example["id"], example["family"], example["entry"], outcomes,
            source_name="local_synthetic_smoke", project="oneiros_smoke",
        ))
    summary = summarise_function_results(results)
    profile = {
        "feedback_rounds": 0,
        "diversity_mode": diversity_mode,
        "holdout_bug_family": None,
        "candidate_budget": 8,
        "k_values": [1, 2, 4, 8],
    }
    return {
        "mode": f"local_smoke_{diversity_mode}",
        "evaluation_split": "synthetic_local_smoke",
        "evaluation_scope_sha256": "local-smoke-v1",
        "tests_per_function": 8,
        "model_artifact_sha256": "deterministic-smoke-policy",
        "evaluation_profile_sha256": evaluation_profile_sha256(profile),
        "seed": 42,
        **summary,
        "function_results": results,
    }


def run_local_smoke() -> Dict[str, Any]:
    standard = _smoke_policy_results("none")
    diverse = _smoke_policy_results("ast")
    kill_rates = [
        standard["kill_at_k"][str(k)]["rate"] for k in (1, 2, 4, 8)
    ]
    pass_rates = [
        standard["pass_at_k"][str(k)]["rate"] for k in (1, 2, 4, 8)
    ]
    if kill_rates != sorted(kill_rates):
        raise RuntimeError("Kill@k smoke invariant failed")
    if pass_rates != sorted(pass_rates):
        raise RuntimeError("Pass@k smoke invariant failed")
    if standard["function_validation_killed"] != 2:
        raise RuntimeError("Synthetic oracle smoke produced an unexpected kill count")

    # Seed aggregation uses scope-compatible copies with controlled rates.
    seed_a = dict(standard)
    seed_b = dict(standard)
    seed_b["seed"] = 43
    seed_aggregate = aggregate_seed_results([seed_a, seed_b])
    comparison = compare_policy_results(standard, diverse)
    return {
        "status": "passed",
        "sealed_test_accessed": False,
        "standard_summary": {
            key: standard[key]
            for key in (
                "function_kill_rate", "kill_at_k", "pass_at_k",
                "candidate_redundancy", "diversity_mean",
            )
        },
        "diversity_ablation": comparison,
        "multi_seed_aggregation": seed_aggregate,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Write the predeclared Modal ablation plan")
    plan_parser.add_argument("--run-name", required=True)
    plan_parser.add_argument("--corpus-version", default=CANONICAL_CORPUS_VERSION)
    plan_parser.add_argument("--output", type=Path, required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run CPU-only synthetic pipeline smoke")
    smoke_parser.add_argument("--output", type=Path)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate compatible seed JSON files")
    aggregate_parser.add_argument("results", nargs="+", type=Path)
    aggregate_parser.add_argument("--output", type=Path)

    compare_parser = subparsers.add_parser("compare", help="Compare two policy result JSON files")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("evaluation", type=Path)
    compare_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "plan":
        payload = build_ablation_plan(args.run_name, args.corpus_version)
    elif args.command == "smoke":
        payload = run_local_smoke()
    elif args.command == "aggregate":
        payload = aggregate_seed_results([_read_json(path) for path in args.results])
    else:
        payload = compare_policy_results(
            _read_json(args.reference), _read_json(args.evaluation)
        )

    output = getattr(args, "output", None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
