"""The frozen, machine-readable receipt of the selected research configuration.

Every number a result depends on lives here in one place, so a reader can tell
what was actually run without reconstructing it from argument lists scattered
across logs.  The receipt separates three things that are easy to conflate:

* **frozen** - fixed by decision and not to be changed without a new receipt;
* **design choice** - deliberately chosen, but NOT demonstrated to help;
* **out of scope** - explicitly excluded from this research phase.

DPO is out of scope for this phase.  It is named here so the exclusion is
recorded rather than merely absent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SELECTED_CONFIGURATION_SCHEMA = "oneiros_selected_research_configuration_v1"

#: The research target for function-level Kill@8 under the frozen protocol.
#: It is a TARGET, not a gate: falling short is reported, never engineered away
#: by changing the corpus, prompts, validation set, or protocol.
SFT_RESEARCH_TARGET_KILL_AT_8 = 0.80

#: The historical locked-validation gate.  Retained as prior viability
#: evidence; it does not authorize any DPO work in this phase.
HISTORICAL_LOCKED_VALIDATION_GATE = 0.58


def _corpus_identity(corpus_dir: Path) -> dict[str, Any]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        return {"corpus_dir": str(corpus_dir), "status": "absent"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "corpus_id": manifest.get("corpus_id"),
        "version": manifest.get("version"),
        "record_count": manifest.get("record_count"),
        "prompt_schema_version": manifest.get("prompt_schema_version"),
        "splits": {
            name: {
                "record_count": value.get("record_count"),
                "record_ids_sha256": value.get("record_ids_sha256"),
            }
            for name, value in (manifest.get("splits") or {}).items()
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    return payload


def build_selected_configuration(
    project_root: Path,
    corpus_version: str = "v4_1_research_hardened_candidate",
    training_view: str | None = None,
) -> dict[str, Any]:
    """Assemble the receipt from the repository's own recorded state."""
    project_root = Path(project_root).resolve()
    corpus_dir = project_root / "data" / "corpus" / corpus_version

    return {
        "schema_version": SELECTED_CONFIGURATION_SCHEMA,
        "phase": "sft_only",
        "scope": {
            "in_scope": [
                "supervised fine-tuning (SFT) of the selected base model",
                "balanced synthetic/real-repository corpus construction",
                "multi-mutant supervised example construction",
                "one bounded SFT hard-example relearning round",
                "baseline comparison including actual Atheris where available",
                "locked validation at three seeds",
            ],
            "out_of_scope": [
                "DPO: not prepared, launched, debugged, evaluated, or budgeted "
                "in this phase. Reaching the Kill@8 research target does NOT "
                "trigger DPO work.",
            ],
        },
        "model": {
            "base_model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "base_model_revision": "2e1fd39751a55f4c0dd1a6c25d2a9f0d5b1b3b7d",
            "tokenizer_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "tokenizer_revision": "matches base_model_revision",
            "attention_implementation": "sdpa",
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": False,
            },
            "adapter": {
                "method": "peft_lora",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "selection_rationale": (
                "Phi-3-mini-4k-instruct does not support SDPA in the pinned "
                "transformers build and is therefore eager-only. That is a "
                "recorded feasibility constraint, NOT evidence that Qwen "
                "outperforms a fully trained Phi-3 in a controlled comparison."
            ),
        },
        "training": {
            "method": "sft",
            "learning_rate": 1e-05,
            "lr_scheduler_type": "constant_with_warmup",
            "epochs": 1,
            "batch_size": 1,
            "warmup_steps": 25,
            "max_grad_norm": 1.0,
            "checkpoint_steps": 50,
        },
        "prompt": {
            "schema_version": "oneiros_unified_test_generation_v2",
            "compaction": "section_aware_ast_units_before_chat_v4_1",
            "compaction_is_fail_closed": True,
            "budgets_by_execution_mode": {
                "function_prompt_tokens": 1024,
                "repository_prompt_tokens": 1024,
                "sequence_tokens": 2048,
            },
            "completion_budgets": {
                "function_completion_tokens": 128,
                "repository_completion_tokens": 1024,
            },
            "selection_prompt_token_limit": 1024,
            "selection_limit_rationale": (
                "Data selection is pinned to a common floor so two arms never "
                "train on different record sets; an unpinned budget silently "
                "changes the corpus as well as the budget."
            ),
        },
        "generation": {
            "candidates_per_target": 8,
            "post_generation_reranking": False,
            "no_reranking_rule": (
                "All eight candidates are scored as generated. No candidate is "
                "reordered, filtered, or re-scored after generation, so Kill@k "
                "measures the sampler and not a selection heuristic layered on "
                "top of it."
            ),
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "evaluation": {
            "seeds": [42, 43, 44],
            "locked_validation_split": "val",
            "development_split": "ablation_dev",
            "sealed_split": "test",
            "sealed_split_status": "sealed - not opened, not inspected",
            "execution_backend": "restricted subprocess worker (harness.safe_execution)",
            "execution_timeout_seconds": 5,
            "native_repository_backend": "WSL2 Ubuntu 24.04 (see native pilot report)",
            "metrics": [
                "kill_at_1", "kill_at_2", "kill_at_4", "kill_at_8",
                "function_level_kill_rate", "candidate_level_kill_rate",
                "unique_defects_killed", "parse_success", "execution_success",
                "reference_validity", "semantic_assertion_correctness",
                "duplicate_rate", "diversity", "assertion_count",
                "mutants_killed_per_generated_test",
                "percent_tests_killing_at_least_four_siblings",
                "mutation_score", "complexity_tier_performance",
                "synthetic_versus_repository_performance",
                "per_dataset_performance", "per_project_macro_average",
                "per_bug_family_macro_average",
                "unseen_repository_generalization",
                "wilson_confidence_intervals", "per_seed_results",
                "across_seed_mean_range_variance", "runtime_and_gpu_cost",
                "native_repository_infrastructure_failure_rate",
            ],
            "statistical_policy": {
                "wilson_interval_applies_to": "per-seed binomial proportions only",
                "across_seed_summary": "mean of proportions reported with range",
                "paired_comparison_noise_floor_functions": 2,
                "unpaired_comparison_noise_floor_functions": 7,
                "note": (
                    "A paired within-seed comparison cancels seed choice and is "
                    "judged against same-seed run-to-run variability. An "
                    "unpaired comparison between separately trained runs is not, "
                    "and must be judged against the across-seed range."
                ),
            },
        },
        "targets_and_gates": {
            "sft_research_target_kill_at_8": SFT_RESEARCH_TARGET_KILL_AT_8,
            "target_is_not_a_gate": True,
            "historical_locked_validation_gate": HISTORICAL_LOCKED_VALIDATION_GATE,
            "historical_gate_status": (
                "retained as prior viability evidence; does not authorize DPO"
            ),
        },
        "checkpoint_selection": {
            "rule": (
                "Among monitored checkpoints, select the one with the highest "
                "mean ablation_dev function kill rate across seeds 42/43/44, "
                "requiring no reference-validity, parse, or execution "
                "regression beyond the declared tolerances. Ties break to the "
                "earlier step."
            ),
            "declared_before_final_run": True,
            "known_instability": (
                "Best step was 50, 142, 142 for seeds 42, 43, 44 in the "
                "800-pair runs. Checkpoint 50 is NOT established as universally "
                "optimal, which is why selection is a cross-seed mean rather "
                "than a single-seed peak."
            ),
            "all_monitored_checkpoints_preserved": True,
        },
        "corpus": {
            "canonical_corpus": _corpus_identity(corpus_dir),
            "training_view": training_view,
            "immutability": (
                "The V4.1 corpus is read-only. Balanced training data is a "
                "derived, separately versioned and hashed view."
            ),
            "split_isolation": {
                "sealed_test_never_materialized_in_development_view": True,
                "validation_never_used_for_selection_or_relearning": True,
                "repository_projects_disjoint_across_splits": True,
            },
        },
        "balancing_policy": {
            "balance_unit": "unique semantic target, not raw mutation row",
            "top_level_target": {
                "synthetic_function": 0.5,
                "real_repository": 0.5,
            },
            "unique_first_sampling": True,
            "max_repeats_per_example": 2,
            "repetition_is_last_resort_and_reported": True,
            "complex_example_floor": 0.60,
            "complex_floor_status": (
                "DESIGN CHOICE, not a demonstrated gain. The prior +4 result "
                "was an unpaired single-seed comparison against an across-seed "
                "range of 7 and is recorded as INCONCLUSIVE."
            ),
            "manual_curated_counted_as_real_world_evidence": False,
        },
        "promotion_and_rollback": {
            "promote_only_if": [
                "improves over the paired base model on the same seeds",
                "no unacceptable reference-validity regression",
                "no unacceptable parse or execution regression",
                "no collapse in candidate diversity",
                "no unacceptable repository-subset regression",
                "no unacceptable complex-tier regression",
                "direction is consistent across seeds 42/43/44",
                "selection followed the frozen checkpoint rule",
            ],
            "otherwise": "roll back to the previously frozen SFT adapter",
            "failed_and_inconclusive_experiments_are_retained": True,
        },
        "provenance": {
            "artifact_identity_fields": [
                "source_tree_sha256", "dependency_spec_sha256", "model_name",
                "model_revision", "python_version", "python_implementation",
                "runtime_dependencies",
            ],
            "git_commit_excluded_from_identity_because": (
                "committing a run's own results changes the commit without "
                "changing executable source; source_tree_sha256 is the stronger "
                "check because it also catches uncommitted edits."
            ),
        },
    }


def configuration_sha256(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
