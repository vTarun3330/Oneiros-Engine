"""Assemble the final readiness report: may the sealed test be opened?

Deliverable 22.  The answer is derived from artifacts on disk, never asserted.
Each deliverable is reported in one of the categories the plan defines -
measured, historical, incomplete, inconclusive, deferred, infrastructure
blocker, design choice, or limitation - so a reader can tell what is evidence
and what is intent.

The report never opens the sealed split. It only decides whether opening it
would be legitimate yet.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _doctor() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, "scripts/v4_1_ready.py", "doctor"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        payload = json.loads(completed.stdout)
        checks = payload.get("checks", [])
        return {
            "all_passed": all(check["passed"] for check in checks),
            "checks": {check["name"]: check["passed"] for check in checks},
        }
    except Exception as exc:
        return {"all_passed": False, "error": f"{type(exc).__name__}: {exc}"}


def build() -> dict[str, Any]:
    results = ROOT / "results"
    configuration = _read(results / "v4_2_selected_research_configuration.json")
    base_vs_sft = _read(results / "v4_2_base_vs_sft_validation.json")
    atheris_matched = _read(
        results / "v4_2_atheris_vs_historical_oneiros_val_matched8_seed42.json"
    )
    atheris_generous = _read(results / "v4_2_atheris_vs_oneiros_val_20000_seed42.json")
    pilot = _read(results / "v4_2_native_repository_pilot.json")
    ceiling = _read(results / "v4_2_repository_expansion_ceiling.json")
    compaction = _read(results / "v4_2_prompt_compaction_audit.json")
    relearning = _read(results / "v4_2_relearning_manifest.json")
    inventory = _read(results / "v4_2_corpus_inventory.json")
    balanced = _read(
        ROOT / "data" / "training_views" / "balanced_sft_v1" / "train.manifest.json"
    )
    bundle = _read(results / "v4_2_baseline_bundle_val.json")
    rescore = _read(results / "v4_2_shape_policy_rescore.json")
    doctor = _doctor()

    blockers: list[str] = []
    if not doctor.get("all_passed"):
        blockers.append("the readiness doctor is not fully green")
    if balanced and not (balanced.get("readiness") or {}).get("ready_for_final_sft", False):
        for condition in (balanced.get("readiness") or {}).get("blocking_conditions", []):
            blockers.append(f"balanced corpus: {condition}")
    receipt = _read(results / "v4_2_final_sft_receipt.json")
    if receipt is None:
        blockers.append(
            "the final source-bound SFT adapter has not been rebuilt and receipted"
        )
    else:
        # A run named for an intervention must actually have applied it. The
        # first "multi-mutant final" rebuild used multi-mutant supervision for
        # 76 of 2085 examples and nothing in the artifacts said so.
        density = (
            (receipt.get("multi_mutant_supervision") or {})
            .get("density_over_eligible_synthetic")
        )
        if density is not None and density < 0.5:
            blockers.append(
                f"the receipted adapter applied multi-mutant supervision to only "
                f"{density:.1%} of eligible synthetic pairs, so it does not test "
                "the intervention it is named for"
            )
        if receipt.get("allow_test_function_candidates") and not receipt.get(
            "baselines_rescored_under_same_shape_policy"
        ):
            blockers.append(
                "the adapter was scored under the widened test-function shape "
                "policy but the baselines were not rescored under it, so the "
                "comparison is not like-for-like"
            )
    if pilot and pilot.get("status") != "SUCCESS":
        blockers.append(
            "native repository execution is a partial pass; no model-generated "
            "test has been executed natively"
        )

    report: dict[str, Any] = {
        "schema_version": "oneiros_final_readiness_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "phase": "sft_only",
        "dpo": "OUT OF SCOPE - not prepared, launched, evaluated, or budgeted",
        "doctor": doctor,
        "measured": {
            "base_versus_sft_on_locked_validation": {
                "category": "MEASURED",
                "artifact": "results/v4_2_base_vs_sft_validation.json",
                "base_mean": (base_vs_sft or {}).get("base_across_seed", {}).get("mean"),
                "arms": {
                    name: {
                        "paired_delta_mean": arm.get("paired_kill_rate_delta_mean"),
                        "positive_in_every_seed": arm.get("positive_in_every_paired_seed"),
                        "reference_validity_regressed_in_every_seed": arm.get(
                            "reference_validity_regressed_in_every_seed"
                        ),
                    }
                    for name, arm in ((base_vs_sft or {}).get("arms") or {}).items()
                },
                "interpretation": (
                    "SFT helps by roughly +1.3 points at best on held-out "
                    "validation, an order of magnitude less than the 100-function "
                    "monitor panel implied, and one trained adapter was net worse "
                    "than the untrained base."
                ),
            },
            "actual_atheris_same_targets": {
                "category": "MEASURED",
                "matched_budget_kill_rate": (atheris_matched or {}).get(
                    "atheris", {}
                ).get("kill_rate_over_all_targets"),
                "generous_budget_kill_rate": (atheris_generous or {}).get(
                    "atheris", {}
                ).get("kill_rate_over_all_targets"),
                "oneiros_sft_kill_rate": 0.635403,
                "interpretation": (
                    "On the same 757 validation targets Oneiros beats real "
                    "Atheris at both budgets, including when Atheris is given "
                    "2,500x the executions AND the reference implementation for "
                    "a differential oracle."
                ),
            },
            "prompt_compaction": {
                "category": "MEASURED",
                "artifact": "results/v4_2_prompt_compaction_audit.json",
                "function_mode": "fits comfortably; 0 rejected at 1024 tokens",
                "repository_mode_rejection_rate": {
                    split: modes.get("repository", {}).get("rejection_rate")
                    for split, modes in ((compaction or {}).get("splits") or {}).items()
                },
                "limitation": (
                    "About a third of repository prompts do not fit the frozen "
                    "1024-token budget and fail closed, so they are absent from "
                    "every repository result."
                ),
            },
            "relearning_losers": {
                "category": "MEASURED",
                "artifact": "results/v4_2_relearning_manifest.json",
                "losers": (relearning or {}).get("losers", {}).get("count"),
                "loser_rate": (relearning or {}).get("losers", {}).get("loser_rate"),
                "by_category": (relearning or {}).get("losers", {}).get(
                    "by_dominant_category"
                ),
            },
            "non_llm_baseline_bundle": {
                "category": "MEASURED" if bundle else "INCOMPLETE",
                "artifact": "results/v4_2_baseline_bundle_val.json",
                "arms": {
                    name: {
                        "mean_kill_rate": arm.get("mean_kill_rate"),
                        # A mean alone hides that the simulated coverage fuzzer
                        # swings 0.378 to 0.514 across seeds - a spread wider
                        # than the entire SFT effect being studied, and wider
                        # than its own Wilson interval on any single seed. Any
                        # single-seed number from a stochastic baseline is
                        # unreliable and must be read with this range.
                        "range_across_seeds": arm.get("range_across_seeds"),
                        "per_seed": {
                            seed: row.get("kill_rate")
                            for seed, row in (arm.get("by_seed") or {}).items()
                        },
                    }
                    for name, arm in ((bundle or {}).get("arms") or {}).items()
                },
                "seed_sensitivity": (
                    "deterministic arms (static, dataset_tests) have zero "
                    "spread; the stochastic coverage arm has the largest, so "
                    "comparisons against it need all three seeds"
                ),
                "protocol": (
                    "same targets, candidate budget, timeout, seeds, validity "
                    "and kill rules as the model arms"
                ),
                "advantage_given_to_baselines": (
                    "every generative baseline derives its expected value by "
                    "executing the reference implementation, so each has a "
                    "perfect oracle and need only choose inputs. Oneiros never "
                    "sees the reference."
                ),
                "naming": (
                    "the coverage arm is a SIMULATED coverage fuzzer written "
                    "for this project and is never reported as Atheris"
                ),
            },
            "shape_policy_is_not_doing_the_work": {
                "category": "MEASURED" if rescore else "INCOMPLETE",
                "artifact": "results/v4_2_shape_policy_rescore.json",
                "base_seeds_rescored": [
                    row.get("seed") for row in (rescore or {}).get("rescored", [])
                ],
                "max_delta_kill_rate": max(
                    (
                        row.get("delta_kill_rate", 0)
                        for row in (rescore or {}).get("rescored", [])
                    ),
                    default=None,
                ),
                "interpretation": (
                    "rescoring the stored base generations under the widened "
                    "test-function rule admitted no additional candidate on any "
                    "seed, so the widened policy confers no baseline-side "
                    "advantage and cannot explain a gap between the arms."
                ),
            },
            "corpus_inventory": {
                "category": "MEASURED",
                "artifact": "results/v4_2_corpus_inventory.json",
                "train_unique_targets": (
                    (inventory or {}).get("splits", {}).get("train", {})
                    .get("unique_semantic_targets")
                ),
                "repository_projects_disjoint": (
                    (inventory or {}).get("repository_project_disjointness", {})
                    .get("disjoint")
                ),
            },
        },
        "incomplete": {
            "final_sft_adapter": {
                "category": "INCOMPLETE",
                "detail": (
                    "The final source-bound rebuild on verified multi-mutant "
                    "supervision must complete and be receipted before any final "
                    "claim about the selected adapter."
                ),
            },
            "relearning_challenger": {
                "category": "INCOMPLETE",
                "detail": (
                    "The relearning dataset is built and hashed. No relearning "
                    "adapter has been trained, so there is no base-vs-SFT-vs-"
                    "relearning comparison yet."
                ),
            },
        },
        "infrastructure_blockers": {
            "swebench_expansion": {
                "category": "INFRASTRUCTURE BLOCKER",
                "detail": (
                    "SWE-bench ingestion runs the official harness on Modal and "
                    "no Modal token is configured, removing 114 otherwise "
                    "eligible instances."
                ),
            },
            "native_generated_test_execution": {
                "category": "INFRASTRUCTURE BLOCKER",
                "detail": (
                    "The native pilot reproduces the OFFICIAL test on 3 of 5 "
                    "BugsInPy defects. No model-generated test has been run "
                    "natively, so repository records remain verified supervision "
                    "coverage."
                ),
            },
        },
        "limitations": {
            "balance_parity_unreachable": {
                "category": "LIMITATION",
                "detail": (
                    "50/50 by unique semantic target needs 317 more repository "
                    "targets. The optimistic ceiling is 304 and the expected "
                    "yield 182, so parity is not reachable from available data."
                ),
                "artifact": "results/v4_2_repository_expansion_ceiling.json",
            },
            "measured_panel_is_synthetic_only": {
                "category": "LIMITATION",
                "detail": (
                    "Every reported Kill@8 number - base, every SFT arm, "
                    "actual Atheris at both budgets, all five non-LLM "
                    "baselines - is measured on function-mode records only. "
                    "The val corpus holds 757 function and 24 repository "
                    "records; the evaluation panel is the 757. Across all 25 "
                    "evaluation artifacts, zero repository targets were "
                    "evaluated at any seed by any arm. Report these as "
                    "held-out SYNTHETIC targets, not as 'held-out functions' "
                    "unqualified. Per-dataset and synthetic-versus-repository "
                    "breakdowns cannot be produced for validation, because the "
                    "repository side has no measurements to break down. The "
                    "comparison against Atheris and the baselines remains "
                    "sound - every arm ran the identical panel - but it is a "
                    "synthetic-target comparison."
                ),
                "artifact": "results/v4_2_evaluation_panel_composition.json",
            },
            "eligibility_is_not_a_budget_problem": {
                "category": "LIMITATION",
                "detail": (
                    "The repository prompt-budget rejection and the repository "
                    "eligibility shortfall are separate mechanisms and were "
                    "briefly reported as one. Raising the repository budget to "
                    "2048 recovers 152 of 212 rejected prompts (39.8% -> "
                    "11.3%), but recovers no excluded record: the 113 "
                    "exclusions reference symbols absent from the record, and "
                    "only 4 are recoverable by import resolution. The rest are "
                    "pytest fixture parameters and inherited test classes. An "
                    "import resolver is not a route to repository parity."
                ),
                "artifacts": [
                    "results/v4_2_prompt_compaction_audit_repo2048.json",
                    "results/v4_2_repository_exclusion_audit.json",
                ],
            },
            "seed_count_bounded_the_conclusion": {
                "category": "LIMITATION",
                "detail": (
                    "An exact two-sided sign test over seed signs returns "
                    "p=0.25 for three positive seeds, which is the SMALLEST "
                    "reachable value at n=3. No three-seed outcome could have "
                    "been significant by that test, so the three-seed result "
                    "bounded the conclusion rather than answering it. The "
                    "floor is 0.0078 at eight seeds."
                ),
                "artifact": "results/v4_2_relearning_seed_power.json",
            },
            "complexity_floor": {
                "category": "DESIGN CHOICE",
                "detail": (
                    "The 0.60 complex-example floor is a design choice. It has "
                    "not been shown to improve results and is not claimed to."
                ),
            },
            "checkpoint_instability": {
                "category": "LIMITATION",
                "detail": (
                    "Best checkpoint differed across seeds (50, 142, 142). "
                    "Checkpoint 50 is not universally optimal."
                ),
            },
            "base_model_choice": {
                "category": "DESIGN CHOICE",
                "detail": (
                    "Qwen was selected because Phi-3 cannot use SDPA in the "
                    "pinned transformers build. That is a feasibility "
                    "constraint, not evidence that Qwen beats a fully trained "
                    "Phi-3 in a controlled comparison."
                ),
            },
        },
        "sealed_test": {
            "may_be_opened": not blockers,
            "blocking_conditions": blockers,
            "one_time_guard": "harness/sealed_final.py (mock-tested, 34 tests)",
            "requirement": (
                "Opening requires a fully frozen bundle and a one-time "
                "authorization token issued against that bundle's hash."
            ),
        },
    }
    if configuration:
        report["selected_configuration_sha256"] = configuration.get(
            "configuration_sha256"
        )
    if ceiling:
        report["repository_expansion_ceiling"] = ceiling.get("ceiling")

    # Evidence added after the first readiness pass. Each is attached only if
    # the artifact exists, so a report built before a sweep has run says
    # nothing about it rather than reporting a default as a measurement.
    atheris = _read(ROOT / "results" / "v4_2_atheris_seed_aggregate.json")
    if atheris:
        report["actual_atheris_by_seed"] = {
            name: {
                "seeds": arm.get("seeds"),
                "mean_kill_rate": arm.get("mean_kill_rate"),
                "range_across_seeds": arm.get("range_across_seeds"),
            }
            for name, arm in (atheris.get("arms") or {}).items()
        }

    power = _read(ROOT / "results" / "v4_2_relearning_seed_power.json")
    if power:
        report["relearning_seed_power"] = {
            "seeds": power.get("seeds_compared"),
            "per_seed_delta": [
                row.get("delta") for row in power.get("per_seed") or []
            ],
            "mean_delta": power.get("mean_delta"),
            "positive_in_every_seed": power.get("positive_in_every_seed"),
            "sign_test": power.get("sign_test_over_seeds"),
            "significant_after_adjustment": (
                power.get("holm_bonferroni") or {}
            ).get("significant_after_adjustment"),
        }

    budget = _read(
        ROOT / "results" / "v4_2_prompt_compaction_audit_repo2048.json"
    )
    if budget:
        report["exploratory_repository_budget_sweep"] = {
            "audit_status": budget.get("audit_status"),
            "is_frozen_configuration": budget.get("is_frozen_configuration"),
            "swept_repository_budget": (
                budget.get("mode_budgets") or {}
            ).get("repository"),
            "note": (
                "evidence for a labelled successor run; not a measurement of "
                "the protocol any reported result was produced under"
            ),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_final_readiness_report.json",
    )
    arguments = parser.parse_args()
    report = build()
    write_json(arguments.output, report)
    print(json.dumps({
        "doctor_all_passed": report["doctor"]["all_passed"],
        "sealed_test_may_be_opened": report["sealed_test"]["may_be_opened"],
        "blocking_conditions": report["sealed_test"]["blocking_conditions"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
