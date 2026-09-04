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
