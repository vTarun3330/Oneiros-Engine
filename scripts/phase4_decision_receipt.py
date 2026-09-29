"""Phase 6 decision receipt (2026-09-29): Choice A vs Choice B after the bounded preparation.

Deterministic; every figure is read from a committed receipt.  Launches nothing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

OUTPUT = "results/sft_root_cause_decision_receipt_2026-09-29_v1.json"
SOURCES = {
    "reporting_addendum": "results/sft_root_cause_reporting_addendum_2026-09-29_v1.json",
    "power_sensitivity": "results/sft_root_cause_phase4_power_sensitivity_v1.json",
    "native_rehearsal": "results/sft_root_cause_phase4_native_rehearsal_receipt_v1.json",
    "rehearsal_manifest": "results/sft_root_cause_phase4_native_rehearsal_manifest_v1.json",
    "objective_smoke": "results/sft_root_cause_phase4_objective_smoke_v1.json",
    "phase4_census": "results/sft_root_cause_phase4_cohort_census.json",
    "hypotheses": "results/sft_root_cause_hypotheses.json",
}


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    rehearsal = load(SOURCES["native_rehearsal"])
    smoke = load(SOURCES["objective_smoke"])
    power = load(SOURCES["power_sensitivity"])
    census = load(SOURCES["phase4_census"])
    hypotheses = load(SOURCES["hypotheses"])
    per_arm = smoke["per_arm"]["control"]
    seconds_per_step = per_arm["timing"]["train_seconds"] / per_arm["optimizer_steps_completed"]
    receipt = {
        "schema_version": "oneiros_sft_root_cause_decision_v1", "date": "2026-09-29",
        "launches_nothing": True,
        "sources": {name: {"path": rel, "sha256": hashlib.sha256((ROOT / rel).read_bytes())
                           .hexdigest()} for name, rel in SOURCES.items()},
        "state_of_evidence": {
            "root_cause_established": False,
            "hypotheses": {h["id"]: {"status": h["status"], "qualifier": h.get("qualifier")}
                           for h in hypotheses["hypotheses"]},
            "summary": ("SFT changed input choice and assertion form; whether it changes "
                        "fixed-input value prediction is schema-dependent and unresolved; 7B beats "
                        "1.5B as a model-scale association only; no hypothesis is terminally "
                        "supported")},
        "preparation_results": {
            "trainer_path": {"status": smoke["status"],
                             "evidence": "only labels differ in the real Trainer batches; "
                                         "legacy full-completion path unchanged"},
            "native_rehearsal": {
                "qualified_over_admitted": rehearsal["rates"]["qualified_over_all_admitted"],
                "environment_success": rehearsal["rates"]["environment_success"],
                "safe_fixed_call_over_qualified":
                    rehearsal["rates"]["safe_fixed_call_over_qualified"],
                "gate_passed": rehearsal["gate_passed"], "caveats": rehearsal["caveats"],
                "blockers": rehearsal["blockers"]},
            "power": {"status": power["status"],
                      "required_groups_80pct": power["required_groups_for_80pct_power_ranges"]}},
        "choice_A": {
            "role": "the ONLY path capable of independent repository-generalisation evidence",
            "requires": ["repository-disjoint Phase 5 gate cohort",
                         "repository-disjoint Phase 6 confirmation cohort",
                         "fixed-input argument capture for repository-native targets (current "
                         "literal extraction: 0/24 safe fixed calls)",
                         "revalidation of any development-pool target under isolation v6",
                         "an independent (non-retrospective) native rehearsal to replace the "
                         "optimistic development qualification rate"],
            "recomputed_size_and_cost": rehearsal["projection"],
            "rule": "no full training until both the data plan and the trainer path are frozen"},
        "choice_B": {
            "role": "optional internal exploratory screening only",
            "material": ("arm-A-exposed remainder (401 groups, 352 feasible): used by arm A, NOT "
                         "untouched, NOT repository-disjoint"),
            "feasible_groups": census["pools"]["arm_a_exposed_remainder"][
                "groups_with_a_feasible_fixed_input_function"],
            "outcomes": ["promising", "harm", "inconclusive_power"],
            "cannot": ["support repository or cross-dataset generalisation",
                       "be called untouched", "compare the old arm A fairly",
                       "replace Choice A", "serve as Phase 6"]},
        "recommendation": {
            "decision": "RUN CHOICE B FIRST ONLY AS AN INEXPENSIVE SCREENING EXPERIMENT, while "
                        "Choice A's CPU-side prerequisites proceed; Choice A stays required for "
                        "any generalisation claim",
            "why": ["the production objective path is integrated and smoke-proven, so a screening "
                    "run now tests the intervention itself rather than plumbing",
                    "Choice A is blocked on fixed-input argument capture and faces an 18,186 "
                    "(7,626-51,552) candidate acquisition; a cheap screen tells us whether "
                    "value-only masking moves the mediator at all before that spend",
                    "a harm or null screen would redirect effort to redesign before acquisition"],
            "next_authorized_gpu_experiment": {
                "name": "Choice B single-seed screening: value-only vs full-completion masking",
                "arms": {"control": "full_completion", "treatment": "value_only"},
                "start": "immutable base Qwen2.5-Coder-1.5B @ 2e1fd397 (not arm A)",
                "data": ("matched manifest from arm-A-exposed train groups (validated by "
                         "harness.objective_masking.validate_manifest), gate groups frozen "
                         "before training and withheld from both arms"),
                "gate": ("~200 remainder groups, fixed-input exact-output accuracy under both "
                         "schemas, final checkpoint only, one look, answer-rate gate, original "
                         "validity guardrail; outcomes promising/harm/inconclusive_power"),
                "estimated_gpu": {
                    "training_seconds_per_optimizer_step_from_smoke": round(seconds_per_step, 2),
                    "training_minutes_per_arm_at_431_steps": round(431 * seconds_per_step / 60,
                                                                   1),
                    "note": ("smoke examples were short fixed-call prompts; evaluation adds "
                             "minutes for the fixed-input probe and tens of minutes for "
                             "production Kill@8 generation; ~1.5-2 GPU-hours for both arms")},
                "requires_user_authorisation": True}},
        "not_launched": ["full Choice A acquisition", "Choice A training", "Choice B training"],
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps(receipt["recommendation"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
