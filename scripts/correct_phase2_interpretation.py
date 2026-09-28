"""Dated Phase 2 interpretation-correction receipt (2026-09-28).

The Phase 2 receipt and the quarantined first receipt are NOT modified.  This
receipt reads the Phase 2 receipt, quotes its numbers, and replaces three
over-strong interpretations with the reviewer-approved wording.  It also
splits the power question into (A) establishing a positive effect and (B)
establishing a lower bound of at least +3 points.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

PHASE2 = "results/sft_root_cause_phase2_decomposition_receipt.json"
QUARANTINED = "results/sft_root_cause/quarantine/phase2_first_run_SUPERSEDED.json"
OUTPUT = "results/sft_root_cause_phase2_interpretation_correction_2026-09-28.json"
Z = NormalDist().inv_cdf


def multiplier(se: float, effect: float, margin: float, power: float) -> float:
    """Cluster multiplier so that P(lower 95% bound > margin) = power, if the true
    difference equals ``effect`` and the SE scales as 1/sqrt(clusters)."""
    needed_se = (effect - margin) / (1.959964 + Z(power))
    return (se / needed_se) ** 2


def main() -> int:
    raw = (ROOT / PHASE2).read_bytes()
    receipt = json.loads(raw)
    m = receipt["primary_locked_validation"]["all_metrics"]

    def q(name):
        x = m[name]
        return {"base": x["a"], "sft": x["b"], "difference_points": x["difference_points"],
                "ci95_points": x["ci95_points"]}

    kill = m["kill_at_8"]
    d, se = kill["difference_points"], kill["bootstrap_se_points"]
    power = {
        "observed_kill_at_8_difference_points": d, "clustered_se_points": se,
        "clusters": receipt["primary_locked_validation"]["clusters"],
        "assumption": ("the true difference equals the observed +3.4346 points and the SE "
                       "scales as 1/sqrt(independent clusters)"),
        "A_effect_greater_than_zero": {
            "current_power": round(NormalDist().cdf(d / se - 1.959964), 3),
            "cluster_multiplier_50pct_power": round(multiplier(se, d, 0.0, 0.5), 2),
            "cluster_multiplier_80pct_power": round(multiplier(se, d, 0.0, 0.8), 2),
        },
        "B_lower_bound_at_least_3_points": {
            "current_power": round(NormalDist().cdf((d - 3.0) / se - 1.959964), 3),
            "cluster_multiplier_50pct_power": round(multiplier(se, d, 3.0, 0.5), 1),
            "cluster_multiplier_80pct_power": round(multiplier(se, d, 3.0, 0.8), 1),
            "why_so_large": ("the point estimate is only 0.4346 points above the +3-point "
                             "practical threshold"),
        },
        "correction": ("the ~296x figure in the Phase 2 receipt is B at 50% power (the "
                       "point estimate's lower bound just reaching +3). It is NOT the sample "
                       "increase needed to show a positive effect, which is question A"),
    }
    correction = {
        "schema_version": "oneiros_sft_root_cause_phase2_correction_v1",
        "date": "2026-09-28",
        "corrects": {"path": PHASE2, "sha256": hashlib.sha256(raw).hexdigest(),
                     "modified": False},
        "quarantined_first_receipt": {
            "path": QUARANTINED,
            "sha256": hashlib.sha256((ROOT / QUARANTINED).read_bytes()).hexdigest(),
            "modified": False},
        "accepted_conclusions": {
            "input_discrimination_improves": {
                "candidate_level_p_disc_given_execute": q("p_disc_given_execute"),
                "unique_candidate_p_disc_given_execute": q("p_disc_given_execute_unique"),
                "function_level_discrimination_recall": q("function_disc_recall_at_8")},
            "assertion_form_shifts_to_exact_equality": q("equality_oracle_given_policy_valid"),
            "no_material_overall_contract_degradation": {
                "parse": q("p_parse"), "policy_acceptance": q("p_policy_valid_given_parse"),
                "execute_given_parse": q("p_execute_given_parse")},
            "unstratified_oracle_validity_regression_partly_form_confounded":
                q("p_correct_oracle_given_disc"),
            "exploratory_exact_equality_subset": q("p_correct_oracle_given_disc_equality"),
        },
        "required_wording": {
            "value_prediction": ("No value-prediction improvement is demonstrated; a >=5-point "
                                 "improvement is excluded, but meaningful degradation remains "
                                 "possible."),
            "form_and_validity": ("The regression is consistent with the large assertion-form "
                                  "composition shift, but causal attribution is not possible "
                                  "from this post-hoc, model-selected stratum."),
            "kill_given_reference_valid": ("The +21.76-point change in P(kill | reference-valid) "
                                           "is descriptive only: reference validity is affected "
                                           "by treatment, so conditioning on it produces "
                                           "survivor/selection bias. It is not evidence that SFT "
                                           "became better at killing."),
        },
        "withdrawn_statements": [
            "'value prediction is unchanged'",
            "'most of the validity loss comes from the form change' (as a causal claim)",
            "any causal reading of the P(kill | reference-valid) change",
            "the ~296x multiplier described as the sample needed to establish the effect",
        ],
        "hypothesis_scope_corrections": {
            "H5": {"status": "weakened",
                   "scope": ("narrowed: syntactic/structural contract degradation is weakened. "
                             "Semantic-oracle validity remains unresolved.")},
            "H6": {"status": "weakened",
                   "scope": ("narrowed: progressive late-checkpoint degradation after step 100 is "
                             "weakened. A rapid early SFT-induced capability shift or "
                             "forgetting (steps 0-100) remains open.")},
            "H2": {"status": "strengthened",
                   "scope": ("input choice and assertion form moved under imitation; no "
                             "value-prediction improvement is demonstrated. Causal support "
                             "requires a Phase 4 arm.")},
            "H4": {"status": "open",
                   "scope": ("a train-only probe cannot support distribution shift; it can only "
                             "weaken it (see protocol amendment 2 decision rules).")},
        },
        "power": power,
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(correction, indent=1) + "\n")
                            .encode("utf-8"))
    print(json.dumps(power, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
