"""Reporting addendum (2026-09-29): multiplicity status, power-evidence status and
acquisition arithmetic.  CPU only; additive; no historical receipt is modified.

Every number is recomputed here from committed sources, and every statement is
tagged as ``observed``, ``simulation_assumption`` or ``projection``.  Output is
deterministic (no timestamps), so a rerun is byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically

OUTPUT = "results/sft_root_cause_reporting_addendum_2026-09-29_v1.json"
DESIGN_3A = "results/sft_root_cause_phase3a_design_receipt.json"
V3_3A = "results/sft_root_cause_phase3a_result_receipt_v3.json"
POWER = "results/sft_root_cause_phase4_power_analysis_v1.json"
CHOICES = "docs/SFT_ROOT_CAUSE_PHASE4_CHOICES.md"
APRIME = "results/v4_3_repository_native_aprime_fresh_confirmation_pilot.json"
PROTOCOL_NATIVE = "docs/REPOSITORY_NATIVE_EVALUATION_PROTOCOL.md"
TARGET_PER_COHORT = 350
MAX_PER_REPOSITORY = 12        # frozen D3 policy
MIN_REPOSITORIES_PER_COHORT = 34   # frozen D3 policy ("N = 400 from at least 34 repositories")
UNAUTHENTICATED_CALLS_PER_HOUR = 60


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def acquisition() -> dict:
    pilot = load(APRIME)
    admitted = pilot["counts"]["admitted"]
    inspected = pilot["counts"]["candidates_inspected"]
    point = admitted / inspected
    low, high = pilot["yield"]["wilson_95"]
    api = pilot["identity"]["api"]
    calls_per_candidate = api["api_calls"] / inspected

    def per(n, rate):
        return math.ceil(n / rate)

    one = {"point": per(TARGET_PER_COHORT, point),
           "range_at_wilson_bounds": [per(TARGET_PER_COHORT, high), per(TARGET_PER_COHORT, low)]}
    both = {"point": 2 * one["point"],
            "range_at_wilson_bounds": [2 * one["range_at_wilson_bounds"][0],
                                       2 * one["range_at_wilson_bounds"][1]]}

    def hours(candidates):
        return round(candidates * calls_per_candidate / UNAUTHENTICATED_CALLS_PER_HOUR, 1)

    cap_minimum = math.ceil(TARGET_PER_COHORT / MAX_PER_REPOSITORY)
    per_cohort_repositories = max(cap_minimum, MIN_REPOSITORIES_PER_COHORT)
    return {
        "observed": {
            "source": APRIME, "admitted": admitted, "inspected": inspected,
            "admission_yield_point": round(point, 5),
            "admission_yield_wilson_95": [low, high],
            "api_calls": api["api_calls"], "api_calls_per_candidate": round(calls_per_candidate, 4),
            "elapsed_seconds_for_pilot": api["elapsed_seconds"],
            "rate_limit_wait_seconds": api["rate_limit_and_backoff_wait_seconds"],
            "note": ("admission precedes native qualification; admitted yield is an UPPER bound "
                     "on the mined-to-qualified yield")},
        "projection": {
            "assumptions": ["the pilot admission yield applies to newly mined repositories",
                            "API calls scale linearly with candidates at the pilot ratio",
                            "unauthenticated REST limit of 60 calls/hour",
                            "no native-qualification losses (all totals are before them)"],
            "target_functions_per_cohort": TARGET_PER_COHORT,
            "candidates_per_cohort": one,
            "candidates_gate_plus_phase6_total": both,
            "unauthenticated_api_hours_per_cohort": {
                "point": hours(one["point"]),
                "range": [hours(one["range_at_wilson_bounds"][0]),
                          hours(one["range_at_wilson_bounds"][1])]},
            "unauthenticated_api_hours_both_cohorts": {
                "point": hours(both["point"]),
                "range": [hours(both["range_at_wilson_bounds"][0]),
                          hours(both["range_at_wilson_bounds"][1])]},
            "repository_minimum": {
                "formula": "ceil(N_gate / 12) + ceil(N_confirmation / 12)",
                "cap_only_minimum_per_cohort": cap_minimum,
                "cap_only_minimum_both": 2 * cap_minimum,
                "policy_minimum_per_cohort": MIN_REPOSITORIES_PER_COHORT,
                "policy_source": f"{PROTOCOL_NATIVE} (D3: at least 34 repositories, at most 12 "
                                 "targets per repository)",
                "required_per_cohort": per_cohort_repositories,
                "required_distinct_repositories_both_disjoint_cohorts":
                    2 * per_cohort_repositories},
            "statement": (f"About {one['point']:,} mined candidates per {TARGET_PER_COHORT} "
                          f"admitted functions at the point estimate (range "
                          f"{one['range_at_wilson_bounds'][0]:,}-"
                          f"{one['range_at_wilson_bounds'][1]:,}); about {both['point']:,} for a "
                          f"{TARGET_PER_COHORT}-function gate plus a similar Phase 6 cohort. "
                          f"About {hours(one['point'])} unauthenticated API hours per cohort and "
                          f"{hours(both['point'])} for both, before native-qualification losses. "
                          f"Two repository-disjoint cohorts need at least "
                          f"{2 * per_cohort_repositories} distinct repositories.")},
    }


def main() -> int:
    v3 = load(V3_3A)
    family = v3["declared_family_zero_effect_tests"]
    addendum = {
        "schema_version": "oneiros_sft_root_cause_reporting_addendum_v1",
        "date": "2026-09-29", "model_calls": 0, "gpu_used": False, "training": False,
        "historical_receipts_modified": False,
        "inputs": {p: sha(p) for p in (DESIGN_3A, V3_3A, POWER, CHOICES, APRIME,
                                        PROTOCOL_NATIVE)},
        "multiplicity_3a": {
            "observed": {
                "design_text_verbatim": load(DESIGN_3A)["inference"]["multiplicity"],
                "design_enumerates_contrasts": False,
                "six_test_family_used_in_v3": list(family),
                "holm_adjusted_monte_carlo_p": {k: f["holm_adjusted_monte_carlo_p"]
                                                for k, f in family.items()}},
            "status": "POST-HOC OPERATIONALISATION / SENSITIVITY ANALYSIS",
            "correction": ("the design names the family only as 'the primary H4 and H1 "
                           "contrasts' and does not enumerate them; the six contrasts were chosen "
                           "when v3 was written, after the results were known. The Holm "
                           "calculation is preserved numerically, but it is not a fully "
                           "predeclared family and supports NO familywise confirmatory claim"),
            "decisions_unchanged": ("H1 (open) and H4 (open, schema-dependent) are decided by the "
                                    "frozen interval and route rules, not by these p-values; "
                                    "both decisions are unchanged"),
            "supersedes_wording_in": V3_3A},
        "power_evidence_status": {
            "status": "PLANNING EVIDENCE, NOT A POWER GUARANTEE",
            "simulation_assumptions": [
                "decisions use cluster-sandwich normal intervals, not the exact paired "
                "cluster-bootstrap gate",
                "injected effects are shared across the two answer schemas by construction and do "
                "not span every possible cross-schema treatment-effect dependence",
                "outcomes are synthetic train-side MBPP/HumanEval Phase 3A outcomes, not "
                "repository-native outcomes",
                "training-seed variance is not modelled",
                "the answer-rate, validity and diversity guardrails are not modelled"],
            "consequence": ("group counts in the power receipt are planning ranges for the "
                            "synthetic distribution; they are not guaranteed repository sample "
                            "sizes, and real power is expected to be lower"),
            "supersedes_wording_in": [POWER, CHOICES]},
        "acquisition_arithmetic": acquisition(),
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(addendum, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(addendum["acquisition_arithmetic"]["projection"]["statement"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
