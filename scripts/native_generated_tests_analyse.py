"""Frozen analysis for native generated tests (protocol v2 s2/s8, amendment v2.1 section E,
amendment v2.3 section D).

    analyse --manifest M --job J --prep P --preflight F --execution-contract C --results R
            (--generations ROOT | --base-generations B --sft-generations S)
            --condition primary_whole_module --study-mode engineering_dress_rehearsal
            --out FILE [--atheris A --atheris-contract AC]

Every input is loaded through its authoritative loader: the exact preparation records, both
generation arms, the execution contract + results (each row re-bound to its generated
candidate and re-verified from the actual module), and Atheris contract + results.
engineering_gate_passed = coverage AND repositories AND stage receipts AND canaries AND
artifact integrity AND no unexplained failure (amendment v2.4 G); any missing evidence fails
the gate and suppresses every arm comparison.

Cohorts (v2.3): the analysed grid is exactly the generation cohort resolved from the job
artifact (scripts/native_generation_io.py) - never all kept/qualified targets. The report
states separately the qualified cohort, the generated/requested cohort, pre-generation
exclusions (never model failures) and post-generation infrastructure exclusions. Every row
must carry its generation telemetry; completion-limit and EOS rates, generated-token and
latency percentiles and duplicates are reported per arm next to every validity figure.
Atheris may cover all qualified targets; joint comparisons use only generation targets that
are infrastructure-eligible and Atheris-eligible; excluded targets appear Atheris-only.

Amendment v2.4: every execution row is re-verified from its retained evidence; the CLI accepts
only the manifest's own study mode (a flag alone never enables confirmation); the inherited
engineering gate (>= ceil(0.9 x qualified) eligible targets across >= 5 repositories) must pass
or every arm comparison is suppressed; Atheris results enter only through the authoritative
loader (scripts/native_atheris_results.py); duplication is reported within rows, per target
across seeds and per arm.

Unit: the TARGET. kill(t, s, m) = 1 if any of the first k slots is a kill; K(t, m) is the mean
over the three seeds; the primary estimand is mean over targets of K(t, SFT) - K(t, base).
Target x seed observations are never pooled as independent samples.

Infrastructure (harness/dependency/environment) rows are never counted as parsed, collected,
executed or reached. A target with any infrastructure row in EITHER arm is excluded from the
paired comparison for BOTH arms, but only if its same-environment canary failed (proving the
failure arm-independent); an infrastructure row whose canary passed refuses the analysis.
Requested and eligible denominators are reported separately. Fixed validity comes from row
evidence (``fixed_valid``), never from a class label alone; nondeterminism is never valid.

study_mode ``engineering_dress_rehearsal`` reports descriptive pipeline metrics only (no
intervals, significance, non-inferiority or promotion decision); ``confirmation`` runs the
frozen inferential analysis (repository-clustered bootstrap, leave-one-repository-out,
validity non-inferiority on the lower bound).

v2.7: study_mode ``confirmation_exploratory_v1`` (three named arms from the frozen registry
cohort) runs the analysis plan frozen in ``ANALYSIS_PLAN_V27`` and bound by the manifest before
any output exists: separate contrasts sft (A@431) - base and relearn - base; per contrast, the
target-level estimand mean_t[K8(t, X) - K8(t, base)] with a repository-clustered bootstrap and
leave-one-repository-out estimates; secondary per-seed paired gained/lost/tied counts, exact
two-sided McNemar tests (Holm-adjusted, descriptive) and Wilson intervals; a frozen
one-target-per-repository sensitivity. Infrastructure exclusions and the coverage/repository
gates are PAIRWISE (an infrastructure failure in one trained arm never changes the other
contrast's denominator); evidence subgates are global. No promotion, equivalence or
generalisation claim is ever made in this mode.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSIS_VERSION = "oneiros_native_generated_tests_analyse_v5"   # v5: full-precision p-values,
#   fixed six-hypothesis Holm family, 34 -> 29 + 5 panel accounting in the exploratory result
ARMS = ("base", "sft")
SEEDS = (42, 43, 44)
SLOTS = 8
KILLS = {"semantic_kill", "crash_kill"}
INFRA = {"harness_failure", "dependency_failure", "environment_failure"}
PARSED_FAIL = {"syntax_failure", "over_limits", "policy_refused"}
COLLECT_FAIL = PARSED_FAIL | {"fabricated_import", "collection_failure", "no_tests_collected"}
EXEC_FAIL = COLLECT_FAIL | {"skipped_or_xfail", "timeout"}
REACH_FAIL = EXEC_FAIL | {"target_not_reached"}
BOOTSTRAP = {"resamples": 10_000, "seed": 20260930, "level": 0.95}
VALIDITY_MARGIN = 3.0
EXPLORATORY = "confirmation_exploratory_v1"
STUDY_MODES = ("engineering_dress_rehearsal", "confirmation", EXPLORATORY)
GATE = {"min_fraction_of_qualified": 0.90, "min_repositories": 5}
LEDGER_SCHEMA = "oneiros_native_quarantine_ledger_v1"
SYNTHETIC_EVIDENCE_SCHEMA = "oneiros_native_synthetic_gate_evidence_v1"
EVIDENCE_SUBGATES = ("stage_receipts_gate_passed", "canaries_gate_passed",
                     "artifact_integrity_gate_passed", "unexplained_failures_gate_passed")
ATHERIS_REQUIRED_CHECKS = (
    "ordinary_different_exceptions_not_a_kill", "ordinary_raise_versus_ok_is_a_kill",
    "ordinary_same_exception_not_a_kill", "every_kill_recomputes_from_confirmations",
    "aggregate_budget_two_busy_children", "worker_and_group_cleanup",
    "live_view_drift_refused", "applicability_versus_infrastructure",
    "real_rows_structurally_valid",
    # Atheris v5 (protocol v2.5 C)
    "atheris_2_3_0_from_overlay_in_target_runtime", "env_only_dependency_imports_in_target_runtime",
    "environment_unchanged_by_every_search", "receiver_constructed_through_public_constructor",
    "union_any_enum_literal_supported", "unsupported_stays_explicit",
    "abi_unavailable_is_infrastructure")
STAGE_RECEIPTS = ("full_suite", "synthetic_pipeline")
CANARY_RECEIPTS = ("sandbox_canaries", "atheris_canaries")
TELEMETRY_FIELDS = ("generated_tokens", "eos_reached", "finish_reason", "hit_completion_limit",
                    "prompt_tokens", "row_wall_seconds")


class AnalysisRefused(ValueError):
    pass


ANALYSIS_PLAN_V27 = {
    "version": EXPLORATORY,
    "unit": "target; kill(t,s,m)=1 if any of the first k of the 8 slots of seed s kills; "
            "K(t,m)=mean of kill over seeds 42, 43, 44; target x seed cells never pooled",
    "contrasts": ["sft_minus_base", "relearn_minus_base"],
    "primary": "per contrast: mean over pair-eligible targets of K8(t, X) - K8(t, base), "
               "percentage points",
    "uncertainty": {"bootstrap": {"resamples": 10_000, "seed": 20260930, "level": 0.95},
                    "cluster": "repository", "leave_one_repository_out": True},
    "secondary": ["Kill@1 and Kill@4 contrasts (same estimator, descriptive)",
                  "per-seed paired Kill@8 gained/lost/tied target counts",
                  "per-seed exact two-sided McNemar p (Holm-adjusted over contrasts x seeds; "
                  "descriptive)",
                  "Wilson 95% intervals of per-arm per-seed Kill@8 over pair-eligible targets",
                  "fixed-valid rate difference (clustered, descriptive)"],
    "sensitivity": {"one_target_per_repository":
                    "per repository, the pair-eligible target with the lexicographically "
                    "smallest sha256(target_key); model-independent and frozen"},
    "infrastructure": "a target with an infrastructure row in either arm of a contrast is "
                      "excluded from THAT contrast only, and only when its same-environment "
                      "canary failed; an infrastructure row with a passing canary refuses",
    "gates": {"per_contrast": {"min_fraction_of_qualified": 0.90, "min_repositories": 5},
              "global": list(("stage_receipts_gate_passed", "canaries_gate_passed",
                              "artifact_integrity_gate_passed",
                              "unexplained_failures_gate_passed"))},
    "multiplicity": "two contrasts x three seeds are reported without inferential claims",
    "claims": "exploratory and underpowered: no promotion, no equivalence from a "
              "non-significant result, no repository-generalisation claim",
}


def analysis_plan_sha256(plan: Mapping[str, Any] = ANALYSIS_PLAN_V27) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


# The FIXED Holm family the frozen plan declares (its contrasts x the three seeds): six
# hypotheses, whatever is later computed or suppressed. Derived from the plan itself, so the
# family can never shrink with the data.
HOLM_FAMILY = tuple(f"{c}:{s}" for c in ANALYSIS_PLAN_V27["contrasts"] for s in SEEDS)


def fixed_family_holm(computed: Mapping[str, float]) -> Dict[str, Any]:
    """Holm over the fixed six-hypothesis family. A suppressed (never computed) hypothesis is
    entered conservatively as p = 1 for the adjustment; its raw and adjusted p are reported as
    null, so no inferential value exists for it."""
    unknown = sorted(set(computed) - set(HOLM_FAMILY))
    if unknown:
        raise AnalysisRefused(f"p-values outside the frozen Holm family: {unknown}")
    adjusted = holm({h: computed.get(h, 1.0) for h in HOLM_FAMILY})
    return {"holm_family_size": len(HOLM_FAMILY),
            "suppressed_entered_as": "p = 1 (conservative); reported as null",
            "hypotheses": [{"label": h,
                            "status": "computed" if h in computed else "suppressed",
                            "raw_p": computed[h] if h in computed else None,
                            "holm_adjusted_p": adjusted[h] if h in computed else None,
                            "raw_p_display": p_display(computed.get(h)),
                            "holm_adjusted_p_display":
                                p_display(adjusted[h] if h in computed else None)}
                           for h in HOLM_FAMILY],
            "descriptive_only": True}


def index(records: Iterable[Mapping[str, Any]], targets: Sequence[str],
          arms: Sequence[str] = ARMS) -> Dict[tuple, dict]:
    grid: Dict[tuple, dict] = {}
    for r in records:
        key = (r["arm"], int(r["seed"]), r["target_key"], int(r["slot"]))
        if key in grid:
            raise AnalysisRefused(f"duplicate record {key}")
        grid[key] = dict(r)
    expected = {(a, s, t, k) for a in arms for s in SEEDS for t in targets for k in range(SLOTS)}
    if set(grid) != expected:
        missing, extra = expected - set(grid), set(grid) - expected
        raise AnalysisRefused(f"incomplete grid: {len(missing)} missing, {len(extra)} extra")
    return grid


def infrastructure_exclusions(grid, targets, arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    excluded, refusals = [], []
    for t in targets:
        rows = [grid[(a, s, t, k)] for a in arms for s in SEEDS for k in range(SLOTS)]
        infra = [r for r in rows if r["class"] in INFRA]
        if not infra:
            continue
        if all(r.get("canary_failed") is True for r in infra):
            excluded.append(t)
        else:
            refusals.append(t)
    if refusals:
        raise AnalysisRefused(f"infrastructure rows without a failed canary (not proven "
                              f"arm-independent): {refusals[:5]}")
    return {"excluded_targets": excluded,
            "rule": "any infrastructure row in either arm excludes the target from both arms"}


def kill_at(grid, arm, seed, target, k) -> float:
    return float(any(grid[(arm, seed, target, i)]["class"] in KILLS for i in range(k)))


def fixed_valid(row) -> bool:
    return row.get("fixed_valid") is True and row["class"] != "nondeterminism"


def _pct(part: int, whole: int) -> float:
    return round(part / whole * 100, 3) if whole else 0.0


def _quantiles(values: Sequence[float], qs: Sequence[int]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs} if len(arr) else {}


def generation_telemetry(grid, targets, arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    """Completion and latency evidence per arm over every generated candidate (v2.3 B/D)."""
    out = {}
    for arm in arms:
        cells = [grid[(arm, s, t, i)] for s in SEEDS for t in targets for i in range(SLOTS)]
        for c in cells:
            g = c.get("generation")
            if not isinstance(g, Mapping) or any(f not in g for f in TELEMETRY_FIELDS):
                raise AnalysisRefused(f"row without generation telemetry: {c.get('key')}")
        gen = [c["generation"] for c in cells]
        hits = sum(g["hit_completion_limit"] is True for g in gen)
        eos = sum(g["eos_reached"] is True for g in gen)
        tokens = [g["generated_tokens"] for g in gen]
        latency = [grid[(arm, s, t, 0)]["generation"]["row_wall_seconds"]
                   for s in SEEDS for t in targets]
        duplicates, unique_total, cross_hashes, cross_candidates, per_target = 0, 0, 0, 0, {}
        for t in targets:
            by_seed = {s: [grid[(arm, s, t, i)]["module_sha256"] for i in range(SLOTS)]
                       for s in SEEDS}
            for hashes in by_seed.values():
                duplicates += len(hashes) - len(set(hashes))
            everything = [h for hashes in by_seed.values() for h in hashes]
            distinct = set(everything)
            unique_total += len(distinct)
            repeated = {h for h in distinct
                        if sum(h in set(hashes) for hashes in by_seed.values()) > 1}
            cross_hashes += len(repeated)
            cross_candidates += sum(h in repeated for h in everything)
            per_target[t] = {"candidates": len(everything), "unique": len(distinct),
                             "cross_seed_repeated_hashes": len(repeated)}
        out[arm] = {"candidates": len(gen), "completion_limit_hits": hits,
                    "completion_limit_hit_rate": _pct(hits, len(gen)),
                    "eos_completions": eos, "eos_rate": _pct(eos, len(gen)),
                    "generated_tokens": {**_quantiles(tokens, (50, 90, 99)),
                                         "max": int(max(tokens)) if tokens else 0},
                    "target_seed_latency_seconds": {**_quantiles(latency, (50, 90)),
                                                    "total": round(float(sum(latency)), 3),
                                                    "rows": len(latency)},
                    "duplicate_candidates": duplicates,
                    "duplicate_rate": _pct(duplicates, len(gen)),
                    "duplication": {
                        "within_target_seed_duplicates": duplicates,
                        "within_target_seed_denominator": len(gen),
                        "unique_candidates": unique_total,
                        "unique_rate": _pct(unique_total, len(gen)),
                        "unique_definition": "distinct module hashes per target across all "
                                             "3 seeds x 8 slots, summed over targets "
                                             "(within-row repeats count once)",
                        "cross_seed_repeated_hashes": cross_hashes,
                        "cross_seed_repeated_candidates": cross_candidates,
                        "per_target": per_target,
                        "policy": "duplicates stay in the primary grid; nothing reranked "
                                  "or deleted"}}
    return out


def denominators(grid, requested, eligible, arms: Sequence[str] = ARMS
                 ) -> Dict[str, Dict[str, int]]:
    out = {}
    for arm in arms:
        req = [grid[(arm, s, t, i)] for s in SEEDS for t in requested for i in range(SLOTS)]
        cells = [grid[(arm, s, t, i)] for s in SEEDS for t in eligible for i in range(SLOTS)]
        classes = [c["class"] for c in cells]
        out[arm] = {"requested": len(req), "eligible": len(cells),
                    "infrastructure_excluded": len(req) - len(cells),
                    "parsed": sum(c not in PARSED_FAIL for c in classes),
                    "collected": sum(c not in COLLECT_FAIL for c in classes),
                    "executed": sum(c not in EXEC_FAIL for c in classes),
                    "reached": sum(c not in REACH_FAIL for c in classes),
                    "fixed_valid": sum(fixed_valid(c) for c in cells),
                    "rates_percent_of_eligible": {
                        name: _pct(sum(c not in fail for c in classes), len(classes))
                        for name, fail in (("parsed", PARSED_FAIL), ("collected", COLLECT_FAIL),
                                           ("executed", EXEC_FAIL), ("reached", REACH_FAIL))},
                    "fixed_valid_percent_of_eligible": _pct(sum(fixed_valid(c) for c in cells),
                                                            len(cells)),
                    "semantic_kills": classes.count("semantic_kill"),
                    "crash_kills": classes.count("crash_kill"),
                    "classes": dict(Counter(classes))}
    return out


def clustered(diffs: Mapping[str, float], repo_of: Mapping[str, str]) -> Dict[str, Any]:
    repos = sorted({repo_of[t] for t in diffs})
    sums = np.array([sum(d for t, d in diffs.items() if repo_of[t] == r) for r in repos])
    counts = np.array([sum(1 for t in diffs if repo_of[t] == r) for r in repos])
    point = float(sums.sum() / counts.sum() * 100)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    pick = rng.integers(0, len(repos), size=(BOOTSTRAP["resamples"], len(repos)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1) * 100
    alpha = (1 - BOOTSTRAP["level"]) / 2 * 100
    low, high = np.percentile(draws, [alpha, 100 - alpha])
    loo = {r: round(float((sums.sum() - sums[i]) / (counts.sum() - counts[i]) * 100), 3)
           for i, r in enumerate(repos) if counts.sum() > counts[i]}
    return {"point": round(point, 3), "low": round(float(low), 3), "high": round(float(high), 3),
            "targets": int(counts.sum()), "repositories": len(repos),
            "leave_one_repository_out": loo}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Dict[str, Any]:
    if n == 0:
        return {"k": k, "n": n, "low": None, "high": None}
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"k": k, "n": n, "rate": round(p * 100, 3), "low": round((centre - half) * 100, 3),
            "high": round((centre + half) * 100, 3)}


def holm(pvalues: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, v) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * v))
        out[k] = running            # full precision: a tiny exact p must never round to 0
    return out


def p_display(p: float | None) -> str | None:
    """Human-readable companion of a p-value; never replaces the numeric value."""
    return None if p is None else f"{p:.3e}"


def one_per_repository(targets: Sequence[str], repo_of: Mapping[str, str]) -> List[str]:
    chosen: Dict[str, str] = {}
    for t in sorted(targets, key=lambda t: hashlib.sha256(t.encode()).hexdigest()):
        chosen.setdefault(repo_of[t], t)
    return sorted(chosen.values())


def exploratory(grid, targets, repo_of, arms, cohort, evidence_gates,
                synthetic_gate: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """The frozen v2.7 exploratory confirmation analysis (ANALYSIS_PLAN_V27).
    ``synthetic_gate`` exists ONLY for the toy synthetic pipeline (one repository); the CLI never
    passes it and any result computed with it is labelled synthetic."""
    if tuple(arms[:1]) != ("base",) or len(arms) < 2:
        raise AnalysisRefused("the exploratory analysis needs base plus trained arms")
    infrastructure_exclusions(grid, targets, arms)      # refuses unexplained infrastructure rows
    evidence_gates = dict(evidence_gates or {})
    global_gates = {k: evidence_gates.get(k) is True for k in EVIDENCE_SUBGATES}
    qualified_n = len(cohort["qualified"]) if cohort else len(targets)
    plan_gate = dict(synthetic_gate) if synthetic_gate else ANALYSIS_PLAN_V27["gates"]["per_contrast"]
    need = math.ceil(plan_gate["min_fraction_of_qualified"] * qualified_n - 1e-9)
    contrasts, pvalues = {}, {}
    for arm in arms[1:]:
        pair = ("base", arm)
        infra = infrastructure_exclusions(grid, targets, pair)
        eligible = [t for t in targets if t not in infra["excluded_targets"]]
        repos = sorted({repo_of[t] for t in eligible})
        gates = {"coverage_gate_passed": len(eligible) >= need,
                 "repository_gate_passed": len(repos) >= plan_gate["min_repositories"],
                 **global_gates}
        name = f"{arm}_minus_base"
        entry: Dict[str, Any] = {
            "pair": list(pair), "requested_targets": len(targets),
            "pair_eligible_targets": len(eligible), "pair_infrastructure_excluded":
                infra["excluded_targets"], "repositories": len(repos),
            "required_eligible": need, "gates": gates,
            "denominators": denominators(grid, targets, eligible, pair)}
        if not all(gates.values()):
            entry["status"] = "SUPPRESSED: " + ", ".join(k for k, v in gates.items() if not v)
            contrasts[name] = entry
            continue
        for k in (1, 4, 8):
            scores = {a: {t: float(np.mean([kill_at(grid, a, s, t, k) for s in SEEDS]))
                          for t in eligible} for a in pair}
            entry[f"kill_at_{k}"] = {
                a: round(float(np.mean(list(scores[a].values()))) * 100, 3) for a in pair}
            entry[f"kill_at_{k}"]["difference_points"] = clustered(
                {t: scores[arm][t] - scores["base"][t] for t in eligible}, repo_of)
        k8 = {a: {t: float(np.mean([kill_at(grid, a, s, t, SLOTS) for s in SEEDS]))
                  for t in eligible} for a in pair}
        entry["primary"] = entry["kill_at_8"]["difference_points"]
        per_seed = {}
        for s in SEEDS:
            kb = {t: kill_at(grid, "base", s, t, SLOTS) for t in eligible}
            kx = {t: kill_at(grid, arm, s, t, SLOTS) for t in eligible}
            gained = sum(1 for t in eligible if kx[t] and not kb[t])
            lost = sum(1 for t in eligible if kb[t] and not kx[t])
            p = mcnemar_exact(gained, lost)
            pvalues[f"{name}:{s}"] = p
            per_seed[str(s)] = {"gained": gained, "lost": lost,
                                "tied": len(eligible) - gained - lost,
                                "mcnemar_exact_two_sided_p": p,
                                "mcnemar_exact_two_sided_p_display": p_display(p),
                                "wilson_kill_at_8": {
                                    "base": wilson(int(sum(kb.values())), len(eligible)),
                                    arm: wilson(int(sum(kx.values())), len(eligible))}}
        entry["per_seed_secondary"] = per_seed
        chosen = one_per_repository(eligible, repo_of)
        entry["one_target_per_repository"] = {
            "targets": chosen, "n": len(chosen),
            "difference_points": round(float(np.mean([k8[arm][t] - k8["base"][t]
                                                       for t in chosen])) * 100, 3)}
        validity = {a: {t: float(np.mean([fixed_valid(grid[(a, s, t, i)]) for s in SEEDS
                                          for i in range(SLOTS)])) for t in eligible}
                    for a in pair}
        entry["fixed_valid_rate"] = {a: round(float(np.mean(list(validity[a].values()))) * 100, 3)
                                     for a in pair}
        entry["fixed_valid_difference_points"] = clustered(
            {t: validity[arm][t] - validity["base"][t] for t in eligible}, repo_of)
        entry["status"] = "computed (exploratory; no claim)"
        contrasts[name] = entry
    multiplicity = fixed_family_holm(pvalues)
    adjusted = {h["label"]: h["holm_adjusted_p"] for h in multiplicity["hypotheses"]}
    for name, entry in contrasts.items():
        for s, row in (entry.get("per_seed_secondary") or {}).items():
            row["holm_adjusted_p"] = adjusted[f"{name}:{s}"]
            row["holm_family_size"] = multiplicity["holm_family_size"]
    panel = (cohort or {}).get("panel")
    policy = list((cohort or {}).get("panel_policy_exclusions") or [])
    return {"analysis_version": ANALYSIS_VERSION, "study_mode": EXPLORATORY,
            "synthetic_gate_override": dict(synthetic_gate) if synthetic_gate else None,
            "analysis_plan_sha256": analysis_plan_sha256(), "arms": list(arms),
            "unit": "target (seeds averaged; never pooled)",
            "cohort": {"frozen_panel_targets": panel["targets"] if panel else None,
                       "frozen_panel": panel,
                       "native_executable_targets": qualified_n,
                       "qualified_targets": qualified_n, "generation_targets": len(targets),
                       "panel_policy_excluded": len(policy),
                       "panel_policy_exclusions": policy,
                       "policy_exclusions_bound_to": (cohort or {}).get("subset_receipt"),
                       "pre_generation_exclusions":
                           list(cohort["pre_generation_exclusions"]) if cohort else [],
                       "post_generation_infrastructure_exclusions": {
                           name: c["pair_infrastructure_excluded"]
                           for name, c in contrasts.items()},
                       "note": "panel policy exclusions (sandbox policy) are outside every "
                               "model denominator and are never model failures; "
                               "post-generation infrastructure exclusions are pair-specific"},
            "multiplicity": multiplicity,
            "grid_cells": len(grid), "global_evidence_gates": global_gates,
            "evidence_problems": evidence_gates.get("problems") or {},
            "generation_telemetry": generation_telemetry(grid, targets, arms),
            "requested_denominators": denominators(grid, targets, targets, arms),
            "contrasts": contrasts,
            "claims": ANALYSIS_PLAN_V27["claims"],
            "root_cause_established": False, "generalization_established": False,
            "sft_benefit_established": False, "relearning_benefit_established": False,
            "atheris_superiority_established": False}


def join_atheris(validated: Mapping[str, Any], grid, targets, eligible, qualified
                 ) -> Dict[str, Any]:
    """Joint Oneiros/Atheris view from the AUTHORITATIVE loader's output only."""
    status = validated["targets"]
    if set(status) != set(qualified):
        raise AnalysisRefused("Atheris targets differ from the qualified cohort")
    usable = {t for t, v in status.items() if v["status"] == "usable"}
    joint = sorted(set(eligible) & usable)
    infra = sorted(t for t, v in status.items() if v["status"] == "infrastructure_excluded")
    only = sorted(set(qualified) - set(targets))
    return {
        "atheris_denominators": {
            "qualified": len(qualified), "atheris_cells": validated["cells"],
            "atheris_usable": len(usable), "atheris_adapter_unsupported": validated["counts"]["adapter_unsupported"],
            "atheris_infrastructure_excluded": len(infra),
            "generation_targets": len(targets), "infrastructure_eligible": len(eligible),
            "joint": len(joint),
            "rule": "joint = generation targets AND Oneiros-infrastructure-eligible AND "
                    "Atheris-usable; an Atheris infrastructure problem excludes the target from "
                    "the joint comparison for both tools"},
        "atheris_contract_sha256": validated["contract_sha256"],
        "atheris_infrastructure_exclusions": {t: status[t].get("problems") for t in infra},
        "atheris_only_not_generated": {
            "targets": only,
            "rule": "pre-generation exclusions: Atheris-only; never a joint comparison",
            "status": {t: status[t]["status"] for t in only},
            "kill_by_mode": {t: status[t].get("kill_by_mode") for t in only}},
        "atheris_jointly_eligible": {
            "targets": len(joint),
            "oneiros_unique_kills": {arm: sum(any(kill_at(grid, arm, s, t, SLOTS) for s in SEEDS)
                                              for t in joint) for arm in ARMS},
            "atheris_unique_kills": {m: sum(status[t]["kill_by_mode"][m] for t in joint)
                                     for m in ("ordinary", "posthoc", "differential")},
            "labels": {"differential": "oracle-assisted upper bound"}}}


# --- evidence subgates (amendment v2.4 G) --------------------------------------------------

def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt_problems(kind: str, entry: Mapping[str, Any] | None, artifact_root: Path) -> List[str]:
    """A receipt counts only if it exists at a safe path, its bytes hash to the recorded value,
    its schema is right, it passed, and it is bound to the CURRENT source."""
    from harness.native_launch_gate import safe_input, source_identity
    if not entry or not entry.get("path") or not entry.get("sha256"):
        return [f"{kind}: missing"]
    path = safe_input(artifact_root, entry["path"])
    if path is None:
        return [f"{kind}: unsafe path"]
    if not path.is_file():
        return [f"{kind}: file missing"]
    if _file_sha(path) != entry["sha256"]:
        return [f"{kind}: bytes differ from the preflight hash"]
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return [f"{kind}: malformed"]
    scripts = ROOT / "scripts"
    bad = []
    if kind == "full_suite":
        if r.get("schema_version") != "oneiros_native_full_suite_receipt_v2":
            bad.append("schema")
        if not (r.get("exit") == 0 and r.get("failed") == 0 and (r.get("passed") or 0) > 0
                and r.get("tree_clean_at_start") is True):
            bad.append("not passed")
        if r.get("executable_tree_sha256") != source_identity(ROOT)["executable_tree_sha256"]:
            bad.append("not bound to the current source")
    elif kind == "synthetic_pipeline":
        from scripts.native_pipeline_synthetic import COMPONENTS
        if r.get("schema_version") != "oneiros_native_pipeline_synthetic_v1":
            bad.append("schema")
        if r.get("passed") is not True:
            bad.append("not passed")
        comps = r.get("components_sha256") or {}
        if set(comps) != set(COMPONENTS) or any(_file_sha(ROOT / c) != h for c, h in comps.items()):
            bad.append("not bound to the current source")
    elif kind == "sandbox_canaries":
        if r.get("schema_version") != "oneiros_native_sandbox_canaries_v2":
            bad.append("schema")
        if r.get("passed") is not True:
            bad.append("not passed")
        if (r.get("executor_sha256"), r.get("inner_sha256"), r.get("prepare_sha256")) != (
                _file_sha(scripts / "native_generated_tests_execute_wsl.py"),
                _file_sha(scripts / "native_sandbox_inner.sh"),
                _file_sha(scripts / "native_rehearsal_prepare_wsl.py")):
            bad.append("not bound to the current source")
    elif kind == "atheris_canaries":
        from scripts.native_atheris_results import DESIGN_VERSION
        if r.get("schema_version") != "oneiros_native_atheris_canaries_v4" or \
                r.get("design_version") != DESIGN_VERSION:
            bad.append("schema")
        checks = r.get("checks") or {}
        if r.get("passed") is not True or not all(checks.get(k) is True
                                                  for k in ATHERIS_REQUIRED_CHECKS):
            bad.append("not passed")
        if (r.get("script_sha256"), r.get("inner_sha256"), r.get("verdicts_sha256")) != (
                _file_sha(scripts / "native_generated_tests_atheris_wsl.py"),
                _file_sha(scripts / "native_sandbox_inner.sh"),
                _file_sha(scripts / "native_atheris_results.py")):
            bad.append("not bound to the current source")
    else:
        bad.append("unknown receipt kind")
    return [f"{kind}: {b}" for b in bad]


def ledger_problems(entry: Mapping[str, Any] | None, artifact_root: Path,
                    quarantined: Mapping[str, str]) -> List[str]:
    """Every quarantined artifact must be explained by a complete ledger entry."""
    from harness.native_launch_gate import safe_input
    if not entry or not entry.get("path") or not entry.get("sha256"):
        return ["quarantine ledger: missing"]
    path = safe_input(artifact_root, entry["path"])
    if path is None or not path.is_file() or _file_sha(path) != entry["sha256"]:
        return ["quarantine ledger: missing or differs from the preflight hash"]
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return ["quarantine ledger: malformed"]
    if ledger.get("schema_version") != LEDGER_SCHEMA or not isinstance(ledger.get("entries"), list):
        return ["quarantine ledger: schema"]
    problems, explained = [], set()
    for i, e in enumerate(ledger["entries"]):
        sup = e.get("superseded_by") or {}
        if not all(isinstance(e.get(f), str) and e.get(f) for f in
                   ("artifact", "sha256", "reason", "corrective_change")) or \
                not sup.get("path") or not sup.get("sha256"):
            problems.append(f"quarantine ledger entry {i}: incomplete")
            continue
        artifact = safe_input(artifact_root, e["artifact"])
        if artifact is not None and artifact.is_file() and _file_sha(artifact) != e["sha256"]:
            problems.append(f"quarantine ledger entry {i}: artifact hash differs")
        superseding = safe_input(artifact_root, sup["path"])
        if superseding is None or not superseding.is_file() or \
                _file_sha(superseding) != sup["sha256"]:
            problems.append(f"quarantine ledger entry {i}: superseding artifact missing or differs")
        explained.add(e["sha256"])
    for rel, sha in sorted(quarantined.items()):
        if sha not in explained:
            problems.append(f"unexplained quarantined artifact: {rel}")
    return problems


def evaluate_evidence(preflight_path: Path, artifact_root: Path, *, job_path: Path,
                      manifest_path: Path, synthetic_allowed: bool,
                      quarantined: Mapping[str, str]) -> Dict[str, Any]:
    """Stage-receipt, canary and unexplained-failure subgates from the exact v2.4 preflight
    (or, for the synthetic pipeline only, explicitly synthetic evidence that can never pass
    the stage-receipt subgate)."""
    from harness import native_launch_gate as gate
    problems: Dict[str, List[str]] = {"preflight": [], "stage_receipts": [], "canaries": [],
                                      "unexplained_failures": []}
    try:
        pre = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pre = {}
        problems["preflight"].append("preflight missing or malformed")
    real = pre.get("schema_version") in gate.PREFLIGHT_SCHEMAS
    if real:
        if pre.get("pipeline_ready") is not True:
            problems["preflight"].append("preflight not pipeline_ready")
        problems["preflight"] += gate.input_problems(artifact_root, pre)
        for field, given in (("job", job_path), ("manifest", manifest_path)):
            named = gate.safe_input(artifact_root, (pre.get(field) or {}).get("path"))
            if named is None or named.resolve() != Path(given).resolve():
                problems["preflight"].append(f"preflight names a different {field}")
    elif pre.get("schema_version") == SYNTHETIC_EVIDENCE_SCHEMA and synthetic_allowed:
        problems["preflight"].append("synthetic evidence is not a v2.4 preflight")
    elif pre:
        problems["preflight"].append(f"preflight schema {pre.get('schema_version')!r}")
    evidence = pre.get("gate_evidence") or {}
    receipts = evidence.get("receipts") or {}
    inputs = pre.get("inputs") or {}

    def bound(entry):
        return not real or (entry and inputs.get(entry.get("path")) == entry.get("sha256"))
    for kind in STAGE_RECEIPTS + CANARY_RECEIPTS:
        entry = receipts.get(kind)
        found = receipt_problems(kind, entry, artifact_root)
        if not found and not bound(entry):
            found = [f"{kind}: not bound by the preflight inputs"]
        problems["stage_receipts" if kind in STAGE_RECEIPTS else "canaries"] += found
    ledger = evidence.get("quarantine_ledger")
    problems["unexplained_failures"] += ledger_problems(ledger, artifact_root, quarantined)
    if not problems["unexplained_failures"] and not bound(ledger):
        problems["unexplained_failures"].append("quarantine ledger not bound by the preflight")
    return {"stage_receipts_gate_passed": not problems["preflight"] and not problems["stage_receipts"],
            "canaries_gate_passed": not problems["canaries"],
            "unexplained_failures_gate_passed": not problems["unexplained_failures"],
            "problems": {k: v for k, v in problems.items() if v}}


def quarantined_artifacts(artifact_root: Path, dirs: Sequence[Path]) -> Dict[str, str]:
    out = {}
    for d in dirs:
        q = Path(d) / "quarantine"
        if q.is_dir():
            for f in sorted(q.rglob("*")):
                if f.is_file():
                    try:
                        rel = f.resolve().relative_to(Path(artifact_root).resolve()).as_posix()
                    except ValueError:
                        rel = f.as_posix()
                    out[rel] = _file_sha(f)
    return out


def analyse(records: Iterable[Mapping[str, Any]], targets: Sequence[str],
            repo_of: Mapping[str, str], study_mode: str,
            atheris: Mapping[str, Any] | None = None,
            cohort: Mapping[str, Any] | None = None,
            evidence_gates: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """``targets`` is the generation cohort. With ``cohort`` (from resolve_cohort) it must be
    exactly the job's generation targets and the report carries the 24/23/1 breakdown."""
    if study_mode not in STUDY_MODES:
        raise AnalysisRefused(f"unknown study mode {study_mode!r}")
    arms = tuple(cohort["arms"]) if cohort and cohort.get("arms") else ARMS
    targets = sorted(targets)
    if cohort is not None and targets != sorted(cohort["generation"]):
        raise AnalysisRefused("analysed targets differ from the job's generation cohort")
    unmapped = [t for t in targets if t not in repo_of]
    if unmapped:
        raise AnalysisRefused(f"repository mapping missing for generated targets {unmapped[:3]}")
    grid = index(records, targets, arms)
    if study_mode == EXPLORATORY:
        return exploratory(grid, targets, repo_of, arms, cohort, evidence_gates)
    if arms != ARMS:
        raise AnalysisRefused(f"study mode {study_mode!r} supports only the base/sft pair")
    infra = infrastructure_exclusions(grid, targets)
    eligible = [t for t in targets if t not in infra["excluded_targets"]]
    if not eligible:
        raise AnalysisRefused("no eligible targets")
    exclusions = list(cohort["pre_generation_exclusions"]) if cohort else []
    qualified_n = len(cohort["qualified"]) if cohort else len(targets)
    need = math.ceil(GATE["min_fraction_of_qualified"] * qualified_n - 1e-9)
    repositories = sorted({repo_of[t] for t in eligible})
    gate_failures = []
    if len(eligible) < need:
        gate_failures.append(f"{len(eligible)} of {qualified_n} qualified targets eligible "
                             f"(< {need})")
    if len(repositories) < GATE["min_repositories"]:
        gate_failures.append(f"{len(repositories)} repositories (< {GATE['min_repositories']})")
    evidence_gates = dict(evidence_gates or {})          # absent evidence fails closed
    subgates = {"coverage_gate_passed": len(eligible) >= need,
                "repository_gate_passed": len(repositories) >= GATE["min_repositories"],
                **{k: evidence_gates.get(k) is True for k in EVIDENCE_SUBGATES}}
    for k in EVIDENCE_SUBGATES:
        if not subgates[k]:
            gate_failures.append(f"{k[:-7]} not established" +
                                 (": " + "; ".join(sum((evidence_gates.get("problems") or {})
                                                       .values(), []))[:300]
                                  if evidence_gates.get("problems") else ""))
    gate = {**GATE, "qualified": qualified_n, "required_eligible": need,
            "eligible": len(eligible), "repositories": len(repositories),
            "represented_repositories": repositories, "subgates": subgates,
            "evidence_problems": evidence_gates.get("problems") or {},
            "failures": gate_failures, "passed": all(subgates.values())}
    result: Dict[str, Any] = {"analysis_version": ANALYSIS_VERSION, "study_mode": study_mode,
                              "unit": "target (seeds averaged; never pooled)",
                              "cohort": {
                                  "qualified_targets": len(cohort["qualified"]) if cohort else None,
                                  "generation_targets": len(targets),
                                  "pre_generation_excluded": len(exclusions),
                                  "pre_generation_exclusions": exclusions,
                                  "post_generation_infrastructure_excluded":
                                      len(infra["excluded_targets"]),
                                  "final_infrastructure_eligible": len(eligible),
                                  "represented_repositories": len(repositories),
                                  "note": "pre-generation exclusions are not model failures and "
                                          "are not in any model denominator"},
                              "requested_targets": len(targets), "eligible_targets": len(eligible),
                              "grid_cells": len(grid),
                              "infrastructure": infra,
                              "generation_telemetry": generation_telemetry(grid, targets),
                              "denominators": denominators(grid, targets, eligible),
                              "engineering_gate": gate, "engineering_gate_passed": gate["passed"]}
    if not gate["passed"]:
        result["arm_comparison"] = ("SUPPRESSED: engineering gate failed: "
                                    + "; ".join(gate_failures))
        result["decisions"] = "SUPPRESSED: engineering gate failed"
        result.update(root_cause_established=False, generalization_established=False,
                      sft_benefit_established=False, atheris_superiority_established=False)
        return result
    for k in (1, 4, 8):
        scores = {arm: {t: float(np.mean([kill_at(grid, arm, s, t, k) for s in SEEDS]))
                        for t in eligible} for arm in ARMS}
        entry = {arm: round(float(np.mean(list(scores[arm].values()))) * 100, 3) for arm in ARMS}
        entry["per_seed"] = {str(s): {arm: round(float(np.mean([kill_at(grid, arm, s, t, k)
                                                                 for t in eligible])) * 100, 3)
                                      for arm in ARMS} for s in SEEDS}
        if study_mode == "confirmation":
            entry["sft_minus_base_points"] = clustered(
                {t: scores["sft"][t] - scores["base"][t] for t in eligible}, repo_of)
        result[f"kill_at_{k}"] = entry
    result["unique_bugs_killed"] = {arm: sum(any(kill_at(grid, arm, s, t, SLOTS) for s in SEEDS)
                                             for t in eligible) for arm in ARMS}
    validity = {arm: {t: np.mean([fixed_valid(grid[(arm, s, t, i)]) for s in SEEDS
                                  for i in range(SLOTS)]) for t in eligible} for arm in ARMS}
    result["fixed_valid_rate"] = {arm: round(float(np.mean(list(validity[arm].values()))) * 100, 3)
                                  for arm in ARMS}
    # a validity figure is never reported without the completion-limit rates beside it
    result["fixed_valid_rate"]["completion_limit_hit_rate"] = {
        arm: result["generation_telemetry"][arm]["completion_limit_hit_rate"] for arm in ARMS}
    if study_mode == "confirmation":
        result["primary"] = result["kill_at_8"]["sft_minus_base_points"]
        vi = clustered({t: validity["sft"][t] - validity["base"][t] for t in eligible}, repo_of)
        result["validity_non_inferiority"] = {
            **vi, "margin_points": -VALIDITY_MARGIN,
            "non_inferiority_shown": vi["low"] >= -VALIDITY_MARGIN,
            "operational_stop": vi["point"] < -VALIDITY_MARGIN and vi["low"] < -VALIDITY_MARGIN}
    else:
        result["decisions"] = ("SUPPRESSED: engineering dress rehearsal - descriptive pipeline "
                               "metrics only; no significance, non-inferiority or promotion")
    if atheris:
        result.update(join_atheris(atheris, grid, targets, eligible,
                                   set(cohort["qualified"]) if cohort else set(targets)))
    result["root_cause_established"] = False
    result["generalization_established"] = False
    result["sft_benefit_established"] = False
    result["atheris_superiority_established"] = False
    return result


def main(argv=None, root: Path | None = None) -> int:
    from harness.atomic_publish import publish_file_atomically
    from scripts import native_generation_io as gio
    from scripts.native_execution_results import load_execution
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("analyse",))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--prep", required=True, help="the manifest-declared records.jsonl")
    parser.add_argument("--preflight", required=True, help="the v2.4 preflight receipt")
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--generations", default=None, help="root containing base/ and sft/")
    parser.add_argument("--base-generations", default=None)
    parser.add_argument("--sft-generations", default=None)
    parser.add_argument("--arm-generations", action="append", default=[], metavar="ARM=DIR")
    parser.add_argument("--condition", required=True, choices=("primary_whole_module",))
    parser.add_argument("--study-mode", required=True, choices=STUDY_MODES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--atheris", default=None, help="Atheris results JSONL")
    parser.add_argument("--atheris-contract", default=None, help="its atheris_contract.json")
    args = parser.parse_args(argv)
    artifact_root = Path(root or ROOT)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cohort = gio.resolve_cohort(Path(args.job), Path(args.manifest), args.condition)
    targets = cohort["generation"]
    repo_of = cohort["repo_of"]
    declared = cohort.get("study_mode")                 # v2.4 F: the manifest decides
    if declared not in STUDY_MODES or args.study_mode != declared:
        raise AnalysisRefused(f"study mode {args.study_mode!r} refused: the manifest declares "
                              f"{declared!r} ({cohort.get('nature')!r})")
    if args.study_mode == EXPLORATORY:
        bound = manifest.get("analysis_plan") or {}
        if bound.get("version") != EXPLORATORY or \
                bound.get("sha256") != analysis_plan_sha256():
            raise AnalysisRefused("the manifest does not bind this frozen exploratory analysis "
                                  "plan (changed after freezing?)")
    if args.study_mode == "confirmation":
        freeze = manifest.get("confirmation_authorization") or {}
        path = Path(args.manifest).parent / str(freeze.get("path", ""))
        if not freeze.get("sha256") or not path.is_file() or \
                hashlib.sha256(path.read_bytes()).hexdigest() != freeze["sha256"]:
            raise AnalysisRefused("confirmation needs a separately frozen confirmation manifest "
                                  "and authorisation; none is bound")
    opt = lambda value: Path(value) if value else None  # noqa: E731
    try:                                  # every artifact through its authoritative loader
        prepared = gio.resolve_prep(Path(args.prep), Path(args.manifest), artifact_root)
        if args.base_generations or args.sft_generations:
            arm_paths = gio.arm_paths(opt(args.generations), opt(args.base_generations),
                                      opt(args.sft_generations), args.condition)
        else:
            explicit = dict(spec.split("=", 1) for spec in args.arm_generations)
            arm_paths = gio.arm_dirs(opt(args.generations), explicit, cohort["arms"],
                                     args.condition)
        generations = gio.load_arm_generations(arm_paths, cohort)
        execution = load_execution(Path(args.results), Path(args.execution_contract),
                                   cohort=cohort, prepared=prepared, generations=generations)
    except SystemExit as exc:
        raise AnalysisRefused(str(exc)) from None
    atheris = None
    if args.atheris or args.atheris_contract:
        if not (args.atheris and args.atheris_contract):
            raise AnalysisRefused("Atheris results and contract must be given together")
        from scripts.native_atheris_results import load_results
        scripts = ROOT / "scripts"
        try:
            atheris = load_results(
                Path(args.atheris), Path(args.atheris_contract), qualified=cohort["qualified"],
                manifest_sha256=cohort["manifest_sha256"],
                prep={"path": prepared["path"], "sha256": prepared["sha256"]},
                prep_rows=prepared["rows"],
                script_sha256=_file_sha(scripts / "native_generated_tests_atheris_wsl.py"),
                inner_sha256=_file_sha(scripts / "native_sandbox_inner.sh"),
                verdicts_sha256=_file_sha(scripts / "native_atheris_results.py"))
        except SystemExit as exc:
            raise AnalysisRefused(str(exc)) from None
    quarantined = quarantined_artifacts(
        artifact_root, [p["dir"] for p in arm_paths.values()] + [Path(args.results).parent])
    evidence = evaluate_evidence(Path(args.preflight), artifact_root, job_path=Path(args.job),
                                 manifest_path=Path(args.manifest),
                                 synthetic_allowed="SYNTHETIC" in str(cohort.get("nature")),
                                 quarantined=quarantined)
    evidence["artifact_integrity_gate_passed"] = True   # every loader above succeeded
    result = analyse(execution["rows"], targets, repo_of, args.study_mode, atheris, cohort,
                     evidence)
    result["inputs"] = {name: _file_sha(Path(p))
                        for name, p in (("manifest", args.manifest), ("job", args.job),
                                        ("prep", args.prep), ("preflight", args.preflight),
                                        ("execution_contract", args.execution_contract),
                                        ("results", args.results), ("atheris", args.atheris),
                                        ("atheris_contract", args.atheris_contract)) if p}
    result["generations"] = {"files_sha256": generations["files_sha256"],
                             "contracts_sha256": generations["contracts_sha256"]}
    result["job_sha256"] = cohort["job_sha256"]
    result["execution_contract_sha256"] = execution["contract_sha256"]
    result["quarantined_artifacts"] = quarantined
    result["condition"] = args.condition
    publish_file_atomically(Path(args.out), (json.dumps(result, indent=1, sort_keys=True)
                                             + "\n").encode("utf-8"))
    if args.study_mode == EXPLORATORY:
        print(json.dumps({"out": args.out, "study_mode": args.study_mode,
                          "global_evidence_gates": result["global_evidence_gates"],
                          "contrasts": {k: v["status"] for k, v in result["contrasts"].items()}}))
        return 0
    print(json.dumps({"out": args.out, "eligible_targets": result["eligible_targets"],
                      "engineering_gate_passed": result["engineering_gate_passed"],
                      "subgates": result["engineering_gate"]["subgates"],
                      "study_mode": args.study_mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
