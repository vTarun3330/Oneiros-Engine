"""Prospective power analysis for the repository-native evaluation (CPU only).

Inputs are permitted historical evidence only: paired per-record Kill@8 from
the train-derived retention panel (613 records, 100 lineages) and the
tool-assisted panel (264 records, 110 lineages).  Before anything is computed,
every evidence file is verified: it exists, its hash equals the hash recorded
in the closed pilot's decision receipt (evaluation envelope AND raw rehearsal
result), paired arms evaluate exactly the same record IDs, the panel's lineage
map covers exactly those records, and every record is a train-view record of a
train-derived panel with no protected access.  All paths are
repository-relative POSIX paths; source hashes are canonical (LF).

The gate is the FROZEN rule and is not changed: a comparison passes only if
the paired Kill@8 gain is >= +5 pp AND its 90% two-sided (one-sided 95%) lower
bound is > 0.  So a pass needs estimate >= max(5, 1.645 * SE).

    SE = sqrt((p_d - delta^2) / N) * sqrt(DE),  DE = 1 + (m - 1) * rho,

where p_d is the paired discordance, delta the true gain, m the targets per
repository and rho the within-repository correlation of paired differences.
Because a pass requires the estimate itself to reach +5 pp, a true gain of
exactly +5 pp passes with probability at most 50% at every N.

A clustered Monte Carlo simulation (repository random effects, cluster-robust
lower bound) cross-checks the analytic power for rho = 0, 0.05, 0.10, 0.20.

The artifact is deterministic.  It is published only after it has been
recomputed and verified (``verify_power_artifact``), with a single atomic file
replacement (``publish_file_atomically``).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import PublicationRefused, publish_file_atomically
from harness.closed_pilot_evidence import (
    EXECUTION_DOSE_RECEIPT, TOOL_ASSISTED_RECEIPT, evaluation_entries, load_json, sha256_file,
    verify_evaluation,
)
from harness.source_identity import canonical_sha256

SCHEMA = "oneiros_repository_native_power_analysis_v2"
SOURCE = "scripts/build_repository_native_power_analysis.py"
OUTPUT = "results/v4_3_repository_native_power_analysis.json"
CORPUS = "data/corpus/v4_1_research_hardened_candidate"
GATE_PP = 5.0
Z_ONE_SIDED_95 = 1.6448536269514722
N_VALUES = (100, 150, 200, 300, 400, 500)
EFFECTS_PP = (5.0, 8.0, 10.0, 15.0)
DISCORDANCE = (0.15, 0.21, 0.30, 0.40)
CENTRAL_DISCORDANCE = 0.21
#: The planning scenario the recommended N is powered for.
PLANNING_EFFECT_PP = 8.0
PLANNING_RHO = 0.05
RHO = (0.0, 0.05, 0.10, 0.20)
#: Repository scenarios: how many targets share a repository (m in the design effect).
REPOSITORY_SCENARIOS = {
    "fixed_25_repositories": lambda n: n / 25,
    "12_targets_per_repository": lambda n: 12.0,
    "cap_bound_8pct": lambda n: max(1.0, 0.08 * n),
}
REPOSITORY_COUNTS = (10, 25, 50, 100)
CAP_SHARES = (0.04, 0.08, 0.15)
MONTE_CARLO_RUNS = 4000
MONTE_CARLO_SEED = 20260925
YIELDS = (0.07, 0.10, 0.15)
#: Per-unit resource rates, from the design estimates (docs/next_direction_design_content.json).
RATES = {"network_hours_per_1000_candidates": (0.67, 1.33),
         "mining_cpu_hours_per_1000_candidates": 0.67,
         "clone_gb_per_1000_candidates": (2.7, 5.3),
         "heuristic_candidates_per_target": 3, "environment_build_minutes": 4,
         "parallel_jobs": 16, "native_test_seconds": 45, "qualification_repeats": 3,
         "environment_gb_per_target": 0.3, "generation_seconds_per_record": 6,
         "models": 2, "samples": 8, "atheris_cpu_seconds": 600, "atheris_seeds": 3,
         "atheris_modes": 2, "evaluation_gb_per_300_targets": 2}
TRAIN_DERIVED_PANELS = ("results/v4_3_execution_dose_retention_panel.json",
                        "results/v4_3_tool_assisted_panel.json")
EVIDENCE = {
    "retention_control_vs_base": ("execution_dose/base", "execution_dose/control",
                                  TRAIN_DERIVED_PANELS[0]),
    "retention_treatment_vs_control": ("execution_dose/control",
                                       "execution_dose/dose_treatment", TRAIN_DERIVED_PANELS[0]),
    "toolassist_b_vs_a": ("tool_assisted/A", "tool_assisted/B", TRAIN_DERIVED_PANELS[1]),
}


class EvidenceRefused(ValueError):
    pass


# --- evidence ----------------------------------------------------------------------

def recorded_panel_hash(root: Path, panel: str) -> str:
    if panel == TRAIN_DERIVED_PANELS[1]:
        return load_json(root, TOOL_ASSISTED_RECEIPT)["panel"]["sha256"]
    return load_json(root, EXECUTION_DOSE_RECEIPT)["tracked_design_artifacts"][panel]


def kill_at_8(path: Path) -> tuple[dict[str, int], list[str]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["record_id"] for item in result["function_results"]]
    values = {item["record_id"]: int(any(o.get("killed") for o in item["candidate_outcomes"]
                                         if int(o.get("rank", 0)) <= 8))
              for item in result["function_results"]}
    return values, ids


def train_record_ids(root: Path) -> set[str]:
    from harness.corpus_view import load_development_split
    return {str(record["id"]) for record in
            load_development_split(root / CORPUS, "train", include_excluded=True)}


def paired_evidence(root: Path, first: str, second: str, panel_path: str,
                    train_ids: set[str]) -> dict[str, Any]:
    entries = evaluation_entries(root)
    problems = verify_evaluation(root, entries[first]) + verify_evaluation(root, entries[second])
    panel_file = root / panel_path
    if not panel_file.is_file():
        raise EvidenceRefused(f"panel missing: {panel_path}")
    if sha256_file(panel_file) != recorded_panel_hash(root, panel_path):
        problems.append(f"panel hash differs from the decision receipt: {panel_path}")
    if problems:
        raise EvidenceRefused("; ".join(problems))
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    for name in (first, second):
        envelope = load_json(root, entries[name]["envelope"]["path"])
        if envelope.get("panel_record_ids_sha256") != panel["record_ids_sha256"]:
            problems.append(f"{name} was not evaluated on this panel")
    a, a_ids = kill_at_8(root / entries[first]["rehearsal_result"]["path"])
    b, b_ids = kill_at_8(root / entries[second]["rehearsal_result"]["path"])
    lineages = panel["record_lineages"]
    checks = {
        "no_duplicate_record_ids": len(a_ids) == len(set(a_ids)) and len(b_ids) == len(set(b_ids)),
        "paired_arms_have_identical_record_ids": set(a) == set(b),
        "panel_record_ids_equal_evaluated_records": set(panel["record_ids"]) == set(a),
        "panel_lineages_cover_exactly_the_evaluated_records": set(lineages) == set(a),
        "panel_is_train_derived": "train-derived" in panel["label"]
                                   and panel_path in TRAIN_DERIVED_PANELS,
        "panel_reports_no_protected_access": not any(
            value for value in panel["leakage"].values() if isinstance(value, bool)),
        "every_record_is_a_train_view_record": set(a) <= train_ids,
    }
    problems += [f"{first} vs {second}: {name}" for name, ok in checks.items() if not ok]
    if problems:
        raise EvidenceRefused("; ".join(problems))
    ids = sorted(a)
    diffs = {i: b[i] - a[i] for i in ids}
    groups: dict[str, list[int]] = defaultdict(list)
    for i in ids:
        groups[lineages[i]].append(diffs[i])
    n, k = len(ids), len(groups)
    mean = sum(diffs.values()) / n
    ssb = sum(len(v) * (statistics.mean(v) - mean) ** 2 for v in groups.values())
    ssw = sum((x - statistics.mean(v)) ** 2 for v in groups.values() for x in v)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - sum(len(v) ** 2 for v in groups.values()) / n) / (k - 1)
    denominator = msb + (n0 - 1) * msw
    inputs = {path: sha256_file(root / path) for path in sorted({
        entries[first]["envelope"]["path"], entries[first]["rehearsal_result"]["path"],
        entries[second]["envelope"]["path"], entries[second]["rehearsal_result"]["path"],
        panel_path})}
    return {"first": first, "second": second, "panel": panel_path,
            "records": n, "clusters": k, "mean_cluster_size": round(n / k, 3),
            "discordance": round(sum(1 for d in diffs.values() if d) / n, 4),
            "difference_pp": round(100 * mean, 3),
            "icc_of_paired_differences": round((msb - msw) / denominator, 4)
            if denominator else 0.0,
            "checks": checks, "inputs_sha256": inputs}


def collect_evidence(root: Path) -> dict[str, Any]:
    train_ids = train_record_ids(root)
    return {name: paired_evidence(root, first, second, panel, train_ids)
            for name, (first, second, panel) in EVIDENCE.items()}


def permitted_inputs(root: Path) -> set[str]:
    entries = evaluation_entries(root)
    paths = {*TRAIN_DERIVED_PANELS}
    for first, second, _ in EVIDENCE.values():
        for name in (first, second):
            paths |= {entries[name]["envelope"]["path"], entries[name]["rehearsal_result"]["path"]}
    return paths


# --- power ------------------------------------------------------------------------

def se_pp(n: int, discordance: float, delta_pp: float, rho: float, m: float) -> float:
    delta = delta_pp / 100
    variance = max(discordance - delta * delta, 1e-9) / n
    return 100 * math.sqrt(variance * (1 + (m - 1) * rho))


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def power(n: int, discordance: float, delta_pp: float, rho: float, m: float) -> float:
    se = se_pp(n, discordance, delta_pp, rho, m)
    threshold = max(GATE_PP, Z_ONE_SIDED_95 * se)
    return 1 - normal_cdf((threshold - delta_pp) / se)


def minimum_detectable_effect(n: int, discordance: float, rho: float, m: float) -> float:
    """Smallest true gain with >= 80% probability of passing the frozen gate."""
    low, high = 0.0, 60.0
    for _ in range(60):
        middle = (low + high) / 2
        if power(n, discordance, middle, rho, m) >= 0.8:
            high = middle
        else:
            low = middle
    return math.ceil(high * 100) / 100  # round UP: the reported MDE must reach 80%


def monte_carlo_power(n: int, discordance: float, delta_pp: float, runs: int = 4000,
                      seed: int = MONTE_CARLO_SEED) -> float:
    """Unclustered simulation of the exact paired rule."""
    return clustered_monte_carlo_power(n, discordance, delta_pp, 0.0, n, runs, seed)


def clustered_monte_carlo_power(n: int, discordance: float, delta_pp: float, rho: float,
                                repositories: int, runs: int = MONTE_CARLO_RUNS,
                                seed: int = MONTE_CARLO_SEED) -> float:
    """Simulated pass probability with repository random effects.

    Repository r has true gain delta + u_r, u_r ~ N(0, rho * (p_d - delta^2)),
    clipped to [-p_d, p_d], so the within-repository correlation of paired
    differences is rho.  Each target gains (+1) or loses (-1) with probabilities
    (p_d +/- delta_r) / 2.  The lower bound uses the cluster-robust (repository)
    variance, a normal approximation of the frozen cluster bootstrap.
    """
    rng = np.random.default_rng([seed, n, int(round(delta_pp * 100)), int(round(rho * 1000)),
                                 repositories])
    sizes = np.full(repositories, n // repositories)
    sizes[: n % repositories] += 1
    delta = delta_pp / 100
    sigma = math.sqrt(max(rho * (discordance - delta * delta), 0.0))
    offsets = rng.normal(0.0, sigma, size=(runs, repositories)) if sigma else \
        np.zeros((runs, repositories))
    true_gain = np.clip(delta + offsets, -discordance, discordance)
    gain = (discordance + true_gain) / 2
    loss = (discordance - true_gain) / 2
    gained = rng.binomial(sizes, gain)
    lost = rng.binomial(sizes - gained, np.divide(loss, 1 - gain, out=np.zeros_like(loss),
                                                  where=(1 - gain) > 0))
    sums = gained - lost
    estimate = sums.sum(axis=1) / n
    residual = sums - sizes * estimate[:, None]
    variance = repositories / (repositories - 1) * (residual ** 2).sum(axis=1) / (n * n)
    passed = (100 * estimate >= GATE_PP - 1e-9) & (
        estimate - Z_ONE_SIDED_95 * np.sqrt(variance) > 0)
    return round(float(passed.mean()), 4)


def resource_estimates(n: int) -> dict[str, Any]:
    r = RATES
    candidates = {f"yield={y}": math.ceil(n / y) for y in YIELDS}
    central = math.ceil(n / 0.10)
    heuristic = r["heuristic_candidates_per_target"] * n
    build_hours = heuristic * r["environment_build_minutes"] / 60 / r["parallel_jobs"]
    qualification_hours = (heuristic * 2 * r["qualification_repeats"] * r["native_test_seconds"]
                           / r["parallel_jobs"] / 3600)
    native_hours = (r["models"] * n * r["samples"] * 2 * r["native_test_seconds"]
                    / r["parallel_jobs"] / 3600)
    atheris_hours = (r["atheris_modes"] * n * r["atheris_cpu_seconds"] * r["atheris_seeds"]
                     / r["parallel_jobs"] / 3600)
    low, high = r["network_hours_per_1000_candidates"]
    clone_low, clone_high = r["clone_gb_per_1000_candidates"]
    return {
        "mined_candidates_needed": candidates,
        "repositories_needed": {"at_12_targets_each": math.ceil(n / 12),
                                "at_8pct_cap_minimum": math.ceil(1 / 0.08)},
        "mining": {"network_hours": [round(central / 1000 * low, 1),
                                     round(central / 1000 * high, 1)],
                   "cpu_hours": round(central / 1000 * r["mining_cpu_hours_per_1000_candidates"], 1),
                   "clone_storage_gb": [round(central / 1000 * clone_low),
                                        round(central / 1000 * clone_high)]},
        "environment_build_wall_hours": round(build_hours, 1),
        "environment_storage_gb": {"kept": round(n * r["environment_gb_per_target"]),
                                   "peak": round(heuristic * r["environment_gb_per_target"])},
        "qualification_wall_hours": round(qualification_hours, 1),
        "one_time_evaluation": {
            "gpu_minutes": round(r["models"] * n * r["generation_seconds_per_record"] / 60),
            "native_execution_wall_hours": round(native_hours, 1),
            "atheris_wall_hours": round(atheris_hours, 1),
            "storage_gb": round(r["evaluation_gb_per_300_targets"] * n / 300, 1)},
        "total_cpu_wall_hours": round(build_hours + qualification_hours + native_hours
                                      + atheris_hours + 1 + central / 1000
                                      * r["mining_cpu_hours_per_1000_candidates"], 1),
        "peak_storage_gb": round(heuristic * r["environment_gb_per_target"]
                                 + central / 1000 * clone_high + 3),
    }


def planning_repositories(n: int) -> int:
    return math.ceil(n / 12)


def smallest_powered_n(effect: float, rho: float, scenario: str,
                       discordance: float = CENTRAL_DISCORDANCE) -> int | None:
    m_of = REPOSITORY_SCENARIOS[scenario]
    for n in N_VALUES:
        if power(n, discordance, effect, rho, m_of(n)) >= 0.8:
            return n
    return None


def build_report(root: Path) -> dict[str, Any]:
    evidence = collect_evidence(root)
    table: dict[str, Any] = {}
    for n in N_VALUES:
        m25 = REPOSITORY_SCENARIOS["fixed_25_repositories"](n)
        table[str(n)] = {
            "power_analytic": {scenario: {f"rho={rho}": {
                f"{e:g}pp": round(power(n, CENTRAL_DISCORDANCE, e, rho, m_of(n)), 3)
                for e in EFFECTS_PP} for rho in RHO}
                for scenario, m_of in REPOSITORY_SCENARIOS.items()},
            "power_clustered_monte_carlo_25_repositories": {f"rho={rho}": {
                f"{e:g}pp": clustered_monte_carlo_power(n, CENTRAL_DISCORDANCE, e, rho, 25)
                for e in EFFECTS_PP} for rho in RHO},
            "power_clustered_monte_carlo_12_targets_per_repository": {f"rho={rho}": {
                f"{e:g}pp": clustered_monte_carlo_power(n, CENTRAL_DISCORDANCE, e, rho,
                                                        planning_repositories(n))
                for e in EFFECTS_PP} for rho in RHO},
            "mde_80pct_pp": {scenario: {f"d={d}": {f"rho={rho}": minimum_detectable_effect(
                n, d, rho, m_of(n)) for rho in RHO} for d in DISCORDANCE}
                for scenario, m_of in REPOSITORY_SCENARIOS.items()},
            "lower_bound_binds_above_gate_rho0.05_25_repositories": Z_ONE_SIDED_95 * se_pp(
                n, CENTRAL_DISCORDANCE, 0.0, 0.05, m25) > GATE_PP,
            "resources": resource_estimates(n),
        }
    sensitivity = {str(n): {
        "mde_by_repository_count": {f"repositories={count}": {
            f"rho={rho}": minimum_detectable_effect(n, CENTRAL_DISCORDANCE, rho,
                                                    max(1.0, n / count))
            for rho in (0.05, 0.10)} for count in REPOSITORY_COUNTS},
        "mde_by_per_repository_cap_worst_case": {f"cap={cap}": {
            f"rho={rho}": minimum_detectable_effect(n, CENTRAL_DISCORDANCE, rho,
                                                    max(1.0, cap * n))
            for rho in (0.05, 0.10)} for cap in CAP_SHARES}} for n in N_VALUES}
    smallest = {scenario: {f"{effect:g}pp": {f"rho={rho}": smallest_powered_n(
        effect, rho, scenario) for rho in RHO} for effect in EFFECTS_PP}
        for scenario in REPOSITORY_SCENARIOS}
    at_gate = {str(n): round(power(n, CENTRAL_DISCORDANCE, GATE_PP, 0.0, 1.0), 3)
               for n in N_VALUES}
    planning = {"true_effect_pp": PLANNING_EFFECT_PP, "discordance": CENTRAL_DISCORDANCE,
                "rho": PLANNING_RHO, "repository_scenario": "12_targets_per_repository",
                "per_repository_cap_targets": 12}

    def planning_power(n: int) -> dict[str, float]:
        mc = table[str(n)]["power_clustered_monte_carlo_12_targets_per_repository"]
        return {"analytic": round(power(n, CENTRAL_DISCORDANCE, PLANNING_EFFECT_PP,
                                        PLANNING_RHO, 12.0), 3),
                "clustered_monte_carlo": mc[f"rho={PLANNING_RHO}"][f"{PLANNING_EFFECT_PP:g}pp"]}

    # Recommended: the smallest N at which BOTH the analytic and the clustered
    # simulation reach 80% for the planning scenario.
    planned_n = next((n for n in N_VALUES if min(planning_power(n).values()) >= 0.8), None)
    recommendation = {
        "planning_scenario": planning,
        "recommended_final_n": planned_n,
        "power_at_recommendation": planning_power(planned_n) if planned_n else None,
        "power_at_planning_effect_by_n": {str(n): planning_power(n) for n in N_VALUES},
        "repositories_required_at_recommendation": planning_repositories(planned_n)
        if planned_n else None,
        "composition_requirement": ("at most 12 targets per repository; the draft 8% cap "
                                    "allows 32 per repository at N=400 and is insufficient"),
        "feasibility_option_n": 200,
        "feasibility_option_label": ("lower-power compromise, not adequately powered for the "
                                     "planning scenario; usable only with explicit approval"),
        "feasibility_option_power_at_planning_effect": planning_power(200),
        "effect_exactly_at_gate": (
            "a true gain of exactly +5 pp passes with probability at most 50% at every N, "
            "because the estimate itself must reach +5 pp; 80% power is impossible there"),
        "reasoning": [
            "the planning effect is +8 pp: the smallest round effect meaningfully above the "
            "+5 pp gate (the largest paired gain observed so far is +6.4 pp, post hoc)",
            "rho = 0.05 is the planning clustering; historical lineage clustering of paired "
            "differences is near zero, and 0.10/0.20 are reported as sensitivity",
            "12 targets per repository keeps the design effect fixed as N grows; with a fixed "
            "25 repositories the design effect grows with N and larger N buys little",
            "the +5 pp gate is not changed to fit the sample",
        ],
    }
    return {
        "schema_version": SCHEMA,
        "label": "prospective design calculation from permitted train-derived evidence only",
        "gate": {"min_gain_pp": GATE_PP, "lower_bound": "one-sided 95% (two-sided 90%) > 0"},
        "evidence": evidence,
        "permitted_inputs": sorted(permitted_inputs(root)),
        "scenarios": {"discordance": DISCORDANCE, "central_discordance": CENTRAL_DISCORDANCE,
                      "rho": RHO, "repository_scenarios": sorted(REPOSITORY_SCENARIOS),
                      "design_effect": "1 + (m - 1) * rho",
                      "monte_carlo": {"runs": MONTE_CARLO_RUNS, "seed": MONTE_CARLO_SEED,
                                      "repositories": "25, and ceil(N / 12)",
                                      "lower_bound": "cluster-robust normal approximation"}},
        "power_by_n": table,
        "sensitivity": sensitivity,
        "smallest_n_reaching_80pct_power": smallest,
        "power_at_true_effect_equal_to_gate": at_gate,
        "resource_rates": RATES,
        "recommendation": recommendation,
        "source_sha256": canonical_sha256(root / SOURCE),
    }


def encode(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_power_artifact(root: Path, data: bytes) -> dict[str, Any]:
    """Refuse unless the artifact is exactly what the verified evidence produces now."""
    report = json.loads(data.decode("utf-8"))
    problems = []
    if report.get("schema_version") != SCHEMA:
        problems.append("schema_version")
    if report.get("source_sha256") != canonical_sha256(root / SOURCE):
        problems.append("source hash differs from the current script")
    if (report.get("gate") or {}).get("min_gain_pp") != 5.0 or GATE_PP != 5.0:
        problems.append("the +5 pp gate changed")
    permitted = permitted_inputs(root)
    for name, entry in (report.get("evidence") or {}).items():
        for path, digest in (entry.get("inputs_sha256") or {}).items():
            if path not in permitted or Path(path).is_absolute() or "\\" in path:
                problems.append(f"{name}: input not permitted or not repository-relative: {path}")
            elif sha256_file(root / path) != digest:
                problems.append(f"{name}: evidence hash differs: {path}")
    if set(report.get("evidence") or {}) != set(EVIDENCE):
        problems.append("evidence set differs")
    if not problems and report != json.loads(encode(build_report(root)).decode("utf-8")):
        problems.append("artifact differs from a fresh recomputation")
    if problems:
        raise EvidenceRefused("power artifact refused: " + "; ".join(problems))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args(argv)
    try:
        data = encode(build_report(ROOT))
        publish_file_atomically(args.output, data, verify=lambda staged: verify_power_artifact(
            ROOT, staged.read_bytes()))
    except (EvidenceRefused, PublicationRefused) as exc:
        print(f"REFUSED: {exc}")
        return 2
    report = json.loads(data)
    print(json.dumps({"recommendation": report["recommendation"]["recommended_final_n"],
                      "sha256": sha256_file(args.output)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
