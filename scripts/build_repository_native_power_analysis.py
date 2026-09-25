"""Prospective power analysis for the repository-native evaluation (CPU only).

Inputs are permitted historical evidence only: paired per-record Kill@8 from
the train-derived retention panel (613 records, 100 lineages) and the
tool-assisted panel (264 records, 110 lineages).  From them it measures paired
discordance and the within-cluster correlation of paired differences.

The gate is the FROZEN rule and is not changed: a comparison passes only if
the paired Kill@8 gain is >= +5 pp AND its 90% two-sided (one-sided 95%) lower
bound is > 0.  So a pass needs estimate >= max(5, 1.645 * SE).

    SE = sqrt((p_d - delta^2) / N) * sqrt(DE),  DE = 1 + (m - 1) * rho,

where p_d is the paired discordance, delta the true gain, m the mean number of
targets per repository and rho the within-repository correlation of paired
differences.  Repository clustering in NEW data is unknown, so rho is swept
conservatively rather than taken from the near-zero historical lineage value.
A Monte Carlo simulation cross-checks the analytic power.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GATE_PP = 5.0
Z_ONE_SIDED_95 = 1.6448536269514722
Z_POWER_80 = 0.8416212335729143
N_VALUES = (100, 150, 200, 300)
EFFECTS_PP = (5.0, 8.0, 10.0, 15.0)
DISCORDANCE = (0.15, 0.21, 0.30, 0.40)
RHO = (0.0, 0.05, 0.10, 0.20)
REPOSITORIES = 25
CAP_SHARE = 0.08
YIELDS = (0.07, 0.10, 0.15)
EVIDENCE = {
    "retention_control_vs_base": ("results/v4_3_execution_dose_v1/retention_base/rehearsal_result.json",
                                  "results/v4_3_execution_dose_v1/retention_control/rehearsal_result.json",
                                  "results/v4_3_execution_dose_retention_panel.json"),
    "retention_treatment_vs_control": (
        "results/v4_3_execution_dose_v1/retention_control/rehearsal_result.json",
        "results/v4_3_execution_dose_v1/retention_dose_treatment/rehearsal_result.json",
        "results/v4_3_execution_dose_retention_panel.json"),
    "toolassist_b_vs_a": ("results/v4_3_tool_assisted_v1/eval_A/rehearsal_result.json",
                          "results/v4_3_tool_assisted_v1/eval_B/rehearsal_result.json",
                          "results/v4_3_tool_assisted_panel.json"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kill_at_8(path: Path) -> dict[str, int]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return {item["record_id"]: int(any(o.get("killed") for o in item["candidate_outcomes"]
                                       if int(o.get("rank", 0)) <= 8))
            for item in result["function_results"]}


def paired_evidence(first: Path, second: Path, panel: Path) -> dict:
    a, b = kill_at_8(first), kill_at_8(second)
    lineages = json.loads(panel.read_text(encoding="utf-8"))["record_lineages"]
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
    return {"records": n, "clusters": k, "mean_cluster_size": round(n / k, 3),
            "discordance": round(sum(1 for d in diffs.values() if d) / n, 4),
            "difference_pp": round(100 * mean, 3),
            "icc_of_paired_differences": round((msb - msw) / denominator, 4) if denominator else 0.0,
            "inputs_sha256": {str(p): sha(ROOT / p) for p in (first, second, panel)}}


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
                      seed: int = 20260925) -> float:
    """Unclustered simulation of the exact paired rule (cross-check of the analytic)."""
    generator = random.Random(seed)
    delta = delta_pp / 100
    gain, loss = (discordance + delta) / 2, (discordance - delta) / 2
    passes = 0
    for _ in range(runs):
        gained = lost = 0
        for _ in range(n):
            u = generator.random()
            if u < gain:
                gained += 1
            elif u < gain + loss:
                lost += 1
        estimate = (gained - lost) / n
        variance = (gained + lost - (gained - lost) ** 2 / n) / (n * n)
        if 100 * estimate >= GATE_PP and estimate - Z_ONE_SIDED_95 * math.sqrt(variance) > 0:
            passes += 1
    return passes / runs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_repository_native_power_analysis.json")
    args = parser.parse_args(argv)
    evidence = {name: paired_evidence(ROOT / a, ROOT / b, ROOT / panel)
                for name, (a, b, panel) in EVIDENCE.items()}
    central_d = 0.21
    central_rho = 0.05
    table = {}
    for n in N_VALUES:
        m_central, m_worst = n / REPOSITORIES, max(1.0, CAP_SHARE * n)
        table[str(n)] = {
            "targets_per_repository_central": m_central,
            "targets_per_repository_at_cap": m_worst,
            "power_central": {f"{e:g}pp": round(power(n, central_d, e, central_rho, m_central), 3)
                              for e in EFFECTS_PP},
            "mde_80pct_pp": {f"d={d}": {f"rho={r}": minimum_detectable_effect(
                n, d, r, m_central) for r in RHO} for d in DISCORDANCE},
            "mde_80pct_pp_at_cap_rho_0.10": {f"d={d}": minimum_detectable_effect(
                n, d, 0.10, m_worst) for d in DISCORDANCE},
            "lower_bound_binds_above_gate_central": Z_ONE_SIDED_95 * se_pp(
                n, central_d, 0.0, central_rho, m_central) > GATE_PP,
            "monte_carlo_unclustered_power_d0.21": {
                f"{e:g}pp": monte_carlo_power(n, central_d, e) for e in (8.0, 10.0)},
            "analytic_unclustered_power_d0.21": {
                f"{e:g}pp": round(power(n, central_d, e, 0.0, 1.0), 3) for e in (8.0, 10.0)},
        }
    attrition = {f"N={n}": {f"yield={y}": math.ceil(n / y) for y in YIELDS} for n in N_VALUES}
    recommendation = {
        "recommended_final_n": 300,
        "minimum_acceptable_n": 200,
        "reasoning": [
            "the frozen gate needs the estimate to reach +5 pp AND a positive lower bound; in "
            "the central scenario the lower-bound requirement binds above 5 pp at every N "
            "considered (about 8.1 pp at N=100, 5.4 pp at N=300), so small N passes only large "
            "true effects",
            "at N=300 (central discordance 0.21, rho 0.05, 25 repositories) the 80%-power "
            "minimum detectable effect is about 8 pp; at N=100 it is about 12 pp",
            "historical lineage clustering of paired differences is near zero, but repository "
            "clustering in new data is unknown, so rho up to 0.20 is reported",
            "the +5 pp gate is not changed to fit the sample; with the expected mining yield, "
            "N=300 needs roughly 2,000-4,300 mined candidate fix commits",
        ],
    }
    report = {
        "schema_version": "oneiros_repository_native_power_analysis_v1",
        "label": "prospective design calculation from permitted train-derived evidence only",
        "gate": {"min_gain_pp": GATE_PP, "lower_bound": "one-sided 95% (two-sided 90%) > 0"},
        "evidence": evidence,
        "scenarios": {"discordance": DISCORDANCE, "central_discordance": central_d,
                      "rho": RHO, "central_rho": central_rho,
                      "repositories_central": REPOSITORIES, "per_repository_cap": CAP_SHARE,
                      "design_effect": "1 + (m - 1) * rho"},
        "power_by_n": table,
        "attrition_required_mining_pool": attrition,
        "recommendation": recommendation,
        "source_sha256": sha(Path(__file__)),
    }
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({n: {"power_central": row["power_central"],
                          "mde_d0.21_rho0.05": row["mde_80pct_pp"]["d=0.21"]["rho=0.05"],
                          "mc_vs_analytic": [row["monte_carlo_unclustered_power_d0.21"],
                                             row["analytic_unclustered_power_d0.21"]]}
                      for n, row in table.items()}, indent=1))
    print(json.dumps({k: {x: y for x, y in v.items() if x != "inputs_sha256"}
                      for k, v in evidence.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
