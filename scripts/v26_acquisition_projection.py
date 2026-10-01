"""v2.6 Phase 1B: corrected acquisition projection (successor of the v2.5 projection, which is
kept unchanged as history; its "upper_bound" field was an expected value, not a bound).

Separates: OBSERVED counts; DETERMINISTIC structural maxima (only where one exists); EXPECTED
(point) values; UNCERTAINTY intervals; PROBABILITIES of reaching thresholds; and pessimistic /
central / optimistic planning scenarios. Method: repository-clustered bootstrap posterior-
predictive simulation (seeded):
- admission clusters: the 20 repositories the earlier fresh confirmation pilot scanned (5
  admitted one fix commit each, 15 admitted none);
- qualification clusters: the v2.5 r2 train projects (qualified / attempted environments);
- positive clusters: the v2.5 r2 projects with qualified environments (verified unique tests /
  qualified lineages).
Each draw resamples clusters for every rate, then draws per-repository admissions for the new
pool from the resampled pilot repositories and binomial qualification and positives.

LIMITATIONS: historical rates come from different repositories and mechanisms and may not
transfer; the empirical pilot never admitted more than one fix commit per repository, so the
simulated per-repository admissions cannot exceed one - an artefact of the data, NOT a
deterministic bound on reality. A projection never passes or fails an observed-data gate.

    python scripts/v26_acquisition_projection.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v26_acquisition_projection.json"
PILOT = "results/v4_3_repository_native_aprime_fresh_confirmation_pilot.json"
VERIFY = "results/sft_root_cause_v25_repository_verification_r2.json"
PARTITION = "results/sft_root_cause_v25_universe_partition.json"
SEED, DRAWS = 20261002, 10_000
THEFUCK = {"attempted": 24, "qualified": 24}       # v2.5 Phase 6 (not rebuilt in r2)


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def clusters() -> dict:
    pilot = json.loads((ROOT / PILOT).read_text(encoding="utf-8"))
    scanned = sorted(pilot["repositories"]["scans"])
    admitted = pilot["admitted_per_repository"]
    adm = [int(admitted.get(r, 0)) for r in scanned]
    v = json.loads((ROOT / VERIFY).read_text(encoding="utf-8"))
    cats = {p: c for p, c in v["environment_categories"].items() if sum(c.values())}
    qual = [(sum(c.values()), c.get("qualified", 0)) for c in cats.values()]
    qual.append((THEFUCK["attempted"], THEFUCK["qualified"]))
    qualified = {p: c.get("qualified", 0) for p, c in cats.items()}
    qualified["thefuck"] = THEFUCK["qualified"]
    pos_by = v["attainable_gate"]["by_project"]
    pos = [(n, pos_by.get(p, 0)) for p, n in qualified.items() if n]
    return {"admissions_per_scanned_repository": adm, "qualification": qual,
            "positives": pos}


def percentile(sorted_values: list, q: float) -> float:
    i = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[i]


def simulate(pool: int, c: dict, draws: int = DRAWS, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    adm_c, qual_c, pos_c = (c["admissions_per_scanned_repository"], c["qualification"],
                            c["positives"])
    out = {"admitted": [], "qualified": [], "positives": [], "repos_with_positive": []}
    rates = {"admit": [], "qualify": [], "positive": []}
    for _ in range(draws):
        a_boot = [rng.choice(adm_c) for _ in adm_c]
        q_boot = [rng.choice(qual_c) for _ in qual_c]
        p_boot = [rng.choice(pos_c) for _ in pos_c]
        q_rate = sum(q for _, q in q_boot) / max(1, sum(n for n, _ in q_boot))
        p_rate = sum(p for _, p in p_boot) / max(1, sum(n for n, _ in p_boot))
        rates["admit"].append(sum(a_boot) / len(a_boot))
        rates["qualify"].append(q_rate)
        rates["positive"].append(p_rate)
        per_repo = [rng.choice(a_boot) for _ in range(pool)]
        admitted = sum(per_repo)
        qualified_repo = [sum(rng.random() < q_rate for _ in range(n)) for n in per_repo]
        positive_repo = [sum(rng.random() < p_rate for _ in range(n)) for n in qualified_repo]
        out["admitted"].append(admitted)
        out["qualified"].append(sum(qualified_repo))
        out["positives"].append(sum(positive_repo))
        out["repos_with_positive"].append(sum(1 for n in positive_repo if n))
    return {"draws": out, "rates": rates}


def summary(values: list) -> dict:
    s = sorted(values)
    return {"expected": round(sum(s) / len(s), 2), "p05": percentile(s, 0.05),
            "median": percentile(s, 0.5), "p95": percentile(s, 0.95)}


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    c = clusters()
    part = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    v = json.loads((ROOT / VERIFY).read_text(encoding="utf-8"))
    observed = v["attainable_gate"]
    short = observed["shortfall"]
    train_pool, conf_pool = len(part["training_expansion_pool"]), len(part["confirmation_only_pool"])
    tr = simulate(train_pool, c)
    cf = simulate(conf_pool, c, seed=SEED + 1)
    rates = {k: sorted(vals) for k, vals in tr["rates"].items()}
    scen = {name: {k: round(percentile(rates[k], q), 4) for k in rates}
            for name, q in (("pessimistic", 0.05), ("central", 0.5), ("optimistic", 0.95))}
    for name, r in scen.items():
        r["expected_positives_training_pool"] = round(
            train_pool * r["admit"] * r["qualify"] * r["positive"], 2)
        r["expected_qualified_targets_confirmation_pool"] = round(
            conf_pool * r["admit"] * r["qualify"], 2)
    d = tr["draws"]
    receipt = {
        "schema_version": "oneiros_v26_acquisition_projection_v1",
        "supersedes_wording_of": {"receipt": "results/sft_root_cause_v25_acquisition_projection.json",
                                  "sha256": sha("results/sft_root_cause_v25_acquisition_projection.json"),
                                  "erratum": "'upper_bound_if_every_admitted_commit_became_a_"
                                             "positive' (10.6) was an EXPECTED admitted count, "
                                             "not an upper bound; 1.9 was a point estimate"},
        "observed": {"verified_unique_repository_tests": observed["repository_tests"],
                     "repositories": observed["repositories"],
                     "lineages": observed["lineages"], "shortfall_to_gate": short},
        "deterministic_structural_maximum": {
            "existing_13_repositories": {"eligible_fragments_after_dedup_and_caps": 150,
                                         "source": "results/sft_root_cause_v25_attainable_gate_decision.json"},
            "new_repositories": "none: the number of fix commits a new repository can supply is "
                                "not bounded by any recorded quantity"},
        "method": {"type": "repository-clustered bootstrap posterior-predictive simulation",
                   "draws": DRAWS, "seed": SEED, "clusters": {k: len(v_) for k, v_ in c.items()},
                   "cluster_data": c},
        "training_pool": {"repositories": train_pool,
                          "admitted_fix_commits": summary(d["admitted"]),
                          "qualified": summary(d["qualified"]),
                          "verified_positives": summary(d["positives"]),
                          "new_repositories_with_a_positive": summary(d["repos_with_positive"]),
                          "probability": {
                              "positives_>=_shortfall_122": sum(x >= short["repository_tests"]
                                                                for x in d["positives"]) / DRAWS,
                              "new_repositories_>=_5": sum(x >= short["repositories"]
                                                           for x in d["repos_with_positive"])
                              / DRAWS}},
        "confirmation_pool": {"repositories": conf_pool,
                              "qualified_targets": summary(cf["draws"]["qualified"]),
                              "probability_>=_80_targets": sum(
                                  x >= 80 for x in cf["draws"]["qualified"]) / DRAWS},
        "scenarios": scen,
        "rate_intervals_90": {k: [round(percentile(v_, 0.05), 4), round(percentile(v_, 0.95), 4)]
                              for k, v_ in rates.items()},
        "limitations": [
            "historical rates come from other repositories and mechanisms and may not transfer "
            "to the new pools",
            "the pilot never admitted more than one fix commit per scanned repository, so "
            "simulated per-repository admissions cannot exceed one: a data artefact, not a bound",
            "a projection never passes or fails an observed-data gate; only observed, verified, "
            "deduplicated rows can"],
        "is_observed_evidence": False,
        "inputs_sha256": {p: sha(p) for p in (PILOT, VERIFY, PARTITION)}}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "training_pool": receipt["training_pool"],
                      "confirmation_pool": receipt["confirmation_pool"],
                      "scenarios": scen,
                      "sha256": sha(RECEIPT) if (ROOT / RECEIPT).exists() else None},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
