"""v2.6 Phase 2: acquisition projection v2 (successor; v2.5 and v2.6-v1 projections are kept).

Method: REPOSITORY-CLUSTER BOOTSTRAP PLANNING SIMULATION (exploratory planning evidence only;
not a Bayesian posterior-predictive analysis). Clusters: the 20 repositories scanned by the
earlier fresh pilot (admissions), the v2.6 a3 projects (environment qualification), and the a3
projects with qualified environments (verified unique tests per qualified lineage). Each draw
resamples clusters, then draws per-repository admissions and binomial qualification/positives.

Corrections over v1:
- shortfalls are DERIVED from the current observed selection (conservation receipt) and the
  frozen 150/8/60 gate, never hard-coded;
- every threshold is checked against the SIMULATOR'S SUPPORT: the pilot never admitted more
  than one fix commit per repository, so per-repository admissions in this simulator lie in
  {0, 1} and every count is bounded by the pool size. A threshold outside [support_min,
  support_max] gets probability null and status NOT_ESTIMABLE_OUTSIDE_SUPPORT - which means
  the simulator cannot speak to it, NOT that it is impossible;
- a probability may be 0.0 only for a threshold inside the support;
- the number of production lineages per repository (several fix commits per repository under
  the real per-function / per-repository caps) is UNKNOWN: the pilot cannot estimate it.
A projection never passes or fails an observed-data gate.

    python scripts/v26_projection_v2.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v26_projection_v2.json"
PILOT = "results/v4_3_repository_native_aprime_fresh_confirmation_pilot.json"
A3 = "results/sft_root_cause_v26_repository_verification_a3.json"
CONSERVATION = "results/sft_root_cause_v26_selection_conservation.json"
PARTITION = "results/sft_root_cause_v25_universe_partition.json"
METHOD = "repository-cluster bootstrap planning simulation"
SEED, DRAWS = 20261003, 10_000
GATE = {"repository_tests": 150, "repositories": 8, "lineages": 60}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def shortfall(observed: Dict[str, int], gate: Dict[str, int] = GATE) -> Dict[str, int]:
    return {k: max(0, gate[k] - int(observed[k])) for k in gate}


def threshold_probability(draws: Sequence[int], threshold: int, support: Sequence[int]) -> Dict:
    lo, hi = support
    exceed = sum(d >= threshold for d in draws)
    if threshold > hi or threshold < lo:
        return {"threshold": threshold, "probability": None,
                "status": "NOT_ESTIMABLE_OUTSIDE_SUPPORT", "support": [lo, hi],
                "simulated_exceedances": exceed,
                "meaning": "outside what this simulator can produce; NOT impossible"}
    return {"threshold": threshold, "probability": exceed / len(draws), "status": "ESTIMATED",
            "support": [lo, hi], "simulated_exceedances": exceed}


def summary(values: List[int]) -> Dict:
    s = sorted(values)
    q = lambda p: s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    return {"expected": round(sum(s) / len(s), 2), "p05": q(0.05), "median": q(0.5),
            "p95": q(0.95), "observed_min_draw": s[0], "observed_max_draw": s[-1]}


def clusters() -> Dict:
    pilot = json.loads((ROOT / PILOT).read_text(encoding="utf-8"))
    scanned = sorted(pilot["repositories"]["scans"])
    adm = [int(pilot["admitted_per_repository"].get(r, 0)) for r in scanned]
    a3 = json.loads((ROOT / A3).read_text(encoding="utf-8"))
    cats = a3["environment_categories"]
    qual = [(sum(c.values()), c.get("qualified", 0)) for c in cats.values() if sum(c.values())]
    sel = json.loads((ROOT / CONSERVATION).read_text(encoding="utf-8"))
    selected = [json.loads(l) for l in (ROOT / sel["sidecar"]["path"]).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    pos_by = {}
    for r in selected:
        if r["outcome"] == "selected":
            pos_by[r["project"]] = pos_by.get(r["project"], 0) + 1
    pos = [(c.get("qualified", 0), pos_by.get(p, 0)) for p, c in cats.items()
           if c.get("qualified", 0)]
    return {"admissions_per_scanned_repository": adm, "qualification": qual, "positives": pos}


def simulate(pool: int, c: Dict, draws: int = DRAWS, seed: int = SEED) -> Dict:
    rng = random.Random(seed)
    out = {"admitted": [], "qualified": [], "positives": [], "repos_with_positive": [],
           "repos_with_qualified": []}
    for _ in range(draws):
        a = [rng.choice(c["admissions_per_scanned_repository"])
             for _ in c["admissions_per_scanned_repository"]]
        q = [rng.choice(c["qualification"]) for _ in c["qualification"]]
        p = [rng.choice(c["positives"]) for _ in c["positives"]]
        q_rate = sum(x for _, x in q) / max(1, sum(n for n, _ in q))
        p_rate = sum(x for _, x in p) / max(1, sum(n for n, _ in p))
        per_repo = [rng.choice(a) for _ in range(pool)]
        qual = [sum(rng.random() < q_rate for _ in range(n)) for n in per_repo]
        posi = [sum(rng.random() < p_rate for _ in range(n)) for n in qual]
        out["admitted"].append(sum(per_repo))
        out["qualified"].append(sum(qual))
        out["positives"].append(sum(posi))
        out["repos_with_positive"].append(sum(1 for n in posi if n))
        out["repos_with_qualified"].append(sum(1 for n in qual if n))
    return out


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    cons = json.loads((ROOT / CONSERVATION).read_text(encoding="utf-8"))
    sel = cons["funnel"]["selected"]
    observed = {"repository_tests": sel["rows"], "repositories": sel["repositories"],
                "lineages": sel["lineages"]}
    short = shortfall(observed)
    c = clusters()
    part = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    train_pool = len(part["training_expansion_pool"])
    conf_pool = len(part["confirmation_only_pool"])
    per_repo_max = max(c["admissions_per_scanned_repository"])
    tr, cf = simulate(train_pool, c), simulate(conf_pool, c, seed=SEED + 1)
    tr_support, cf_support = (0, train_pool * per_repo_max), (0, conf_pool * per_repo_max)
    receipt = {
        "schema_version": "oneiros_v26_projection_v2",
        "method": METHOD, "evidence_class": "exploratory planning evidence only",
        "is_observed_evidence": False, "draws": DRAWS, "seed": SEED,
        "observed_base": {**observed, "source": CONSERVATION},
        "derived_shortfall": short,
        "simulator_support": {"per_repository_admissions": [0, per_repo_max],
                              "restriction": "the pilot admitted at most one fix commit per "
                                             "scanned repository; resampling cannot exceed it",
                              "training_pool_counts": list(tr_support),
                              "confirmation_pool_counts": list(cf_support)},
        "training_pool": {
            "repositories": train_pool,
            "admitted_fix_commits": summary(tr["admitted"]),
            "verified_positives": summary(tr["positives"]),
            "new_repositories_with_a_positive": summary(tr["repos_with_positive"]),
            "thresholds": {
                "additional_tests": threshold_probability(tr["positives"],
                                                          short["repository_tests"], tr_support),
                "additional_repositories": threshold_probability(
                    tr["repos_with_positive"], short["repositories"], tr_support),
                "additional_lineages": threshold_probability(tr["positives"], short["lineages"],
                                                             tr_support)}},
        "confirmation_pool": {
            "repositories": conf_pool,
            "qualified_targets": summary(cf["qualified"]),
            "thresholds": {"targets_80": threshold_probability(cf["qualified"], 80, cf_support),
                           # a panel needs QUALIFIED targets, not training positives
                           "repositories_10": threshold_probability(
                               cf["repos_with_qualified"], 10, cf_support)}},
        "unknown_quantities": [
            "fix commits admitted per repository under production acquisition (pilot: <= 1)",
            "qualifying lineages per repository under the real per-function / per-repository "
            "caps", "transfer of historical rates to the new pools"],
        "cluster_data": c,
        "inputs_sha256": {p: sha(p) for p in (PILOT, A3, CONSERVATION, PARTITION)}}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "observed": observed, "shortfall": short,
                      "training": receipt["training_pool"],
                      "confirmation": receipt["confirmation_pool"],
                      "sha256": sha(RECEIPT) if (ROOT / RECEIPT).exists() else None}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
