"""v2.5 CPU program 2, Phase 11: reproducible power simulation for the confirmation panel
(addendum 1 section 3). Stdlib only; fully seeded; every parameter is in the receipt.

Generative model per simulated panel:
- R repositories, n targets each; repository random effect u_r ~ N(0, SD^2) on the logit scale
  (SD = 1.1), shared by both arms;
- per target and seed, arm A kills (Kill@8 for that seed) with p_A = logistic(logit(base) + u_r)
  and arm C with p_C = logistic(logit(base) + delta + u_r), where delta is chosen so that the
  marginal effect at u = 0 is ``effect`` points (logit(base + effect) - logit(base));
- target Kill@8 = mean over 3 seeds; paired difference C - A per target; repository means;
  two-sided sign-flip test (exact over 2^R sign vectors, harness/v25_inference.py) at alpha.

    python scripts/v25_power_simulation.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = "results/sft_root_cause_v25_power_simulation.json"
PARAMS = {"repository_logit_sd": 1.1, "seeds_per_target": 3, "alpha": 0.05,
          "simulations": 2000, "seed": 20261001,
          "panels": [{"repositories": 10, "per_repository": 8},
                     {"repositories": 8, "per_repository": 8},
                     {"repositories": 5, "per_repository": 8}],
          "base_rates": [0.0, 0.05, 0.10], "effects": [0.0, 0.10, 0.15]}


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def simulate_panel(rng: random.Random, repositories: int, per_repository: int, base: float,
                   effect: float, sd: float, seeds: int) -> list:
    """-> [(repository, paired target difference)]."""
    base = min(max(base, 1e-4), 1 - 1e-4)
    delta = logit(min(base + effect, 1 - 1e-4)) - logit(base)
    rows = []
    for r in range(repositories):
        u = rng.gauss(0.0, sd)
        pa, pc = logistic(logit(base) + u), logistic(logit(base) + delta + u)
        for _ in range(per_repository):
            a = sum(rng.random() < pa for _ in range(seeds)) / seeds
            c = sum(rng.random() < pc for _ in range(seeds)) / seeds
            rows.append((f"r{r}", c - a))
    return rows


def power(repositories: int, per_repository: int, base: float, effect: float,
          params=PARAMS) -> float:
    from harness.v25_inference import repository_means, sign_flip_test
    seed_text = f"{params['seed']}:{repositories}:{per_repository}:{base}:{effect}"
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    hits = 0
    for _ in range(params["simulations"]):
        rows = simulate_panel(rng, repositories, per_repository, base, effect,
                              params["repository_logit_sd"], params["seeds_per_target"])
        means = list(repository_means(rows).values())
        if all(m == 0 for m in means):
            continue                                   # no information: never a rejection
        if sign_flip_test(means)["p_value"] <= params["alpha"]:
            hits += 1
    return hits / params["simulations"]


def main() -> int:
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    table = []
    for panel in PARAMS["panels"]:
        for base in PARAMS["base_rates"]:
            for effect in PARAMS["effects"]:
                pw = power(panel["repositories"], panel["per_repository"], base, effect)
                table.append({**panel, "targets": panel["repositories"] *
                              panel["per_repository"], "base": base, "effect": effect,
                              "power": round(pw, 4)})
                print(json.dumps(table[-1]), flush=True)
    claims = {
        "addendum1_claim_base<=0.05_effect0.10_power>=0.83": next(
            r["power"] for r in table if r["repositories"] == 10 and r["base"] == 0.05
            and r["effect"] == 0.10),
        "addendum1_claim_base<=0.05_effect0.15_power>=0.95": next(
            r["power"] for r in table if r["repositories"] == 10 and r["base"] == 0.05
            and r["effect"] == 0.15),
        "addendum1_claim_base0.10_effect0.10_power~0.7": next(
            r["power"] for r in table if r["repositories"] == 10 and r["base"] == 0.10
            and r["effect"] == 0.10)}
    receipt = {"schema_version": "oneiros_v25_power_simulation_v1", "parameters": PARAMS,
               "model": __doc__.split("Generative model per simulated panel:")[1]
               .split("python scripts")[0].strip(),
               "table": table, "addendum1_claims_reproduced": claims,
               "note": "the panel thresholds of addendum 1 are NOT changed by this "
                       "simulation; any discrepancy is reported, never acted on after "
                       "model outcomes",
               "scripts_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                  for p in ("scripts/v25_power_simulation.py",
                                            "harness/v25_inference.py")}}
    status = publish_once(RECEIPT, receipt)
    print(json.dumps({"status": status, "claims": claims,
                      "sha256": hashlib.sha256((ROOT / RECEIPT).read_bytes()).hexdigest()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
