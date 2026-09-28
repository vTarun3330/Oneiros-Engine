"""Simulation-based power for the Phase 4 fixed-input mediator gate (CPU only).

Empirical basis: the permitted, train-side Phase 3A retained outcomes.  For each
semantic group we keep the paired outcomes of two different 1.5B models (base and
arm A) on the SAME items under BOTH answer schemas, so the simulation preserves the
real group structure, within-group correlation, between-model discordance and the
cross-schema coupling.

One simulated gate experiment with G groups:

1. draw G groups with replacement from the empirical groups (all items, both schemas
   and both models of a group travel together);
2. control = the base outcomes; the treatment starts from the arm A outcomes, i.e.
   a realistic second model on the same items;
3. the treatment is shifted to a target mean effect ``delta`` (percentage points,
   the same in both schemas) by flipping treatment outcomes, either independently
   per item ("item" scenario) or for whole groups at once ("group" scenario, strong
   intragroup correlation of the effect);
4. per schema: difference in accuracy and a cluster-robust (sandwich) standard error
   over groups, the large-sample equivalent of the paired cluster bootstrap; 95%
   two-sided interval = difference +/- 1.96 SE.

Rules evaluated on each simulated experiment:
* ``positive_both``: lower bound > 0 under both schemas, no sign flip;
* ``confirmatory_5``: lower bound > +5 under both schemas, no sign flip (the
  corrected confirmatory five-point rule);
* ``old_rule``: point >= 5 and lower bound > 0 under both schemas (the superseded
  V1-draft gate, reported for comparison only).

Not modelled (stated, not invented): training-seed variance, the answer-rate gate,
and validity/diversity guardrails.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

SCHEMAS = ("A0", "A0_answer_schema")
Z = 1.959963984540054


def empirical_groups(rows: Sequence[Mapping[str, Any]], control: str, treatment: str
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Arrays [group, item, schema] of control and treatment correctness (items padded)."""
    by_group: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["level"] not in SCHEMAS or r["arm"] not in (control, treatment):
            continue
        by_group.setdefault(r["group_id"], {}).setdefault(r["item_key"], {})[
            (r["arm"], r["level"])] = bool(r["correct"])
    groups = sorted(by_group)
    width = max(len(items) for items in by_group.values())
    c = np.full((len(groups), width, len(SCHEMAS)), np.nan)
    t = np.full_like(c, np.nan)
    for gi, g in enumerate(groups):
        for ii, key in enumerate(sorted(by_group[g])):
            cell = by_group[g][key]
            for si, s in enumerate(SCHEMAS):
                if (control, s) not in cell or (treatment, s) not in cell:
                    raise ValueError(f"item {key} lacks a paired {s} outcome")
                c[gi, ii, si] = cell[(control, s)]
                t[gi, ii, si] = cell[(treatment, s)]
    return c, t, groups


def _shift(t: np.ndarray, mask: np.ndarray, target: np.ndarray, rng: np.random.Generator,
           scenario: str) -> np.ndarray:
    """Flip treatment outcomes so each schema's expected accuracy moves by ``target``."""
    out = t.copy()
    for s in range(t.shape[-1]):
        cur = np.nanmean(np.where(mask[..., s], t[..., s], np.nan))
        need = target[s] - cur
        if need == 0:
            continue
        source = 0.0 if need > 0 else 1.0
        eligible = mask[..., s] & (t[..., s] == source)
        rate = min(1.0, abs(need) * mask[..., s].sum() / max(eligible.sum(), 1))
        if scenario == "item":
            flip = rng.random(eligible.shape) < rate
        else:   # whole-group flips: one draw per (simulation, group), shared by its items
            flip = np.broadcast_to((rng.random(eligible.shape[0]) < rate)[:, None],
                                   eligible.shape)
        out[..., s] = np.where(eligible & flip, 1.0 - source, out[..., s])
    return out


def simulate(c: np.ndarray, t: np.ndarray, groups: int, delta_points: float, sims: int,
             seed: int, scenario: str = "item") -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.shape[0], size=(sims, groups))
    cs, ts = c[idx], t[idx]                       # [sims, G, items, schemas]
    mask = ~np.isnan(cs)
    target = np.nanmean(np.where(mask, cs, np.nan), axis=(1, 2)) + delta_points / 100.0
    ts = np.stack([_shift(ts[k], mask[k], target[k], rng, scenario) for k in range(sims)])
    d = np.where(mask, ts - cs, 0.0)
    n_g = mask.sum(axis=2)                         # [sims, G, schemas]
    d_g = d.sum(axis=2)
    N = n_g.sum(axis=1)
    D = d_g.sum(axis=1) / N                        # [sims, schemas]
    resid = d_g - n_g * D[:, None, :]
    se = np.sqrt(groups / (groups - 1) * (resid ** 2).sum(axis=1)) / N
    low, point = (D - Z * se) * 100, D * 100
    no_flip = np.sign(point[:, 0]) == np.sign(point[:, 1])
    rules = {"positive_both": (low > 0).all(axis=1) & no_flip,
             "confirmatory_5": (low > 5).all(axis=1) & no_flip,
             "old_rule": ((point >= 5) & (low > 0)).all(axis=1)}
    out = {"groups": groups, "delta_points": delta_points, "scenario": scenario, "sims": sims,
           "mean_simulated_effect_points": [round(float(x), 3) for x in point.mean(axis=0)],
           "mean_se_points": [round(float(x), 3) for x in (se * 100).mean(axis=0)]}
    for name, hit in rules.items():
        p = float(hit.mean())
        out[name] = {"power": round(p, 4),
                     "monte_carlo_se": round(float(np.sqrt(p * (1 - p) / sims)), 4)}
    return out


def required_groups(results: Sequence[Mapping[str, Any]], rule: str, target: float = 0.8
                    ) -> int | None:
    """Smallest simulated group count whose power reaches ``target`` (None if none)."""
    hits = sorted(r["groups"] for r in results if r[rule]["power"] >= target)
    return hits[0] if hits else None
