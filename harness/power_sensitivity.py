"""Power sensitivity for the Phase 4 mediator gate (successor to harness/power_simulation.py,
which is kept unchanged so its receipt stays reproducible).

Adds, relative to the v1 simulation:

* cross-schema coupling of the injected treatment effect: ``coupling`` in [0, 1] is the
  probability that an item's (or group's) flip draw is shared between the two answer
  schemas (1 = fully shared, 0.5 = partial, 0 = independent);
* an EXACT-GATE check: the planned paired semantic-group cluster-bootstrap percentile
  interval, computed on the very same simulated datasets as the fast cluster-sandwich
  normal interval, so the two decisions can be compared dataset by dataset;
* sign-flip frequency (point estimates of opposite sign across schemas).

Not modelled (stated, not invented): training-seed variance, the answer-rate gate and
the validity/diversity guardrails.  Outcomes are synthetic train-side Phase 3A outcomes.
"""
from __future__ import annotations

from typing import Any

import numpy as np

Z = 1.959963984540054


def _draws(rng, shape_items, level, coupling):
    """Uniform flip draws [sims, G, items, schemas] with cross-schema coupling."""
    sims, groups, items, schemas = shape_items
    base_shape = (sims, groups, 1) if level == "group" else (sims, groups, items)
    shared = rng.random(base_shape)
    own = rng.random(base_shape + (schemas,))
    use_shared = rng.random(base_shape + (schemas,)) < coupling
    u = np.where(use_shared, shared[..., None], own)
    if level == "group":
        u = np.broadcast_to(u, (sims, groups, items, schemas))
    return u


def simulate_datasets(c: np.ndarray, t: np.ndarray, groups: int, delta_points: float,
                      sims: int, seed: int, level: str, coupling: float):
    """Resample groups and shift the treatment to control + delta in each schema."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.shape[0], size=(sims, groups))
    cs, ts = c[idx], t[idx].copy()
    mask = ~np.isnan(cs)
    u = _draws(rng, cs.shape, level, coupling)
    n = mask.sum(axis=(1, 2))                                        # [sims, schemas]
    target = np.nansum(np.where(mask, cs, 0), axis=(1, 2)) / n + delta_points / 100.0
    current = np.nansum(np.where(mask, ts, 0), axis=(1, 2)) / n
    need = target - current                                          # [sims, schemas]
    up = need > 0
    source = np.where(up, 0.0, 1.0)[:, None, None, :]
    eligible = mask & (ts == source)
    count = eligible.sum(axis=(1, 2))
    rate = np.clip(np.abs(need) * n / np.maximum(count, 1), 0, 1)[:, None, None, :]
    flip = eligible & (u < rate)
    ts = np.where(flip, 1.0 - source, ts)
    return cs, ts, mask


def sandwich_decisions(cs, ts, mask) -> dict[str, np.ndarray]:
    d = np.where(mask, ts - cs, 0.0)
    n_g, d_g = mask.sum(axis=2), d.sum(axis=2)                       # [sims, G, schemas]
    N = n_g.sum(axis=1)
    D = d_g.sum(axis=1) / N
    G = cs.shape[1]
    se = np.sqrt(G / (G - 1) * ((d_g - n_g * D[:, None, :]) ** 2).sum(axis=1)) / N
    return {"point": D * 100, "low": (D - Z * se) * 100, "high": (D + Z * se) * 100}


def bootstrap_decisions(cs, ts, mask, resamples: int, seed: int) -> dict[str, np.ndarray]:
    """The planned gate: paired cluster-bootstrap percentile 95% interval per schema."""
    rng = np.random.default_rng(seed)
    sims, groups = cs.shape[:2]
    c_g = np.where(mask, cs, 0).sum(axis=2)                          # [sims, G, schemas]
    t_g = np.where(mask, ts, 0).sum(axis=2)
    n_g = mask.sum(axis=2)
    low = np.empty((sims, cs.shape[3]))
    high = np.empty_like(low)
    point = (t_g.sum(axis=1) - c_g.sum(axis=1)) / n_g.sum(axis=1) * 100
    for k in range(sims):
        draws = rng.integers(0, groups, size=(resamples, groups))
        w = np.zeros((resamples, groups))
        np.add.at(w, (np.repeat(np.arange(resamples), groups), draws.ravel()), 1)
        diff = ((w @ t_g[k]) - (w @ c_g[k])) / (w @ n_g[k]) * 100
        low[k], high[k] = np.percentile(diff, [2.5, 97.5], axis=0)
    return {"point": point, "low": low, "high": high}


def rules(dec: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    point, low = dec["point"], dec["low"]
    flip = np.sign(point[:, 0]) * np.sign(point[:, 1]) < 0
    return {"positive_both": (low > 0).all(axis=1) & ~flip,
            "confirmatory_5": (low > 5).all(axis=1) & ~flip,
            "sign_flip": flip}


def summarise(hits: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for name, hit in hits.items():
        p = float(hit.mean())
        n = len(hit)
        # Wilson 95% interval for the Monte Carlo probability
        centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
        half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
        out[name] = {"probability": round(p, 4),
                     "monte_carlo_wilson_95": [round(max(0.0, centre - half), 4),
                                               round(min(1.0, centre + half), 4)]}
    return out
