"""Power sensitivity v2: corrected cross-schema coupling.

``harness/power_sensitivity.py`` (v1) is kept byte-identical so its receipt stays
reproducible. It drew a shared/not-shared selector independently for EACH schema,
so the probability that both schemas used the common draw was ``coupling ** 2``
(0.25 at the declared "partial" 0.5), not ``coupling``.

v2 draws ONE Boolean selector per item (item level) or per semantic group (group
level) with probability ``coupling``. If selected, both schemas receive the same
latent draw; otherwise each schema receives its own independent draw.

The draw order is unchanged from v1 (common draw, then per-schema draws, then the
selector), and ``u < 0`` / ``u < 1`` are constant, so at coupling 0 and 1 v2
consumes identical random numbers for everything that matters and reproduces v1
exactly; only partial coupling changes.

The decision, rule and summary functions are v1's, re-exported unchanged.
"""
from __future__ import annotations

import numpy as np

from harness.power_sensitivity import (  # noqa: F401  (re-exported unchanged)
    Z, bootstrap_decisions, rules, sandwich_decisions, summarise,
)

DESIGN_VERSION = "oneiros_power_sensitivity_v2_common_selector"


def coupled_draws(rng, shape_items, level, coupling, return_selector=False):
    """Uniform flip draws [sims, G, items, schemas] with ONE shared/independent selector.

    ``level="item"``: one selector and one common draw per item.
    ``level="group"``: one selector and one common draw per semantic group,
    broadcast to all its items.
    """
    if level not in ("item", "group"):
        raise ValueError(f"unknown effect level {level!r}")
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must lie in [0, 1]")
    sims, groups, items, schemas = shape_items
    base_shape = (sims, groups, 1) if level == "group" else (sims, groups, items)
    shared = rng.random(base_shape)
    own = rng.random(base_shape + (schemas,))
    selector = rng.random(base_shape) < coupling
    u = np.where(selector[..., None], shared[..., None], own)
    if level == "group":
        u = np.broadcast_to(u, (sims, groups, items, schemas))
        selector = np.broadcast_to(selector, (sims, groups, items))
    return (u, selector) if return_selector else u


def simulate_datasets(c: np.ndarray, t: np.ndarray, groups: int, delta_points: float,
                      sims: int, seed: int, level: str, coupling: float):
    """v1's resample-and-shift procedure with the corrected coupled draws."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.shape[0], size=(sims, groups))
    cs, ts = c[idx], t[idx].copy()
    mask = ~np.isnan(cs)
    u = coupled_draws(rng, cs.shape, level, coupling)
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
