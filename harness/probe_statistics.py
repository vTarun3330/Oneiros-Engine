"""Clustered inference for the fixed-input probes (Phase 3 v2 analyses).

Rows are scored items: ``{"arm", "level", "group_id", "correct", ...}``.  A
*cell* is an (arm, level) pair; a *contrast* compares two cells on the same
items.  Everything resamples or flips whole semantic groups, never items.

* ``bootstrap_contrast`` - paired cluster bootstrap percentile interval;
* ``bootstrap_interaction`` - difference-in-differences of two contrasts, with
  the same cluster draws for all four cells;
* ``sign_flip_test`` - paired semantic-group sign-flip randomisation test,
  p = (extreme + 1) / (resamples + 1), so a finite test never reports zero;
* ``holm`` - Holm step-down adjustment (monotone, never below the raw value);
* ``schema_dependent`` - the frozen rule: the SFT effect changes sign between
  the prefilled-assertion schema and the ANSWER schema *in the stratum used*.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

RESAMPLES = 10_000
SEED = 20260928
Row = Mapping[str, Any]
Cell = tuple[str, str]


def _sums(rows: Sequence[Row], cell: Cell, groups: Sequence[str],
          select: Callable[[Row], bool]) -> np.ndarray:
    index = {g: i for i, g in enumerate(groups)}
    sums = np.zeros((len(groups), 2))
    for r in rows:
        if r["arm"] == cell[0] and r["level"] == cell[1] and select(r):
            sums[index[r["group_id"]], 0] += bool(r["correct"])
            sums[index[r["group_id"]], 1] += 1
    return sums


def _weights(n_groups: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(resamples, n_groups))
    weights = np.zeros((resamples, n_groups))
    np.add.at(weights, (np.repeat(np.arange(resamples), n_groups), draws.ravel()), 1)
    return weights


def _rate(sums: np.ndarray) -> float:
    return float(sums[:, 0].sum() / sums[:, 1].sum()) if sums[:, 1].sum() else float("nan")


def _groups(rows: Sequence[Row], select: Callable[[Row], bool]) -> list[str]:
    return sorted({r["group_id"] for r in rows if select(r)})


def bootstrap_contrast(rows: Sequence[Row], a: Cell, b: Cell,
                       select: Callable[[Row], bool] = lambda r: True,
                       resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """b minus a, percentage points, paired cluster bootstrap over semantic groups."""
    groups = _groups(rows, select)
    sa, sb = _sums(rows, a, groups, select), _sums(rows, b, groups, select)
    w = _weights(len(groups), resamples, seed)
    ba, bb = w @ sa, w @ sb
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (bb[:, 0] / bb[:, 1] - ba[:, 0] / ba[:, 1]) * 100
    diff = diff[~np.isnan(diff)]
    low, high = np.percentile(diff, [2.5, 97.5])
    return {"a": {"cell": list(a), "accuracy": round(_rate(sa), 4), "items": int(sa[:, 1].sum())},
            "b": {"cell": list(b), "accuracy": round(_rate(sb), 4), "items": int(sb[:, 1].sum())},
            "groups": len(groups),
            "difference_points": round((_rate(sb) - _rate(sa)) * 100, 3),
            "ci95_points": [round(float(low), 3), round(float(high), 3)]}


def bootstrap_interaction(rows: Sequence[Row], first: tuple[Cell, Cell],
                          second: tuple[Cell, Cell],
                          select: Callable[[Row], bool] = lambda r: True,
                          resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """(second contrast) minus (first contrast), same semantic-group draws for all cells."""
    groups = _groups(rows, select)
    cells = [*first, *second]
    sums = [_sums(rows, c, groups, select) for c in cells]
    w = _weights(len(groups), resamples, seed)
    boot = [w @ s for s in sums]
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = [b[:, 0] / b[:, 1] for b in boot]
    diff = ((rates[3] - rates[2]) - (rates[1] - rates[0])) * 100
    diff = diff[~np.isnan(diff)]
    point = ((_rate(sums[3]) - _rate(sums[2])) - (_rate(sums[1]) - _rate(sums[0]))) * 100
    low, high = np.percentile(diff, [2.5, 97.5])
    return {"groups": len(groups), "clustered_by": "semantic group_id",
            "first_contrast_points": round((_rate(sums[1]) - _rate(sums[0])) * 100, 3),
            "second_contrast_points": round((_rate(sums[3]) - _rate(sums[2])) * 100, 3),
            "interaction_points": round(point, 3),
            "ci95_points": [round(float(low), 3), round(float(high), 3)]}


def sign_flip_test(rows: Sequence[Row], a: Cell, b: Cell,
                   select: Callable[[Row], bool] = lambda r: True,
                   resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Two-sided paired sign-flip test over semantic groups.

    Under H0 (a and b exchangeable within each group), each group's paired
    difference in correct counts is equally likely to have either sign.  The
    statistic is the pooled accuracy difference sum(d_g) / sum(n_g).
    """
    groups = _groups(rows, select)
    sa, sb = _sums(rows, a, groups, select), _sums(rows, b, groups, select)
    if not np.array_equal(sa[:, 1], sb[:, 1]):
        raise ValueError("sign-flip test needs identical items in both cells")
    d, n = sb[:, 0] - sa[:, 0], sa[:, 1].sum()
    observed = abs(d.sum() / n)
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(resamples, len(groups)))
    extreme = int((np.abs(signs @ d / n) >= observed - 1e-12).sum())
    p = (extreme + 1) / (resamples + 1)
    return {"p_two_sided": p, "extreme": extreme, "resamples": resamples,
            "minimum_attainable_p": 1 / (resamples + 1),
            "report": (f"p <= {p:.5f} (no resample as extreme; smallest attainable with "
                       f"{resamples} resamples)" if extreme == 0 else f"p = {p:.5f}")}


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values: monotone in rank and never below the raw value."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(ordered), 0.0, {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = running
    return out


def schema_dependent(prefill_points: float, answer_points: float) -> bool:
    """The frozen control rule: the SFT effect changes sign between the two schemas."""
    return prefill_points * answer_points < 0


#: Strata the Phase 3A v2 analysis must report, and the one each H4 rule reads.
H4_STRATA = ("all", "exposed", "unexposed", "exposed_upstream_sft_target", "exposed_novel",
             "unexposed_upstream", "unexposed_novel")


def decide_h4(strata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen H4 rules with the schema rule checked on the strata they read.

    ``strata[name]`` holds ``prefill`` and ``answer`` contrasts (SFT minus base,
    with ``difference_points`` and ``ci95_points``).  Rules 1-2 read the exposed
    and unexposed strata, so a sign flip in EITHER blocks the exposure verdict,
    whatever the pooled stratum does.
    """
    missing = [s for s in H4_STRATA if s not in strata]
    if missing:
        raise ValueError(f"missing strata: {missing}")
    used = ("exposed", "unexposed")
    flips = {s: schema_dependent(strata[s]["prefill"]["difference_points"],
                                 strata[s]["answer"]["difference_points"]) for s in used}
    if any(flips.values()):
        return {"status": "open", "qualifier": "schema_dependent_exposure_result",
                "verdict": "inconclusive: the exposure result is answer-schema dependent",
                "schema_dependent_strata_used": [s for s, f in flips.items() if f],
                "distribution_shift": "open: no external diagnostic cohort exists"}
    exposed_improves = strata["exposed"]["prefill"]["ci95_points"][0] > 0
    unexposed_improves = strata["unexposed"]["prefill"]["ci95_points"][0] > 0
    if not exposed_improves:
        verdict, status = ("rule 1: no improvement on exposed functions", "weakened")
    elif not unexposed_improves:
        verdict, status = ("rule 2: memorisation/exposure failure", "strengthened")
    else:
        verdict, status = ("both train cohorts improve; distribution shift untestable", "open")
    return {"status": status, "qualifier": None, "verdict": verdict,
            "schema_dependent_strata_used": [],
            "distribution_shift": "open: no external diagnostic cohort exists"}
