"""Clustered inference for the fixed-input probes (Phase 3 v3 analyses).

Rows are scored items: ``{"arm", "level", "item_key", "group_id", "correct",
...}``.  A *cell* is an (arm, level) pair; a *contrast* compares two cells on
the SAME items.  Everything resamples or flips whole semantic groups, never
items.

v3 repair: every paired comparison validates exact item identity
(``paired_cells``).  Each cell must hold each item key exactly once, the cells
must hold the identical key set, and each key must sit in the same semantic
group in every cell.  Equal per-group counts are never sufficient.

* ``bootstrap_contrast`` - paired cluster bootstrap percentile interval;
* ``bootstrap_interaction`` - difference-in-differences of two contrasts, with
  the same cluster draws for all four cells;
* ``sign_flip_test`` - paired semantic-group sign-flip randomisation test,
  Monte Carlo p = (extreme + 1) / (resamples + 1);
* ``holm`` - Holm step-down adjustment (monotone, never below the raw value);
* ``schema_dependent`` - the frozen rule: the SFT effect changes sign between
  the prefilled-assertion schema and the ANSWER schema *in the stratum used*;
* ``decide_h4`` - route-exact H4 rules;
* ``answer_rate_gate`` / ``capacity_interpretation`` - the frozen Phase 3C gate,
  recomputed from rows and enforced before any interpretation.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

RESAMPLES = 10_000
SEED = 20260928
Row = Mapping[str, Any]
Cell = tuple[str, str]


class PairingError(ValueError):
    """Paired cells do not hold exactly the same items in the same groups."""


def _cell_items(rows: Sequence[Row], cell: Cell,
                select: Callable[[Row], bool]) -> dict[str, Row]:
    items: dict[str, Row] = {}
    for r in rows:
        if r["arm"] != cell[0] or r["level"] != cell[1] or not select(r):
            continue
        key = r.get("item_key")
        if not key:
            raise PairingError(f"row in {cell} has no item_key")
        if key in items:
            raise PairingError(f"duplicate item {key!r} in {cell}")
        items[key] = r
    return items


def paired_cells(rows: Sequence[Row], cells: Sequence[Cell],
                 select: Callable[[Row], bool] = lambda r: True
                 ) -> tuple[list[str], list[str], np.ndarray]:
    """Validate exact item pairing; return (item keys, their groups, correct[cell, item])."""
    per_cell = [_cell_items(rows, c, select) for c in cells]
    keys = sorted(per_cell[0])
    if not keys:
        raise PairingError(f"no items in {cells[0]}")
    for cell, items in zip(cells[1:], per_cell[1:]):
        if set(items) != set(keys):
            missing = sorted(set(keys) - set(items))[:3]
            extra = sorted(set(items) - set(keys))[:3]
            raise PairingError(f"{cell} is not paired with {cells[0]}: missing {missing}, "
                               f"substituted/extra {extra}")
    groups = [per_cell[0][k]["group_id"] for k in keys]
    for cell, items in zip(cells[1:], per_cell[1:]):
        moved = [k for k, g in zip(keys, groups) if items[k]["group_id"] != g]
        if moved:
            raise PairingError(f"items {moved[:3]} change semantic group in {cell}")
    correct = np.array([[bool(items[k]["correct"]) for k in keys] for items in per_cell],
                       dtype=float)
    return keys, groups, correct


def _group_sums(groups: Sequence[str], correct: np.ndarray) -> tuple[list[str], np.ndarray]:
    """[cell, group, (correct, items)] sums, groups sorted."""
    order = sorted(set(groups))
    index = np.array([order.index(g) for g in groups])
    sums = np.zeros((correct.shape[0], len(order), 2))
    for c in range(correct.shape[0]):
        np.add.at(sums[c, :, 0], index, correct[c])
        np.add.at(sums[c, :, 1], index, 1)
    return order, sums


def _weights(n_groups: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(resamples, n_groups))
    weights = np.zeros((resamples, n_groups))
    np.add.at(weights, (np.repeat(np.arange(resamples), n_groups), draws.ravel()), 1)
    return weights


def _rate(sums: np.ndarray) -> float:
    return float(sums[:, 0].sum() / sums[:, 1].sum())


def _interval(samples: np.ndarray, level: float = 95.0) -> list[float]:
    tail = (100.0 - level) / 2
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return [round(float(low), 3), round(float(high), 3)]


def bootstrap_contrast(rows: Sequence[Row], a: Cell, b: Cell,
                       select: Callable[[Row], bool] = lambda r: True,
                       resamples: int = RESAMPLES, seed: int = SEED,
                       extra_levels: Sequence[float] = ()) -> dict[str, Any]:
    """b minus a, percentage points, paired cluster bootstrap over semantic groups."""
    keys, groups, correct = paired_cells(rows, (a, b), select)
    order, sums = _group_sums(groups, correct)
    w = _weights(len(order), resamples, seed)
    ba, bb = w @ sums[0], w @ sums[1]
    diff = (bb[:, 0] / bb[:, 1] - ba[:, 0] / ba[:, 1]) * 100
    out = {"a": {"cell": list(a), "accuracy": round(_rate(sums[0]), 4),
                 "correct": int(sums[0][:, 0].sum()), "items": int(sums[0][:, 1].sum())},
           "b": {"cell": list(b), "accuracy": round(_rate(sums[1]), 4),
                 "correct": int(sums[1][:, 0].sum()), "items": int(sums[1][:, 1].sum())},
           "paired_items": len(keys), "groups": len(order),
           "difference_points": round((_rate(sums[1]) - _rate(sums[0])) * 100, 3),
           "ci95_points": _interval(diff)}
    for level in extra_levels:
        out[f"ci{level:g}_points"] = _interval(diff, level)
    return out


def bootstrap_interaction(rows: Sequence[Row], first: tuple[Cell, Cell],
                          second: tuple[Cell, Cell],
                          select: Callable[[Row], bool] = lambda r: True,
                          resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """(second contrast) minus (first contrast), same semantic-group draws for all cells."""
    keys, groups, correct = paired_cells(rows, (*first, *second), select)
    order, sums = _group_sums(groups, correct)
    w = _weights(len(order), resamples, seed)
    rates = [(w @ sums[c])[:, 0] / (w @ sums[c])[:, 1] for c in range(4)]
    diff = ((rates[3] - rates[2]) - (rates[1] - rates[0])) * 100
    point = [_rate(sums[c]) for c in range(4)]
    return {"paired_items": len(keys), "groups": len(order),
            "clustered_by": "semantic group_id",
            "first_contrast_points": round((point[1] - point[0]) * 100, 3),
            "second_contrast_points": round((point[3] - point[2]) * 100, 3),
            "interaction_points": round(((point[3] - point[2]) - (point[1] - point[0])) * 100, 3),
            "ci95_points": _interval(diff)}


def monte_carlo_report(extreme: int, resamples: int) -> str:
    p = (extreme + 1) / (resamples + 1)
    if extreme == 0:
        return (f"Monte Carlo p = {p:.5f}; 0/{resamples} sampled permutations were as "
                f"extreme; {p:.5f} is the minimum reportable resolution at this resample "
                "count.")
    return (f"Monte Carlo p = {p:.5f}; {extreme}/{resamples} sampled permutations were as "
            "extreme.")


def sign_flip_test(rows: Sequence[Row], a: Cell, b: Cell,
                   select: Callable[[Row], bool] = lambda r: True,
                   resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Two-sided paired sign-flip test over semantic groups on exactly paired items.

    Under H0 (a and b exchangeable within each group), each group's paired
    difference in correct counts is equally likely to have either sign.  The
    statistic is the pooled accuracy difference sum(d_g) / sum(n_g).
    """
    keys, groups, correct = paired_cells(rows, (a, b), select)
    order, sums = _group_sums(groups, correct)
    d, n = sums[1][:, 0] - sums[0][:, 0], sums[0][:, 1].sum()
    observed = abs(d.sum() / n)
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(resamples, len(order)))
    extreme = int((np.abs(signs @ d / n) >= observed - 1e-12).sum())
    return {"monte_carlo_p": (extreme + 1) / (resamples + 1), "extreme": extreme,
            "resamples": resamples, "paired_items": len(keys), "groups": len(order),
            "report": monte_carlo_report(extreme, resamples)}


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


#: Strata the Phase 3A analysis must report.
H4_STRATA = ("all", "exposed", "unexposed", "exposed_upstream_sft_target", "exposed_novel",
             "unexposed_upstream", "unexposed_novel")


def decide_h4(strata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Route-exact frozen H4 rules.

    1. exposed schema flip -> open (schema_dependent_exposure_result);
    2. rule 1: exposed not improved (prefill lower bound <= 0) -> weakened;
    3. only now consult unexposed: schema flip -> open (schema_dependent_unexposed_result);
    4. rule 2: unexposed not improved -> strengthened (memorisation/exposure failure);
    5. rule 3: both improve -> open (distribution shift untestable: no external cohort).
    """
    missing = [s for s in H4_STRATA if s not in strata]
    if missing:
        raise ValueError(f"missing strata: {missing}")

    def flips(name):
        return schema_dependent(strata[name]["prefill"]["difference_points"],
                                strata[name]["answer"]["difference_points"])

    def improves(name):
        return strata[name]["prefill"]["ci95_points"][0] > 0

    base = {"distribution_shift": "open: no external diagnostic cohort exists"}
    route = ["exposed_schema_check"]
    if flips("exposed"):
        return {**base, "route": route, "status": "open",
                "qualifier": "schema_dependent_exposure_result",
                "verdict": "inconclusive: the exposed result is answer-schema dependent"}
    route.append("rule_1")
    if not improves("exposed"):
        return {**base, "route": route, "status": "weakened", "qualifier": None,
                "verdict": "rule 1: no improvement on exposed functions"}
    route.append("unexposed_schema_check")
    if flips("unexposed"):
        return {**base, "route": route, "status": "open",
                "qualifier": "schema_dependent_unexposed_result",
                "verdict": "inconclusive: the unexposed result is answer-schema dependent"}
    route.append("rule_2")
    if not improves("unexposed"):
        return {**base, "route": route, "status": "strengthened",
                "qualifier": "memorisation_exposure_failure",
                "verdict": "rule 2: improves on exposed but not unexposed functions"}
    route.append("rule_3")
    return {**base, "route": route, "status": "open",
            "qualifier": "distribution_shift_untestable",
            "verdict": "rule 3: both train cohorts improve; no external cohort"}


def answer_rate(rows: Sequence[Row], cell: Cell, method: str | None = None) -> float:
    sel = [r for r in rows if r["arm"] == cell[0] and r["level"] == cell[1]]
    if not sel:
        raise ValueError(f"no rows for {cell}")
    hit = [r for r in sel if r["answer"] and (method is None or r["method"] == method)]
    return len(hit) / len(sel)


def answer_rate_gate(rows: Sequence[Row], large: str, small: str, levels: Sequence[str],
                     control: str, margin: float = 0.02, strict_min: float = 0.95
                     ) -> dict[str, Any]:
    """Frozen Phase 3C gate, recomputed from rows: non-inferior answer rate at every level
    and a strict ANSWER-schema rate of at least ``strict_min`` on the control."""
    per_level = {level: {"large": round(answer_rate(rows, (large, level)), 4),
                         "small": round(answer_rate(rows, (small, level)), 4)}
                 for level in levels}
    for block in per_level.values():
        block["non_inferior"] = block["large"] >= block["small"] - margin
    strict = answer_rate(rows, (large, control), method="strict")
    return {"margin": margin, "strict_min": strict_min, "per_level": per_level,
            "control_strict_rate": round(strict, 4),
            "passed": all(b["non_inferior"] for b in per_level.values())
            and strict >= strict_min}


def capacity_interpretation(gate: Mapping[str, Any],
                            primary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Model-scale association statement, or nothing if the recomputed gate failed."""
    if not gate["passed"]:
        return {"available": False, "H3": "unchanged",
                "reason": "the recomputed answer-rate gate failed; no capacity or model-scale "
                          "conclusion is drawn"}
    positive = all(c["ci95_points"][0] > 0 for c in primary.values())
    return {"available": True,
            "H3": {"status": "strengthened" if positive else "open",
                   "qualifier": "model_scale_association",
                   "statement": ("Within the Qwen2.5-Coder family, the 7B checkpoint is "
                                 "substantially better than the 1.5B checkpoint on this "
                                 "fixed-input panel." if positive else
                                 "7B is not clearly better than 1.5B on this panel."),
                   "not_established": ("parameter count, LoRA rank, adapter capacity, "
                                       "pretraining data or recipe as the causal factor")}}
