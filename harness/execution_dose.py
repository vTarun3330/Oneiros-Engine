"""Pure planning logic for the execution-dose-and-retention experiment.

Everything here is deterministic and side-effect free so it can be unit
tested without data, a tokenizer or a GPU.  The builder script supplies rows
and token counts; this module decides which frozen-control positions are
replaced (balanced replay) and which verified train execution rows fill them.

Two ideas govern the design:

* **Balanced replay.**  The treatment keeps the frozen control's canonical
  rows at every position it does not replace.  Which positions are replaced is
  decided by largest-remainder quotas over (source dataset x complexity tier),
  with bounded bug-family and execution-mode loss, so a larger auxiliary share never removes a
  whole dataset, tier, bug family or the real-repository rows.
* **Supervised token mass.**  The trainer normalises loss per supervised token
  over each accumulation window, so the auxiliary task's gradient share is its
  share of supervised tokens, not of examples.  The dose is therefore stated in
  both units, and the treatment's total mass is held near the control's.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

ARM_SIZE = 1024
DOSE_DESIGNS = {"d25": 256, "d50": 512}
REPOSITORY_SOURCES = frozenset({"SWE-bench Verified", "BugsInPy"})

#: The treatment is not "a 25% dose": it raises both the example share and the
#: total supervised tokens.  Every report uses this label.
INTERVENTION_LABEL = "25%-example / 58%-supervised-token execution intervention"
#: Frozen after the token-matched control proved infeasible at 1%, 2% and 5%
#: (results/v4_3_execution_dose_matched_control.json).
INFERENCE_LIMITATION = (
    "Any observed effect is attributable to the combined 25%-example, "
    "58%-token execution-supervision intervention and cannot isolate supervision "
    "type from supervised-token exposure."
)


def largest_remainder(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Apportion ``total`` over ``weights`` by largest remainder, ties by key."""
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("cannot apportion over empty weights")
    raw = {key: total * weight / denominator for key, weight in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    left = total - sum(result.values())
    for key in sorted(weights, key=lambda key: (-(raw[key] - result[key]), key))[:left]:
        result[key] += 1
    return result


def stratum(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row["source_dataset"]), str(row["complexity_tier"]))


def replay_removal_quotas(rows: Sequence[Mapping[str, Any]], remove: int) -> dict:
    """Removal count per (source, tier), keeping at least one row per stratum."""
    sizes = Counter(stratum(row) for row in rows)
    quotas = largest_remainder(remove, {f"{s}\x1f{t}": n for (s, t), n in sizes.items()})
    result = {tuple(key.split("\x1f")): count for key, count in quotas.items()}
    # A stratum may never be emptied. Move any excess to the stratum with the
    # most remaining headroom (deterministic), which keeps the total exact.
    for key in sorted(result):
        while result[key] > sizes[key] - 1 and sizes[key] >= 1:
            result[key] -= 1
            donors = [other for other in result if result[other] < sizes[other] - 1]
            if not donors:
                raise ValueError("cannot remove that many rows without emptying a stratum")
            donor = max(donors, key=lambda other: (sizes[other] - 1 - result[other], other))
            result[donor] += 1
    return result


#: A bug family or execution mode may lose at most this factor times its
#: proportional share.  Exact proportionality cannot hold simultaneously with
#: exact (source x tier) quotas, because categories are unevenly spread.
FAMILY_REMOVAL_TOLERANCE = 1.15


def removal_ceilings(rows: Sequence[Mapping[str, Any]], remove: int,
                     field: str) -> dict[str, int]:
    """Bound each category's loss near its share; every category keeps one row."""
    share = remove / len(rows)
    sizes = Counter(str(row[field]) for row in rows)
    return {key: min(size - 1, math.ceil(share * size * FAMILY_REMOVAL_TOLERANCE))
            for key, size in sizes.items()}


#: Dimensions whose loss is bounded in addition to the exact stratum quotas.
BOUNDED_FIELDS = ("bug_family", "execution_mode")


def plan_replay(
    rows: Sequence[Mapping[str, Any]],
    remove: int,
    order_key: Callable[[int], Any],
) -> list[int]:
    """Positions of the frozen control to replace.

    ``order_key(position)`` gives the within-stratum removal preference (lower
    goes first).  Stratum quotas are exact; bug-family and execution-mode
    ceilings are honoured by skipping rows whose category is exhausted.  Raises when the constraints
    cannot be met rather than silently relaxing them.
    """
    quotas = replay_removal_quotas(rows, remove)
    ceilings = {field: removal_ceilings(rows, remove, field) for field in BOUNDED_FIELDS}
    by_stratum: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        by_stratum[stratum(row)].append(position)
    # Strata with the fewest family alternatives are filled first, so a
    # family's allowance is not spent where other families could have served.
    def flexibility(key: tuple[str, str]) -> tuple[int, tuple[str, str]]:
        families = {str(rows[position]["bug_family"]) for position in by_stratum[key]}
        return (len(families), key)

    removed: dict[str, Counter] = {field: Counter() for field in BOUNDED_FIELDS}
    chosen: list[int] = []
    for key in sorted(by_stratum, key=flexibility):
        taken = 0
        for position in sorted(by_stratum[key], key=order_key):
            if taken == quotas.get(key, 0):
                break
            values = {field: str(rows[position][field]) for field in BOUNDED_FIELDS}
            if any(removed[field][value] >= ceilings[field][value]
                   for field, value in values.items()):
                continue
            chosen.append(position)
            for field, value in values.items():
                removed[field][value] += 1
            taken += 1
        if taken != quotas.get(key, 0):
            raise ValueError(f"replay quota for {key} unfillable under category ceilings")
    return sorted(chosen)


def replay_balance_report(rows: Sequence[Mapping[str, Any]],
                          removed: Iterable[int]) -> dict[str, Any]:
    """Before/after counts for every dimension balanced replay must preserve."""
    removed = set(removed)
    kept = [row for position, row in enumerate(rows) if position not in removed]

    def compare(field: Callable[[Mapping[str, Any]], str]) -> dict[str, dict[str, Any]]:
        before = Counter(field(row) for row in rows)
        after = Counter(field(row) for row in kept)
        return {key: {"control": before[key], "kept": after[key],
                      "kept_fraction": round(after[key] / before[key], 4)}
                for key in sorted(before)}

    report = {
        "by_source": compare(lambda row: str(row["source_dataset"])),
        "by_complexity_tier": compare(lambda row: str(row["complexity_tier"])),
        "by_source_and_tier": compare(lambda row: "/".join(stratum(row))),
        "by_bug_family": compare(lambda row: str(row["bug_family"])),
        "by_execution_mode": compare(lambda row: str(row["execution_mode"])),
        "real_repository": compare(
            lambda row: "repository" if row["source_dataset"] in REPOSITORY_SOURCES
            else "function"),
    }
    cells = [(section, key, entry) for section, values in report.items()
             for key, entry in values.items()]
    report["no_category_removed"] = all(entry["kept"] >= 1 for _, _, entry in cells)
    lowest = min(cells, key=lambda item: (item[2]["kept_fraction"], item[0], item[1]))
    report["minimum_kept_fraction"] = {"section": lowest[0], "category": lowest[1],
                                       **lowest[2]}
    report["integer_granularity_note"] = (
        "kept counts are integers, so a cell of n rows can only keep k/n; a 3-row "
        "cell keeps 2/3 = 66.7% or 3/3, never 75%. Large groups sit near the 75% "
        "target; small cells can fall below it. No 75% floor is claimed.")
    return report


def select_execution_rows(
    pool: Sequence[Mapping[str, Any]],
    count: int,
    *,
    lineage_cap: int,
    tier_quotas: Mapping[str, int],
    family_cap: int,
    order_key: Callable[[Mapping[str, Any]], Any],
) -> list[Mapping[str, Any]]:
    """Deterministic, diversity-constrained selection of execution rows.

    One row per record (``pool`` is expected to hold one candidate per record
    already) and unique trace targets.  Tiers are filled to exact quotas and
    sources are interleaved least-filled-first, so a dominant source cannot
    crowd out the others while supply remains.  Raises if the quotas cannot be
    met.
    """
    queues: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(pool, key=order_key):
        queues[(str(row["complexity_tier"]), str(row["source_dataset"]))].append(row)
    selected: list[Mapping[str, Any]] = []
    per_lineage: Counter = Counter()
    per_family: Counter = Counter()
    per_source: Counter = Counter()
    seen_targets: set[str] = set()
    for tier in sorted(tier_quotas):
        need = tier_quotas[tier]
        sources = sorted(source for t, source in queues if t == tier)
        cursors = {source: 0 for source in sources}
        taken = 0
        while taken < need:
            open_sources = [s for s in sources if cursors[s] < len(queues[(tier, s)])]
            if not open_sources:
                raise ValueError(f"tier {tier} exhausted at {taken}/{need}")
            source = min(open_sources, key=lambda s: (per_source[s], s))
            row = queues[(tier, source)][cursors[source]]
            cursors[source] += 1
            lineage = str(row["function_lineage"])
            family = str(row["bug_family"])
            target = str(row["trace_completion_sha256"])
            if (per_lineage[lineage] >= lineage_cap or per_family[family] >= family_cap
                    or target in seen_targets):
                continue
            selected.append(row)
            per_lineage[lineage] += 1
            per_family[family] += 1
            per_source[source] += 1
            seen_targets.add(target)
            taken += 1
    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} execution rows, expected {count}")
    return selected


def dose_summary(control_tokens: Sequence[int], removed: Iterable[int],
                 execution_tokens: Sequence[int]) -> dict[str, Any]:
    """Example share, supervised-token share and total mass versus control."""
    removed = list(removed)
    control_mass = sum(control_tokens)
    removed_mass = sum(control_tokens[position] for position in removed)
    execution_mass = sum(execution_tokens)
    treatment_mass = control_mass - removed_mass + execution_mass
    return {
        "execution_examples": len(execution_tokens),
        "execution_example_share": round(len(execution_tokens) / len(control_tokens), 6),
        "control_supervised_tokens": control_mass,
        "removed_canonical_supervised_tokens": removed_mass,
        "execution_supervised_tokens": execution_mass,
        "treatment_supervised_tokens": treatment_mass,
        "execution_token_share": round(execution_mass / treatment_mass, 6),
        "treatment_to_control_mass_ratio": round(treatment_mass / control_mass, 6),
    }


def cluster_bootstrap_difference(
    pairs: Sequence[tuple[str, bool, bool]],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.90,
) -> dict[str, Any]:
    """Percentile interval for mean(b) - mean(a) resampling whole clusters.

    ``pairs`` holds (cluster, a_success, b_success) per unit.  Records from one
    lineage are mutants of the same function and are not independent, so the
    lineage is the resampling unit.  Returned values are percentage points.
    """
    import random

    clusters: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for cluster, first, second in pairs:
        clusters[str(cluster)].append((int(bool(first)), int(bool(second))))
    keys = sorted(clusters)
    sums = [(sum(b - a for a, b in clusters[key]), len(clusters[key])) for key in keys]
    total_units = sum(size for _, size in sums)
    point = 100 * sum(delta for delta, _ in sums) / total_units
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        delta = units = 0
        for _ in keys:
            cluster_delta, cluster_units = sums[generator.randrange(len(sums))]
            delta += cluster_delta
            units += cluster_units
        draws.append(100 * delta / units)
    draws.sort()
    tail = (1 - confidence) / 2
    low = draws[max(0, math.floor(tail * replicates))]
    high = draws[min(replicates - 1, math.ceil((1 - tail) * replicates) - 1)]
    return {"difference_pp": point, "low_pp": low, "high_pp": high,
            "clusters": len(keys), "units": total_units,
            "replicates": replicates, "seed": seed, "confidence": confidence}


MATCH_TOLERANCES = (0.01, 0.02, 0.05)


def plan_token_matched_swaps(
    base_tokens: Mapping[int, int],
    base_cells: Mapping[int, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    current_total: int,
    target_total: int,
    tolerance: float,
    admissible: Callable[[Mapping[str, Any], int, dict], bool],
    order_key: Callable[[int, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Fewest same-cell swaps that bring a control's token mass within tolerance.

    ``base_tokens``/``base_cells`` describe the control rows that may be
    swapped (position -> supervised tokens / representation cell).  Each
    candidate carries ``cell``, ``tokens`` and an ``id``; a candidate may only
    replace a row of the identical cell, so every representation count is
    unchanged.  Swaps are taken largest-gain first (fewest rows changed), then
    one best-fit swap closes the remaining gap.  ``admissible(candidate,
    position, state)`` enforces uniqueness and lineage caps against the
    evolving arm, with the row at ``position`` already released.
    Nothing is repeated, padded or fabricated; if the tolerance cannot be met
    the result says so.
    """
    band = tolerance * target_total
    by_cell: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_cell[candidate["cell"]].append(candidate)
    pairs = []
    for position, cell in base_cells.items():
        for candidate in by_cell.get(cell, ()):
            gain = int(candidate["tokens"]) - int(base_tokens[position])
            if gain > 0:
                pairs.append((gain, position, candidate))
    pairs.sort(key=lambda item: (-item[0], order_key(item[1], item[2])))
    state: dict[str, Any] = {"swaps": {}, "used": set(), "total": current_total}

    def apply(gain: int, position: int, candidate: Mapping[str, Any]) -> bool:
        if position in state["swaps"] or candidate["id"] in state["used"]:
            return False
        if not admissible(candidate, position, state):
            return False
        state["swaps"][position] = candidate
        state["used"].add(candidate["id"])
        state["total"] += gain
        return True

    # Largest gains while the gap exceeds the largest remaining single fix.
    for gain, position, candidate in pairs:
        gap = target_total - state["total"]
        if gap <= band:
            break
        if gain <= gap:
            apply(gain, position, candidate)
    # Best fit: the single admissible swap that lands closest to the target.
    gap = target_total - state["total"]
    if abs(gap) > band:
        remaining = sorted(pairs, key=lambda item: (abs(gap - item[0]),
                                                    order_key(item[1], item[2])))
        for gain, position, candidate in remaining:
            if abs(gap - gain) >= abs(gap):
                break
            if apply(gain, position, candidate):
                break
    ratio = state["total"] / target_total
    return {"feasible": abs(state["total"] - target_total) <= band,
            "tolerance": tolerance, "total": state["total"], "target": target_total,
            "ratio_to_target": ratio, "swaps": state["swaps"]}
