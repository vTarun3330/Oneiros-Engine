"""Cap shares by dropping surplus, and report whether the cap actually held.

The selector this replaces checked each stratum's share BEFORE adding the row
and skipped the check entirely for the first 200 rows. Both make the declared
cap unenforceable: a share measured before the increment is not the share that
results from it, and 200 unchecked rows can put a stratum past its ceiling
before any checking begins. It could therefore report "source cap 0.70" beside
a final source share above 0.70 and contradict itself without noticing.

This works in two auditable passes.

1. SELECT: deterministic order, exact-duplicate targets dropped, lineage cap
   applied, and a share cap tested on the PROSPECTIVE share - what the share
   would become if this row were added.
2. ENFORCE: the caps are then re-checked against the FINAL selection, and any
   stratum still over its ceiling has its most-recently-added rows dropped
   until it is not. This repeats until the selection is stable.

If a cap cannot be met without deleting scarce rows, it is reported as
violated rather than quietly relaxed. Nothing is ever duplicated: a cap only
ever removes rows.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

#: Below this many rows a share is not meaningful - the first row is always
#: 100% of its stratum. Caps bind from here on, and the number is declared
#: rather than hidden inside the loop.
MIN_DENOMINATOR_FOR_SHARE_CAPS = 20


def _shares(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    total = len(rows) or 1
    return {value: count / total
            for value, count in Counter(r[key] for r in rows).items()}


def select(rows: Sequence[dict[str, Any]], *,
           target_of: Callable[[dict[str, Any]], str],
           lineage_key: str,
           max_per_lineage: int,
           share_caps: dict[str, float],
           order_key: Callable[[dict[str, Any]], Any] | None = None,
           ) -> dict[str, Any]:
    """Unique-first selection with enforceable share caps."""
    dropped: Counter = Counter()
    seen: set[str] = set()
    per_lineage: Counter = Counter()
    kept: list[dict[str, Any]] = []

    if order_key is None:
        # Rarest stratum first across every capped dimension, so a cap removes
        # surplus from crowded strata rather than starving scarce ones.
        sizes = {dimension: Counter(r[dimension] for r in rows)
                 for dimension in share_caps}

        def order_key(row):                                    # noqa: F811
            return (tuple(sizes[d][row[d]] for d in sorted(share_caps)),
                    str(target_of(row)))

    for row in sorted(rows, key=order_key):
        digest = str(target_of(row))
        if digest in seen:
            dropped["exact_duplicate_target"] += 1
            continue
        if per_lineage[row[lineage_key]] >= max_per_lineage:
            dropped["function_lineage_cap"] += 1
            continue
        prospective_total = len(kept) + 1
        if prospective_total > MIN_DENOMINATOR_FOR_SHARE_CAPS:
            over = None
            for dimension, cap in share_caps.items():
                current = sum(1 for r in kept if r[dimension] == row[dimension])
                # The share this row would CREATE, not the one it follows.
                if (current + 1) / prospective_total > cap:
                    over = dimension
                    break
            if over is not None:
                dropped[f"{over}_cap"] += 1
                continue
        seen.add(digest)
        per_lineage[row[lineage_key]] += 1
        kept.append(row)

    # Pass 2: the prospective test admits rows that a LATER drop can push back
    # over the line, so the caps are re-checked against the final set.
    enforcement_passes = 0
    while kept:
        violations = []
        for dimension, cap in share_caps.items():
            for value, share in _shares(kept, dimension).items():
                if share > cap and len(kept) > MIN_DENOMINATOR_FOR_SHARE_CAPS:
                    violations.append((share - cap, dimension, value))
        if not violations:
            break
        enforcement_passes += 1
        _, dimension, value = max(violations)
        for index in range(len(kept) - 1, -1, -1):
            if kept[index][dimension] == value:
                per_lineage[kept[index][lineage_key]] -= 1
                dropped[f"{dimension}_cap_enforcement"] += 1
                del kept[index]
                break
        else:                                                  # pragma: no cover
            break

    final_shares = {dimension: _shares(kept, dimension) for dimension in share_caps}
    violations = {
        dimension: {value: round(share, 4)
                    for value, share in shares.items() if share > share_caps[dimension]}
        for dimension, shares in final_shares.items()
    }
    violations = {d: v for d, v in violations.items() if v}

    return {
        "selected": kept,
        "dropped": dict(dropped.most_common()),
        "share_caps": dict(share_caps),
        "max_per_lineage": max_per_lineage,
        "min_denominator_for_share_caps": MIN_DENOMINATOR_FOR_SHARE_CAPS,
        "enforcement_passes": enforcement_passes,
        "final_shares": {d: {k: round(v, 4) for k, v in s.items()}
                         for d, s in final_shares.items()},
        "cap_violations": violations,
        "caps_satisfied": not violations,
        "max_rows_from_one_lineage": max(per_lineage.values()) if per_lineage else 0,
        "method": (
            "unique-first. Exact-duplicate targets and over-cap rows are "
            "DROPPED; no row is ever repeated. Share caps are tested on the "
            "prospective share and then re-enforced against the final "
            "selection, so caps_satisfied reflects the shares actually "
            "achieved rather than the ones intended."
        ),
    }


def weights(rows: Sequence[dict[str, Any]], dimensions: Sequence[str]
            ) -> dict[str, dict[str, float]]:
    """Inverse-frequency weights, as METADATA for a later sampler.

    Balancing by repeating a scarce row makes it look plentiful while teaching
    nothing new. A weight lets a sampler decide later, without changing what
    the dataset contains.
    """
    total = len(rows) or 1
    out: dict[str, dict[str, float]] = {}
    for dimension in dimensions:
        counts = Counter(r[dimension] for r in rows)
        out[dimension] = {
            value: round(total / (len(counts) * count), 4)
            for value, count in counts.items()
        } if counts else {}
    return out
