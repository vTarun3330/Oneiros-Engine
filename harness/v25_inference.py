"""Protocol v2.5 confirmation inference (addendum 1 section 2), stdlib only.

- ``repository_means``: paired target-level differences averaged within each repository;
- ``sign_flip_test``: two-sided sign-flip test on repository-mean differences; exact (all
  2^k sign vectors) when 2^k <= ``flips``, otherwise ``flips`` Monte-Carlo vectors (seeded);
- ``cluster_bootstrap_ci``: percentile interval of the mean target difference, resampling
  repositories with replacement (seeded).
"""
from __future__ import annotations

from collections import defaultdict
import itertools
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

SEED = 20261001
FLIPS = 10_000
RESAMPLES = 10_000


def repository_means(rows: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    """``rows``: (repository, paired target difference) -> {repository: mean difference}."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for repo, diff in rows:
        acc[repo].append(float(diff))
    return {r: sum(v) / len(v) for r, v in sorted(acc.items())}


def sign_flip_test(means: Sequence[float], flips: int = FLIPS, seed: int = SEED) -> Dict:
    values = [float(m) for m in means]
    k = len(values)
    if k == 0:
        raise ValueError("no repositories")
    observed = abs(sum(values) / k)
    tol = 1e-12
    if 2 ** k <= flips:
        vectors = itertools.product((1, -1), repeat=k)
        total, extreme, method = 2 ** k, 0, "exact"
        for signs in vectors:
            if abs(sum(s * v for s, v in zip(signs, values)) / k) >= observed - tol:
                extreme += 1
        p = extreme / total
    else:
        rng = random.Random(seed)
        extreme = sum(abs(sum(v if rng.random() < 0.5 else -v for v in values) / k)
                      >= observed - tol for _ in range(flips))
        p, method = (extreme + 1) / (flips + 1), "monte_carlo"
    return {"statistic": sum(values) / k, "p_value": p, "repositories": k, "method": method}


def cluster_bootstrap_ci(rows: Iterable[Tuple[str, float]], resamples: int = RESAMPLES,
                         seed: int = SEED, level: float = 0.95) -> Dict:
    clusters: Dict[str, List[float]] = defaultdict(list)
    for repo, diff in rows:
        clusters[repo].append(float(diff))
    keys = sorted(clusters)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        picked = [clusters[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [d for c in picked for d in c]
        stats.append(sum(flat) / len(flat))
    stats.sort()
    lo = stats[int((1 - level) / 2 * resamples)]
    hi = stats[min(resamples - 1, int((1 + level) / 2 * resamples))]
    flat = [d for c in clusters.values() for d in c]
    return {"estimate": sum(flat) / len(flat), "lower": lo, "upper": hi, "level": level}


def leave_one_repository_out(rows: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    repos = sorted({r for r, _ in rows})
    out = {}
    for left in repos:
        kept = [d for r, d in rows if r != left]
        out[left] = sum(kept) / len(kept) if kept else float("nan")
    return out
