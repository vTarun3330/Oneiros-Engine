"""Report achieved supervision weights, including ineffective oversampling."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from utils.dataset_identity import DATASET_IDENTITY_POLICY


def _identity(example: Any) -> tuple[str, ...]:
    return tuple(str(getattr(example, name, "")) for name in (
        "function_id", "prompt", "completion", "execution_mode",
    ))


def summarize_sampling_weights(raw: Sequence[Any], effective: Sequence[Any]) -> dict[str, Any]:
    """Weights after budget filtering/deduplication, before optimizer padding.

    A unique example is a record/prompt/completion/mode tuple. This makes
    duplicate counts explicit without mistaking 18 repetitions for 18 tests.
    """
    raw_counts = Counter(_identity(item) for item in raw)
    counts = Counter(_identity(item) for item in effective)
    if counts.keys() - raw_counts.keys():
        raise ValueError("Effective sampling contains an example outside the raw pool")
    metadata = {}
    for item in [*raw, *effective]:
        key = _identity(item)
        labels = tuple(str(getattr(item, attr, "unknown")) for attr in ("dataset", "bug_family", "dataset_family"))
        if key in metadata and metadata[key] != labels:
            raise ValueError("One example identity has conflicting dataset/family labels")
        metadata[key] = labels

    def table(attribute: str) -> dict[str, Any]:
        buckets: dict[str, list[Any]] = defaultdict(list)
        for example in raw:
            buckets[str(getattr(example, attribute, "") or "unknown")].append(example)
        effective_groups = Counter(
            str(getattr(item, attribute, "") or "unknown") for item in effective
        )
        result = {}
        for group, items in sorted(buckets.items()):
            identities = {_identity(item) for item in items}
            group_counts = [counts[key] for key in identities if counts[key]]
            effective_count = effective_groups[group]
            result[group] = {
                "raw_examples": len(items),
                "unique_examples": len(identities),
                "effective_unique_examples": len(group_counts),
                "effective_examples": effective_count,
                "extra_repetitions": effective_count - len(group_counts),
                "max_repeat_count": max(group_counts, default=0),
                "raw_weight": round(len(items) / len(raw), 8) if raw else 0.0,
                "effective_weight": round(effective_count / len(effective), 8) if effective else 0.0,
            }
        return result

    tables = {attribute: table(attribute) for attribute in ("dataset", "bug_family", "dataset_family")}
    return {
        "schema_version": "oneiros_sampling_weights_v1",
        "dataset_identity_policy": DATASET_IDENTITY_POLICY,
        "weight_scope": "post_budget_filter_and_dedup_before_optimizer_padding",
        "raw_examples": len(raw),
        "unique_examples": len(raw_counts),
        "effective_examples": len(effective),
        "effective_unique_examples": len(counts),
        "excluded_unique_examples": len(raw_counts.keys() - counts.keys()),
        "extra_repetitions": len(effective) - len(counts),
        "repeat_histogram": dict(sorted(Counter(str(value) for value in counts.values()).items())),
        "max_repeat_count": max(counts.values(), default=0),
        "unknown_dataset_examples": sum(getattr(item, "dataset", "unknown") == "unknown" for item in raw),
        "per_dataset": tables["dataset"],
        "per_mutation_family": tables["bug_family"],
        "per_dataset_family": tables["dataset_family"],
        "dataset_weights_changed": any(
            abs(row["raw_weight"] - row["effective_weight"]) > 1e-8
            for row in tables["dataset"].values()
        ),
    }
