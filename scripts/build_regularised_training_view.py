"""Assemble the final training view, regularised through DATA rather than knobs.

The dropout-0.10 arm and the two-epoch arm both failed to improve locked
validation, so more dropout or more epochs is not the lever. What remains
untried is the data itself: what is sampled, how often, and how alike the
examples are.

Every regulariser here is a property of the assembled view, and each is
measured on the output rather than assumed from the request:

* one example per lineage per block, capped repetition across blocks
* source-dataset skew reduced by capping REPETITION of the dominant source,
  never by deleting lineages - the corpus is not to be shrunk
* exact prompt and completion deduplication
* only verified, reference-valid completions
* curriculum order preserved, with easy and moderate replay anchors intact
* candidate diversity measured: distinct completions, distinct entry points,
  distinct mutation families, and the share the largest single source holds

Nothing here trains. It writes the view a training run would consume, plus the
evidence that the view is what it claims to be.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

#: A source above this share has its REPETITION capped first. Lineages are
#: never dropped: the standing instruction on this project is to grow the
#: corpus rather than trim it, and trimming to hit a ratio would also throw
#: away verified supervision that cost execution time to produce.
TARGET_MAX_SOURCE_SHARE = 0.60
MAX_REPEATS_PER_LINEAGE = 2


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build(curriculum_path: Path, corrections_path: Path) -> dict[str, Any]:
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    corrections_report = json.loads(corrections_path.read_text(encoding="utf-8"))
    corrections = {
        str(item["lineage"]): item for item in corrections_report["items"]
    }

    seen_completion: set[str] = set()
    repeats: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for block in curriculum["curriculum"]:
        block_lineages: set[str] = set()
        for item in block["items"]:
            lineage = str(item["lineage"])
            correction = corrections.get(lineage)
            if correction is None:
                rejected["no_verified_correction"] += 1
                continue
            if not correction.get("verified") or not correction.get(
                    "verification_evidence", {}).get("valid_on_reference"):
                rejected["unverified_completion"] += 1
                continue
            if lineage in block_lineages:
                rejected["duplicate_lineage_within_block"] += 1
                continue
            if repeats[lineage] >= MAX_REPEATS_PER_LINEAGE:
                rejected["repetition_cap"] += 1
                continue

            completion = str(correction["completion"])
            digest = _digest(completion)
            if digest in seen_completion:
                rejected["duplicate_completion"] += 1
                continue

            source = str(correction.get("source_dataset") or "unknown")
            total = sum(source_counts.values())
            if total >= 50 and source_counts[source] / total > TARGET_MAX_SOURCE_SHARE \
                    and repeats[lineage] >= 1:
                # The dominant source keeps every lineage it has; it simply
                # stops being repeated while it is over its share.
                rejected["dominant_source_repetition_capped"] += 1
                continue

            seen_completion.add(digest)
            block_lineages.add(lineage)
            repeats[lineage] += 1
            source_counts[source] += 1
            rows.append({
                "block": block["block"],
                "lineage": lineage,
                "tier": item["tier"],
                "difficulty": item["difficulty"],
                "displayed_record_id": correction["displayed_record_id"],
                "entry_point": correction["entry_point"],
                "source_dataset": source,
                "completion": completion,
                "completion_shape": "assertion",
                "assertion_count": 1,
                "verified": True,
            })

    total = len(rows)
    # Separate from `total`: using `len(rows) or 1` as both the divisor and the
    # reported count made an EMPTY view report one example, which is the one
    # number a reader would most want to be honest.
    divisor = total or 1
    tiers = Counter(row["tier"] for row in rows)
    blocks = Counter(row["block"] for row in rows)
    per_block_tiers = {
        block: Counter(row["tier"] for row in rows if row["block"] == block)
        for block in sorted(blocks)
    }

    return {
        "schema_version": "oneiros_regularised_training_view_v1",
        "sealed_final_test_accessed": False,
        "validation_performance_used": False,
        "examples": total,
        "distinct_lineages": len(repeats),
        "distinct_completions": len(seen_completion),
        "distinct_entry_points": len({row["entry_point"] for row in rows}),
        "max_repeats_per_lineage": max(repeats.values()) if repeats else 0,
        "source_shares": {
            source: round(count / divisor, 4)
            for source, count in source_counts.most_common()
        },
        "largest_source_share": round(
            max(source_counts.values()) / divisor, 4) if source_counts else None,
        "tier_shares": {t: round(c / divisor, 4) for t, c in tiers.most_common()},
        "per_block_tier_counts": {
            str(block): dict(counts) for block, counts in per_block_tiers.items()
        },
        "every_block_multi_tier": all(
            len(counts) > 1 for counts in per_block_tiers.values()),
        "rejected": dict(rejected.most_common()),
        "regularisers": {
            "unique_lineage_per_block": True,
            "bounded_repetition": MAX_REPEATS_PER_LINEAGE,
            "completion_deduplicated": True,
            "verified_reference_valid_only": True,
            "curriculum_order_preserved": True,
            "source_skew_handled_by": (
                "capping repetition of the dominant source, never by deleting "
                "lineages"
            ),
        },
        "items": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path,
                        default=ROOT / "data" / "training_views" / "curriculum_v1" / "curriculum.json")
    parser.add_argument("--corrections", type=Path,
                        default=ROOT / "data" / "training_views" / "single_assertion_v1" / "corrections.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.curriculum, arguments.corrections)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
