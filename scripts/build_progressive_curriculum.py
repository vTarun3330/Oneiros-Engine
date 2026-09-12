"""Tier training examples by difficulty, using training-only information.

Difficulty is scored from signals available before any evaluation: the shape
of the code, the rarity of the mutation family, how hard the untrained base
model found the TRAIN panel, how much work the verified oracle needed, and how
structured the inputs are. Locked validation and the sealed split are never
consulted - a curriculum tuned on validation performance would be selecting on
the thing it is later measured by.

The schedule is progressive and MIXED, never hard-first or easy-first:

* early blocks are mostly easy and moderate, so the model is not learning the
  task and its hardest cases simultaneously;
* middle blocks raise the hard share;
* late blocks are hard-heavy but keep easy and moderate replay anchors,
  because a final block of a single tier is how a model forgets what it could
  already do.

The realised mixture is recorded per block rather than assumed from the
requested weights, since deduplication and repetition caps can bend one into
the other.
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json

TIERS = ("easy", "moderate", "hard")

#: Requested tier weights per block. Late blocks stay mixed on purpose.
#:
#: Block 1 opens at 40/35/25 rather than the original 50/35/15. The earlier
#: opening was not wrong - it was easy-first and mixed, which is what the
#: curriculum literature supports - but it spent half the first block on the
#: tier the model already handles, and the realised mixture is measured below
#: rather than assumed, so the change can be checked instead of argued.
#:
#: Block 4 keeps 35% non-hard. That is the replay anchor, and REPLAY_FLOOR
#: below makes it a checked property rather than a comment: a final block of a
#: single tier is how a model forgets what it could already do.
SCHEDULE = (
    {"block": 1, "easy": 0.40, "moderate": 0.35, "hard": 0.25},
    {"block": 2, "easy": 0.30, "moderate": 0.40, "hard": 0.30},
    {"block": 3, "easy": 0.15, "moderate": 0.35, "hard": 0.50},
    {"block": 4, "easy": 0.10, "moderate": 0.25, "hard": 0.65},
)

#: No block may fall below this share of non-hard examples. Checked against the
#: REALISED mixture, because deduplication and the repetition cap can bend a
#: requested mixture into a different one.
REPLAY_FLOOR = 0.20

MAX_REPEATS_PER_LINEAGE = 2


def assert_schedule_is_progressive_and_mixed(schedule=SCHEDULE) -> None:
    """A hard-first or single-tier schedule is refused at import time.

    Stated as an executable rule because it is the one curriculum property
    this project was explicitly instructed to preserve, and a comment does not
    survive someone editing the weights.
    """
    hard = [block["hard"] for block in schedule]
    if hard != sorted(hard):
        raise ValueError(
            "the hard share must be non-decreasing: a hard-first schedule asks "
            "the model to learn the task and its hardest cases at once")
    for block in schedule:
        replay = block["easy"] + block["moderate"]
        if replay < REPLAY_FLOOR:
            raise ValueError(
                f"block {block['block']} keeps only {replay:.0%} non-hard "
                f"examples, below the {REPLAY_FLOOR:.0%} replay floor")


assert_schedule_is_progressive_and_mixed()


def _structural_complexity(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    branching = sum(
        isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp,
                          ast.Compare, ast.comprehension))
        for node in ast.walk(tree)
    )
    return len(list(ast.walk(tree))) + 5 * branching


def _arity_and_shape(code: str, entry_point: str) -> tuple[int, int]:
    """(argument count, structured-input weight)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return (0, 0)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == entry_point:
            arguments = list(node.args.posonlyargs) + list(node.args.args)
            annotated = sum(1 for a in arguments if a.annotation is not None)
            # Unannotated parameters must have their shape inferred, which is
            # where the fuzzer's adapter failed and where the model has least
            # to go on.
            return (len(arguments), len(arguments) - annotated)
    return (0, 0)


def _base_train_difficulty(path: Path) -> dict[str, bool]:
    """Which TRAIN records the untrained base model failed.

    Train only. Using validation here would tune the curriculum on the panel
    it is later judged by.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_split") not in (None, "train"):
        raise SystemExit(
            f"{path} is split {payload.get('evaluation_split')!r}; the "
            "curriculum may only use train-split difficulty"
        )
    return {
        str(row.get("record_id")): not bool(row.get("killed"))
        for row in payload.get("function_results") or []
    }


def score(examples: list[dict[str, Any]], records: dict[str, dict[str, Any]],
          base_failed: dict[str, bool]) -> list[dict[str, Any]]:
    families = Counter(str(e.get("primary_mutation_family")) for e in examples)
    rarest = max(families.values()) if families else 1

    scored: list[dict[str, Any]] = []
    for example in examples:
        record = records.get(str(example.get("displayed_record_id"))) or {}
        reference = str(record.get("reference_code") or "")
        entry = str(example.get("entry_point") or "")
        arity, unannotated = _arity_and_shape(reference, entry)
        family = str(example.get("primary_mutation_family"))
        siblings = int(example.get("mutants_evaluated") or 0)
        killed = int(example.get("mutants_killed") or 0)

        signals = {
            # Bigger, more branching functions are harder to reason about.
            "structural": _structural_complexity(reference) / 100.0,
            # A family the corpus rarely shows is one the model has seen least.
            "family_rarity": 1.0 - (families[family] / rarest),
            # The base model failing this TRAIN record is direct evidence.
            "base_failed": 1.0 if base_failed.get(
                str(example.get("displayed_record_id"))) else 0.0,
            # An oracle that distinguishes few siblings was hard to find.
            "oracle_difficulty": 1.0 - (killed / siblings) if siblings else 0.5,
            "arity": min(arity, 5) / 5.0,
            "unstructured_inputs": min(unannotated, 5) / 5.0,
        }
        total = sum(signals.values())
        scored.append({
            "lineage": str(example.get("lineage")),
            "displayed_record_id": str(example.get("displayed_record_id")),
            "source_dataset": str(example.get("source_dataset")),
            "primary_mutation_family": family,
            "difficulty_signals": {k: round(v, 4) for k, v in signals.items()},
            "difficulty": round(total, 4),
        })
    return scored


def assign_tiers(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = sorted(item["difficulty"] for item in scored)
    if not values:
        return scored
    low = values[len(values) // 3]
    high = values[(2 * len(values)) // 3]
    for item in scored:
        item["tier"] = (
            "easy" if item["difficulty"] <= low
            else "moderate" if item["difficulty"] <= high else "hard"
        )
    return scored


def build_schedule(scored: list[dict[str, Any]]) -> dict[str, Any]:
    pools = {tier: [i for i in scored if i["tier"] == tier] for tier in TIERS}
    for tier in TIERS:
        # Deterministic order: by difficulty then lineage, so two runs of this
        # script produce the same curriculum.
        pools[tier].sort(key=lambda item: (item["difficulty"], item["lineage"]))

    used: Counter[str] = Counter()
    blocks: list[dict[str, Any]] = []
    per_block = max(1, len(scored) // len(SCHEDULE))

    cursors = {tier: 0 for tier in TIERS}
    for spec in SCHEDULE:
        chosen: list[dict[str, Any]] = []
        for tier in TIERS:
            want = int(round(spec[tier] * per_block))
            pool = pools[tier]
            taken = 0
            while taken < want and pool:
                item = pool[cursors[tier] % len(pool)]
                cursors[tier] += 1
                if used[item["lineage"]] >= MAX_REPEATS_PER_LINEAGE:
                    if cursors[tier] % len(pool) == 0 and all(
                            used[p["lineage"]] >= MAX_REPEATS_PER_LINEAGE
                            for p in pool):
                        break
                    continue
                used[item["lineage"]] += 1
                chosen.append({**item, "block": spec["block"]})
                taken += 1
        realised = Counter(item["tier"] for item in chosen)
        total = sum(realised.values()) or 1
        blocks.append({
            "block": spec["block"],
            "requested_mixture": {t: spec[t] for t in TIERS},
            "realised_mixture": {
                t: round(realised.get(t, 0) / total, 4) for t in TIERS},
            "examples": len(chosen),
            "items": chosen,
        })
    return {"blocks": blocks, "repeats": used}


def build(view: Path, corpus_dir: Path, base_train: Path) -> dict[str, Any]:
    payload = json.loads((view / "train.examples.json").read_text(encoding="utf-8"))
    examples = payload["examples"] if isinstance(payload, dict) and "examples" in payload else payload
    records = {
        str(r["id"]): r for r in
        json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    }
    base_failed = _base_train_difficulty(base_train)

    scored = assign_tiers(score(examples, records, base_failed))
    schedule = build_schedule(scored)
    tiers = Counter(item["tier"] for item in scored)
    difficulties = [item["difficulty"] for item in scored]

    return {
        "schema_version": "oneiros_progressive_curriculum_v1",
        "source_view": view.name,
        "sealed_final_test_accessed": False,
        "validation_performance_used": False,
        "signals": [
            "structural complexity of the reference",
            "mutation-family rarity within train",
            "base model failure on the TRAIN panel",
            "verified oracle difficulty (siblings distinguished)",
            "argument arity",
            "unannotated / structured inputs",
        ],
        "lineages": len(scored),
        "tier_counts": dict(tiers),
        "difficulty_range": {
            "min": round(min(difficulties), 4) if difficulties else None,
            "median": round(statistics.median(difficulties), 4) if difficulties else None,
            "max": round(max(difficulties), 4) if difficulties else None,
        },
        "max_repeats_per_lineage": MAX_REPEATS_PER_LINEAGE,
        "distinct_lineages_used": len(schedule["repeats"]),
        "blocks": [
            {k: v for k, v in block.items() if k != "items"}
            for block in schedule["blocks"]
        ],
        "curriculum": [
            {"block": block["block"], "items": [
                {"lineage": i["lineage"], "tier": i["tier"],
                 "difficulty": i["difficulty"],
                 "displayed_record_id": i["displayed_record_id"]}
                for i in block["items"]]}
            for block in schedule["blocks"]
        ],
        "never_single_tier_final_block": all(
            sum(1 for share in block["realised_mixture"].values() if share > 0) > 1
            for block in schedule["blocks"]
        ),
        "replay_floor": REPLAY_FLOOR,
        "realised_replay_share_per_block": {
            str(block["block"]): round(
                block["realised_mixture"]["easy"]
                + block["realised_mixture"]["moderate"], 4)
            for block in schedule["blocks"]
        },
        "every_block_meets_replay_floor": all(
            block["realised_mixture"]["easy"] + block["realised_mixture"]["moderate"]
            >= REPLAY_FLOOR for block in schedule["blocks"]
        ),
        "hard_share_is_non_decreasing": [
            block["realised_mixture"]["hard"] for block in schedule["blocks"]
        ] == sorted(
            block["realised_mixture"]["hard"] for block in schedule["blocks"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path,
                        default=ROOT / "data" / "training_views" / "multi_mutant_v1")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--base-train", type=Path,
                        default=ROOT / "results" / "local_base_qwen_train_seed42"
                        / "base_validation_train_smoke900_seed_42.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.view, arguments.corpus, arguments.base_train)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "curriculum"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
