"""The curriculum must be progressive, mixed, and free of validation signal.

Two failure modes it exists to avoid: a schedule that ends on a single tier,
which is how a model forgets what it could already do; and a difficulty score
tuned on the panel the arm is later judged by, which would be selecting on the
measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_progressive_curriculum import (
    MAX_REPEATS_PER_LINEAGE, SCHEDULE, TIERS, _base_train_difficulty,
    _structural_complexity, assign_tiers, build_schedule,
)

CURRICULUM = ROOT / "data" / "training_views" / "curriculum_v1" / "curriculum.json"


def _scored(n=90):
    return [{"lineage": f"g{i}", "displayed_record_id": f"r{i}",
             "difficulty": float(i), "source_dataset": "mbpp",
             "primary_mutation_family": "boundary",
             "difficulty_signals": {}} for i in range(n)]


def test_tiers_split_the_population_three_ways():
    tiers = {item["tier"] for item in assign_tiers(_scored())}
    assert tiers == set(TIERS)


def test_the_schedule_moves_from_easy_to_hard():
    hard = [spec["hard"] for spec in SCHEDULE]
    assert hard == sorted(hard), "the hard share must not decrease"
    assert hard[0] < hard[-1]


def test_no_block_is_a_single_tier():
    """A final block of one tier is how earlier ability is forgotten."""
    for spec in SCHEDULE:
        present = [tier for tier in TIERS if spec[tier] > 0]
        assert len(present) > 1, f"block {spec['block']} uses only {present}"


def test_the_last_block_keeps_replay_anchors():
    last = SCHEDULE[-1]
    assert last["easy"] > 0 and last["moderate"] > 0


def test_a_lineage_is_not_repeated_beyond_the_cap():
    schedule = build_schedule(assign_tiers(_scored()))
    assert max(schedule["repeats"].values()) <= MAX_REPEATS_PER_LINEAGE


def test_the_schedule_is_deterministic():
    """Two builds must give the same curriculum, or nothing is reproducible."""
    first = build_schedule(assign_tiers(_scored()))
    second = build_schedule(assign_tiers(_scored()))
    order = lambda s: [[i["lineage"] for i in b["items"]] for b in s["blocks"]]
    assert order(first) == order(second)


def test_validation_difficulty_is_refused(tmp_path):
    """The one signal that would make the curriculum select on its own judge."""
    artifact = tmp_path / "val.json"
    artifact.write_text(json.dumps({
        "evaluation_split": "val", "function_results": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        _base_train_difficulty(artifact)


def test_train_difficulty_is_accepted(tmp_path):
    artifact = tmp_path / "train.json"
    artifact.write_text(json.dumps({
        "evaluation_split": "train",
        "function_results": [{"record_id": "r", "killed": False}]}), encoding="utf-8")
    assert _base_train_difficulty(artifact) == {"r": True}


def test_branching_code_scores_harder_than_straight_line():
    simple = "def f(x):\n    return x + 1\n"
    branchy = ("def f(x):\n"
               "    if x > 0:\n"
               "        for i in range(x):\n"
               "            if i % 2:\n"
               "                return i\n"
               "    return 0\n")
    assert _structural_complexity(branchy) > _structural_complexity(simple)


def test_the_committed_curriculum_never_used_validation():
    if not CURRICULUM.exists():
        return
    report = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    assert report["validation_performance_used"] is False
    assert report["sealed_final_test_accessed"] is False
    assert report["never_single_tier_final_block"] is True


def test_the_committed_curriculum_records_realised_not_requested_mixture():
    """Dedup and repetition caps bend requested weights; the record must show it."""
    if not CURRICULUM.exists():
        return
    report = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    for block in report["blocks"]:
        assert "realised_mixture" in block and "requested_mixture" in block
        assert abs(sum(block["realised_mixture"].values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Schedule guards. Added after review raised hard-first curricula and replay
# anchors: the schedule already satisfied both, and a comment saying so does
# not survive someone editing the weights.
# ---------------------------------------------------------------------------

def test_the_shipped_schedule_is_progressive_and_mixed():
    from scripts.build_progressive_curriculum import (
        REPLAY_FLOOR, SCHEDULE, assert_schedule_is_progressive_and_mixed,
    )
    assert_schedule_is_progressive_and_mixed()
    hard = [block["hard"] for block in SCHEDULE]
    assert hard == sorted(hard)
    assert hard[0] < hard[-1], "a flat schedule is not a curriculum"
    for block in SCHEDULE:
        assert block["easy"] + block["moderate"] >= REPLAY_FLOOR


def test_a_hard_first_schedule_is_refused():
    from scripts.build_progressive_curriculum import (
        assert_schedule_is_progressive_and_mixed,
    )
    import pytest
    with pytest.raises(ValueError, match="non-decreasing"):
        assert_schedule_is_progressive_and_mixed((
            {"block": 1, "easy": 0.10, "moderate": 0.20, "hard": 0.70},
            {"block": 2, "easy": 0.40, "moderate": 0.40, "hard": 0.20},
        ))


def test_a_single_tier_final_block_is_refused():
    from scripts.build_progressive_curriculum import (
        assert_schedule_is_progressive_and_mixed,
    )
    import pytest
    with pytest.raises(ValueError, match="replay floor"):
        assert_schedule_is_progressive_and_mixed((
            {"block": 1, "easy": 0.40, "moderate": 0.35, "hard": 0.25},
            {"block": 2, "easy": 0.00, "moderate": 0.05, "hard": 0.95},
        ))
