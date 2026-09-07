"""Pin the source-dataset cap that keeps one benchmark from being the corpus.

Left uncapped, mbpp supplies 555 of 1009 unique targets - more than the other
three sources combined. The measured per-benchmark result makes the cost
concrete: SFT moves HumanEval +10.7 points and mbpp +3.3, so a corpus that is
mostly mbpp is mostly the case where the method does not work.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_balanced_sft_dataset import cap_dataset_share


def _entries(spec: dict[str, int]) -> list[dict]:
    out = []
    for dataset, count in spec.items():
        for i in range(count):
            out.append({"dataset": dataset, "target_key": f"{dataset}-{i}",
                        "bug_family": f"fam{i % 5}", "complexity_tier":
                        ["simple", "moderate", "complex"][i % 3]})
    return out


def test_the_dominant_dataset_is_trimmed_to_the_ceiling():
    kept, report = cap_dataset_share(
        _entries({"mbpp": 555, "swebench": 234, "bugsinpy": 112, "humaneval": 108}),
        0.35,
    )
    assert report["cap_applied"] is True
    assert report["cap_met"] is True
    for dataset, share in report["dataset_shares_after"].items():
        assert share <= 0.35 + 1e-9, f"{dataset} still exceeds the ceiling"
    assert report["dataset_counts_after"]["mbpp"] < 555


def test_scarce_datasets_are_never_padded_up_to_the_ceiling():
    """A ceiling is not a quota. Inventing targets would be manipulation."""
    kept, report = cap_dataset_share(
        _entries({"mbpp": 555, "swebench": 234, "bugsinpy": 112, "humaneval": 108}),
        0.35,
    )
    after = report["dataset_counts_after"]
    assert after["humaneval"] == 108
    assert after["bugsinpy"] == 112
    assert after["swebench"] == 234


def test_an_already_balanced_corpus_is_left_alone():
    entries = _entries({"a": 100, "b": 100, "c": 100, "d": 100})
    kept, report = cap_dataset_share(entries, 0.35)
    assert report["cap_applied"] is False
    assert report["cap_met"] is True
    assert len(kept) == len(entries)


def test_trimming_is_deterministic():
    spec = {"mbpp": 400, "other": 100}
    first, _ = cap_dataset_share(_entries(spec), 0.35)
    second, _ = cap_dataset_share(_entries(spec), 0.35)
    assert [e["target_key"] for e in first] == [e["target_key"] for e in second]


def test_trimming_keeps_rarer_bug_families_first():
    """Trimming must remove the most redundant targets, not an arbitrary tail."""
    entries = (
        [{"dataset": "big", "target_key": f"c{i}", "bug_family": "common",
          "complexity_tier": "simple"} for i in range(90)]
        + [{"dataset": "big", "target_key": f"r{i}", "bug_family": "rare",
            "complexity_tier": "simple"} for i in range(10)]
        + [{"dataset": "small", "target_key": f"s{i}", "bug_family": "common",
            "complexity_tier": "simple"} for i in range(30)]
    )
    kept, report = cap_dataset_share(entries, 0.35)
    survivors = {e["target_key"] for e in kept if e["dataset"] == "big"}
    assert {f"r{i}" for i in range(10)} <= survivors, (
        "every rare-family target should survive before any common one"
    )


def test_no_entries_is_handled():
    kept, report = cap_dataset_share([], 0.35)
    assert kept == []
    assert report["cap_met"] is True
