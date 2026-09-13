"""The O1 sidecar refuses everything that would make arm B uninterpretable.

The failure these guard against is specific and has happened: a supervision
sidecar that reaches almost none of the training batch produces a comparison
that measures ordinary supervision and reports it under the sidecar's name.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts import emit_o1_sidecar as sidecar
from scripts.emit_o1_sidecar import MAX_SIDECAR_SHARE, build, proportional_subsample
from scripts.preflight_o1_sidecar_ab import HELD_CONSTANT, _dig

ROOT = Path(__file__).resolve().parent.parent


def _row(index: int, source: str = "mbpp", origin: str = "synthetic"):
    return {
        "record_id": f"mutation::rec_{index}",
        "sft_target": f"assert f({index}) == {index}",
        "candidate_shape": "assertion",
        "candidate_position": index % 8,
        "supervision_role": "positive_original" if index % 2 else
                            "positive_verified_correction",
        "label": "valid_killing_test" if index % 2 else "wrong_oracle",
        "source_dataset": source,
        "bug_family": f"family_{index % 4}",
        "complexity_tier": "simple",
        "origin": origin,
        "function_lineage": f"lineage_{index % 25}",
    }


def _dataset(tmp_path: Path, rows, **manifest_over):
    directory = tmp_path / "o1"
    directory.mkdir()
    positives = directory / "positives.json"
    positives.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "positives_sha256": hashlib.sha256(positives.read_bytes()).hexdigest(),
        "candidates_sha256": "c" * 64,
        "source_derived_sha256": "d" * 64,
        "evaluation_split": "train",
        "sealed_final_test_accessed": False,
        "canonical_records_json_opened": False,
    }
    manifest.update(manifest_over)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return directory


@pytest.fixture
def train_shard(monkeypatch):
    """Every synthetic record id is in the train shard unless a test says not."""
    ids = {f"mutation::rec_{index}" for index in range(500)}

    def fake(corpus_dir, split, include_excluded=False):
        assert split == "train", "the sidecar may only read the train shard"
        return [{"id": value} for value in sorted(ids)]

    monkeypatch.setattr(sidecar, "load_development_split", fake)
    return ids


# --------------------------------------------------------- subsample honesty

def test_the_subsample_preserves_source_shares():
    """Truncating a sorted list would silently rewrite the capped shares."""
    rows = ([_row(i, "mbpp") for i in range(700)]
            + [_row(1000 + i, "humaneval") for i in range(300)])
    kept = proportional_subsample(rows, 200, "source_dataset")
    shares = Counter(r["source_dataset"] for r in kept)
    assert len(kept) == 200
    assert shares["mbpp"] == 140 and shares["humaneval"] == 60


def test_the_subsample_never_duplicates():
    rows = [_row(i) for i in range(300)]
    kept = proportional_subsample(rows, 120, "source_dataset")
    targets = [r["sft_target"] for r in kept]
    assert len(targets) == len(set(targets)) == 120


def test_the_subsample_is_deterministic_under_input_order():
    rows = ([_row(i, "mbpp") for i in range(200)]
            + [_row(500 + i, "humaneval") for i in range(90)])
    first = proportional_subsample(rows, 77, "source_dataset")
    second = proportional_subsample(list(reversed(rows)), 77, "source_dataset")
    assert [r["sft_target"] for r in first] == [r["sft_target"] for r in second]


def test_asking_for_everything_returns_everything():
    rows = [_row(i) for i in range(40)]
    assert len(proportional_subsample(rows, 999, "source_dataset")) == 40


# ------------------------------------------------------------ refusal matrix

def test_a_positives_hash_mismatch_is_refused(tmp_path, train_shard):
    rows = [_row(i) for i in range(50)]
    directory = _dataset(tmp_path, rows, positives_sha256="0" * 64)
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    assert any("hashes to" in p for p in result["problems"])


def test_a_record_outside_the_train_shard_is_refused(tmp_path, train_shard):
    rows = [_row(i) for i in range(20)]
    rows.append(_row(9999))                       # not in the fixture's shard
    directory = _dataset(tmp_path, rows)
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    assert any("outside the train shard" in p for p in result["problems"])


def test_a_non_train_dataset_is_refused(tmp_path, train_shard):
    directory = _dataset(tmp_path, [_row(i) for i in range(20)],
                         evaluation_split="val")
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    assert any("split is 'val'" in p for p in result["problems"])


def test_a_dataset_not_denying_sealed_access_is_refused(tmp_path, train_shard):
    directory = _dataset(tmp_path, [_row(i) for i in range(20)],
                         sealed_final_test_accessed=None)
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    assert any("sealed-test" in p for p in result["problems"])


def test_a_ratio_above_the_auxiliary_ceiling_is_refused(tmp_path, train_shard):
    """O1 is a bounded component. A majority sidecar is a different experiment."""
    directory = _dataset(tmp_path, [_row(i) for i in range(200)])
    result = build(directory, tmp_path, baseline_pairs=800,
                   ratio=MAX_SIDECAR_SHARE + 0.01)
    assert any("auxiliary ceiling" in p for p in result["problems"])


def test_a_non_positive_row_is_refused(tmp_path, train_shard):
    rows = [_row(i) for i in range(20)]
    rows[3]["supervision_role"] = "negative"
    directory = _dataset(tmp_path, rows)
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    assert any("not a positive role" in p for p in result["problems"])


def test_an_empty_target_is_refused(tmp_path, train_shard):
    rows = [_row(i) for i in range(20)]
    rows[5]["sft_target"] = "   "
    directory = _dataset(tmp_path, rows)
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    assert any("empty sft_target" in p for p in result["problems"])


# ------------------------------------------------------------ ratio and rows

def test_the_achieved_ratio_matches_the_request(tmp_path, train_shard):
    directory = _dataset(tmp_path, [_row(i) for i in range(400)])
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    assert result["problems"] == []
    report = result["report"]
    assert report["sidecar_rows"] == 200            # 200 / (800 + 200) = 0.20
    assert report["achieved_sidecar_share"] == pytest.approx(0.20, abs=0.005)


def test_a_short_supply_reports_the_share_it_actually_reached(tmp_path, train_shard):
    """Fewer positives than the ratio wants must not silently claim the ratio."""
    directory = _dataset(tmp_path, [_row(i) for i in range(30)])
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20)
    report = result["report"]
    assert report["sidecar_rows"] == 30
    assert report["achieved_sidecar_share"] < 0.20


def test_every_row_carries_exactly_one_repeat(tmp_path, train_shard):
    directory = _dataset(tmp_path, [_row(i) for i in range(120)])
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.10)
    assert {row["repeats"] for row in result["sidecar"]} == {1}
    assert result["report"]["duplicated_rows"] == 0
    completions = [row["completion"] for row in result["sidecar"]]
    assert len(completions) == len(set(completions))


def test_the_report_counts_real_repository_rows_separately(tmp_path, train_shard):
    rows = [_row(i) for i in range(90)]
    for index in range(3):
        rows[index]["origin"] = "real_repository"
    directory = _dataset(tmp_path, rows)
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.10)
    assert "real_repository_rows" in result["report"]
    assert result["report"]["shares"]["origin"].get("real_repository") is not None


def test_verification_is_not_extended_to_siblings(tmp_path, train_shard):
    """Each positive killed its own displayed mutant, and only that one."""
    directory = _dataset(tmp_path, [_row(i) for i in range(40)])
    row = build(directory, tmp_path, baseline_pairs=800, ratio=0.10)["sidecar"][0]
    assert "never extended to sibling mutants" in row["verification"]


# --------------------------------------------------- held-constant contract

def test_every_held_constant_field_is_read_from_arm_a():
    """A field arm B sets itself is a field the two arms can differ on."""
    report = {
        "tokenization": {"model_name": "m", "model_revision": "main",
                         "prompt_token_limit": 1024,
                         "completion_token_limit": 1024,
                         "sequence_token_limit": 2048,
                         "prompt_information_variant": "full",
                         "output_instruction_variant": "self_contained",
                         "prompt_compaction_strategy": "s"},
        "training": {"epochs": 1, "batch_size": 1, "learning_rate": 1e-5,
                     "lr_scheduler_type": "constant_with_warmup",
                     "warmup_steps_requested": 25, "checkpoint_steps": 50},
        "evaluation_panel": {"evaluation_split": "ablation_dev"},
    }
    for name, path in HELD_CONSTANT:
        assert _dig(report, path) is not None, name


def test_a_missing_held_constant_field_is_visible():
    assert _dig({"training": {}}, ("training", "epochs")) is None
    assert _dig({}, ("tokenization", "model_name")) is None


def test_the_held_constant_list_covers_what_the_brief_named():
    names = {name for name, _ in HELD_CONSTANT}
    assert {"model_name", "learning_rate", "epochs", "prompt_token_limit",
            "completion_token_limit", "evaluation_split"} <= names


# ------------------------------------------------- token budget, before draw

def test_an_over_budget_row_is_dropped_before_the_subsample(tmp_path, train_shard):
    """Filtering after the draw would make the ratio depend on the draw.

    One long row in a 1,316-row supply either lands in the sample or does not.
    Dropped afterwards it yields 1,304 rows on some days and 1,305 on others,
    from the same requested ratio.
    """
    rows = [_row(i) for i in range(400)]
    rows[7]["sft_target"] = "assert f(7) == " + ("9" * 400)
    directory = _dataset(tmp_path, rows)
    long_target = rows[7]["sft_target"]
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.20,
                   completion_tokens=lambda text: len(text),
                   max_completion_tokens=100)
    report = result["report"]
    assert report["dropped_over_token_budget"] == 1
    assert report["token_budget_eligible_positives"] == 399
    assert report["sidecar_rows"] == 200
    assert long_target not in {r["completion"] for r in result["sidecar"]}


def test_nothing_is_ever_truncated_to_fit(tmp_path, train_shard):
    rows = [_row(i) for i in range(60)]
    rows[3]["sft_target"] = "assert f(3) == " + ("8" * 300)
    directory = _dataset(tmp_path, rows)
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.05,
                   completion_tokens=lambda text: len(text),
                   max_completion_tokens=80)
    assert result["report"]["completions_truncated"] == 0
    for row in result["sidecar"]:
        assert len(row["completion"]) <= 80


def test_the_exact_requested_count_survives_an_over_budget_row(tmp_path, train_shard):
    """The 16.00% decision: 1,305 must be 1,305 regardless of which rows drew."""
    rows = [_row(i) for i in range(300)]
    rows[11]["sft_target"] = "assert f(11) == " + ("7" * 500)
    directory = _dataset(tmp_path, rows)
    result = build(directory, tmp_path, baseline_pairs=1000, ratio=0.10,
                   completion_tokens=lambda text: len(text),
                   max_completion_tokens=120)
    # 0.10 * 1000 / 0.90 = 111.1 -> 111
    assert result["report"]["sidecar_rows"] == 111
    assert len(result["sidecar"]) == 111


def test_unused_positives_are_counted(tmp_path, train_shard):
    """Eleven rows are deliberately left on the table; that must be visible."""
    directory = _dataset(tmp_path, [_row(i) for i in range(300)])
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.10)
    report = result["report"]
    assert report["unused_verified_positives"] == \
        report["available_positives"] - report["sidecar_rows"]


def test_no_token_filter_means_no_rows_dropped_for_budget(tmp_path, train_shard):
    directory = _dataset(tmp_path, [_row(i) for i in range(80)])
    result = build(directory, tmp_path, baseline_pairs=800, ratio=0.05)
    assert result["report"]["dropped_over_token_budget"] == 0
    assert result["report"]["max_completion_tokens"] is None
