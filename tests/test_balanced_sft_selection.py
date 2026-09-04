from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.corpus import sha256_file, write_json
from scripts.build_balanced_sft_dataset import build
from scripts.train_on_dataset import apply_balanced_sft_selection
from utils.reproducibility import source_tree_sha256


ROOT = Path(__file__).resolve().parent.parent


def _pair(record_id: str, repository: bool = False) -> dict:
    return {
        "id": record_id,
        "execution_mode": (
            "repository_pytest_fragment" if repository else "function_assertion"
        ),
    }


def _ready_selection(tmp_path: Path, readiness: bool = True) -> Path:
    view = tmp_path / "balanced"
    view.mkdir()
    selection = [
        {"record_id": "s1", "origin_group": "synthetic_function"},
        {"record_id": "r1", "origin_group": "real_repository"},
        {"record_id": "r1", "origin_group": "real_repository"},
    ]
    write_json(view / "train.selection.json", selection)
    write_json(view / "train.manifest.json", {
        "split": "train",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "selection_sha256": sha256_file(view / "train.selection.json"),
        "policy": {"max_repeats": 2},
        "readiness": {
            "ready_for_final_sft": readiness,
            "blocking_conditions": ["need more unique repository targets"],
        },
    })
    return view


def test_exact_balanced_selection_preserves_bounded_replay_order(tmp_path: Path) -> None:
    selected = apply_balanced_sft_selection(
        [_pair("s1"), _pair("r1", repository=True)],
        _ready_selection(tmp_path),
    )
    assert [pair["id"] for pair in selected] == ["s1", "r1", "r1"]


def test_final_sft_refuses_provisional_effective_only_balance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provisional"):
        apply_balanced_sft_selection(
            [_pair("s1"), _pair("r1", repository=True)],
            _ready_selection(tmp_path, readiness=False),
        )


def test_builder_keeps_unique_repository_before_repeating(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory"
    multi = tmp_path / "multi"
    corpus_view = tmp_path / "view"
    output = tmp_path / "output"
    for path in (inventory, multi, corpus_view):
        path.mkdir()

    annotations = [
        {
            "record_id": "s1", "origin_group": "synthetic_function",
            "complexity_tier": "complex", "source_dataset": "humaneval",
        },
        {
            "record_id": "s2", "origin_group": "synthetic_function",
            "complexity_tier": "moderate", "source_dataset": "mbpp",
        },
        {
            "record_id": "r1", "origin_group": "real_repository",
            "function_lineage": "repo-defect-1", "primary_bug_family": "API_contract",
            "secondary_bug_tags": [], "complexity_tier": "complex",
            "source_dataset": "BugsInPy", "project": "project-a",
        },
    ]
    write_json(inventory / "train.annotations.json", annotations)
    write_json(multi / "train.examples.json", [
        {
            "displayed_record_id": "s1", "lineage": "function-1",
            "assertion_count": 2, "mutants_killed": 4,
            "primary_mutation_family": "boundary",
            "covered_mutation_families": ["boundary"],
            "completion": "def test_f():\n    assert f(0) == 0\n",
        },
        {
            "displayed_record_id": "s2", "lineage": "function-2",
            "assertion_count": 2, "mutants_killed": 3,
            "primary_mutation_family": "arithmetic",
            "covered_mutation_families": ["arithmetic"],
            "completion": "def test_g():\n    assert g(0) == 0\n",
        },
    ])
    write_json(corpus_view / "training_exclusions.json", [])

    manifest = build(inventory, multi, corpus_view, output, "train")
    assert manifest["stages"]["repository_after_project_audit"] == 1
    assert manifest["project_cap"]["unique_targets_dropped"] == 0
    assert manifest["unique_target_balance"]["target_met"] is False
    assert manifest["unique_target_balance"]["repository_unique_targets"] == 1
    assert manifest["effective_training_balance"]["target_met"] is True
