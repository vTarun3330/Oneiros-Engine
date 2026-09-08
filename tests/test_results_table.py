"""Every figure in the writeup comes from this table, so pin its derivation.

The one report on this project that was typed by hand quoted a base control
from a different run than the arms it compared. These tests are about the
properties that would let that happen again: a missing evaluation being
silently dropped, a gain computed against the wrong panel, or sealed data
being read at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_results_table import ARMS, build

TABLE = ROOT / "results" / "v4_2_results_table.json"


def _evaluation(path: Path, rate: float, benchmark: str = "mbpp",
                functions: int = 10, sealed: bool = False,
                split: str = "val") -> None:
    killed = round(rate * functions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "evaluation_split": split,
        "final_test_measurement": sealed,
        "evaluation_split_records": functions,
        "kill_at_k": {"8": {"rate": rate, "wilson_95": [rate - 0.01, rate + 0.01]}},
        "function_results": [
            {"dataset_name": benchmark, "killed": index < killed}
            for index in range(functions)
        ],
    }), encoding="utf-8")


def _tree(tmp_path: Path, arm_val: float = 0.70, base_val: float = 0.60) -> Path:
    _evaluation(tmp_path / "local_base_qwen_val_seed42"
                / "base_validation_standard_seed_42.json", base_val)
    _evaluation(tmp_path / "local_base_qwen_adev_seed42"
                / "base_validation_ablation-dev_seed_42.json", 0.50)
    _evaluation(tmp_path / "local_base_qwen_train_seed42"
                / "base_validation_train_smoke900_seed_42.json", 0.65)
    run = ARMS["relearning"]["run"]
    _evaluation(tmp_path / run / "sft_validation_standard_seed_42.json", arm_val)
    _evaluation(tmp_path / run / "sft_validation_train_smoke900_seed_42.json", 0.85)
    return tmp_path


def test_gains_are_computed_against_the_same_panel(tmp_path):
    table = build(42, results=_tree(tmp_path))
    arm = table["arms"]["relearning"]
    assert arm["val"]["gain_over_base"] == pytest.approx(0.10)
    assert "gain_over_base" not in arm.get("ablation_dev", {}), (
        "an arm with no ablation_dev evaluation must not borrow a gain from "
        "another panel"
    )


def test_a_missing_evaluation_is_named_not_dropped(tmp_path):
    table = build(42, results=_tree(tmp_path))
    assert "ablation_dev" in table["arms_missing"]["relearning"]
    assert "full_density" in table["arms_missing"]


def test_the_generalisation_gap_needs_both_panels(tmp_path):
    table = build(42, results=_tree(tmp_path))
    assert table["arms"]["relearning"]["generalisation_gap_points"] == 15.0
    assert "generalisation_gap_points" not in table["arms"]["long_two_epoch"]


def test_sealed_final_test_data_is_refused(tmp_path):
    tree = _tree(tmp_path)
    _evaluation(tree / ARMS["relearning"]["run"]
                / "sft_validation_standard_seed_42.json", 0.7, sealed=True)
    with pytest.raises(SystemExit):
        build(42, results=tree)


def test_a_test_split_artifact_is_refused(tmp_path):
    tree = _tree(tmp_path)
    _evaluation(tree / ARMS["relearning"]["run"]
                / "sft_validation_standard_seed_42.json", 0.7, split="test")
    with pytest.raises(SystemExit):
        build(42, results=tree)


def test_the_committed_table_matches_the_committed_artifacts():
    """A hand-edit or a replaced artifact fails here rather than being cited."""
    if not TABLE.exists():
        return
    committed = json.loads(TABLE.read_text(encoding="utf-8"))
    rebuilt = build(committed["seed"])
    assert committed["arms"] == rebuilt["arms"]
    assert committed["base"] == rebuilt["base"]


def test_the_table_states_that_repository_targets_are_unevaluated():
    """Every rate is synthetic-only; omitting that would overstate coverage."""
    if not TABLE.exists():
        return
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    assert table["repository_targets_evaluated"] == 0
    assert any("SYNTHETIC" in caveat for caveat in table["caveats"])
    assert any("prompt-copying" in caveat for caveat in table["caveats"])
