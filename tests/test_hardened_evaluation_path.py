from engine.generator import Phi3Generator
import pytest

import scripts.train_on_dataset as training
from scripts.train_on_dataset import (
    evaluate_pair,
    resolve_sft_dataset_fingerprint,
    sft_validation_results_filename,
)


def test_generator_parser_applies_candidate_policy():
    generator = Phi3Generator.__new__(Phi3Generator)
    generator.stats = {
        "total_generated": 0,
        "valid_generated": 0,
        "invalid_generated": 0,
    }
    valid = generator._parse_output("assert add(2, 3) == 5", "add")
    invalid = generator._parse_output("assert True", "add")
    assert valid.is_valid is True
    assert invalid.is_valid is False
    assert invalid.parse_error == "target_entry_point_not_called"


def test_evaluate_pair_counts_only_policy_and_reference_valid_candidates():
    winners, losers = evaluate_pair(
        [
            "assert add(2, 3) == 5",
            "assert add(2, 3) == 99",
            "assert True",
            "value = add(2, 3)\nassert value == 5",
        ],
        "def add(a, b): return a + b",
        "def add(a, b): return a - b",
        "add",
    )
    assert winners == ["assert add(2, 3) == 5"]
    assert losers == []


def test_evaluate_pair_retains_reference_valid_survivor_as_loser():
    winners, losers = evaluate_pair(
        ["assert add(2, 3) == 5"],
        "def add(a, b): return a + b",
        "def add(a, b): return a + b",
        "add",
    )
    assert winners == []
    assert losers == ["assert add(2, 3) == 5"]


def test_validation_seed_results_do_not_overwrite_each_other():
    assert sft_validation_results_filename(42) == "sft_validation_hardened_results_seed_42.json"
    assert sft_validation_results_filename(43) == "sft_validation_hardened_results_seed_43.json"


def test_final_test_phase_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(training, "TRAINING_PHASE", "dpo_eval")
    monkeypatch.setattr(training, "CONFIRM_FINAL_TEST", False)
    with pytest.raises(RuntimeError, match="final test split is sealed"):
        training.run_training(use_mock=True)


def test_evaluation_reuses_bounded_adapter_training_scope(tmp_path):
    manifest = {
        "corpus_id": "v3",
        "schema_version": "3",
        "files": {
            "records.json": {"sha256": "records"},
            "splits.json": {"sha256": "splits"},
        },
    }
    frozen = "v3:3:records:splits:sft_scope=first_800_train_records"
    version_file = tmp_path / "dataset_manifest.sha256"
    version_file.write_text(frozen + "\n", encoding="utf-8")
    assert resolve_sft_dataset_fingerprint(
        manifest, "full_train_split", version_file, run_sft=False
    ) == frozen
