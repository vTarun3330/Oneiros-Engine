"""The last development-side measurement must spend its freedom in advance.

Locked validation is the promotion decision. After it there is nothing left but
the sealed test. So the interesting failures are not crashes - they are the
quiet ones: a threshold that could be edited after the fact, an output path
that lands on a development artifact, a validation record read during the
"preflight", an adapter swapped for a different checkpoint, or an evaluator
that drifted since the development measurement this decision inherits.

Each test below pins one of those.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import preflight_locked_validation as pf

ROOT = Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SHA = "2e1fd397ee46e1388853d2af2c993145b0f1098a"


@pytest.fixture(scope="module")
def receipt():
    problems: list[str] = []
    return pf.collect(problems), problems


# --------------------------------------------------------- it actually passes

def test_the_preflight_passes_on_the_real_repository(receipt):
    built, problems = receipt
    assert problems == [], problems
    assert built["ready_to_launch"] is True


def test_the_preflight_launches_nothing(receipt):
    built, _ = receipt
    assert built["launched"] is False
    assert built["cpu_only_preflight"] is True


# ------------------------------------------------------------ the two arms

def test_exactly_two_arms(receipt):
    built, _ = receipt
    assert set(built["arms"]) == {"A_base_control", "B_arm_a_checkpoint_431"}


def test_the_base_arm_carries_no_adapter(receipt):
    built, _ = receipt
    assert built["arms"]["A_base_control"]["adapter"] is None
    assert "--phase" in built["arms"]["A_base_control"]["command"]
    cmd = built["arms"]["A_base_control"]["command"]
    assert cmd[cmd.index("--phase") + 1] == "base_eval"


def test_the_arm_a_adapter_is_pinned_to_the_selected_checkpoint(receipt):
    built, _ = receipt
    arm = built["arms"]["B_arm_a_checkpoint_431"]
    assert arm["adapter_checkpoint_step"] == 431
    assert arm["adapter_sha256_expected"] == pf.ARM_A_431_ADAPTER_SHA256
    assert arm["adapter_sha256_found"] == pf.ARM_A_431_ADAPTER_SHA256


def test_a_substituted_adapter_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "ARM_A_431_ADAPTER_SHA256", "f" * 64)
    problems: list[str] = []
    pf.collect(problems)
    assert any("adapter hash does not match" in p for p in problems)


def test_a_missing_adapter_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "ARM_A_431_ADAPTER", "checkpoints/does_not_exist/adapter.safetensors")
    problems: list[str] = []
    pf.collect(problems)
    assert any("adapter is missing" in p for p in problems)


# ------------------------------------------------------------ model identity

def test_the_model_revision_is_immutable(receipt):
    built, _ = receipt
    identity = built["model_identity"]
    assert identity["base_model_revision"] == SHA
    assert identity["selection_tokenizer_revision"] == SHA
    assert identity["base_model_name"] == QWEN


@pytest.mark.parametrize("moving", ["main", "master", "latest", "HEAD"])
def test_a_moving_model_reference_is_refused(monkeypatch, moving):
    monkeypatch.setattr(pf, "immutable_revision_for", lambda name: moving)
    problems: list[str] = []
    pf.collect(problems)
    assert any("immutable" in p or "base_model_revision" in p for p in problems)


def test_a_shortened_sha_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "immutable_revision_for", lambda name: SHA[:12])
    problems: list[str] = []
    pf.collect(problems)
    assert problems


# ------------------------------------------------- split identity and sealing

def test_the_split_is_val_and_the_sealed_test_is_not_measured(receipt):
    built, _ = receipt
    assert built["protocol"]["evaluation_split"] == "val"
    assert built["protocol"]["final_test_measurement"] is False
    assert built["sealed_final_test"]["accessed"] is False
    assert built["sealed_final_test"]["split"] == "test"


def test_the_sealed_split_is_absent_from_the_corpus_view(receipt):
    built, _ = receipt
    assert "test" not in built["corpus"]["included_splits"]
    assert built["corpus"]["sealed_splits_excluded"] == ["test"]


def test_split_identity_is_bound_without_reading_a_validation_record(receipt):
    """The whole point: identity from manifest metadata, not from payload."""
    built, _ = receipt
    ident = built["evaluation_split_identity"]
    assert ident["records_read_by_this_preflight"] == 0
    assert ident["split"] == "val"
    assert len(ident["shard_sha256"]) == 64
    assert len(ident["record_ids_sha256"]) == 64
    assert ident["record_count"] == 781


def test_the_recorded_shard_hash_is_the_real_one(receipt):
    """A hash nobody checks is decoration."""
    built, _ = receipt
    shard = (ROOT / "data" / "corpus" / pf.CORPUS_VERSION / "development_view"
             / built["evaluation_split_identity"]["shard_filename"])
    assert shard.is_file()
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == built["evaluation_split_identity"]["shard_sha256"]


def test_a_view_that_includes_the_sealed_split_is_refused():
    bad = {"included_splits": ["train", "val", "test"],
           "sealed_splits_excluded": [], "splits": {"val": {}}}
    found = pf.view_problems(bad)
    assert any("includes the sealed split" in p for p in found)
    assert any("does not declare the sealed split excluded" in p for p in found)


def test_a_view_without_the_validation_shard_is_refused():
    found = pf.view_problems(
        {"included_splits": ["train"], "sealed_splits_excluded": ["test"], "splits": {}})
    assert any("has no 'val' shard" in p for p in found)


def test_the_real_view_has_no_problems():
    view = json.loads(
        (ROOT / "data" / "corpus" / pf.CORPUS_VERSION / "development_view" / "manifest.json")
        .read_text(encoding="utf-8"))
    assert pf.view_problems(view) == []


# --------------------------------------------------------- output isolation

def test_output_run_names_are_not_development_run_names(receipt):
    built, _ = receipt
    assert pf.BASE_RUN_NAME not in pf.DEVELOPMENT_RUN_NAMES
    assert pf.ARM_A_RUN_NAME not in pf.DEVELOPMENT_RUN_NAMES
    assert built["output_isolation"]["collides_with_development_artifacts"] is False


def test_an_output_name_that_would_overwrite_development_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "BASE_RUN_NAME", "local_sft_armA_baseline_successor_s42")
    problems: list[str] = []
    pf.collect(problems)
    assert any("collides with a development run" in p or "already exists" in p for p in problems)


def test_every_development_run_is_listed_as_untouchable(receipt):
    built, _ = receipt
    listed = built["output_isolation"]["development_run_names_that_must_not_be_touched"]
    for name in ("local_base_qwen_ablationdev_successor_s42_v3",
                 "local_sft_armA_baseline_successor_s42",
                 "local_sft_armB_o1_nocollide_s42",
                 "local_eval_armA_ckpt150_matched"):
        assert name in listed


# --------------------------------------------------------- protocol identity

def test_the_protocol_is_the_named_successor_protocol(receipt):
    built, _ = receipt
    p = built["protocol"]
    assert p["protocol_name"] == "oneiros_successor_generation_protocol_v1"
    assert p["candidate_parse_mode"] == "whole_output"
    assert p["retain_raw_output"] is True
    assert p["candidates_per_function"] == 8
    assert p["generation_seed"] == 42
    assert p["temperature"] == 0.7
    assert p["top_p"] == 0.9
    assert p["function_generation_completion_limit"] == 1024


def test_no_reranking_and_no_feedback_rounds(receipt):
    built, _ = receipt
    assert built["protocol"]["feedback_rounds"] == 0
    assert built["protocol"]["diversity_mode"] == "none"
    assert "none" in built["protocol"]["reranking"]


def test_the_commands_carry_the_successor_protocol_flag(receipt):
    built, _ = receipt
    for arm in built["arms"].values():
        assert "--successor-protocol" in arm["command"]
        assert "--evaluation-split" in arm["command"]
        cmd = arm["command"]
        assert cmd[cmd.index("--evaluation-split") + 1] == "val"


def test_both_arms_use_an_identical_command_apart_from_phase_and_run_name(receipt):
    built, _ = receipt
    a = list(built["arms"]["A_base_control"]["command"])
    b = list(built["arms"]["B_arm_a_checkpoint_431"]["command"])
    for cmd in (a, b):
        del cmd[cmd.index("--run-name"): cmd.index("--run-name") + 2]
        del cmd[cmd.index("--phase"): cmd.index("--phase") + 2]
    assert a == b


# --------------------------------------------- evaluator drift since dev run

def test_the_evaluator_has_not_drifted_since_the_development_measurement(receipt):
    built, _ = receipt
    assert built["contract_sources_match_development_measurement"] is True


def test_evaluator_drift_is_refused(monkeypatch, tmp_path):
    """If the scorer changed, locked validation is not comparable with dev."""
    frozen = json.loads((ROOT / pf.FROZEN_DEVELOPMENT_RECEIPT).read_text(encoding="utf-8"))
    frozen["contract_source_hashes"]["evaluator"] = "0" * 64
    # Point only the receipt lookup at tmp; everything else stays real. An
    # absolute right-hand operand wins in Path.__truediv__, so ROOT / target
    # resolves to target.
    target = tmp_path / "drifted_frozen_receipt.json"
    target.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(pf, "FROZEN_DEVELOPMENT_RECEIPT", str(target))
    problems: list[str] = []
    built = pf.collect(problems)
    assert any("changed since the development measurement" in p for p in problems)
    assert any("metrics/research_evaluation.py" in p for p in problems)
    assert built["contract_sources_match_development_measurement"] is False


def test_runtime_components_are_hashed_file_by_file(receipt):
    built, _ = receipt
    hashes = built["runtime_component_hashes"]
    for name in ("scripts/train_on_dataset.py", "engine/generator.py",
                 "metrics/research_evaluation.py", "harness/candidate_policy.py"):
        assert name in hashes
        assert hashes[name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_tooling_files_are_not_treated_as_runtime(receipt):
    """Editing a preflight must not read as a runtime change."""
    assert "scripts/preflight_locked_validation.py" not in pf.RUNTIME_COMPONENTS
    assert "scripts/preflight_sft_run.py" not in pf.RUNTIME_COMPONENTS


# ------------------------------------------------------------ decision rule

def test_the_decision_rule_is_frozen_before_results(receipt):
    built, _ = receipt
    assert built["decision_rule"]["frozen_before_results"] is True
    assert built["decision_rule"]["all_criteria_must_hold"] is True


def test_the_rule_requires_more_than_a_kill_gain(receipt):
    built, _ = receipt
    c = built["decision_rule"]["criteria"]
    assert set(c) == {
        "1_practical_kill_gain", "2_reference_validity", "3_execution_and_parse",
        "4_diversity", "5_integrity",
    }


def test_the_kill_bar_is_stricter_than_the_development_screen(receipt):
    """Promotion is not screening; ablation_dev already flattered this checkpoint."""
    built, _ = receipt
    k = built["decision_rule"]["criteria"]["1_practical_kill_gain"]
    assert k["kill_at_8_absolute_points"] == ">= +3.0"
    assert k["net_functions_gained"] == ">= +15"
    assert k["paired_mcnemar_two_sided_p"] == "< 0.01"


def test_the_reference_validity_tolerance_is_declared_and_explained(receipt):
    """The known ~11-point SFT regression is tolerated explicitly, not silently."""
    built, _ = receipt
    r = built["decision_rule"]["criteria"]["2_reference_validity"]
    assert r["max_absolute_drop_points"] == 12.0
    assert "-11.05" in r["why"]
    assert "must be reported every time" in r["why"]


def test_failure_retains_the_base_model(receipt):
    built, _ = receipt
    assert "Retain the immutable base model" in built["decision_rule"]["if_any_criterion_fails"]


def test_post_hoc_changes_are_named_and_forbidden(receipt):
    built, _ = receipt
    forbidden = built["decision_rule"]["forbidden_after_seeing_results"]
    for item in ("retraining of any kind", "prompt changes", "threshold changes",
                 "selecting a different checkpoint", "opening the sealed final test"):
        assert item in forbidden


# ------------------------------------------------ inherited development state

def test_the_preflight_inherits_the_recorded_development_decision(receipt):
    built, _ = receipt
    inherited = built["inherited_development_decision"]
    assert inherited["selected_candidate"] == "Arm A checkpoint 431"
    assert inherited["arm_b_rejected"] == "REJECTED"
    assert len(inherited["receipt_sha256"]) == 64


def test_a_selection_receipt_that_does_not_select_arm_a_is_refused(monkeypatch, tmp_path):
    sel = json.loads((ROOT / pf.DEVELOPMENT_SELECTION_RECEIPT).read_text(encoding="utf-8"))
    sel["decision"]["selected_development_sft_candidate"] = "Arm B checkpoint 150"
    target = tmp_path / "selection.json"
    target.write_text(json.dumps(sel), encoding="utf-8")
    monkeypatch.setattr(pf, "DEVELOPMENT_SELECTION_RECEIPT", str(target))
    problems: list[str] = []
    pf.collect(problems)
    assert any("does not select Arm A checkpoint 431" in p for p in problems)


# ------------------------------------------------------------- the artifact

def test_the_written_receipt_round_trips(tmp_path):
    out = tmp_path / "preflight.json"
    problems: list[str] = []
    built = pf.collect(problems)
    out.write_bytes((json.dumps(built, indent=2) + "\n").encode("utf-8"))
    again = json.loads(out.read_text(encoding="utf-8"))
    assert again["arms"]["B_arm_a_checkpoint_431"]["adapter_sha256_expected"] == pf.ARM_A_431_ADAPTER_SHA256
    assert again["decision_rule"]["frozen_before_results"] is True


def test_the_published_preflight_receipt_is_clean_if_present():
    path = ROOT / "results" / "v4_2_locked_validation_preflight.json"
    if not path.exists():
        pytest.skip("preflight receipt not yet generated")
    built = json.loads(path.read_text(encoding="utf-8"))
    assert built["preflight_problems"] == []
    assert built["ready_to_launch"] is True
    assert built["launched"] is False
    assert built["sealed_final_test"]["accessed"] is False
    assert built["evaluation_split_identity"]["records_read_by_this_preflight"] == 0
