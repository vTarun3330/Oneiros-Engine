"""The bug this contract exists to prevent, pinned as a test.

Locked validation failed a frozen integrity criterion because the criterion
demanded both arms share a ``run_contract_sha256``, and a later change put the
adapter hash inside that contract. A base arm and an adapter arm can never
share such a hash. The comparison was fine; the identity model was wrong.

So the load-bearing test here is the one that would have caught it:
a base-versus-adapter comparison must be valid *while* the arms' adapter
hashes differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from harness.comparison_contract import (
    ARM_FIELDS, BASE_MODEL_PROVENANCE, COMPARISON_CONTRACT_VERSION, SHARED_FIELDS,
    arm_sha256, base_model_adapter_identity, build, comparison_problems,
    missing_fields, shared_sha256,
)

ROOT = Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REV = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ADAPTER = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"


def _shared(**over):
    payload = {name: f"value-for-{name}" for name in SHARED_FIELDS}
    payload.update(
        corpus_version="v4_1_research_hardened_candidate",
        evaluation_split="val",
        protocol_name="oneiros_successor_generation_protocol_v1",
        candidate_parse_mode="whole_output",
        retain_raw_output=True,
        candidates_per_function=8,
        temperature=0.7, top_p=0.9, generation_seed=42,
        generation_completion_token_limit=1024,
        prompt_token_limit=1024, max_sequence_tokens=3072,
        feedback_rounds=0, diversity_mode="none",
    )
    payload.update(over)
    return payload


def _base_arm(**over):
    arm = dict(
        arm_name="base_control",
        base_model_name=QWEN, base_model_revision=REV,
        adapter_provenance=BASE_MODEL_PROVENANCE,
        adapter_path=None,
        adapter_sha256=base_model_adapter_identity(QWEN, REV),
        adapter_source_run=None, adapter_checkpoint_step=None,
    )
    arm.update(over)
    return arm


def _adapter_arm(**over):
    arm = dict(
        arm_name="arm_a_checkpoint_431",
        base_model_name=QWEN, base_model_revision=REV,
        adapter_provenance="external_evaluation_adapter",
        adapter_path="checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter",
        adapter_sha256=ADAPTER,
        adapter_source_run="local_sft_armA_baseline_successor_s42",
        adapter_checkpoint_step=431,
    )
    arm.update(over)
    return arm


# ----------------------------------------- THE regression this module exists for

def test_a_base_versus_adapter_comparison_is_valid_despite_differing_adapter_hashes():
    """The exact shape that failed locked validation's criterion 5."""
    built = build(_shared(), [_base_arm(), _adapter_arm()])
    assert built["problems"] == [], built["problems"]
    assert built["valid"] is True
    base, arm = built["arms"]
    assert base["adapter_sha256"] != arm["adapter_sha256"]
    assert base["arm_sha256"] != arm["arm_sha256"]
    assert base["shared_sha256"] == arm["shared_sha256"]


def test_nothing_requires_arm_identities_to_match():
    built = build(_shared(), [_base_arm(), _adapter_arm()])
    assert built["arm_identities_are_expected_to_differ"] is True
    joined = " ".join(built["problems"])
    assert "adapter_sha256" not in joined
    assert "arm_sha256" not in joined


def test_two_arms_with_identical_weights_are_refused():
    """The suspicious case is sameness, not difference."""
    duplicate = _adapter_arm(arm_name="arm_a_again")
    found = comparison_problems(_shared(), [_adapter_arm(), duplicate])
    assert any("same weights measured twice" in p for p in found)


# ------------------------------------------------- shared identity must match

def test_arms_measured_under_different_shared_contracts_are_refused():
    shared = _shared()
    other = shared_sha256(_shared(evaluation_split="ablation_dev"))
    arm = _adapter_arm(shared_sha256=other)
    found = comparison_problems(shared, [_base_arm(shared_sha256=shared_sha256(shared)), arm])
    assert any("different shared contract" in p for p in found)


@pytest.mark.parametrize("field", [
    "evaluation_split", "protocol_sha256", "candidate_parse_mode",
    "candidates_per_function", "temperature", "top_p", "generation_seed",
    "prompt_token_limit", "evaluator_source_canonical_sha256",
])
def test_every_measurement_defining_field_moves_the_shared_hash(field):
    before = shared_sha256(_shared())
    after = shared_sha256(_shared(**{field: "changed"}))
    assert before != after, field


@pytest.mark.parametrize("field", [
    "base_model_revision", "adapter_sha256", "adapter_provenance",
    "adapter_source_run", "adapter_checkpoint_step",
])
def test_every_weight_defining_field_moves_the_arm_hash(field):
    before = arm_sha256(_adapter_arm())
    after = arm_sha256(_adapter_arm(**{field: "changed"}))
    assert before != after, field


def test_the_arm_hash_ignores_the_label():
    """Renaming an arm must not change which weights it says answered."""
    assert arm_sha256(_adapter_arm(arm_name="renamed")) == arm_sha256(_adapter_arm())
    assert "arm_name" not in ARM_FIELDS


def test_the_two_hash_families_are_disjoint():
    """A field in both buckets would recreate the original defect."""
    assert set(SHARED_FIELDS).isdisjoint(set(ARM_FIELDS))
    for weighty in ("adapter_sha256", "base_model_revision", "adapter_provenance"):
        assert weighty not in SHARED_FIELDS, weighty
    for measured in ("evaluation_split", "protocol_sha256", "candidates_per_function"):
        assert measured not in ARM_FIELDS, measured


def test_an_adapter_hash_change_does_not_move_the_shared_hash():
    a = shared_sha256(_shared())
    # changing every arm-level fact must leave the shared identity alone
    _ = arm_sha256(_adapter_arm(adapter_sha256="f" * 64))
    assert shared_sha256(_shared()) == a


# ---------------------------------------------------------- base-arm identity

def test_a_base_arm_uses_the_evaluators_own_identity_digest():
    expected = hashlib.sha256(f"{QWEN}@{REV}".encode()).hexdigest()
    assert base_model_adapter_identity(QWEN, REV) == expected
    assert expected == "320a87cc423b34ed8bb26502dbc503d163076b76edc4078a6d62089aa9d480e0"


def test_a_base_arm_with_a_wrong_identity_digest_is_refused():
    found = comparison_problems(_shared(), [_base_arm(adapter_sha256="0" * 64), _adapter_arm()])
    assert any("not the base-model identity digest" in p for p in found)


def test_an_unknown_provenance_is_refused():
    found = comparison_problems(_shared(), [_base_arm(), _adapter_arm(adapter_provenance="vibes")])
    assert any("unknown adapter_provenance" in p for p in found)


def test_every_arm_can_answer_which_weights_answered():
    built = build(_shared(), [_base_arm(), _adapter_arm()])
    for arm in built["arms"]:
        assert arm["adapter_sha256"], arm["arm_name"]
        assert arm["adapter_provenance"], arm["arm_name"]


# ------------------------------------------------------------- basic refusals

def test_a_single_arm_is_not_a_comparison():
    assert any("at least two arms" in p for p in comparison_problems(_shared(), [_base_arm()]))


def test_duplicate_arm_names_are_refused():
    found = comparison_problems(_shared(), [_base_arm(), _base_arm()])
    assert any("not unique" in p for p in found)


def test_missing_shared_fields_are_named():
    shared = _shared()
    del shared["protocol_sha256"]
    found = missing_fields(shared, [_base_arm(), _adapter_arm()])
    assert "shared.protocol_sha256" in found


def test_missing_arm_fields_are_named():
    found = missing_fields(_shared(), [_adapter_arm(base_model_revision=None)])
    assert any("base_model_revision" in f for f in found)


def test_optional_adapter_fields_are_optional_for_a_base_arm():
    """A base arm has no source run or checkpoint step; that is not a defect."""
    assert missing_fields(_shared(), [_base_arm()]) == []


# ------------------------------------ it must not touch what is already recorded

def test_the_contract_is_separately_versioned():
    assert COMPARISON_CONTRACT_VERSION == "oneiros_comparison_contract_v2"
    built = build(_shared(), [_base_arm(), _adapter_arm()])
    assert built["applies_to"] == "future measurements only"


def test_the_contract_names_what_it_will_not_re_score():
    built = build(_shared(), [_base_arm(), _adapter_arm()])
    for artifact in ("results/v4_2_locked_validation_result_receipt.json",
                     "results/v4_2_development_selection_receipt.json"):
        assert artifact in built["does_not_re_score"]


@pytest.mark.parametrize("relative,expected", [
    ("results/v4_2_frozen_development_evaluation_receipt.json",
     "062774028693cfcbf5346eafeed232027748720940af7a47b5ac1a591055c4fe"),
    ("results/v4_2_locked_validation_preflight.json",
     "97be6855d41b1186a5c8a59a8a89efa71e1f40f5e1c7f29ab387402b7598912c"),
])
def test_historical_receipts_are_byte_for_byte_unchanged(relative, expected):
    """If this fails, someone edited evidence to fit a newer convention."""
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_the_locked_validation_defect_is_still_recorded_as_a_failure():
    """The defect is history. It must not be quietly repaired by v2 existing."""
    path = ROOT / "results/v4_2_locked_validation_result_receipt.json"
    if not path.exists():
        pytest.skip("locked-validation result receipt absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    criterion = receipt["promotion_criteria"]["5_integrity"]
    assert criterion["passed"] is False
    assert criterion["shared_run_contract"] is False
    assert "unsatisfiable" in criterion["why_it_failed"]
    assert receipt["promote_arm_a_431"] is False
    assert receipt["decision"] == "RETAIN the immutable base model"


def test_the_new_contract_does_not_import_or_rewrite_locked_artifacts():
    source = (ROOT / "harness" / "comparison_contract.py").read_text(encoding="utf-8")
    for forbidden in ("write_text", "write_bytes", "open(", "json.load("):
        assert forbidden not in source, forbidden
