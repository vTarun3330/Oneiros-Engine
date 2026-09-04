"""Tests for the one-time authorized sealed-final path.

Mocks only. No test here reads, names, or counts a sealed-test record: the
point of the guard is that the split stays closed until everything is frozen,
and a test suite that opened it would defeat that.
"""
from __future__ import annotations

import json

import pytest

from harness.sealed_final import (
    REQUIRED_BUNDLE_FIELDS,
    SEALED_SPLIT,
    Authorization,
    FinalBundle,
    SealedAccessError,
    SealedFinalGuard,
    issue_authorization,
    refuse_sealed_split_for_development,
)


def _fields(**overrides):
    payload = {
        "adapter_path": "checkpoints/final",
        "adapter_source_tree_sha256": "a" * 64,
        "base_model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "base_model_revision": "2e1fd397",
        "corpus_version": "v4_1_research_hardened_candidate",
        "corpus_records_sha256": "b" * 64,
        "split_ids_sha256": "c" * 64,
        "prompt_schema_version": "oneiros_unified_test_generation_v2",
        "prompt_budgets": {"function": 1024, "repository": 1024},
        "candidates_per_target": 8,
        "seeds": [42, 43, 44],
        "sampling": {"temperature": 0.7, "top_p": 0.9},
        "timeout_seconds": 5,
        "evaluator_version": "research_metrics_v3",
        "baseline_versions": {"atheris": "2.3.0"},
        "checkpoint_selection_rule": "highest mean ablation_dev kill across seeds",
    }
    payload.update(overrides)
    return payload


def _guard(tmp_path):
    return SealedFinalGuard(tmp_path / "state.json", tmp_path / "audit.log")


# --------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------

def test_a_complete_bundle_is_frozen():
    bundle = FinalBundle(_fields())
    assert bundle.missing_fields() == []
    assert bundle.to_dict()["frozen"] is True


@pytest.mark.parametrize("field", REQUIRED_BUNDLE_FIELDS)
def test_every_required_field_must_be_present(field):
    bundle = FinalBundle(_fields(**{field: None}))
    assert field in bundle.missing_fields()


def test_bundle_hash_changes_when_any_frozen_field_changes():
    first = FinalBundle(_fields()).sha256()
    assert FinalBundle(_fields(candidates_per_target=4)).sha256() != first
    assert FinalBundle(_fields(seeds=[42])).sha256() != first
    assert FinalBundle(_fields(timeout_seconds=10)).sha256() != first


def test_bundle_hash_ignores_creation_time():
    """Two identical configurations must hash the same, whenever made."""
    left = FinalBundle(_fields(), created_utc="2026-01-01T00:00:00+00:00")
    right = FinalBundle(_fields(), created_utc="2026-09-04T12:00:00+00:00")
    assert left.sha256() == right.sha256()


# --------------------------------------------------------------------------
# authorization
# --------------------------------------------------------------------------

def test_an_unfrozen_bundle_cannot_be_authorized():
    bundle = FinalBundle(_fields(seeds=None))
    with pytest.raises(SealedAccessError, match="not frozen"):
        issue_authorization(bundle, "researcher", "final measurement")


def test_authorization_requires_an_issuer_and_a_reason():
    bundle = FinalBundle(_fields())
    with pytest.raises(SealedAccessError):
        issue_authorization(bundle, "  ", "final measurement")
    with pytest.raises(SealedAccessError):
        issue_authorization(bundle, "researcher", "")


def test_authorization_binds_to_one_bundle_hash():
    bundle = FinalBundle(_fields())
    authorization = issue_authorization(bundle, "researcher", "final")
    assert authorization.bundle_sha256 == bundle.sha256()


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def test_development_splits_pass_through_without_a_token(tmp_path):
    guard = _guard(tmp_path)
    bundle = FinalBundle(_fields())
    for split in ("train", "ablation_dev", "val"):
        assert guard.open_sealed_split(split, bundle, None, "trainer") is None


def test_sealed_access_without_a_token_is_refused_and_audited(tmp_path):
    guard = _guard(tmp_path)
    bundle = FinalBundle(_fields())
    with pytest.raises(SealedAccessError, match="no authorization token"):
        guard.open_sealed_split(SEALED_SPLIT, bundle, None, "trainer")
    events = [entry["event"] for entry in guard.audit_entries()]
    assert "sealed_access_refused" in events


def test_an_unknown_token_is_refused(tmp_path):
    guard = _guard(tmp_path)
    bundle = FinalBundle(_fields())
    with pytest.raises(SealedAccessError, match="unknown authorization token"):
        guard.open_sealed_split(SEALED_SPLIT, bundle, "not-a-real-token", "evaluator")


def test_a_valid_token_opens_the_split_exactly_once(tmp_path):
    guard = _guard(tmp_path)
    bundle = FinalBundle(_fields())
    authorization = issue_authorization(bundle, "researcher", "final")
    guard.register(authorization)

    guard.open_sealed_split(SEALED_SPLIT, bundle, authorization.token, "evaluator")

    with pytest.raises(SealedAccessError, match="already been spent"):
        guard.open_sealed_split(SEALED_SPLIT, bundle, authorization.token, "evaluator")


def test_changing_the_bundle_after_authorization_invalidates_the_token(tmp_path):
    """Anything tuned after authorization would contaminate the measurement."""
    guard = _guard(tmp_path)
    bundle = FinalBundle(_fields())
    authorization = issue_authorization(bundle, "researcher", "final")
    guard.register(authorization)

    tuned = FinalBundle(_fields(candidates_per_target=16))
    with pytest.raises(SealedAccessError, match="bundle changed"):
        guard.open_sealed_split(SEALED_SPLIT, tuned, authorization.token, "evaluator")


def test_a_token_cannot_be_used_with_an_unfrozen_bundle(tmp_path):
    guard = _guard(tmp_path)
    bundle = FinalBundle(_fields())
    authorization = issue_authorization(bundle, "researcher", "final")
    guard.register(authorization)
    with pytest.raises(SealedAccessError, match="fully frozen"):
        guard.open_sealed_split(
            SEALED_SPLIT, FinalBundle(_fields(evaluator_version=None)),
            authorization.token, "evaluator",
        )


def test_granted_access_is_audited_with_the_bundle_hash(tmp_path):
    guard = _guard(tmp_path)
    bundle = FinalBundle(_fields())
    authorization = issue_authorization(bundle, "researcher", "final")
    guard.register(authorization)
    guard.open_sealed_split(SEALED_SPLIT, bundle, authorization.token, "evaluator")

    granted = [
        entry for entry in guard.audit_entries()
        if entry["event"] == "sealed_access_granted"
    ]
    assert len(granted) == 1
    assert granted[0]["bundle_sha256"] == bundle.sha256()
    assert granted[0]["caller"] == "evaluator"


def test_audit_log_is_append_only_across_guard_instances(tmp_path):
    bundle = FinalBundle(_fields())
    first = _guard(tmp_path)
    with pytest.raises(SealedAccessError):
        first.open_sealed_split(SEALED_SPLIT, bundle, None, "one")
    second = _guard(tmp_path)
    with pytest.raises(SealedAccessError):
        second.open_sealed_split(SEALED_SPLIT, bundle, None, "two")
    assert len(second.audit_entries()) == 2


# --------------------------------------------------------------------------
# development refusal
# --------------------------------------------------------------------------

def test_development_commands_refuse_the_sealed_split_by_name():
    with pytest.raises(SealedAccessError, match="may not read the sealed"):
        refuse_sealed_split_for_development(SEALED_SPLIT, "train_on_dataset.py")


@pytest.mark.parametrize("split", ["train", "ablation_dev", "val"])
def test_development_commands_allow_development_splits(split):
    assert refuse_sealed_split_for_development(split, "train_on_dataset.py") is None
