"""A receipt must say what it used, not point at another receipt that does.

An audit found the identity was present but scattered: Arm A carried the model
name and revision under ``tokenization`` and its source tree under
``run_identity``; Arm B carried neither its own source tree nor its own
tokenizer identity and described the model only by linking to Arm A. Nothing
was wrong with the values. The problem was that checking them required knowing
where to look and trusting a link, and a link is what an audit cannot verify.

Scattering is also how the revision defect survived three inline resolvers for
weeks: no single place stated the answer, so no single place could be wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.model_identity import (
    MOVING_REFS, SCHEMA, build, describe_revision, disagreements,
    is_immutable_revision, problems,
)

ROOT = Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SHA = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
TREE = "43250787467580e8a6c7527ec2ddf9412a9217754e02a17263a063ce98d71fc0"
PROTOCOL = "oneiros_successor_generation_protocol_v1"


def _identity(**over):
    base = dict(model_name=QWEN, model_revision=SHA, tokenizer_name=QWEN,
                tokenizer_revision=SHA, source_tree_sha256=TREE,
                protocol_name=PROTOCOL, protocol_sha256="a" * 64)
    base.update(over)
    return build(**base)


# ------------------------------------------------ every required field exists

@pytest.mark.parametrize("field", [
    "base_model_name", "base_model_revision", "selection_tokenizer_name",
    "selection_tokenizer_revision", "source_tree_sha256",
    "successor_protocol_name", "successor_protocol_sha256",
    "expected_training_model_identity",
])
def test_the_block_records_every_required_field(field):
    assert field in _identity()


def test_the_recorded_values_are_the_ones_asked_for():
    identity = _identity()
    assert identity["base_model_name"] == QWEN
    assert identity["base_model_revision"] == SHA
    assert identity["selection_tokenizer_name"] == QWEN
    assert identity["selection_tokenizer_revision"] == SHA
    assert identity["source_tree_sha256"] == TREE
    assert identity["successor_protocol_name"] == PROTOCOL


def test_the_expected_training_identity_is_stated_not_implied():
    """A later run has to be checkable against something explicit."""
    expected = _identity()["expected_training_model_identity"]
    assert expected == {"model_name": QWEN, "model_revision": SHA,
                        "tokenizer_name": QWEN, "tokenizer_revision": SHA}


def test_a_clean_identity_raises_nothing():
    assert problems(_identity(), label="arm A") == []


# --------------------------------------------------------- refusal matrix

def test_an_absent_identity_is_refused():
    assert any("no model identity block" in p
               for p in problems(None, label="arm B"))


@pytest.mark.parametrize("moving", sorted(MOVING_REFS - {""}))
def test_a_branch_or_tag_revision_blocks_readiness(moving):
    found = problems(_identity(model_revision=moving), label="arm A")
    assert any("base_model_revision" in p for p in found), moving


@pytest.mark.parametrize("short", ["2e1fd39", "2e1fd397ee46", "2e1f"])
def test_a_shortened_sha_blocks_readiness(short):
    found = problems(_identity(model_revision=short), label="arm A")
    assert any("shortened SHA" in p for p in found)


def test_an_empty_revision_blocks_readiness():
    found = problems(_identity(model_revision=""), label="arm A")
    assert any("empty" in p for p in found)


def test_an_uppercase_sha_is_not_accepted():
    found = problems(_identity(model_revision=SHA.upper()), label="arm A")
    assert found


def test_a_branch_tokenizer_revision_blocks_readiness():
    found = problems(_identity(tokenizer_revision="main"), label="arm A")
    assert any("selection_tokenizer_revision" in p for p in found)


def test_a_missing_source_tree_blocks_readiness():
    found = problems(_identity(source_tree_sha256=""), label="arm B")
    assert any("source_tree_sha256" in p for p in found)


def test_a_missing_protocol_sha_blocks_readiness():
    found = problems(_identity(protocol_sha256=""), label="arm B")
    assert any("successor_protocol_sha256" in p for p in found)


def test_a_tokenizer_that_is_not_the_models_own_blocks_readiness():
    found = problems(_identity(tokenizer_name="other/model"), label="arm A")
    assert any("tokenizer identities differ" in p for p in found)


def test_a_wrong_schema_blocks_readiness():
    identity = dict(_identity(), schema_version="something_else")
    assert any("schema" in p for p in problems(identity, label="arm A"))


# ------------------------------------------------- the arms must agree

def test_identical_arms_do_not_disagree():
    assert disagreements(_identity(), _identity()) == []


@pytest.mark.parametrize("field,value", [
    ("model_name", "other/model"),
    ("model_revision", "b" * 40),
    ("tokenizer_name", "other/tokenizer"),
    ("tokenizer_revision", "c" * 40),
    ("protocol_name", "other_protocol"),
    ("protocol_sha256", "d" * 64),
])
def test_any_identity_difference_between_arms_is_refused(field, value):
    found = disagreements(_identity(), _identity(**{field: value}))
    assert found, f"{field} difference was not caught"


# --------------------------------------------------------- helpers

def test_is_immutable_revision_accepts_only_a_full_lowercase_sha():
    assert is_immutable_revision(SHA)
    for bad in ("main", "", None, SHA.upper(), "2e1fd39", "v1.0"):
        assert not is_immutable_revision(bad), bad


def test_describe_revision_explains_the_failure():
    assert describe_revision("") == "empty"
    assert "moving reference" in describe_revision("main")
    assert "shortened SHA" in describe_revision("2e1fd39")
    assert describe_revision(SHA) == "immutable"


# ------------------------------------- the preflights actually record it

def test_the_arm_a_preflight_builds_the_block():
    source = (ROOT / "scripts" / "preflight_sft_run.py").read_text(encoding="utf-8")
    assert "build_model_identity(" in source
    assert '"model_identity": build_model_identity(' in source


def test_the_arm_b_preflight_builds_its_own_block_and_checks_arm_a():
    source = (ROOT / "scripts" / "preflight_o1_sidecar_ab.py").read_text(encoding="utf-8")
    assert "arm_b_identity = build_model_identity(" in source
    assert "identity_problems(arm_a_identity" in source
    assert "identity_problems(arm_b_identity" in source
    assert "identity_disagreements(arm_a_identity, arm_b_identity)" in source
    assert "source_tree_sha256(ROOT)" in source
    assert '"arm_a_preflight_sha256"' in source


def test_the_arm_b_preflight_refuses_runtime_drift_not_tooling_edits():
    """A whole-tree hash is the wrong test and was replaced.

    Editing preflight, emitter or test code moves the tree hash without
    changing a byte the trainer executes. Failing on that either blocks
    honest tooling fixes or teaches people to wave the difference through -
    and a real runtime change could then hide inside the same delta. So the
    comparison is made file by file over the components that decide what a
    training run does.
    """
    source = (ROOT / "scripts" / "preflight_o1_sidecar_ab.py").read_text(encoding="utf-8")
    assert "RUNTIME COMPONENTS CHANGED SINCE ARM A" in source
    assert "Arm A must be retrained" in source
    assert "RUNTIME_COMPONENTS" in source
    # The tree difference is still recorded, just not fatal by itself.
    assert '"source_trees_identical"' in source
    assert '"runtime_components_identical"' in source
    # Tooling files must NOT be in the runtime list.
    from scripts.preflight_o1_sidecar_ab import RUNTIME_COMPONENTS
    for tooling in ("scripts/preflight_o1_sidecar_ab.py",
                    "scripts/preflight_sft_run.py",
                    "scripts/emit_o1_sidecar.py"):
        assert tooling not in RUNTIME_COMPONENTS, tooling
    # The files the trainer actually executes must be.
    for runtime in ("scripts/train_on_dataset.py", "engine/generator.py",
                    "engine/sft_trainer.py", "harness/o1_sidecar.py"):
        assert runtime in RUNTIME_COMPONENTS, runtime


def test_arm_a_records_the_runtime_component_hashes_it_was_built_on():
    source = (ROOT / "scripts" / "preflight_sft_run.py").read_text(encoding="utf-8")
    assert '"runtime_component_hashes"' in source


@pytest.mark.parametrize("receipt,label", [
    ("results/v4_2_armA_successor_preflight.json", "arm A"),
    ("results/v4_2_armB_successor_preflight.json", "arm B"),
])
def test_the_published_receipts_carry_a_clean_identity(receipt, label):
    """Against the real receipts on disk, not a fixture."""
    path = ROOT / receipt
    if not path.exists():
        pytest.skip(f"{receipt} not yet regenerated")
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = payload.get("model_identity")
    if identity is None:
        pytest.skip(f"{receipt} predates the identity block")
    assert problems(identity, label=label) == []
    assert identity["base_model_name"] == QWEN
    assert identity["base_model_revision"] == SHA
    assert identity["selection_tokenizer_revision"] == SHA
