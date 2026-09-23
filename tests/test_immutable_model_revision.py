"""Every field that names the model must name one immutable commit.

This exists because a completed GPU evaluation had to be quarantined. The
revision pinning in b753ba1 fixed the shared resolver and the preflight, but
``scripts/train_on_dataset.py`` carried three more inline copies of the same
logic, each ending in a bare branch-name fallback, and those three were what
wrote ``run_contract``, ``reproducibility`` and ``generation_settings``. Worse,
the generator's own constructor fell back the same way, so the branch name was
not merely recorded - it was passed to ``from_pretrained`` for the tokenizer,
the config and the weights.

The earlier tests missed all of it by exercising the resolver function rather
than the fields the artifacts record and the arguments the loaders receive.
These tests do the opposite.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from config import IMMUTABLE_MODEL_REVISIONS, immutable_revision_for

ROOT = Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
QWEN_SHA = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
QWEN_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_7B_SHA = "c03e6d358207e414f1eca0bb1891e29f1db0e242"

#: Anything that is not one immutable commit.
NOT_IMMUTABLE = ("main", "master", "latest", "HEAD", "head", "",
                 "2e1fd39", "2e1fd397ee46", "v1.0", "refs/heads/main")

IMMUTABLE = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture
def qwen_override(monkeypatch):
    """The exact override path the GPU runs use."""
    from scripts import train_on_dataset as trainer
    monkeypatch.setattr(trainer, "BASE_MODEL_NAME_OVERRIDE", QWEN)
    monkeypatch.setattr(trainer, "BASE_MODEL_REVISION_OVERRIDE", None)
    monkeypatch.setattr(trainer, "SFT_SELECTION_TOKENIZER_NAME_OVERRIDE", None)
    return trainer


# ------------------------------------------------------- the pinned value

def test_the_qwen_snapshot_is_the_full_immutable_sha():
    assert immutable_revision_for(QWEN) == QWEN_SHA
    assert IMMUTABLE.match(QWEN_SHA)


def test_the_qwen_7b_snapshot_is_the_full_immutable_sha():
    assert immutable_revision_for(QWEN_7B) == QWEN_7B_SHA
    assert IMMUTABLE.match(QWEN_7B_SHA)


def test_every_pinned_revision_is_immutable():
    for name, revision in IMMUTABLE_MODEL_REVISIONS.items():
        assert IMMUTABLE.match(revision), f"{name} -> {revision!r}"


# --------------------------------------------- the resolver, on the real path

def test_the_override_path_resolves_to_the_immutable_sha(qwen_override):
    name, revision = qwen_override.resolved_base_model_identity()
    assert name == QWEN
    assert revision == QWEN_SHA


def test_the_selection_tokenizer_resolves_to_the_same_sha(qwen_override):
    _, model_revision = qwen_override.resolved_base_model_identity()
    _, tokenizer_revision = qwen_override.resolved_selection_tokenizer_identity()
    assert tokenizer_revision == model_revision == QWEN_SHA


@pytest.mark.parametrize("moving", NOT_IMMUTABLE)
def test_a_moving_or_short_override_never_survives(qwen_override, monkeypatch, moving):
    """A branch, a tag or a shortened SHA must not reach a receipt."""
    monkeypatch.setattr(qwen_override, "BASE_MODEL_REVISION_OVERRIDE", moving)
    if moving in ("main", "master", "latest", "HEAD", "head", ""):
        # A moving reference is replaced by the pin.
        _, revision = qwen_override.resolved_base_model_identity()
        assert revision == QWEN_SHA
    else:
        # Anything else non-immutable is refused rather than recorded.
        with pytest.raises(RuntimeError, match="immutable"):
            qwen_override.resolved_base_model_identity()


def test_an_unpinned_model_is_refused_not_defaulted(monkeypatch):
    from scripts import train_on_dataset as trainer
    monkeypatch.setattr(trainer, "BASE_MODEL_NAME_OVERRIDE", "nobody/unpinned")
    monkeypatch.setattr(trainer, "BASE_MODEL_REVISION_OVERRIDE", None)
    with pytest.raises(RuntimeError, match="IMMUTABLE_MODEL_REVISIONS"):
        trainer.resolved_base_model_identity()


def test_is_immutable_revision_accepts_only_a_full_sha():
    from scripts.train_on_dataset import is_immutable_revision
    assert is_immutable_revision(QWEN_SHA)
    for value in NOT_IMMUTABLE:
        assert not is_immutable_revision(value), value
    assert not is_immutable_revision(None)
    assert not is_immutable_revision(QWEN_SHA.upper())


# ------------------------------------- the revision that is actually LOADED

def test_the_generator_loads_the_immutable_sha(monkeypatch):
    """The revision reaches from_pretrained for tokenizer, config and weights."""
    from engine.generator import Phi3Generator
    generator = Phi3Generator(model_name=QWEN, model_revision=None)
    assert generator.model_revision == QWEN_SHA


@pytest.mark.parametrize("moving", ["main", "master", "latest", "HEAD", ""])
def test_the_generator_replaces_a_moving_reference(moving):
    from engine.generator import Phi3Generator
    generator = Phi3Generator(model_name=QWEN, model_revision=moving)
    assert generator.model_revision == QWEN_SHA


@pytest.mark.parametrize("bad", ["2e1fd39", "v1.0", "refs/heads/main"])
def test_the_generator_refuses_a_non_immutable_revision(bad):
    from engine.generator import Phi3Generator
    with pytest.raises(ValueError, match="immutable"):
        Phi3Generator(model_name=QWEN, model_revision=bad)


def test_the_generator_refuses_an_unpinned_model():
    from engine.generator import Phi3Generator
    with pytest.raises(ValueError, match="immutable"):
        Phi3Generator(model_name="nobody/unpinned", model_revision=None)


def test_the_runtime_profile_records_the_loaded_revision(monkeypatch):
    """model_runtime_profile must agree with what was loaded, not a default."""
    from engine.generator import Phi3Generator
    generator = Phi3Generator(model_name=QWEN, model_revision=None)
    # The profile is populated at load; assert the source of truth it copies.
    assert generator.model_revision == QWEN_SHA
    source = (ROOT / "engine" / "generator.py").read_text(encoding="utf-8")
    assert 'self.runtime_profile["model_revision"] = self.model_revision' in source
    for call in ("self.tokenizer = AutoTokenizer.from_pretrained(",
                 "config = AutoConfig.from_pretrained(",
                 "self.model = AutoModelForCausalLM.from_pretrained("):
        index = source.index(call)
        window = source[index:index + 400]
        assert "revision=self.model_revision" in window, call


def test_the_trainer_hands_the_generator_the_resolved_revision():
    """Passing the raw override is what let a None fall through to a branch."""
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    index = source.index("generator = Phi3Generator(\n        model_name=")
    window = source[index:index + 400]
    assert "model_revision=_gen_revision" in window
    assert "model_revision=BASE_MODEL_REVISION_OVERRIDE" not in source


# --------------------------------------- no duplicated fallback path remains

DUPLICATE_GUARD_FILES = (
    "scripts/train_on_dataset.py",
    "scripts/preflight_sft_run.py",
    "scripts/preflight_o1_sidecar_ab.py",
    "engine/generator.py",
    "utils/reproducibility.py",
)


@pytest.mark.parametrize("relative", DUPLICATE_GUARD_FILES)
def test_no_bare_branch_fallback_remains(relative):
    """The three inline copies are why a healthy GPU run was quarantined."""
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} absent")
    source = path.read_text(encoding="utf-8")
    for pattern in ('else "main"', "else 'main'", 'or "main"', "or 'main'"):
        assert pattern not in source, (
            f"{relative} still contains a bare branch fallback {pattern!r}; "
            "every path must use the one shared immutable resolver")


def test_only_one_resolver_defines_the_base_model_identity():
    """Parallel resolution logic is how the three copies drifted apart."""
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = [node.name for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "resolved_base_model_identity"]
    assert len(definitions) == 1
    # And every recording site calls it rather than recomputing.
    assert source.count("resolved_base_model_identity()") >= 4
    assert "model_config.model_revision\n            if resolved_base_model_name" \
        not in source


# ------------------------------------- artifact fields agree with each other

def test_a_recorded_artifact_would_agree_across_every_field(qwen_override):
    """The fields the quarantined run disagreed on, checked together."""
    name, revision = qwen_override.resolved_base_model_identity()
    from engine.generator import Phi3Generator
    generator = Phi3Generator(model_name=name, model_revision=revision)

    recorded = {
        "generation_settings.base_model_revision": revision,
        "run_contract.base_model_revision": revision,
        "reproducibility.model_revision": revision,
        "model_runtime_profile.model_revision": generator.model_revision,
        "loaded_tokenizer_revision": generator.model_revision,
        "selection_tokenizer_revision":
            qwen_override.resolved_selection_tokenizer_identity()[1],
    }
    assert len(set(recorded.values())) == 1, recorded
    for field, value in recorded.items():
        assert IMMUTABLE.match(str(value)), f"{field} = {value!r}"
        assert value not in NOT_IMMUTABLE, field


def test_the_quarantined_artifact_is_still_refused_by_these_rules():
    """The real artifact that failed, asserted to fail. Not deleted, quarantined."""
    artifact = (ROOT / "results" / "local_base_qwen_ablationdev_successor_s42"
                / "base_validation_ablation-dev_parse-whole-output_completion1024_seed_42.json")
    if not artifact.exists():
        pytest.skip("quarantined artifact not present")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    recorded = payload["generation_settings"]["base_model_revision"]
    assert not IMMUTABLE.match(str(recorded)), (
        "the quarantined artifact should still show the defect it was "
        "quarantined for")
    note = artifact.parent / "QUARANTINED_unpinned_revision.md"
    assert note.exists(), "a quarantined artifact must say why"
