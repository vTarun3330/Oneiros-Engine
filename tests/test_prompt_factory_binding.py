"""A prompt that changes without any setting changing.

``scripts/train_on_dataset.build_prompt`` read two module globals at call time,
and the sealed loader imported ``build_pair_prompt``. So a sealed run would have
rendered its prompts under whatever the trainer's CLI state happened to hold,
not under the variants its own receipt froze.

It rendered the right prompt only by luck: the trainer defaults are ``full`` and
``self_contained``, which are also the frozen values. Change a CLI default and
the one measurement that cannot be repeated silently changes, with nothing in
the artifact to show it.

The load-bearing test here is the first one: mutate the trainer globals after
building the sealed factory, and the prompt must not move.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.test_generation_prompt import (
    OUTPUT_INSTRUCTION_VARIANTS, PROMPT_INFORMATION_VARIANTS,
    PROMPT_SCHEMA_VERSION,
)
from harness.generation_adapter import successor_settings
from harness.prompt_factory import (
    PROMPT_DEFINING_SOURCES, PROMPT_FACTORY_VERSION, PromptSettings,
    build_record_prompt, frozen_prompt_settings, prompt_binding_problems,
    prompt_factory, prompt_factory_source_hashes,
)
import scripts.train_on_dataset as trainer

ROOT = Path(__file__).resolve().parent.parent
V6 = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v6.json"


def _record():
    return {
        "prompt_code_under_test": "def add_two(a, b):\n    return a - b\n",
        "mutant_code": "def add_two(a, b):\n    return a - b\n",
        "entry_point": "add_two",
        "specification": "Return the sum of the two arguments.",
        "execution_mode": "function_assertion",
        "support_context": "",
        "target_symbols": ["add_two"],
    }


# ------------------------- THE defect: globals must not reach a frozen factory

def test_mutating_trainer_globals_cannot_change_a_sealed_prompt(monkeypatch):
    """Build the factory, then move the globals underneath it."""
    factory = prompt_factory(frozen_prompt_settings())
    before = factory(_record())

    monkeypatch.setattr(trainer, "PROMPT_INFORMATION_VARIANT", "code_only")
    monkeypatch.setattr(trainer, "OUTPUT_INSTRUCTION_VARIANT", "legacy_exactly_one")

    assert factory(_record()) == before, (
        "the sealed prompt followed a trainer global instead of its frozen settings")


def test_the_sealed_loader_does_not_import_the_trainer_prompt_helper():
    source = (ROOT / "harness" / "sealed_final_loader.py").read_text(encoding="utf-8")
    assert "build_pair_prompt" not in source
    assert "prompt_factory(settings.prompt_settings())" in source
    for forbidden in ("PROMPT_INFORMATION_VARIANT", "OUTPUT_INSTRUCTION_VARIANT"):
        assert forbidden not in source, forbidden


def test_the_factory_carries_its_settings_in_a_closure():
    settings = frozen_prompt_settings()
    factory = prompt_factory(settings)
    assert factory.prompt_settings == settings
    # and the held copy is independent of the caller's object
    assert factory.prompt_settings is not settings


def test_the_builder_reads_no_module_global():
    import inspect
    source = inspect.getsource(build_record_prompt)
    for forbidden in ("PROMPT_INFORMATION_VARIANT", "OUTPUT_INSTRUCTION_VARIANT",
                      "train_on_dataset"):
        assert forbidden not in source, forbidden
    assert "settings.information_variant" in source
    assert "settings.output_instruction_variant" in source


# --------------------------- explicit settings change the prompt, as expected

def test_changing_the_information_variant_changes_the_prompt():
    full = prompt_factory(frozen_prompt_settings())(_record())
    code_only = prompt_factory(PromptSettings(
        "code_only", "self_contained", PROMPT_SCHEMA_VERSION))(_record())
    assert full != code_only
    assert "Return the sum of the two arguments." in full
    assert "Return the sum of the two arguments." not in code_only


def test_changing_the_output_instruction_changes_the_prompt():
    a = prompt_factory(frozen_prompt_settings())(_record())
    b = prompt_factory(PromptSettings(
        "full", "legacy_exactly_one", PROMPT_SCHEMA_VERSION))(_record())
    assert a != b


def test_identical_settings_render_identically():
    first = prompt_factory(frozen_prompt_settings())(_record())
    second = prompt_factory(frozen_prompt_settings())(_record())
    assert first == second


# ---------------------------- the sealed prompt IS the canonical factory output

def test_the_sealed_prompt_matches_the_shared_factory_for_a_synthetic_record():
    from harness.sealed_final_smoke import synthetic_record

    settings = successor_settings()
    sealed = prompt_factory(settings.prompt_settings())
    canonical = build_record_prompt(synthetic_record(), frozen_prompt_settings())
    assert sealed(synthetic_record()) == canonical


def test_the_trainer_and_the_factory_agree_under_the_frozen_configuration():
    """Migration must not have changed what development renders."""
    record = _record()
    assert trainer.build_pair_prompt(record) == build_record_prompt(
        record, trainer.current_prompt_settings())


def test_the_trainer_builds_explicit_settings_from_its_globals(monkeypatch):
    monkeypatch.setattr(trainer, "PROMPT_INFORMATION_VARIANT", "code_specification")
    settings = trainer.current_prompt_settings()
    assert settings.information_variant == "code_specification"
    assert settings.problems() == []


# ------------------------------------- validation from the canonical constants

@pytest.mark.parametrize("variant", PROMPT_INFORMATION_VARIANTS)
def test_every_canonical_information_variant_is_accepted(variant):
    assert PromptSettings(variant, "self_contained", PROMPT_SCHEMA_VERSION).problems() == []


@pytest.mark.parametrize("variant", OUTPUT_INSTRUCTION_VARIANTS)
def test_every_canonical_output_variant_is_accepted(variant):
    assert PromptSettings("full", variant, PROMPT_SCHEMA_VERSION).problems() == []


@pytest.mark.parametrize("invented", ["minimal", "bare", "", "FULL", None])
def test_invented_variants_are_rejected(invented):
    assert PromptSettings(invented, "self_contained", PROMPT_SCHEMA_VERSION).problems()
    assert PromptSettings("full", invented, PROMPT_SCHEMA_VERSION).problems()


def test_the_previously_hand_written_sets_were_wrong_in_both_directions():
    """Pins the defect: 'minimal'/'bare' were invented, four real ones refused."""
    for invented in ("minimal", "bare"):
        assert invented not in PROMPT_INFORMATION_VARIANTS
        assert invented not in OUTPUT_INSTRUCTION_VARIANTS
    for real in ("code_only", "code_specification"):
        assert real in PROMPT_INFORMATION_VARIANTS
    for real in ("legacy_exactly_one", "metamorphic_allowed"):
        assert real in OUTPUT_INSTRUCTION_VARIANTS


def test_a_wrong_schema_version_is_rejected():
    assert PromptSettings("full", "self_contained", "something_else").problems()


def test_generation_settings_validate_prompt_variants_canonically():
    from harness.generation_adapter import GenerationSettings
    base = successor_settings().to_dict()
    assert GenerationSettings(**base).problems() == []
    assert GenerationSettings(**dict(base, prompt_information_variant="minimal")).problems()
    assert GenerationSettings(**dict(base, prompt_information_variant="code_only")).problems() == []
    assert GenerationSettings(**dict(base, output_instruction_variant="bare")).problems()
    assert GenerationSettings(
        **dict(base, output_instruction_variant="metamorphic_allowed")).problems() == []


def test_the_frozen_configuration_remains_valid():
    settings = frozen_prompt_settings()
    assert settings.information_variant == "full"
    assert settings.output_instruction_variant == "self_contained"
    assert settings.prompt_schema_version == PROMPT_SCHEMA_VERSION
    assert settings.problems() == []


def test_a_builder_with_invalid_settings_refuses():
    with pytest.raises(ValueError, match="invalid prompt settings"):
        prompt_factory(PromptSettings("minimal", "self_contained", PROMPT_SCHEMA_VERSION))
    with pytest.raises(ValueError, match="invalid prompt settings"):
        build_record_prompt(_record(), PromptSettings("full", "bare", PROMPT_SCHEMA_VERSION))


# ------------------------------------------------- prompt sources are bound

def test_every_prompt_defining_source_is_hashed():
    from harness.source_identity import canonical_sha256
    binding = prompt_factory_source_hashes(ROOT)
    assert set(PROMPT_DEFINING_SOURCES) <= set(binding)
    for role, relative in PROMPT_DEFINING_SOURCES.items():
        assert binding[role]["path"] == relative
        assert binding[role]["canonical_sha256"] == canonical_sha256(ROOT / relative)


def test_the_binding_covers_engine_compaction_and_record_adaptation():
    assert PROMPT_DEFINING_SOURCES["prompt_engine"] == "engine/test_generation_prompt.py"
    assert PROMPT_DEFINING_SOURCES["prompt_compaction"] == "engine/prompt_budget.py"
    assert PROMPT_DEFINING_SOURCES["record_adaptation"] == "scripts/train_on_dataset.py"
    assert PROMPT_DEFINING_SOURCES["prompt_factory"] == "harness/prompt_factory.py"


def test_a_clean_binding_reports_no_problems():
    assert prompt_binding_problems(prompt_factory_source_hashes(ROOT), ROOT) == []


@pytest.mark.parametrize("role", sorted(PROMPT_DEFINING_SOURCES))
def test_a_changed_prompt_source_hash_invalidates_the_receipt(role):
    recorded = prompt_factory_source_hashes(ROOT)
    recorded[role] = dict(recorded[role], canonical_sha256="0" * 64)
    problems = prompt_binding_problems(recorded, ROOT)
    assert any(role in p for p in problems), problems


def test_a_changed_factory_version_invalidates_the_receipt():
    recorded = dict(prompt_factory_source_hashes(ROOT), prompt_factory_version="other")
    assert any("prompt_factory_version" in p for p in prompt_binding_problems(recorded, ROOT))


def test_a_changed_variant_set_invalidates_the_receipt():
    recorded = dict(prompt_factory_source_hashes(ROOT),
                    information_variants=["full"])
    assert any("information_variants" in p for p in prompt_binding_problems(recorded, ROOT))


def test_an_absent_binding_is_refused():
    assert prompt_binding_problems({}, ROOT) == [
        "receipt records no prompt-factory source identity"]


# --------------------------------------------------------------- v5 receipt

def _v6() -> dict:
    if not V6.exists():
        pytest.skip("v5 receipt not generated")
    return json.loads(V6.read_text(encoding="utf-8"))


def test_the_v5_receipt_binds_the_prompt_sources():
    source = _v6()["final_evaluator_source"]
    assert source["prompt_factory_version"] == PROMPT_FACTORY_VERSION
    assert source["frozen_prompt_settings"] == frozen_prompt_settings().to_dict()
    assert prompt_binding_problems(source["prompt_binding"], ROOT) == []
    for role in PROMPT_DEFINING_SOURCES:
        assert role in source["prompt_binding"], role


def test_the_v5_receipt_refuses_v4_and_earlier():
    import scripts.run_sealed_final_test as entry
    for version in ("v1", "v2", "v3", "v4"):
        assert f"oneiros_sealed_final_readiness_{version}" in entry.REFUSED_SCHEMA_VERSIONS
    assert entry.REQUIRED_SCHEMA_VERSION == "oneiros_sealed_final_readiness_v6"
    assert _v6()["schema_version"] == "oneiros_sealed_final_readiness_v6"


def test_the_v5_binding_matches_the_runtime():
    import scripts.run_sealed_final_test as entry
    assert entry.evaluator_binding_problems(_v6()) == []
    assert entry.settings_binding_problems(_v6()) == []


def test_scope_b_is_still_explicit_in_v5():
    receipt = _v6()
    scope = receipt["baseline_scope"]
    assert scope["comparative_claims_supported"] is False
    assert scope["option_chosen"].startswith("B")
    text = json.dumps(receipt["frozen_bundle"])
    assert "v4_2_atheris_tasks_val.manifest.json" not in text
    assert "v4_2_baseline_bundle_val.json" not in text


def test_v4_is_preserved_byte_for_byte():
    v4 = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v4.json"
    if not v4.exists():
        pytest.skip("v4 receipt absent")
    assert hashlib.sha256(v4.read_bytes()).hexdigest() == \
        "b5d35660bbc6a4a629bce199ff660966c5e1abf5f9d67c7fb468a16105335aeb"
    marker = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v4.SUPERSEDED.md"
    assert marker.exists()
    text = marker.read_text(encoding="utf-8")
    assert "build_pair_prompt" in text
    assert "PROMPT_INFORMATION_VARIANT" in text


def test_no_authorization_has_ever_been_granted():
    assert not (ROOT / "results" / "sealed_final_state.json").exists()
    assert not (ROOT / "results" / "sealed_final_run_state.json").exists()
    audit = ROOT / "results" / "sealed_final_audit.log"
    if audit.exists():
        events = [json.loads(line) for line in
                  audit.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [e for e in events if e.get("event") == "sealed_access_granted"] == []
