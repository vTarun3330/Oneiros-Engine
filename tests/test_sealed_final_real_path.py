"""The defects that mocks could never have caught.

An audit found three in the sealed loader at once: it imported
``build_test_generation_prompt`` (does not exist), called
``Phi3Generator.generate_candidates`` (does not exist), and never set
``parse_mode``, which defaults to ``"first_assertion"`` - so the final
measurement would have been scored by the legacy parser while claiming the
frozen successor protocol.

The first two would have raised *after* the token was spent, because the
imports sat inside a function body and the "loader is importable" check
therefore proved nothing. The third would not have raised at all.

Every test here exists to make one of those impossible to reintroduce. They
resolve real symbols instead of trusting names, they rehearse split parsing on a
permitted split, and they refuse any wiring that is mock-only.

Nothing here opens, enumerates, hashes or renders the sealed split.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tests.sealed_history import (
    assert_no_new_authorization, assert_old_output_is_empty,
    assert_re_execution_is_blocked, guard_fingerprint,
)

from harness.generation_adapter import (
    ADAPTER_VERSION, GenerationSettings, adapter_source_hashes, empty_slots,
    generate_candidate_slots, successor_settings,
)
from harness.sealed_final import SealedAccessError
from harness.sealed_final_loader import (
    REQUIRED_RECORD_FIELDS, SplitSchemaError, build_sealed_generator,
    select_split_records,
)
from harness.sealed_final_smoke import (
    SYNTHETIC_GOLDEN, SYNTHETIC_MUTANT, run_model_smoke, smoke_problems,
    synthetic_record,
)

ROOT = Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REV = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
VIEW = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"


# ---------------------------------------- 1. the nonexistent-symbol defects

def test_every_symbol_the_loader_imports_actually_resolves():
    """Resolve the real objects rather than trusting that a name exists."""
    from engine.generator import Phi3Generator
    from engine.prompt_budget import compact_unified_user_prompt
    from engine.test_generation_prompt import build_unified_user_prompt, format_chat_prompt
    from scripts.train_on_dataset import build_pair_prompt
    for obj in (Phi3Generator, compact_unified_user_prompt,
                build_unified_user_prompt, format_chat_prompt, build_pair_prompt):
        assert callable(obj)


SEALED_MODULES = ("harness/sealed_final_loader.py", "harness/generation_adapter.py",
                  "harness/sealed_final_smoke.py", "harness/sealed_final_evaluator.py")


def _executable_source(relative: str) -> str:
    """Module source with docstrings and comments removed.

    These modules explain the defects they fix, so the dead symbol names appear
    in their prose. Scanning raw text flagged the explanation as the bug.
    """
    import ast
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_the_nonexistent_prompt_function_is_gone():
    """`build_test_generation_prompt` never existed. Nothing may call it."""
    import engine.test_generation_prompt as prompts
    assert not hasattr(prompts, "build_test_generation_prompt")
    for module in SEALED_MODULES:
        assert "build_test_generation_prompt" not in _executable_source(module), module


def test_the_nonexistent_generator_method_is_gone():
    """`Phi3Generator.generate_candidates` never existed."""
    from engine.generator import Phi3Generator
    assert not hasattr(Phi3Generator, "generate_candidates")
    for module in SEALED_MODULES:
        assert "generate_candidates" not in _executable_source(module), module


def test_the_generation_call_the_adapter_makes_exists():
    """The adapter calls model.generate and generator._parse_output; both real."""
    from engine.generator import Phi3Generator
    assert callable(Phi3Generator._parse_output)
    source = inspect.getsource(generate_candidate_slots)
    assert "generator.model.generate(" in source
    assert "generator._parse_output(" in source


# ------------------------------------------- 2. the silent parse-mode defect

def test_the_generator_default_parse_mode_is_still_the_legacy_one():
    """Pins why setting it explicitly matters: the default is NOT whole_output."""
    from engine.generator import Phi3Generator
    assert Phi3Generator.parse_mode == "first_assertion"


def test_the_adapter_sets_parse_mode_explicitly_from_settings():
    source = inspect.getsource(generate_candidate_slots)
    assert "generator.parse_mode = settings.candidate_parse_mode" in source


def test_the_frozen_successor_settings_are_whole_output():
    settings = successor_settings()
    assert settings.candidate_parse_mode == "whole_output"
    assert settings.retain_raw_output is True
    assert settings.candidates_per_function == 8
    assert settings.temperature == 0.7
    assert settings.top_p == 0.9
    assert settings.prompt_token_limit == 1024
    assert settings.generation_completion_token_limit == 1024
    assert settings.max_sequence_tokens == 3072
    assert settings.seed == 42
    assert settings.problems() == []


def test_a_loader_receipt_that_is_not_whole_output_is_refused():
    with pytest.raises(SealedAccessError, match="whole_output"):
        build_sealed_generator(_receipt_stub(candidate_parse_mode="first_assertion"))


def test_a_loader_receipt_with_the_wrong_batch_size_is_refused():
    with pytest.raises(SealedAccessError, match="batch size"):
        build_sealed_generator(_receipt_stub(generation_batch_size=1))


def test_settings_that_do_not_fit_the_sequence_budget_are_refused():
    bad = GenerationSettings(**dict(successor_settings().to_dict(),
                                    prompt_token_limit=2048))
    assert any("does not fit" in p for p in bad.problems())


def _receipt_stub(**overrides):
    """A receipt carrying the frozen settings whole.

    build_sealed_generator now reads
    final_evaluator_source.frozen_generation_settings rather than
    reassembling settings field by field from the bundle - reassembly meant
    every new semantic had to be remembered twice, and the forgotten one was
    the parse mode.
    """
    settings = dict(successor_settings().to_dict(), **overrides)
    return {
        "final_candidate": {"model": QWEN, "model_revision": REV, "adapter": None},
        "final_evaluator_source": {"frozen_generation_settings": settings},
        "frozen_bundle": {"fields": {
            "candidates_per_target": settings["candidates_per_function"]}},
    }


def test_an_adapter_bearing_receipt_is_refused_for_the_base_candidate():
    receipt = _receipt_stub()
    receipt["final_candidate"]["adapter"] = "some/adapter"
    with pytest.raises(SealedAccessError, match="no adapter"):
        build_sealed_generator(receipt)


# --------------------------------- 3. one shared path, not two that diverge

def test_the_trainer_delegates_to_the_shared_adapter():
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    assert "from harness.generation_adapter import" in source
    assert "generate_candidate_slots(" in source


def test_the_sealed_path_uses_the_same_adapter():
    source = (ROOT / "harness" / "sealed_final_loader.py").read_text(encoding="utf-8")
    assert "from harness.generation_adapter import generate_candidate_slots" in source


def test_there_is_only_one_generation_body():
    """Both paths must reach the same function object."""
    import scripts.train_on_dataset as trainer
    import harness.generation_adapter as adapter
    assert "generator.model.generate(" not in inspect.getsource(
        trainer.generate_tests_ai_batched)
    assert "generator.model.generate(" in inspect.getsource(adapter.generate_candidate_slots)


def test_the_smoke_uses_the_real_prompt_builder():
    source = inspect.getsource(run_model_smoke)
    assert "from scripts.train_on_dataset import build_pair_prompt" in source
    assert "generate_candidate_slots(" in source


# ------------------------- 4. non-sealed loader / schema rehearsal

def _permitted_records(limit: int = 12):
    shard = VIEW / "ablation_dev.records.json"
    if not shard.is_file():
        pytest.skip("permitted development shard is unavailable")
    return json.loads(shard.read_text(encoding="utf-8"))[:limit]


def test_split_parsing_rehearses_on_a_permitted_split():
    """Exercises the real record schema without touching the sealed split."""
    records = _permitted_records()
    ids = [str(r["id"]) for r in records]
    selected = select_split_records({"ablation_dev": ids}, records, "ablation_dev")
    assert [str(r["id"]) for r in selected] == ids
    for record in selected:
        for field in REQUIRED_RECORD_FIELDS:
            assert record.get(field), field


def test_split_order_is_preserved_not_set_ordered():
    """The scope hash is taken over the id sequence, so order is load-bearing."""
    records = _permitted_records()
    ids = [str(r["id"]) for r in records]
    reversed_ids = list(reversed(ids))
    selected = select_split_records({"s": reversed_ids}, records, "s")
    assert [str(r["id"]) for r in selected] == reversed_ids


def test_a_missing_record_is_a_loud_failure():
    records = _permitted_records()
    ids = [str(r["id"]) for r in records] + ["definitely-not-a-real-id"]
    with pytest.raises(SplitSchemaError, match="have no record"):
        select_split_records({"s": ids}, records, "s")


def test_a_record_missing_required_fields_is_refused():
    records = [dict(r) for r in _permitted_records(3)]
    records[1]["reference_code"] = ""
    ids = [str(r["id"]) for r in records]
    with pytest.raises(SplitSchemaError, match="missing required fields"):
        select_split_records({"s": ids}, records, "s")


def test_an_absent_split_is_refused():
    with pytest.raises(SplitSchemaError, match="is absent"):
        select_split_records({"train": ["a"]}, [{"id": "a"}], "nope")


def test_an_empty_split_is_refused():
    with pytest.raises(SplitSchemaError, match="empty"):
        select_split_records({"s": []}, [], "s")


def test_duplicate_ids_are_refused():
    with pytest.raises(SplitSchemaError, match="duplicate"):
        select_split_records({"s": ["a", "a"]}, [{"id": "a"}], "s")


def test_the_rehearsal_never_touches_the_sealed_split():
    """Permitted shards are fine; the canonical corpus files are not.

    An earlier version banned the substring "records.json", which also matched
    the permitted ablation_dev shard this rehearsal is supposed to use. The
    distinction that matters is sealed-vs-permitted, not the file extension.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    source = source.replace(
        inspect.getsource(test_the_rehearsal_never_touches_the_sealed_split), "")
    # The canonical files hold every split, including the sealed one.
    assert "splits.json" not in source
    assert "/records.json" not in source
    assert 'corpus_dir / "records.json"' not in source
    # And no test may select the sealed split by name.
    assert 'select_split_records(' in source
    assert ', "test")' not in source
    assert "SEALED_SPLIT" not in source
    # The shard actually used is the permitted development one.
    assert "ablation_dev.records.json" in source


def test_the_loader_refuses_because_the_split_is_consumed():
    """Authorization is no longer the gate, and no longer needs to be.

    ``authorization_granted`` is permanently true - one grant was recorded and
    spent - so it stopped being a refusal. The refusal is now unconditional and
    reached before any corpus file is opened.
    """
    from harness import sealed_final_loader

    assert sealed_final_loader.authorization_granted() is True
    with pytest.raises(SealedAccessError, match="consumed"):
        sealed_final_loader.sealed_records()


# --------------------------------------- 5. the synthetic model smoke

def test_the_smoke_record_is_synthetic_and_self_contained():
    record = synthetic_record()
    assert record["entry_point"] == "add_two"
    assert record["golden_code"] == SYNTHETIC_GOLDEN
    assert record["mutant_code"] == SYNTHETIC_MUTANT
    assert record["source_name"] == "synthetic"


def test_smoke_problems_catch_each_regression():
    settings = successor_settings()
    good = {
        "records_in_batch": 2, "prompts_generated_together": 2,
        "candidate_slots_total": 16,
        "per_record": [
            {"record_id": "a", "candidate_slots": 8, "scored_candidates": 8},
            {"record_id": "b", "candidate_slots": 8, "scored_candidates": 8},
        ],
        "observed_parse_mode": "whole_output",
        "raw_output_hashes_verified": True, "raw_outputs_retained": 16,
        "scored_candidates": 16, "seed_applied": True, "seed": 42,
        "sealed_loader_imported_by_smoke": False,
        "sealed_authorization_granted": False,
        "model_revision": REV, "records_are_synthetic": True,
        "generator_identity": {"is_loaded": True},
    }
    assert smoke_problems(good, settings) == []

    assert any("not the frozen whole_output" in p for p in smoke_problems(
        dict(good, observed_parse_mode="first_assertion"), settings))
    assert any("expected 16" in p for p in smoke_problems(
        dict(good, candidate_slots_total=8), settings))
    assert any("raw-output hashes" in p for p in smoke_problems(
        dict(good, raw_output_hashes_verified=False), settings))
    assert any("imported the sealed loader" in p for p in smoke_problems(
        dict(good, sealed_loader_imported_by_smoke=True), settings))
    assert any("non-immutable model revision" in p for p in smoke_problems(
        dict(good, model_revision="main"), settings))
    assert any("synthetic records" in p for p in smoke_problems(
        dict(good, records_are_synthetic=False), settings))


def test_the_smoke_refuses_a_non_successor_parse_mode():
    settings = GenerationSettings(**dict(successor_settings().to_dict(),
                                         candidate_parse_mode="first_assertion"))
    with pytest.raises(RuntimeError, match="successor protocol"):
        run_model_smoke(settings=settings)


def test_the_smoke_refuses_a_one_record_batch_shape():
    """Locked validation padded two; a one-record smoke proves the wrong thing."""
    settings = GenerationSettings(**dict(successor_settings().to_dict(),
                                         generation_batch_size=1))
    with pytest.raises(RuntimeError, match="frozen batch shape"):
        run_model_smoke(settings=settings)


def test_the_smoke_never_imports_the_sealed_loader():
    """It may *observe* whether the loader was imported; it may not import it."""
    module = _executable_source("harness/sealed_final_smoke.py")
    assert "import harness.sealed_final_loader" not in module
    assert "from harness.sealed_final_loader" not in module
    assert "sealed_records(" not in module


# ------------------------------------ 6. the entrypoint runs the smoke first

def test_the_smoke_runs_before_the_guard_is_called():
    source = (ROOT / "scripts" / "run_sealed_final_test.py").read_text(encoding="utf-8")
    assert source.index("run_model_smoke(") < source.index("guard.open_sealed_split(")
    assert source.index("smoke_problems(") < source.index("guard.open_sealed_split(")


def test_check_only_includes_the_smoke():
    source = (ROOT / "scripts" / "run_sealed_final_test.py").read_text(encoding="utf-8")
    assert source.index("run_model_smoke(") < source.index("if args.check_only:")


def test_superseded_receipts_are_refused():
    import scripts.run_sealed_final_test as entry
    for version in ("v1", "v2", "v3", "v4"):
        assert f"oneiros_sealed_final_readiness_{version}" in entry.REFUSED_SCHEMA_VERSIONS
    assert entry.REQUIRED_SCHEMA_VERSION == "oneiros_sealed_final_readiness_v6"


def test_the_adapter_identity_is_recorded():
    identity = adapter_source_hashes()
    assert identity["adapter_version"] == ADAPTER_VERSION
    assert len(identity["canonical_sha256"]) == 64


def test_empty_slots_preserve_the_denominator():
    slots = empty_slots(8, rank_offset=0)
    assert len(slots) == 8
    assert [s["rank"] for s in slots] == list(range(1, 9))
    assert all(s["parse_valid"] is False and s["code"] is None for s in slots)
