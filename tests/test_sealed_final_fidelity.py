"""Seed, batch, semantics, generator reuse, and baseline scope.

Five defects an audit found in v3, each of which would have degraded the one
measurement that cannot be repeated:

* the frozen seed was recorded and never applied;
* generation ran one target at a time where locked validation padded two;
* several generation semantics were read from mutable module globals;
* the model was loaded a second time *after* the token was spent;
* val baseline artifacts sat inside a sealed-final bundle, implying a
  comparison the run could not support.

None of these would have raised. Four of them would have produced a plausible
number under a different protocol, and the fifth would have produced a true
number presented as something it was not.
"""
from __future__ import annotations

import inspect
import json
import random
from pathlib import Path

import pytest

from harness.generation_adapter import (
    GenerationSettings, chunked, generate_candidate_slots, successor_settings,
)
from harness.generation_rng import (
    SEED_APPLICATION_VERSION, SEEDED_GENERATORS, rng_state_fingerprint,
    seed_generation_rngs,
)
from harness.sealed_final_evaluator import FinalEvaluationError, run_final_evaluation
from harness.sealed_final_smoke import (
    generator_identity, smoke_problems, synthetic_batch, synthetic_record,
    synthetic_record_b,
)
import scripts.run_sealed_final_test as entry

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v5.json"


# ------------------------------------------------------------ 1. seed fidelity

def test_the_shared_seeder_actually_moves_rng_state():
    seed_generation_rngs(42)
    before = rng_state_fingerprint()
    random.random()
    moved = rng_state_fingerprint()
    assert moved != before, "drawing from the RNG must change its state"
    seed_generation_rngs(42)
    assert rng_state_fingerprint() == before, "reseeding must restore the state"


def test_the_seeder_is_deterministic_for_a_given_seed():
    seed_generation_rngs(42)
    first = [random.random() for _ in range(5)]
    seed_generation_rngs(42)
    assert [random.random() for _ in range(5)] == first


def test_the_seeder_reports_what_it_seeded():
    record = seed_generation_rngs(42)
    assert record["applied"] is True
    assert record["seed"] == 42
    assert record["seed_application_version"] == SEED_APPLICATION_VERSION
    assert record["seeded"] == list(SEEDED_GENERATORS)
    assert "python_random" in record["seeded"]
    assert "torch_cpu" in record["seeded"]
    assert "torch_cuda_all" in record["seeded"]


@pytest.mark.parametrize("bad", [-1, "42", None, 1.5])
def test_an_invalid_seed_is_refused(bad):
    with pytest.raises(ValueError):
        seed_generation_rngs(bad)


def test_the_locked_evaluator_uses_the_shared_seeder():
    """Both paths must seed through one initializer, not two inline copies."""
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    assert "from harness.generation_rng import seed_generation_rngs" in source
    assert "seed_generation_rngs(SEED)" in source
    # the inline duplicates are gone
    assert "torch.cuda.manual_seed_all(SEED)" not in source


def test_the_adapter_seeds_before_calling_model_generate():
    """Ordering, in the one place generation happens."""
    source = inspect.getsource(generate_candidate_slots)
    assert source.index("seed_generation_rngs(settings.seed)") < \
        source.index("generator.model.generate(")


def test_the_entrypoint_resets_the_seed_before_sealed_generation():
    source = (ROOT / "scripts" / "run_sealed_final_test.py").read_text(encoding="utf-8")
    reset = source.index("seed_generation_rngs(settings.seed)")
    assert source.index("guard.open_sealed_split(") < reset, \
        "the reset belongs after authorization, immediately before generation"
    assert reset < source.index("run_final_evaluation(")
    assert "reset_immediately_before_sealed_generation" in source
    assert "rng_state_before_reset" in source


# ----------------------------------------------------------- 2. batch fidelity

def test_the_frozen_batch_size_matches_locked_validation():
    assert successor_settings().generation_batch_size == 2


def test_a_batch_size_other_than_two_is_refused_by_the_entrypoint():
    receipt = _v4()
    frozen = dict(receipt["final_evaluator_source"]["frozen_generation_settings"])
    frozen["generation_batch_size"] = 1
    tampered = json.loads(json.dumps(receipt))
    tampered["final_evaluator_source"]["frozen_generation_settings"] = frozen
    problems = entry.settings_binding_problems(tampered)
    assert any("generation_batch_size is 1" in p for p in problems)


def test_chunking_produces_full_batches_then_a_remainder():
    records = [{"id": str(i)} for i in range(5)]
    batches = list(chunked(records, 2))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [r["id"] for b in batches for r in b] == ["0", "1", "2", "3", "4"]


def test_the_evaluator_generates_in_batches_and_scores_per_target(tmp_path):
    seen_batches = []

    def generate_batch(records):
        seen_batches.append(len(records))
        return [[{"raw_output": "assert add(1, 2) == 3", "code": "assert add(1, 2) == 3"}
                 for _ in range(8)] for _ in records]

    records = [{
        "id": f"r{i}", "entry_point": "add",
        "golden_code": "def add(a, b):\n    return a + b\n",
        "mutant_code": "def add(a, b):\n    return a - b\n",
    } for i in range(5)]

    outcome = run_final_evaluation(
        load_records=lambda: records, generate_batch=generate_batch,
        output_dir=tmp_path / "run", bundle_sha256="a" * 64,
        frozen_settings={}, generation_batch_size=2)
    assert seen_batches == [2, 2, 1]
    assert outcome["generation_batches"] == 3
    artifact = json.loads(Path(outcome["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["generation_batch_size"] == 2
    assert artifact["function_validation_records"] == 5
    assert len(artifact["function_results"]) == 5


def test_a_batch_generator_returning_the_wrong_count_is_refused(tmp_path):
    with pytest.raises(FinalEvaluationError, match="batch generation returned"):
        run_final_evaluation(
            load_records=lambda: [{"id": "a", "entry_point": "f",
                                   "golden_code": "def f():\n    return 1\n",
                                   "mutant_code": "def f():\n    return 2\n"}],
            generate_batch=lambda records: [],
            output_dir=tmp_path / "r", bundle_sha256="a" * 64,
            frozen_settings={}, generation_batch_size=2)


def test_the_smoke_batch_has_two_distinct_synthetic_targets():
    batch = synthetic_batch()
    assert len(batch) == 2
    assert batch[0]["id"] != batch[1]["id"]
    assert batch[0]["entry_point"] == "add_two"
    assert batch[1]["entry_point"] == "scale_by"
    for record in batch:
        assert record["source_name"] == "synthetic"


def test_smoke_problems_reject_a_one_record_batch():
    settings = successor_settings()
    good = _good_smoke(settings)
    assert smoke_problems(good, settings) == []
    assert any("frozen batch size is 2" in p for p in smoke_problems(
        dict(good, records_in_batch=1), settings))
    assert any("did not pad and generate the whole batch together" in p
               for p in smoke_problems(dict(good, prompts_generated_together=1), settings))


def _good_smoke(settings):
    return {
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
        "model_revision": settings.base_model_revision,
        "records_are_synthetic": True,
        "generator_identity": {"is_loaded": True},
    }


@pytest.mark.parametrize("field,value,fragment", [
    ("seed_applied", False, "did not apply the frozen generation seed"),
    ("seed", 7, "the frozen seed is 42"),
    ("candidate_slots_total", 8, "expected 16"),
    ("raw_outputs_retained", 8, "raw output for every slot"),
    ("scored_candidates", 8, "score every candidate slot"),
    ("observed_parse_mode", "first_assertion", "not the frozen whole_output"),
])
def test_each_smoke_regression_is_caught(field, value, fragment):
    settings = successor_settings()
    assert any(fragment in p for p in smoke_problems(
        dict(_good_smoke(settings), **{field: value}), settings))


def test_an_unloaded_generator_is_caught():
    settings = successor_settings()
    bad = dict(_good_smoke(settings), generator_identity={"is_loaded": False})
    assert any("cannot be reused" in p for p in smoke_problems(bad, settings))


# ------------------------------------------------ 3. all semantics are bound

REQUIRED_SETTING_FIELDS = (
    "generation_batch_size", "allow_test_function_candidates",
    "prompt_information_variant", "output_instruction_variant",
    "prompt_schema_version", "candidate_parse_mode", "retain_raw_output",
    "candidates_per_function", "temperature", "top_p",
    "generation_completion_token_limit", "prompt_token_limit",
    "max_sequence_tokens", "seed", "base_model_name", "base_model_revision",
    "attention_implementation",
)


@pytest.mark.parametrize("field", REQUIRED_SETTING_FIELDS)
def test_every_generation_semantic_is_a_settings_field(field):
    assert field in successor_settings().to_dict(), field


@pytest.mark.parametrize("field,value", [
    ("prompt_information_variant", "wrong"),
    ("output_instruction_variant", "wrong"),
    ("attention_implementation", "wrong"),
    ("base_model_revision", "main"),
    ("generation_batch_size", 0),
])
def test_an_invalid_semantic_is_refused(field, value):
    settings = GenerationSettings(**dict(successor_settings().to_dict(), **{field: value}))
    assert settings.problems(), field


def test_the_sealed_path_reads_no_mutable_module_globals():
    """Settings arrive as an object; the loader must not reach into the trainer."""
    source = (ROOT / "harness" / "sealed_final_loader.py").read_text(encoding="utf-8")
    for forbidden in ("CANDIDATE_PARSE_MODE", "RETAIN_RAW_OUTPUT", "BATCH_GEN_SIZE",
                      "PROMPT_TOKEN_LIMIT", "MAX_NEW_TOKENS_OVERRIDE"):
        assert forbidden not in source, forbidden


@pytest.mark.parametrize("field", [
    "generation_batch_size", "allow_test_function_candidates",
    "prompt_information_variant", "output_instruction_variant",
    "prompt_schema_version", "attention_implementation", "tokenizer_revision",
    "candidates_per_target", "seeds", "baseline_scope",
])
def test_the_v4_bundle_binds_every_semantic(field):
    receipt = _v4()
    assert field in receipt["frozen_bundle"]["fields"], field


def test_the_v4_bundle_records_the_seed_application_method():
    seeds = _v4()["frozen_bundle"]["fields"]["seeds"]
    assert seeds["generation_seed"] == 42
    assert seeds["seed_application_version"] == SEED_APPLICATION_VERSION
    assert seeds["applied_immediately_before_generation"] is True


# -------------------------------------------- 4. no post-token second load

def test_the_entrypoint_prepares_the_model_once_and_reuses_it():
    source = (ROOT / "scripts" / "run_sealed_final_test.py").read_text(encoding="utf-8")
    assert source.count("prepared.load_model()") == 1
    load = source.index("prepared.load_model()")
    guard = source.index("guard.open_sealed_split(")
    assert load < guard, "the model must be loaded and proven before the token"
    assert "sealed_batch_generator(prepared, settings, build_prompt)" in source
    assert "reused_from_pre_authorization_smoke" in source
    assert "same_object_as_smoke" in source


def test_the_batch_generator_takes_a_prepared_generator():
    from harness.sealed_final_loader import sealed_batch_generator
    source = inspect.getsource(sealed_batch_generator)
    assert "load_model" not in source, "it must not load a second model"
    assert "Phi3Generator" not in source


def test_generator_identity_can_prove_reuse():
    settings = successor_settings()

    class Stub:
        is_loaded = True
        parse_mode = "whole_output"

    stub = Stub()
    first = generator_identity(stub, settings)
    second = generator_identity(stub, settings)
    assert first["python_object_id"] == second["python_object_id"]
    assert generator_identity(Stub(), settings)["python_object_id"] != \
        first["python_object_id"]


def test_the_run_state_records_the_smoke_and_the_reuse():
    source = (ROOT / "scripts" / "run_sealed_final_test.py").read_text(encoding="utf-8")
    assert '"pre_authorization_smoke": smoke_result' in source
    assert "generator_identity=identity" in source


# ------------------------------------------------------- 5. baseline scope

def test_the_v4_bundle_bundles_no_baseline():
    baselines = _v4()["frozen_bundle"]["fields"]["baseline_versions"]
    assert baselines["baselines_bundled"] == []
    assert baselines["comparative_claims_supported"] is False
    assert "CANNOT support any comparative claim" in baselines["statement"]


def test_the_v4_receipt_states_the_scope_explicitly():
    scope = _v4()["baseline_scope"]
    assert scope["comparative_claims_supported"] is False
    assert scope["option_chosen"].startswith("B")
    assert "Atheris" in scope["statement"]


def test_no_validation_baseline_artifact_is_retained_in_the_bundle():
    """The v3 defect: val baselines pinned inside a sealed-final freeze."""
    text = json.dumps(_v4()["frozen_bundle"])
    assert "v4_2_atheris_tasks_val.manifest.json" not in text
    assert "v4_2_baseline_bundle_val.json" not in text


# ------------------------------------------------- refusal of v1, v2 and v3

@pytest.mark.parametrize("version", [
    "oneiros_sealed_final_readiness_v1",
    "oneiros_sealed_final_readiness_v2",
    "oneiros_sealed_final_readiness_v3",
])
def test_every_superseded_schema_is_refused(version):
    assert version in entry.REFUSED_SCHEMA_VERSIONS


def test_the_required_schema_is_v5():
    assert entry.REQUIRED_SCHEMA_VERSION == "oneiros_sealed_final_readiness_v5"


def test_the_v3_receipt_is_preserved_and_marked_superseded():
    v3 = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v3.json"
    marker = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v3.SUPERSEDED.md"
    if not v3.exists():
        pytest.skip("v3 receipt absent")
    import hashlib
    assert hashlib.sha256(v3.read_bytes()).hexdigest() == \
        "70821ef9504f76b7521ba236b5c92dc10275850c86a64341e07b9fbdbece7611"
    assert marker.exists()
    text = marker.read_text(encoding="utf-8")
    for topic in ("Seed recorded but never applied", "Batch shape differed",
                  "second model load after the token", "Implied baseline comparison"):
        assert topic in text


# ------------------------------------------------------- the v4 receipt

def _v4() -> dict:
    if not V4.exists():
        pytest.skip("v4 executable receipt not generated")
    return json.loads(V4.read_text(encoding="utf-8"))


@pytest.mark.parametrize("field", [
    "canonical_sha256", "measurement_logic_canonical_sha256",
    "adapter_canonical_sha256", "smoke_canonical_sha256",
    "loader_canonical_sha256", "rng_canonical_sha256",
    "entrypoint_canonical_sha256", "frozen_generation_settings",
])
def test_the_v4_receipt_binds_every_module(field):
    assert field in _v4()["final_evaluator_source"], field


def test_the_v5_binding_matches_the_runtime():
    assert entry.evaluator_binding_problems(_v4()) == []
    assert entry.settings_binding_problems(_v4()) == []


def test_the_v4_receipt_is_ready_and_untouched_by_sealed_data():
    receipt = _v4()
    assert receipt["schema_version"] == "oneiros_sealed_final_readiness_v5"
    assert receipt["preflight_problems"] == []
    assert receipt["ready_for_authorization"] is True
    assert receipt["final_evaluator_executable"] is True
    assert receipt["sealed_split_accessed"] is False
    assert receipt["authorization_token_issued"] is False
    assert receipt["sealed_records_read"] == 0


def test_no_authorization_has_ever_been_granted():
    assert not (ROOT / "results" / "sealed_final_state.json").exists()
    assert not (ROOT / "results" / "sealed_final_run_state.json").exists()
    audit = ROOT / "results" / "sealed_final_audit.log"
    if audit.exists():
        events = [json.loads(line) for line in
                  audit.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [e for e in events if e.get("event") == "sealed_access_granted"] == []
