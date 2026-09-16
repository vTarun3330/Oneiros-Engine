"""A real end-to-end rehearsal on synthetic data, before any token is spent.

Every defect that reached the sealed design survived because the real path was
never executed: a nonexistent prompt function, a nonexistent generator method,
a parse mode left at its legacy default, a seed recorded but never applied, and
a batch size of one where locked validation used two. Mocks proved the plumbing
around the model and nothing about the model path itself.

So this loads the actual Qwen snapshot, builds the actual frozen prompts,
generates a **two-record batch** through the shared adapter, retains and hashes
every raw output, parses under ``whole_output``, and scores every slot against
synthetic golden/mutant pairs with the same candidate policy and execution
harness the real run uses.

Two records rather than one is deliberate. Locked validation padded two targets
per ``generate()`` call, and left-padding changes what the model samples, so a
one-record smoke would sign off a protocol the sealed run does not follow.

The records are synthetic and public. No corpus is opened, no split enumerated,
and the sealed loader is not imported by this module.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from harness.generation_adapter import (
    ADAPTER_VERSION, GenerationSettings, adapter_source_hashes,
    generate_candidate_slots,
)
from harness.generation_rng import SEED_APPLICATION_VERSION

SMOKE_VERSION = "oneiros_sealed_final_smoke_v2"

#: Synthetic, public targets. Deliberately trivial: the smoke is about whether
#: the path runs under the frozen protocol, not whether the model is clever.
SYNTHETIC_GOLDEN = (
    'def add_two(a, b):\n'
    '    """Return the sum of a and b."""\n'
    '    return a + b\n'
)
SYNTHETIC_MUTANT = (
    'def add_two(a, b):\n'
    '    """Return the sum of a and b."""\n'
    '    return a - b\n'
)
SYNTHETIC_GOLDEN_B = (
    'def scale_by(values, factor):\n'
    '    """Return each value multiplied by factor."""\n'
    '    return [v * factor for v in values]\n'
)
SYNTHETIC_MUTANT_B = (
    'def scale_by(values, factor):\n'
    '    """Return each value multiplied by factor."""\n'
    '    return [v + factor for v in values]\n'
)


def _record(identifier: str, entry_point: str, golden: str, mutant: str,
            specification: str) -> Dict[str, Any]:
    return {
        "id": identifier,
        "entry_point": entry_point,
        "golden_code": golden,
        "mutant_code": mutant,
        "prompt_code_under_test": mutant,
        "specification": specification,
        "execution_mode": "function_assertion",
        "support_context": "",
        "target_symbols": [entry_point],
        "bug_family": "arithmetic",
        "source_name": "synthetic",
        "project": "synthetic",
        "dataset_name": "synthetic",
    }


def synthetic_record() -> Dict[str, Any]:
    """The first synthetic target."""
    return _record("synthetic-smoke-add-two", "add_two", SYNTHETIC_GOLDEN,
                   SYNTHETIC_MUTANT, "Return the sum of the two arguments.")


def synthetic_record_b() -> Dict[str, Any]:
    """The second synthetic target, so the batch has two members."""
    return _record("synthetic-smoke-scale-by", "scale_by", SYNTHETIC_GOLDEN_B,
                   SYNTHETIC_MUTANT_B, "Return each value multiplied by the factor.")


def synthetic_batch() -> List[Dict[str, Any]]:
    """Exactly the batch shape the sealed run will use."""
    return [synthetic_record(), synthetic_record_b()]


def prepare_generator(settings: GenerationSettings):
    """Load the frozen model once.

    Returned so the caller can keep it: the model smoke-tested before
    authorization must be the model that runs after it. Building a second one
    post-token meant a failure there would waste the single authorization on a
    load that had never been proven.
    """
    from engine.generator import Phi3Generator

    generator = Phi3Generator(
        model_name=settings.base_model_name,
        model_revision=settings.base_model_revision,
        attention_implementation=settings.attention_implementation,
    )
    generator.temperature = settings.temperature
    generator.top_p = settings.top_p
    generator.parse_mode = settings.candidate_parse_mode
    generator.load_model()
    return generator


def generator_identity(generator, settings: GenerationSettings) -> Dict[str, Any]:
    """Enough to prove the prepared generator is the one that later ran."""
    return {
        "python_object_id": id(generator),
        "model_name": settings.base_model_name,
        "model_revision": settings.base_model_revision,
        "attention_implementation": settings.attention_implementation,
        "parse_mode": getattr(generator, "parse_mode", None),
        "is_loaded": bool(getattr(generator, "is_loaded", False)),
    }


def run_model_smoke(
    *,
    settings: GenerationSettings,
    build_pair_prompt=None,
    generator=None,
) -> Dict[str, Any]:
    """Load the model and run a two-record synthetic batch end to end.

    ``generator`` is injectable so unit tests can exercise the assertions
    without a GPU. The pre-authorization path passes a prepared generator and
    keeps it.
    """
    from metrics.research_evaluation import evaluate_candidate_slots

    problems = settings.problems()
    if problems:
        raise RuntimeError(f"smoke settings are invalid: {problems}")
    if settings.candidate_parse_mode != "whole_output":
        raise RuntimeError(
            "smoke must run the frozen successor protocol; parse mode is "
            f"{settings.candidate_parse_mode!r}, expected 'whole_output'")
    if settings.generation_batch_size != 2:
        raise RuntimeError(
            "smoke must exercise the frozen batch shape; generation_batch_size "
            f"is {settings.generation_batch_size}, expected 2")

    if build_pair_prompt is None:
        from scripts.train_on_dataset import build_pair_prompt as build_pair_prompt

    if generator is None:
        generator = prepare_generator(settings)

    records = synthetic_batch()
    if len(records) != settings.generation_batch_size:
        raise RuntimeError("the synthetic batch does not match the frozen batch size")

    started = datetime.now(timezone.utc)

    import sys as _sys
    loader_already_imported = "harness.sealed_final_loader" in _sys.modules

    # Seeded through the same shared initializer the sealed run uses, so the
    # smoke exercises seeding rather than merely asserting that it exists.
    accounting = generate_candidate_slots(
        generator, records, settings, build_pair_prompt,
        seed_before_generation=True)

    per_record: List[Dict[str, Any]] = []
    retained = 0
    scored = 0
    killed = 0
    executed = 0
    for index, record in enumerate(records):
        slots = accounting[index]["candidate_slots"]
        if len(slots) != settings.candidates_per_function:
            raise RuntimeError(
                f"target {record['id']} produced {len(slots)} slots, expected "
                f"{settings.candidates_per_function}")
        for slot in slots:
            digest = slot.get("raw_output_sha256")
            if digest is None:
                continue
            raw = slot.get("raw_output")
            if raw is None:
                raise RuntimeError(
                    "retain_raw_output is set but a slot carries no raw output")
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
                raise RuntimeError("raw-output hash does not match its text")
            retained += 1
        outcomes = evaluate_candidate_slots(
            slots,
            golden_code=record["golden_code"],
            mutant_code=record["mutant_code"],
            entry_point=record["entry_point"],
            allow_test_function=settings.allow_test_function_candidates,
        )
        scored += len(outcomes)
        executed += sum(bool(o.get("execution_valid")) for o in outcomes)
        killed += sum(bool(o.get("killed")) for o in outcomes)
        per_record.append({
            "record_id": record["id"],
            "candidate_slots": len(slots),
            "parsed_candidates": accounting[index]["parsed_candidates"],
            "scored_candidates": len(outcomes),
        })

    observed_parse_mode = getattr(generator, "parse_mode", None)
    if observed_parse_mode != "whole_output":
        raise RuntimeError(
            f"generator parse mode after generation is {observed_parse_mode!r}, "
            "expected 'whole_output'")

    loader_imported_now = "harness.sealed_final_loader" in _sys.modules
    sealed_authorization_granted = False
    if loader_imported_now:
        sealed_authorization_granted = bool(
            _sys.modules["harness.sealed_final_loader"].authorization_granted())

    return {
        "smoke_version": SMOKE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "seed_application_version": SEED_APPLICATION_VERSION,
        "adapter_source": adapter_source_hashes(),
        "ran_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": settings.base_model_name,
        "model_revision": settings.base_model_revision,
        "records_in_batch": len(records),
        "prompts_generated_together": len(records),
        "record_ids": [r["id"] for r in records],
        "records_are_synthetic": True,
        "candidates_requested_per_target": settings.candidates_per_function,
        "candidate_slots_total": sum(item["candidate_slots"] for item in per_record),
        "per_record": per_record,
        "raw_outputs_retained": retained,
        "raw_output_hashes_verified": True,
        "observed_parse_mode": observed_parse_mode,
        "scored_candidates": scored,
        "executed_candidates": executed,
        "killing_candidates": killed,
        "seed_applied": True,
        "seed": settings.seed,
        "sealed_loader_imported_by_smoke": (
            loader_imported_now and not loader_already_imported),
        "sealed_authorization_granted": sealed_authorization_granted,
        "sealed_data_touched": False,
        "generator_identity": generator_identity(generator, settings),
        "settings": settings.to_dict(),
        "passed": True,
    }


def smoke_problems(result: Dict[str, Any], settings: GenerationSettings) -> List[str]:
    """Judge a smoke result. Separated so it can be tested without a GPU."""
    found: List[str] = []
    expected_slots = settings.candidates_per_function * settings.generation_batch_size

    if result.get("records_in_batch") != settings.generation_batch_size:
        found.append(
            f"smoke ran {result.get('records_in_batch')} records, the frozen batch "
            f"size is {settings.generation_batch_size}")
    if result.get("prompts_generated_together") != settings.generation_batch_size:
        found.append("smoke did not pad and generate the whole batch together")
    if result.get("candidate_slots_total") != expected_slots:
        found.append(
            f"smoke produced {result.get('candidate_slots_total')} slots, expected "
            f"{expected_slots}")
    for item in result.get("per_record") or []:
        if item.get("candidate_slots") != settings.candidates_per_function:
            found.append(
                f"target {item.get('record_id')} got {item.get('candidate_slots')} "
                f"candidates, expected {settings.candidates_per_function}")
        if item.get("scored_candidates") != settings.candidates_per_function:
            found.append(f"target {item.get('record_id')} was not fully scored")
    if result.get("observed_parse_mode") != "whole_output":
        found.append(
            f"smoke ran under parse mode {result.get('observed_parse_mode')!r}, "
            "not the frozen whole_output")
    if not result.get("raw_output_hashes_verified"):
        found.append("smoke did not verify raw-output hashes")
    if result.get("raw_outputs_retained") != expected_slots:
        found.append("smoke did not retain a raw output for every slot")
    if result.get("scored_candidates") != expected_slots:
        found.append("smoke did not score every candidate slot")
    if not result.get("seed_applied"):
        found.append("smoke did not apply the frozen generation seed")
    if result.get("seed") != settings.seed:
        found.append(
            f"smoke seeded {result.get('seed')}, the frozen seed is {settings.seed}")
    if result.get("sealed_loader_imported_by_smoke"):
        found.append("the smoke test itself imported the sealed loader")
    if result.get("sealed_authorization_granted"):
        found.append("a sealed authorization is already granted before the smoke")
    if result.get("model_revision") and len(str(result["model_revision"])) != 40:
        found.append("smoke ran against a non-immutable model revision")
    if not result.get("records_are_synthetic"):
        found.append("smoke did not run on synthetic records")
    identity = result.get("generator_identity") or {}
    if not identity.get("is_loaded"):
        found.append("the smoke generator is not loaded; it cannot be reused")
    return found
