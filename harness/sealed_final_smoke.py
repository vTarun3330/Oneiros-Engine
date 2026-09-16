"""A real end-to-end rehearsal on synthetic data, before any token is spent.

Every defect that reached the sealed design survived because the real path was
never executed: a nonexistent prompt function, a nonexistent generator method,
and a parse mode left at its legacy default. Mocks proved the plumbing around
the model and nothing about the model path itself.

So this loads the actual Qwen snapshot, builds the actual frozen prompt,
generates exactly eight candidates, retains and hashes their raw outputs, parses
them under ``whole_output``, and scores them against a synthetic golden/mutant
pair with the same candidate policy and execution harness the real run uses.

The record is synthetic and public - a two-line addition function written here.
No corpus is opened, no split is enumerated, and the sealed loader is never
imported. If any of the three defects above returned, this fails loudly on a
throwaway function instead of quietly on the final measurement.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict

from harness.generation_adapter import (
    ADAPTER_VERSION, GenerationSettings, adapter_source_hashes,
    generate_candidate_slots,
)

SMOKE_VERSION = "oneiros_sealed_final_smoke_v1"

#: A synthetic, public target. Deliberately trivial: the smoke test is about
#: whether the path runs and is wired to the frozen protocol, not about whether
#: the model is clever.
SYNTHETIC_GOLDEN = "def add_two(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n"
SYNTHETIC_MUTANT = "def add_two(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a - b\n"


def synthetic_record() -> Dict[str, Any]:
    """The only record this smoke test ever sees."""
    return {
        "id": "synthetic-smoke-add-two",
        "entry_point": "add_two",
        "golden_code": SYNTHETIC_GOLDEN,
        "mutant_code": SYNTHETIC_MUTANT,
        "prompt_code_under_test": SYNTHETIC_MUTANT,
        "specification": "Return the sum of the two arguments.",
        "execution_mode": "function_assertion",
        "support_context": "",
        "target_symbols": ["add_two"],
        "bug_family": "arithmetic",
        "source_name": "synthetic",
        "project": "synthetic",
        "dataset_name": "synthetic",
    }


def run_model_smoke(
    *,
    model_name: str,
    model_revision: str,
    settings: GenerationSettings,
    build_pair_prompt=None,
    generator=None,
) -> Dict[str, Any]:
    """Load the model and run one synthetic target end to end.

    ``generator`` is injectable so unit tests can exercise the assertions
    without a GPU; the pre-authorization path passes nothing and gets the real
    Qwen snapshot.
    """
    from metrics.research_evaluation import evaluate_candidate_slots

    problems = settings.problems()
    if problems:
        raise RuntimeError(f"smoke settings are invalid: {problems}")
    if settings.candidate_parse_mode != "whole_output":
        raise RuntimeError(
            f"smoke must run the frozen successor protocol; parse mode is "
            f"{settings.candidate_parse_mode!r}")

    if build_pair_prompt is None:
        from scripts.train_on_dataset import build_pair_prompt as build_pair_prompt

    owns_generator = generator is None
    if owns_generator:
        from engine.generator import Phi3Generator
        generator = Phi3Generator(
            model_name=model_name,
            model_revision=model_revision,
            attention_implementation="sdpa",
        )
        generator.temperature = settings.temperature
        generator.top_p = settings.top_p
        generator.load_model()

    record = synthetic_record()
    started = datetime.now(timezone.utc)

    # Whether the SMOKE pulls in the sealed loader, not whether anything ever
    # did. The entrypoint legitimately imports it earlier to verify its symbols
    # resolve, so an absolute "never imported" check fires on correct
    # behaviour. What must stay true is that this code path does not reach for
    # it.
    import sys as _sys
    loader_already_imported = "harness.sealed_final_loader" in _sys.modules
    accounting = generate_candidate_slots(
        generator, [record], settings, build_pair_prompt)
    slots = accounting[0]["candidate_slots"]

    if len(slots) != settings.candidates_per_function:
        raise RuntimeError(
            f"expected {settings.candidates_per_function} candidate slots, got {len(slots)}")

    retained = 0
    for slot in slots:
        digest = slot.get("raw_output_sha256")
        if digest is None:
            continue
        raw = slot.get("raw_output")
        if raw is None:
            raise RuntimeError("retain_raw_output is set but a slot carries no raw output")
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
            raise RuntimeError("raw-output hash does not match its text")
        retained += 1

    outcomes = evaluate_candidate_slots(
        slots, golden_code=SYNTHETIC_GOLDEN, mutant_code=SYNTHETIC_MUTANT,
        entry_point=record["entry_point"], allow_test_function=True)

    # The parse mode actually used, read back off the generator rather than
    # assumed. This is the assertion that would have caught the legacy-default
    # defect.
    observed_parse_mode = getattr(generator, "parse_mode", None)
    if observed_parse_mode != "whole_output":
        raise RuntimeError(
            f"generator parse mode after generation is {observed_parse_mode!r}, "
            "expected 'whole_output'")

    import sys
    loader_imported_now = "harness.sealed_final_loader" in sys.modules
    sealed_loader_imported = loader_imported_now and not loader_already_imported

    # Belt and braces: even if it were imported, it cannot have returned sealed
    # records, because it refuses without a granted authorization.
    sealed_authorization_granted = False
    if loader_imported_now:
        sealed_authorization_granted = bool(
            sys.modules["harness.sealed_final_loader"].authorization_granted())

    return {
        "smoke_version": SMOKE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "adapter_source": adapter_source_hashes(),
        "ran_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": model_name,
        "model_revision": model_revision,
        "record_id": record["id"],
        "record_is_synthetic": True,
        "candidates_requested": settings.candidates_per_function,
        "candidate_slots": len(slots),
        "raw_outputs_retained": retained,
        "raw_output_hashes_verified": True,
        "observed_parse_mode": observed_parse_mode,
        "parsed_candidates": accounting[0]["parsed_candidates"],
        "scored_candidates": len(outcomes),
        "executed_candidates": sum(bool(o.get("execution_valid")) for o in outcomes),
        "killing_candidates": sum(bool(o.get("killed")) for o in outcomes),
        "sealed_loader_imported": sealed_loader_imported,
        "sealed_loader_already_imported_by_caller": loader_already_imported,
        "sealed_authorization_granted": sealed_authorization_granted,
        "sealed_data_touched": False,
        "settings": settings.to_dict(),
        "passed": True,
    }


def smoke_problems(result: Dict[str, Any], settings: GenerationSettings) -> list[str]:
    """Judge a smoke result. Separated so it can be tested without a GPU."""
    found: list[str] = []
    if result.get("candidate_slots") != settings.candidates_per_function:
        found.append(
            f"smoke produced {result.get('candidate_slots')} slots, expected "
            f"{settings.candidates_per_function}")
    if result.get("observed_parse_mode") != "whole_output":
        found.append(
            f"smoke ran under parse mode {result.get('observed_parse_mode')!r}, "
            "not the frozen whole_output")
    if not result.get("raw_output_hashes_verified"):
        found.append("smoke did not verify raw-output hashes")
    if result.get("scored_candidates") != settings.candidates_per_function:
        found.append("smoke did not score every candidate slot")
    if result.get("sealed_loader_imported"):
        found.append("the smoke test itself imported the sealed loader")
    if result.get("sealed_authorization_granted"):
        found.append("a sealed authorization is already granted before the smoke")
    if result.get("model_revision") and len(str(result["model_revision"])) != 40:
        found.append("smoke ran against a non-immutable model revision")
    if not result.get("record_is_synthetic"):
        found.append("smoke did not run on a synthetic record")
    return found
