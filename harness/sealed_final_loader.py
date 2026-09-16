"""Sealed-split access and generation wiring for an authorized final run.

The previous version of this file was wrong in three ways at once, and none of
them could be caught because every test injected a mock generator:

* it imported ``build_test_generation_prompt``, which does not exist;
* it called ``Phi3Generator.generate_candidates``, which does not exist;
* it never set ``parse_mode``, which defaults to ``"first_assertion"`` - so the
  final measurement would have been scored by the legacy parser while claiming
  the frozen successor protocol.

The first two would have raised *after* the token was spent, because the
imports sat inside a function body and the entrypoint's "loader is importable"
check therefore proved nothing. The third would not have raised at all. It
would have produced a plausible number under the wrong parser, on the one
measurement that cannot be repeated.

So this version invents no API. Generation goes through
``harness.generation_adapter``, the same code locked validation runs, with
settings taken from the frozen successor protocol rather than from defaults.
Split parsing is factored into a pure function that can be rehearsed against a
non-sealed split, because the schema assumption was the other thing nobody had
ever executed.

Importing this module reads nothing. Only ``sealed_records`` touches the sealed
shard, and it refuses unless the guard has already recorded a granted
authorization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from harness.sealed_final import SEALED_SPLIT, SealedAccessError

ROOT = Path(__file__).resolve().parent.parent
CORPUS_VERSION = "v4_1_research_hardened_candidate"
STATE_PATH = ROOT / "results" / "sealed_final_state.json"

#: Fields a CANONICAL record must carry. These are the corpus field names -
#: reference_code and code_under_test - not the adapted evaluation names
#: golden_code and mutant_code. An earlier version required the adapted names,
#: which no canonical record has, so the sealed run would have loaded records
#: the evaluator could not score. The non-sealed rehearsal is what caught it.
REQUIRED_RECORD_FIELDS = (
    "id", "entry_point", "reference_code", "code_under_test", "specification",
    "task_type", "tests",
)


class SplitSchemaError(RuntimeError):
    """Raised when a split or its records do not have the expected shape."""


# --------------------------------------------------------------------------
# Pure split parsing. No file access, no sealed knowledge - so it can be
# rehearsed against a permitted split or a synthetic fixture.
# --------------------------------------------------------------------------

def select_split_records(
    splits: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    """Return ``split_name``'s records, in split order, or explain the failure.

    Split order is preserved deliberately: the evaluation scope hash is taken
    over the id sequence, so a set-ordered result would produce a different
    scope digest for identical content.
    """
    if split_name not in splits:
        raise SplitSchemaError(
            f"split {split_name!r} is absent; available: {sorted(splits)}")
    ids = [str(item) for item in splits[split_name]]
    if not ids:
        raise SplitSchemaError(f"split {split_name!r} is empty; refusing to measure nothing")
    if len(set(ids)) != len(ids):
        raise SplitSchemaError(f"split {split_name!r} contains duplicate ids")

    wanted = set(ids)
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        identifier = str(record.get("id"))
        if identifier in wanted:
            by_id[identifier] = dict(record)

    missing = [identifier for identifier in ids if identifier not in by_id]
    if missing:
        raise SplitSchemaError(
            f"{len(missing)} id(s) in split {split_name!r} have no record; "
            f"first missing: {missing[:3]}")

    incomplete = []
    for identifier in ids:
        absent = [f for f in REQUIRED_RECORD_FIELDS if not by_id[identifier].get(f)]
        if absent:
            incomplete.append((identifier, absent))
    if incomplete:
        raise SplitSchemaError(
            f"{len(incomplete)} record(s) are missing required fields; "
            f"first: {incomplete[0]}")

    return [by_id[identifier] for identifier in ids]


def adapt_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Convert canonical records into the evaluation shape.

    Delegates to the trainer's own ``_record_to_pair`` rather than
    reimplementing the mapping, for the same reason generation is shared: a
    second adaptation is a second thing to be wrong, and its output would not
    be comparable with locked validation's.
    """
    from scripts.train_on_dataset import _record_to_pair
    return [_record_to_pair(dict(record)) for record in records]


# --------------------------------------------------------------------------
# Sealed access, gated.
# --------------------------------------------------------------------------

def authorization_granted() -> bool:
    """True only once the guard has actually granted and spent an authorization."""
    if not STATE_PATH.is_file():
        return False
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(state.get("runs"))


def _require_authorization() -> None:
    if not authorization_granted():
        raise SealedAccessError(
            "sealed records requested without a granted authorization. Only "
            "scripts/run_sealed_final_test.py may call this, and only after the "
            "guard has granted the single one-time token. Tests must use "
            "select_split_records against a permitted split or a fixture.")


def sealed_records(corpus_version: str = CORPUS_VERSION) -> List[Dict[str, Any]]:
    """Load the sealed split. Refuses without a granted authorization."""
    _require_authorization()
    corpus_dir = ROOT / "data" / "corpus" / corpus_version
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    return adapt_records(select_split_records(splits, records, SEALED_SPLIT))


# --------------------------------------------------------------------------
# Generation, through the shared adapter.
# --------------------------------------------------------------------------

def build_sealed_generator(receipt: Mapping[str, Any]):
    """Construct the frozen generator described by an approved receipt.

    Returns (generator, settings, build_prompt). Nothing is loaded at import
    time; the model is loaded when this is called, which happens only after
    every pre-authorization check has passed.
    """
    from engine.generator import Phi3Generator
    from harness.generation_adapter import GenerationSettings, successor_settings

    candidate = receipt["final_candidate"]
    if candidate.get("adapter") is not None:
        raise SealedAccessError(
            "the approved final candidate is the base model with no adapter; "
            "refusing to load one")

    fields = receipt["frozen_bundle"]["fields"]
    settings = successor_settings()
    # The receipt is authoritative over the protocol module: if they disagree,
    # the approved thing is what was approved.
    settings = GenerationSettings(
        candidate_parse_mode=fields["sampling"]["candidate_parse_mode"],
        retain_raw_output=bool(fields["sampling"]["retain_raw_output"]),
        candidates_per_function=int(fields["candidates_per_target"]),
        temperature=float(fields["sampling"]["temperature"]),
        top_p=float(fields["sampling"]["top_p"]),
        prompt_token_limit=int(fields["prompt_budgets"]["prompt_token_limit"]),
        generation_completion_token_limit=int(
            fields["prompt_budgets"]["generation_completion_token_limit"]),
        max_sequence_tokens=int(fields["prompt_budgets"]["max_sequence_tokens"]),
        seed=int(fields["seeds"]["generation_seed"]),
    )
    problems = settings.problems()
    if problems:
        raise SealedAccessError(f"approved generation settings are invalid: {problems}")
    if settings.candidate_parse_mode != "whole_output":
        raise SealedAccessError(
            f"the frozen successor protocol requires whole_output parsing, the "
            f"receipt declares {settings.candidate_parse_mode!r}")

    generator = Phi3Generator(
        model_name=candidate["model"],
        model_revision=candidate["model_revision"],
        attention_implementation="sdpa",
    )
    generator.temperature = settings.temperature
    generator.top_p = settings.top_p
    generator.parse_mode = settings.candidate_parse_mode

    from scripts.train_on_dataset import build_pair_prompt
    return generator, settings, build_pair_prompt


def sealed_generator(receipt: Mapping[str, Any]) -> Callable[
        [Mapping[str, Any], int], Sequence[Dict[str, Any]]]:
    """A per-record generate callable for the final evaluator.

    Wraps the shared adapter so the sealed run and locked validation execute
    the same generation body.
    """
    from harness.generation_adapter import generate_candidate_slots

    generator, settings, build_prompt = build_sealed_generator(receipt)
    generator.load_model()

    def generate(record: Mapping[str, Any], candidates: int) -> List[Dict[str, Any]]:
        if candidates != settings.candidates_per_function:
            raise SealedAccessError(
                f"asked for {candidates} candidates, the frozen protocol declares "
                f"{settings.candidates_per_function}")
        accounting = generate_candidate_slots(
            generator, [dict(record)], settings, build_prompt)
        slots = accounting[0]["candidate_slots"]
        return [{"raw_output": slot.get("raw_output", ""), "code": slot.get("code")}
                for slot in slots]

    return generate
