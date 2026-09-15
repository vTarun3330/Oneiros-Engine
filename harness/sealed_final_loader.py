"""Sealed-split access, used only by an authorized final run.

This module is committed so the sealed procedure is reviewable in advance, and
so its presence and identity can be checked *before* a token is spent. Importing
it reads nothing. Only calling ``sealed_records`` touches the sealed shard, and
it refuses to do so unless the guard has already recorded a granted
authorization - so an accidental import, a stray call from a notebook, or a
test that wandered off its mocks cannot open the split.

The development corpus view deliberately contains no sealed shard, so this is
the one place that reads the canonical corpus files. That is the point of it
being separate, small, and gated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from harness.sealed_final import SEALED_SPLIT, SealedAccessError

ROOT = Path(__file__).resolve().parent.parent
CORPUS_VERSION = "v4_1_research_hardened_candidate"
STATE_PATH = ROOT / "results" / "sealed_final_state.json"


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
            "sealed records requested without a granted authorization. This module "
            "may only be called by scripts/run_sealed_final_test.py after the guard "
            "has granted the single one-time token. Tests must inject mock records "
            "instead.")


def sealed_records(corpus_version: str = CORPUS_VERSION) -> List[Dict[str, Any]]:
    """Load the sealed split. Refuses without a granted authorization."""
    _require_authorization()
    corpus_dir = ROOT / "data" / "corpus" / corpus_version
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    sealed_ids = [str(item) for item in splits.get(SEALED_SPLIT, [])]
    if not sealed_ids:
        raise SealedAccessError("the sealed split is empty; refusing to measure nothing")

    wanted = set(sealed_ids)
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record.get("id")): record for record in records
             if str(record.get("id")) in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise SealedAccessError(
            f"{len(missing)} sealed ids are absent from the canonical records")
    # Preserve split order so the evaluation scope hash is deterministic.
    return [by_id[identifier] for identifier in sealed_ids]


def sealed_generator(receipt: Mapping[str, Any]) -> Callable[
        [Mapping[str, Any], int], Sequence[Dict[str, Any]]]:
    """Build the frozen generator described by an approved executable receipt.

    Returns a callable of (record, candidates) -> ordered candidate dicts, each
    carrying ``raw_output`` and the parsed ``code``. Constructed lazily so that
    merely importing this module loads no model.
    """
    fields = receipt["frozen_bundle"]["fields"]
    candidate = receipt["final_candidate"]
    if candidate.get("adapter") is not None:
        raise SealedAccessError(
            "the approved final candidate is the base model with no adapter; "
            "refusing to load one")

    from engine.generator import Phi3Generator
    from engine.test_generation_prompt import build_test_generation_prompt

    generator = Phi3Generator(
        model_name=candidate["model"],
        model_revision=candidate["model_revision"],
        attention_implementation="sdpa",
    )
    generator.load_model()
    generator.max_new_tokens = int(
        fields["prompt_budgets"]["generation_completion_token_limit"])

    def generate(record: Mapping[str, Any], candidates: int) -> List[Dict[str, Any]]:
        prompt = build_test_generation_prompt(
            record, prompt_token_limit=int(fields["prompt_budgets"]["prompt_token_limit"]),
        )
        outputs = generator.generate_candidates(prompt, candidates)
        return [{"raw_output": item.get("raw_output", ""), "code": item.get("code")}
                for item in outputs]

    return generate
