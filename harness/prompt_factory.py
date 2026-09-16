"""One prompt builder, taking its settings as arguments.

``scripts/train_on_dataset.build_prompt`` reads two module globals -
``PROMPT_INFORMATION_VARIANT`` and ``OUTPUT_INSTRUCTION_VARIANT`` - at call
time. The sealed loader imported ``build_pair_prompt``, so a sealed run would
have rendered its prompts under whatever those globals happened to hold, not
under the variants its own receipt froze.

Today that renders the right prompt by luck: the trainer's defaults are ``full``
and ``self_contained``, which are also the frozen values. Luck is not a binding.
Anyone changing a CLI default would silently change the prompts of a measurement
that cannot be repeated, and nothing in the artifact would show it.

So the settings travel as an argument. Development code may read its CLI globals
to *construct* a ``PromptSettings``; the builder itself never reads a global.

Validation comes from ``engine.test_generation_prompt``'s own constants rather
than a hand-written copy. The hand-written copy was wrong in both directions: it
invented ``minimal`` and ``bare``, which the prompt engine rejects, while
refusing ``code_only``, ``code_specification``, ``legacy_exactly_one`` and
``metamorphic_allowed``, which it accepts.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from engine.test_generation_prompt import (
    OUTPUT_INSTRUCTION_VARIANTS, PROMPT_INFORMATION_VARIANTS,
    PROMPT_SCHEMA_VERSION, build_unified_user_prompt,
)

PROMPT_FACTORY_VERSION = "oneiros_prompt_factory_v1"

#: Files whose contents can change a rendered prompt. Bound by hash into the
#: sealed receipt, because a prompt that changed between freeze and run would
#: change the measurement without changing any recorded setting.
PROMPT_DEFINING_SOURCES = {
    "prompt_factory": "harness/prompt_factory.py",
    "prompt_engine": "engine/test_generation_prompt.py",
    "prompt_compaction": "engine/prompt_budget.py",
    "record_adaptation": "scripts/train_on_dataset.py",
}


@dataclass(frozen=True)
class PromptSettings:
    """Everything that decides what the model is shown."""

    information_variant: str
    output_instruction_variant: str
    prompt_schema_version: str

    def problems(self) -> List[str]:
        """Validated against the prompt engine's own constants."""
        found: List[str] = []
        if self.information_variant not in PROMPT_INFORMATION_VARIANTS:
            found.append(
                f"unknown prompt information_variant {self.information_variant!r}; "
                f"the prompt engine declares {list(PROMPT_INFORMATION_VARIANTS)}")
        if self.output_instruction_variant not in OUTPUT_INSTRUCTION_VARIANTS:
            found.append(
                f"unknown output_instruction_variant {self.output_instruction_variant!r}; "
                f"the prompt engine declares {list(OUTPUT_INSTRUCTION_VARIANTS)}")
        if self.prompt_schema_version != PROMPT_SCHEMA_VERSION:
            found.append(
                f"prompt_schema_version {self.prompt_schema_version!r} is not the "
                f"engine's {PROMPT_SCHEMA_VERSION!r}")
        return found

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def frozen_prompt_settings() -> PromptSettings:
    """The configuration every reported Oneiros result was produced under."""
    return PromptSettings(
        information_variant="full",
        output_instruction_variant="self_contained",
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
    )


def build_record_prompt(record: Mapping[str, Any], settings: PromptSettings) -> str:
    """Render one adapted record's prompt under explicit settings.

    The fixed implementation, mutation diff, dataset identity, oracle result and
    expected completion are absent from this API by construction, so a caller
    cannot leak them into the model-visible prompt.
    """
    problems = settings.problems()
    if problems:
        raise ValueError(f"invalid prompt settings: {problems}")
    return build_unified_user_prompt(
        code_under_test=(record.get("prompt_code_under_test")
                         or record.get("mutant_code") or ""),
        execution_mode=record.get("execution_mode") or "function_assertion",
        specification=record.get("specification", ""),
        support_context=record.get("support_context", ""),
        target_symbols=record.get("target_symbols", []),
        entry_point=record.get("entry_point", ""),
        information_variant=settings.information_variant,
        output_instruction_variant=settings.output_instruction_variant,
    )


def prompt_factory(settings: PromptSettings) -> Callable[[Mapping[str, Any]], str]:
    """A one-argument builder closed over EXPLICIT settings.

    The returned callable is what the generation adapter receives. It holds its
    settings in a closure, so a later change to any module global - trainer CLI
    state included - cannot reach it.
    """
    problems = settings.problems()
    if problems:
        raise ValueError(f"invalid prompt settings: {problems}")
    frozen = PromptSettings(**settings.to_dict())

    def build(record: Mapping[str, Any]) -> str:
        return build_record_prompt(record, frozen)

    build.prompt_settings = frozen  # type: ignore[attr-defined]
    return build


def prompt_factory_source_hashes(root=None) -> Dict[str, Any]:
    """Identity of every source that can change a rendered prompt."""
    from harness.source_identity import canonical_sha256, raw_sha256

    base = Path(root) if root else Path(__file__).resolve().parent.parent
    sources: Dict[str, Any] = {
        "prompt_factory_version": PROMPT_FACTORY_VERSION,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "information_variants": list(PROMPT_INFORMATION_VARIANTS),
        "output_instruction_variants": list(OUTPUT_INSTRUCTION_VARIANTS),
    }
    for role, relative in sorted(PROMPT_DEFINING_SOURCES.items()):
        path = base / relative
        sources[role] = {
            "path": relative,
            "raw_sha256": raw_sha256(path),
            "canonical_sha256": canonical_sha256(path),
        }
    return sources


def prompt_binding_problems(recorded: Mapping[str, Any], root=None) -> List[str]:
    """Refuse if any prompt-defining source differs from the approved receipt."""
    if not recorded:
        return ["receipt records no prompt-factory source identity"]
    current = prompt_factory_source_hashes(root)
    found: List[str] = []
    for key in ("prompt_factory_version", "prompt_schema_version"):
        if recorded.get(key) != current[key]:
            found.append(
                f"{key} differs from the approved receipt: "
                f"{recorded.get(key)!r} vs {current[key]!r}")
    for key in ("information_variants", "output_instruction_variants"):
        if list(recorded.get(key) or []) != current[key]:
            found.append(f"{key} differs from the approved receipt")
    for role in sorted(PROMPT_DEFINING_SOURCES):
        was = (recorded.get(role) or {}).get("canonical_sha256")
        now = current[role]["canonical_sha256"]
        if was != now:
            found.append(
                f"prompt-defining source {role} "
                f"({PROMPT_DEFINING_SOURCES[role]}) differs from the approved "
                f"receipt: {was} vs {now}")
    return found
