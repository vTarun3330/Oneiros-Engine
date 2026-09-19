"""Experimental prompt variants, kept out of the frozen prompt engine.

``engine/test_generation_prompt.py`` is bound by hash into the rehearsal
receipt (``b522d352...``). Adding a variant there changes that hash, so every
receipt referring to it becomes unverifiable and the prompt-binding tests fail.
An exploratory variant is not worth invalidating the frozen baseline for.

So a variant lives here instead. This module never edits the prompt engine: it
calls the canonical builder and substitutes exactly one known block of text,
asserting first that the block is present verbatim. If the canonical wording
ever changes, the substitution fails loudly rather than silently rendering a
prompt nobody chose.

Nothing here is part of the frozen protocol. A number produced with one of
these variants is a development observation.
"""
from __future__ import annotations

from typing import Sequence

from engine.test_generation_prompt import build_unified_user_prompt

VARIANTS = ("oracle_first",)

#: The canonical ``self_contained`` task instruction, verbatim. A variant is
#: rendered by replacing exactly this text, so a drift in the prompt engine is
#: caught here instead of producing a silently different measurement.
CANONICAL_SELF_CONTAINED = (
    "Generate one minimal, self-contained bug-revealing Python test case. The test\n"
    "case may contain setup and assertions needed to demonstrate one behavioral\n"
    "defect, but it must not contain multiple independent test cases."
)

#: Measurement, not intuition. On the locked panel 65.5% of the tuned arm's
#: candidates fail against the CORRECT implementation, while only 9.1% fail by
#: probing an input where reference and mutant agree: the model finds the bug
#: and then misstates what should happen there.
#:
#: Every other variant asks for the assertion directly, so the expected value is
#: committed mid-expression with nothing to check it against. This one forces
#: the value to be decided first, in isolation, then written down. The staging
#: is the intervention.
#:
#: Measured on ablation_dev at seed 42 against the recorded base run: Kill@1
#: +10.52 points (p=0.000003), Kill@8 +4.24 (p=0.052) after applying one
#: extraction rule to both arms. The gain is in discrimination, not in oracle
#: correctness - reference failures rose from 38.4% to 45.2%.
ORACLE_FIRST = (
    "Work in two steps, and show both.\n"
    "\n"
    "STEP 1 - State the intended behaviour. In one or two sentences, say what a\n"
    "CORRECT implementation should return for the input you are about to test.\n"
    "Decide this from the specification and the function's purpose. The code shown\n"
    "to you may be defective, so do not describe what it does - describe what it\n"
    "ought to do. Write this as a Python comment.\n"
    "\n"
    "STEP 2 - Write one minimal, self-contained bug-revealing Python test case\n"
    "whose assertion matches the behaviour you just stated. The test may contain\n"
    "setup and assertions needed to demonstrate one behavioral defect, but it must\n"
    "not contain multiple independent test cases.\n"
    "\n"
    "If step 1 and the code disagree, trust step 1: that disagreement is the bug."
)

_INSTRUCTIONS = {"oracle_first": ORACLE_FIRST}


class VariantError(RuntimeError):
    """Raised when a variant cannot be rendered from the canonical prompt."""


def build_variant_prompt(
    variant: str,
    *,
    code_under_test: str,
    execution_mode: str,
    specification: str = "",
    support_context: str = "",
    target_symbols: Sequence[str] | str | None = None,
    entry_point: str = "",
    information_variant: str = "full",
) -> str:
    """Render an experimental variant from the canonical ``self_contained`` prompt."""
    if variant not in _INSTRUCTIONS:
        raise VariantError(f"unknown experimental variant: {variant!r}; "
                           f"known: {tuple(_INSTRUCTIONS)}")
    prompt = build_unified_user_prompt(
        code_under_test=code_under_test,
        execution_mode=execution_mode,
        specification=specification,
        support_context=support_context,
        target_symbols=target_symbols,
        entry_point=entry_point,
        information_variant=information_variant,
        output_instruction_variant="self_contained",
    )
    return apply_variant(prompt, variant)


def apply_variant(prompt: str, variant: str) -> str:
    """Substitute the variant instruction into an already-rendered prompt.

    ``harness.prompt_factory.build_record_prompt`` calls the canonical builder
    directly and applies no compaction, so substituting here is byte-identical
    to having rendered the variant inside the prompt engine. Budget compaction
    runs later, in the generation adapter, on the finished prompt - which is
    where it ran for the inline version too.
    """
    if variant not in _INSTRUCTIONS:
        raise VariantError(f"unknown experimental variant: {variant!r}; "
                           f"known: {tuple(_INSTRUCTIONS)}")
    if prompt.count(CANONICAL_SELF_CONTAINED) != 1:
        raise VariantError(
            "the canonical self_contained instruction is not present exactly once "
            "in the rendered prompt; engine/test_generation_prompt.py has changed "
            "and this variant must be re-derived rather than silently applied")
    return prompt.replace(CANONICAL_SELF_CONTAINED, _INSTRUCTIONS[variant])


def variant_builder(canonical_build, variant: str):
    """Wrap a ``prompt_factory`` builder so it renders an experimental variant."""
    def build(record):
        return apply_variant(canonical_build(record), variant)

    build.prompt_settings = getattr(canonical_build, "prompt_settings", None)
    build.experimental_variant = variant
    return build
