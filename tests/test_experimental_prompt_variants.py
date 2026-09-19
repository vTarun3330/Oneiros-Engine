"""Guards for development-only prompt variants kept outside the frozen engine."""
from __future__ import annotations

import pytest

from engine.test_generation_prompt import build_unified_user_prompt
from research.experimental.prompt_variants import (
    CANONICAL_SELF_CONTAINED,
    ORACLE_FIRST,
    VariantError,
    apply_variant,
    build_variant_prompt,
)


def _canonical() -> str:
    return build_unified_user_prompt(
        code_under_test="def f(x):\n    return x + 1\n",
        execution_mode="function_assertion",
        specification="Return the successor of x.",
        target_symbols=("f",),
        entry_point="f",
        output_instruction_variant="self_contained",
    )


def test_oracle_first_replaces_exactly_the_canonical_instruction():
    canonical = _canonical()
    rendered = apply_variant(canonical, "oracle_first")
    assert CANONICAL_SELF_CONTAINED in canonical
    assert CANONICAL_SELF_CONTAINED not in rendered
    assert ORACLE_FIRST in rendered
    assert rendered == canonical.replace(CANONICAL_SELF_CONTAINED, ORACLE_FIRST)


def test_direct_builder_is_byte_identical_to_post_render_substitution():
    direct = build_variant_prompt(
        "oracle_first",
        code_under_test="def f(x):\n    return x + 1\n",
        execution_mode="function_assertion",
        specification="Return the successor of x.",
        target_symbols=("f",),
        entry_point="f",
    )
    assert direct == apply_variant(_canonical(), "oracle_first")


def test_variant_refuses_prompt_drift_or_unknown_variant():
    with pytest.raises(VariantError, match="not present exactly once"):
        apply_variant("a prompt whose canonical instruction changed", "oracle_first")
    with pytest.raises(VariantError, match="unknown experimental variant"):
        apply_variant(_canonical(), "does_not_exist")
