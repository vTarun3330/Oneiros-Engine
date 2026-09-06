"""Pin the two changes that make the multi-mutant capability measurable.

The frozen path collapses each model output to its first assertion, so a
multi-assertion test cannot survive to be scored, and only a hash of the raw
output is kept, so what the model emitted cannot be recovered. Both are opt-in
to change, because either one alters what Kill@8 means.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.generator import Phi3Generator

TEST_FUNCTION = (
    "def test_clamp_boundaries():\n"
    "    assert clamp(-1, 0, 10) == 0\n"
    "    assert clamp(5, 0, 10) == 5\n"
    "    assert clamp(11, 0, 10) == 10"
)


def _generator(mode: str) -> Phi3Generator:
    generator = Phi3Generator.__new__(Phi3Generator)
    generator.stats = {"total_generated": 0, "valid_generated": 0, "invalid_generated": 0}
    generator.parse_mode = mode
    return generator


def test_the_frozen_mode_keeps_only_the_first_assertion():
    """This is the defect, asserted so it cannot be fixed by accident."""
    parsed = _generator("first_assertion")._parse_output(
        TEST_FUNCTION, "clamp", "clamp"
    )
    assert parsed.input_code == "assert clamp(-1, 0, 10) == 0"
    assert "clamp(11, 0, 10)" not in parsed.input_code


def test_whole_output_mode_keeps_every_assertion():
    parsed = _generator("whole_output")._parse_output(
        TEST_FUNCTION, "clamp", "clamp"
    )
    assert parsed.input_code.count("assert ") == 3
    assert parsed.input_code.startswith("def test_clamp_boundaries")
    assert parsed.is_valid, parsed.parse_error


def test_whole_output_mode_still_accepts_a_bare_assertion():
    """Widening must not reject what the frozen rule already accepted."""
    parsed = _generator("whole_output")._parse_output(
        "assert clamp(5, 0, 10) == 5", "clamp", "clamp"
    )
    assert parsed.is_valid, parsed.parse_error
    assert parsed.input_code == "assert clamp(5, 0, 10) == 5"


def test_whole_output_mode_unwraps_a_fenced_code_block():
    fenced = "```python\n" + TEST_FUNCTION + "\n```"
    parsed = _generator("whole_output")._parse_output(fenced, "clamp", "clamp")
    assert not parsed.input_code.startswith("```")
    assert parsed.input_code.count("assert ") == 3


def test_whole_output_mode_rejects_output_that_never_calls_the_target():
    parsed = _generator("whole_output")._parse_output(
        "def test_other():\n    assert something_else(1) == 1", "clamp", "clamp"
    )
    assert not parsed.is_valid


def test_the_default_mode_is_the_frozen_one():
    """An unset parse_mode must behave exactly as every reported run did."""
    generator = Phi3Generator.__new__(Phi3Generator)
    generator.stats = {"total_generated": 0, "valid_generated": 0, "invalid_generated": 0}
    parsed = generator._parse_output(TEST_FUNCTION, "clamp", "clamp")
    assert parsed.input_code == "assert clamp(-1, 0, 10) == 0"


def test_the_driver_defaults_preserve_the_frozen_protocol():
    import scripts.train_on_dataset as driver

    assert driver.CANDIDATE_PARSE_MODE == "first_assertion"
    assert driver.RETAIN_RAW_OUTPUT is False

    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    assert 'default="first_assertion"' in source
    assert '"--retain-raw-output", action="store_true"' in source
    assert 'slot["raw_output"] = text' in source
