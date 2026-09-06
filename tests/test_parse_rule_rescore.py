"""Pin the parse-rule rescore, including the bug that made it meaningless.

The first version of this rescore executed test functions without invoking
them. `def test_x(): assert ...` run on its own defines a name and asserts
nothing, so it passed on the reference and the mutant alike and every
candidate looked like a survivor: it reported a 0.01 kill rate against the
pipeline's own recorded 0.795. A rescore that does not reproduce the pipeline
measures nothing, and it fails loudly here instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import executable_candidate, validate_generated_test
from scripts.rescore_under_parse_rule import first_assertion, kills, whole_output

REFERENCE = "def f(x):\n    return x + 1\n"
MUTANT = "def f(x):\n    return x + 2\n"
TEST_FUNCTION = "def test_f():\n    assert f(1) == 2\n    assert f(3) == 4\n"


def test_a_test_function_is_invoked_not_merely_defined():
    """Without the appended call, nothing runs and nothing can be detected."""
    valid, killed = kills(TEST_FUNCTION, "f", REFERENCE, MUTANT)
    assert valid, "the test must pass on the reference implementation"
    assert killed, "the test must fail on the mutant; if not, it was never called"


def test_the_invocation_is_what_makes_the_difference():
    policy = validate_generated_test(TEST_FUNCTION, "f", allow_test_function=True)
    runnable = executable_candidate(TEST_FUNCTION, policy.shape)
    assert runnable.rstrip().endswith("test_f()")
    # The model emits the definition alone. Checking for the bare substring
    # would pass trivially, because "test_f()" also occurs inside "def
    # test_f():" - the call has to be looked for as its own statement.
    lines = [line.strip() for line in TEST_FUNCTION.splitlines()]
    assert "test_f()" not in lines, "the completion must not already call itself"
    assert "test_f()" in [line.strip() for line in runnable.splitlines()]


def test_first_assertion_reproduces_the_frozen_parser():
    assert first_assertion(TEST_FUNCTION) == "assert f(1) == 2"
    assert first_assertion("no assertions here") == ""
    assert first_assertion("  assert f(1) == 2  ") == "assert f(1) == 2"


def test_whole_output_unwraps_a_fenced_block():
    fenced = "```python\n" + TEST_FUNCTION + "```"
    assert not whole_output(fenced).startswith("```")
    assert whole_output(fenced).count("assert ") == 2


def test_a_bare_assertion_scores_the_same_under_both_rules():
    code = "assert f(1) == 2"
    assert first_assertion(code) == code
    assert whole_output(code) == code
    assert kills(code, "f", REFERENCE, MUTANT) == (True, True)


def test_a_reference_invalid_test_is_not_a_kill():
    """A test that fails on correct code would flag correct code as buggy.

    This is the mechanism behind the measured result: a longer test has more
    chances to detect the mutant AND more chances to be wrong about an
    expected value, and being wrong discards the whole candidate.
    """
    wrong = "def test_f():\n    assert f(1) == 99\n"
    valid, killed = kills(wrong, "f", REFERENCE, MUTANT)
    assert not valid
    assert not killed
