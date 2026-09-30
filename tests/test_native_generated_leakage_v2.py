"""Leakage scanner v2: exact-line provenance for issue text (amendment v2.2 section A)."""
from __future__ import annotations

import hashlib
import textwrap

import pytest

from harness import native_generated_test_leakage as leak

BUGGY = textwrap.dedent('''
    def intword(value, powers, largest_ordinal=False):
        for ordinal, power in enumerate(powers):
            rounded_value = round(value / power, 1)
            if not largest_ordinal and rounded_value * power == powers[ordinal + 1]:
                return "next"
        return "same"
''')
BUGGY_LINE = "if not largest_ordinal and rounded_value * power == powers[ordinal + 1]:"
FIXED = BUGGY.replace(BUGGY_LINE, "if not largest_ordinal and rounded_value >= powers[ordinal + 1] / power:")
FIXED_LINE = "if not largest_ordinal and rounded_value >= powers[ordinal + 1] / power:"
PROSE = "The comparison in intword never promotes the value to the next power correctly."


def _sealed(text: str) -> dict:
    return {"prompt": text, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}


def _prompt(*extra: str) -> dict:
    return _sealed("### TEST GENERATION TASK\n\nCode under test:\n" + BUGGY + "\n"
                   + "\n".join(extra) + "\n")


def _verifier(**extra):
    return {"buggy_source": BUGGY, "fixed_source": FIXED, "patch": "", "official_tests": [],
            "issue_text": "", "commit_message": "", **extra}


def test_version_bumped():
    assert leak.SCANNER_VERSION == "oneiros_native_generated_test_leakage_v2"


def test_exact_issue_quotation_of_a_buggy_line_is_allowed():
    issue = f"intword is wrong.\n\n```python\n        {BUGGY_LINE}\n```\nPlease fix."
    result = leak.scan(_prompt(), _verifier(issue_text=issue))
    assert result["ok"] is True, result["reasons"]
    # the whole-line match is whitespace-normalised on both sides
    result = leak.scan(_prompt(), _verifier(issue_text="\t" + BUGGY_LINE.replace(" ", "   ")))
    assert result["ok"] is True


def test_a_substring_of_a_buggy_line_is_not_enough():
    fragment = "if not largest_ordinal and rounded_value * power == powers[ordinal"
    assert len(fragment) >= leak.MIN_SENTENCE
    result = leak.scan(_prompt(), _verifier(issue_text=fragment))
    assert result["ok"] is False and any(r.startswith("issue_text") for r in result["reasons"])


def test_buggy_line_plus_prose_on_the_same_line_is_not_exempt():
    line = f"{BUGGY_LINE} is where it goes wrong for large inputs, see traceback"
    result = leak.scan(_prompt(line), _verifier(issue_text=line))
    assert result["ok"] is False and any(r.startswith("issue_text") for r in result["reasons"])


def test_surrounding_issue_prose_remains_refused():
    issue = f"{PROSE}\n{BUGGY_LINE}\n"
    assert leak.scan(_prompt(), _verifier(issue_text=issue))["ok"] is True
    result = leak.scan(_prompt(PROSE), _verifier(issue_text=issue))
    assert result["ok"] is False and result["reasons"] == [f"issue_text: {PROSE[:80]}"]


def test_issue_line_that_is_in_the_prompt_but_not_buggy_source_is_refused():
    boilerplate = "Infer intended behavior conservatively from the supplied execution context."
    result = leak.scan(_prompt(boilerplate), _verifier(issue_text=boilerplate))
    assert result["ok"] is False


@pytest.mark.parametrize("injected, extra, reason", [
    (FIXED_LINE, {"issue_text": FIXED_LINE}, "fixed_line"),
    (FIXED_LINE, {"issue_text": FIXED_LINE}, "issue_text"),
    ("return 'promoted-to-next-power'", {"patch": "+    return 'promoted-to-next-power'\n"},
     "patch_line"),
    ("assert intword(999999, [1000, 1000000]) == 'next'", {"official_tests": [
        "def test_it():\n    assert intword(999999, [1000, 1000000]) == 'next'\n"]},
     "official_test_line"),
    ("see 'one point zero million'", {"official_tests": [
        "def test_it():\n    assert f() == 'one point zero million'\n"]}, "expected_literal"),
    ("Fix intword promotion at power boundaries", {
        "commit_message": "Fix intword promotion at power boundaries"}, "commit_message"),
])
def test_other_leaks_remain_refused_even_when_quoted_in_the_issue(injected, extra, reason):
    result = leak.scan(_prompt(injected), _verifier(**extra))
    assert result["ok"] is False and any(r.startswith(reason) for r in result["reasons"])


def test_malformed_and_seal_mismatched_input_refuses():
    good = _prompt()
    assert leak.scan({**good, "prompt_sha256": "0" * 64}, _verifier())["reasons"] == \
        ["prompt does not match its seal"]
    for bad in ({}, {"prompt": None, "prompt_sha256": "x"}, {"prompt": "x"}, None):
        assert leak.scan(bad, _verifier())["ok"] is False
    assert leak.scan(good, None)["ok"] is False
