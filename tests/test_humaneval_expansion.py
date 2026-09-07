"""Pin the HumanEval expansion, especially the invocation it depends on.

The corpus is grown rather than trimmed, so every new record has to earn its
place by execution. The first version of this expansion verified nothing at
all: HumanEval ships tests as `def check(candidate): ...` with no call, so
running one defines a name, asserts nothing, and passes on the reference AND
the mutant. It reported zero verified mutants out of nineteen and the reason
was the harness, not the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.expand_humaneval_coverage import _problem_number, runnable, verify

REFERENCE = "def f(x):\n    return x + 1\n"
MUTANT = "def f(x):\n    return x + 2\n"
CHECK_TEST = "def check(candidate):\n    assert candidate(1) == 2\n"
KILLING = ["assert f(1) == 2"]


def test_a_check_style_test_is_invoked():
    """Without the appended call nothing runs and no mutant is detectable."""
    made = runnable(CHECK_TEST, "f")
    assert made.rstrip().endswith("check(f)")
    assert "assert candidate(1) == 2" in made


def test_a_test_that_already_calls_check_is_not_called_twice():
    already = CHECK_TEST + "check(f)\n"
    assert runnable(already, "f").count("check(f)") == 1


def test_a_plain_assertion_is_left_alone():
    assert runnable("assert f(1) == 2", "f") == "assert f(1) == 2"


def test_a_detectable_mutant_verifies():
    labelled, evidence = verify(REFERENCE, MUTANT, KILLING)
    assert evidence["verified"] is True
    assert evidence["killing_tests"] == 1
    assert evidence["reference_valid_tests"] == 1
    assert labelled[0]["kills"] is True


def test_an_undetectable_mutant_is_rejected():
    """A mutant no test distinguishes is a rewrite, not a defect."""
    equivalent = "def f(x):\n    y = x\n    return y + 1\n"
    labelled, evidence = verify(REFERENCE, equivalent, KILLING)
    assert evidence["verified"] is False
    assert evidence["killing_tests"] == 0
    assert labelled and labelled[0]["kills"] is False, (
        "a non-distinguishing assertion is still a valid test and is retained, "
        "just labelled as not distinguishing"
    )


def test_a_test_failing_on_the_reference_is_not_counted():
    """It would flag correct code as buggy, so it cannot certify a mutant."""
    labelled, evidence = verify(REFERENCE, MUTANT, ["assert f(1) == 99"])
    assert evidence["reference_valid_tests"] == 0
    assert evidence["reference_invalid_excluded"] == 1
    assert evidence["verified"] is False
    assert labelled == []


def test_problem_numbers_are_parsed_from_record_ids():
    assert _problem_number("mutation::humaneval_HumanEval_143_mut_01387") == 143
    assert _problem_number("official-repository::django::1::x") is None


def test_expansion_records_never_reuse_an_existing_problem(tmp_path):
    """The leakage guarantee must be checked against the corpus, not assumed."""
    from scripts.expand_humaneval_coverage import used_problem_ids
    used = used_problem_ids()
    assert used, "the corpus should report the problems it already contains"
    import json
    staged = ROOT / "data" / "humaneval_expansion_staging" / "records.json"
    if not staged.exists():
        return
    report = json.loads(staged.read_text(encoding="utf-8"))
    import re
    for record in report["records"]:
        upstream = record["provenance"]["upstream_record_id"]
        number = int(re.search(r"(\d+)", upstream).group(1))
        assert number not in used, (
            f"HumanEval_{number} is already in {used[number]}; adding it to "
            "train would leak a held-out problem into training"
        )
