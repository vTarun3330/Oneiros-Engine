"""Pin the specification-quality measurement and its sealed-split refusal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measure_specification_quality import WORKED_EXAMPLE, measure

REPORT = ROOT / "results" / "v4_2_specification_quality.json"


@pytest.mark.parametrize("text", [
    "Return the sum.\n>>> add(1, 2)\n3",
    "add([1,2]) => 3",
    "For example: add(1,2) returns 3",
    "Examples:\nadd(1, 2) == 3",
])
def test_worked_examples_are_recognised(text):
    assert WORKED_EXAMPLE.search(text)


@pytest.mark.parametrize("text", [
    "Write a function to find the demlo number for the given number.",
    "Write a function to extract maximum and minimum k elements in the given tuple.",
    "This is an example of a poorly specified task.",
])
def test_prose_without_an_input_output_pair_is_not_an_example(text):
    assert not WORKED_EXAMPLE.search(text), (
        "counting a bare mention of the word example would erase the very gap "
        "this measurement exists to show"
    )


def test_the_sealed_split_cannot_be_profiled():
    from scripts import measure_specification_quality as module
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(module.__file__)), "--splits", "test"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "sealed" in (result.stdout + result.stderr).lower()


def test_the_measurement_reads_specifications_not_code(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "records.json").write_text(json.dumps([
        {"id": "mutation::mbpp_1_mut_1", "specification": "Write a function.",
         "reference_code": "def f():\n    '''>>> f()'''\n"},
        {"id": "mutation::humaneval_1_mut_1",
         "specification": "Do a thing.\n>>> f(1)\n2", "reference_code": ""},
    ]), encoding="utf-8")
    (corpus / "splits.json").write_text(json.dumps({
        "train": ["mutation::mbpp_1_mut_1", "mutation::humaneval_1_mut_1"],
    }), encoding="utf-8")

    report = measure(corpus, ("train",))["by_split"]["train"]
    assert report["mbpp"]["worked_example_rate"] == 0.0, (
        "an example hiding in reference_code is not visible to the model in "
        "the specification, and must not be counted as one"
    )
    assert report["humaneval"]["worked_example_rate"] == 1.0


def test_the_committed_report_shows_mbpp_has_no_worked_examples():
    """The finding the next phase of work is planned against."""
    if not REPORT.exists():
        return
    by_split = json.loads(REPORT.read_text(encoding="utf-8"))["by_split"]
    assert "test" not in by_split, "the sealed split leaked into the report"
    for split, benchmarks in by_split.items():
        if "mbpp" not in benchmarks:
            continue
        assert benchmarks["mbpp"]["worked_example_rate"] == 0.0, (
            f"mbpp in {split} now has worked examples; the specification-gap "
            "diagnosis needs re-deriving before it is cited again"
        )
        assert benchmarks["humaneval"]["worked_example_rate"] > 0.5
