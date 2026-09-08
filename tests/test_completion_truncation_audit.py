"""Pin the truncation audit, including the artifact that first fooled it.

The first run of this audit reported 4207 of 4336 base completions as
unparseable, against a measured kill@8 of 0.5959. Both cannot be true. The
cause was in the audit, not the model: it parsed the raw text without
unwrapping the markdown fence a chat model puts around code, so it was
measuring its own omission.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_completion_truncation import _drop_last_line, _parses, analyse


def _artifact(tmp_path: Path, raws: list[str], split: str = "ablation_dev",
              sealed: bool = False) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({
        "evaluation_split": split,
        "final_test_measurement": sealed,
        "function_results": [{
            "record_id": "r",
            "candidate_outcomes": [{"raw_output": raw} for raw in raws],
        }],
    }), encoding="utf-8")
    return path


def test_a_fenced_completion_is_not_counted_as_unparseable(tmp_path):
    """The defect that produced 4207 of 4336 unparseable for the base model."""
    fenced = "```python\nassert clamp(1, 0, 10) == 1\n```"
    report = analyse(_artifact(tmp_path, [fenced]), 128,
                     "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    assert report["unparseable_completions"] == 0
    assert report["assertions_present"] == 1


def test_a_truncated_test_function_is_recognised_as_truncation(tmp_path):
    """Parses once the trailing partial line goes: the signature of a cut."""
    cut = ("def test_clamp():\n"
           "    assert clamp(1, 0, 10) == 1\n"
           "    assert clamp(2, 0, 1")
    report = analyse(_artifact(tmp_path, [cut]), 128,
                     "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    assert report["unparseable_completions"] == 1
    assert report["unparseable_that_parse_after_dropping_the_last_line"] == 1
    assert report["assertions_recoverable_from_truncated_completions"] == 1


def test_genuinely_malformed_output_is_not_called_truncation(tmp_path):
    """A bigger budget would not fix this, so it must not argue for one."""
    broken = "def ((( not python at all"
    report = analyse(_artifact(tmp_path, [broken]), 128,
                     "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    assert report["unparseable_completions"] == 1
    assert report["unparseable_that_parse_after_dropping_the_last_line"] == 0


def test_sealed_data_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        analyse(_artifact(tmp_path, ["assert f(1) == 1"], sealed=True), 128,
                "Qwen/Qwen2.5-Coder-1.5B-Instruct")


def test_a_test_split_artifact_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        analyse(_artifact(tmp_path, ["assert f(1) == 1"], split="test"), 128,
                "Qwen/Qwen2.5-Coder-1.5B-Instruct")


def test_dropping_the_last_line_of_a_single_line_yields_nothing():
    assert _drop_last_line("assert f(1) == 1") == ""
    assert _parses("assert f(1) == 1")


COMMITTED = ROOT / "results" / "v4_2_completion_truncation_relearn.json"


def test_the_committed_audit_supports_the_budget_argument():
    """Both conditions of the verdict rule, so neither is quoted alone."""
    if not COMMITTED.exists():
        return
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert report["share_at_or_over_the_limit"] > 0.10, (
        "the case for raising the completion budget rests on a material share "
        "of completions reaching it"
    )
    rescued = report["unparseable_that_parse_after_dropping_the_last_line"]
    assert rescued > 0.5 * report["unparseable_completions"], (
        "most unparseable completions must be truncations rather than bad "
        "generation, or a larger budget fixes nothing"
    )
