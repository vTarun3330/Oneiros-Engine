"""Pin the stratification, because a bug in it produced a false headline.

A quick inline version of this sorted records into "states a killing value"
and "everything else". That swept records whose docstring examples cannot be
verified against the reference into the clean stratum. Those records kill at
0.22, they dragged the clean average from 0.7381 down to 0.5833, and the
conclusion drawn from it - that humaneval and mbpp are equally hard once the
giveaway is removed - was wrong by about sixteen points.

The three-way split is therefore the point of this module, and the test below
reconstructs that exact failure so it cannot come back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measure_prompt_giveaway_effect import _states_a_killing_value, stratify

REFERENCE = "def f(x):\n    return x + 1\n"
MUTANT = "def f(x):\n    return x + 2\n"


def _record(record_id: str, specification: str, mutant: str = MUTANT) -> dict:
    return {
        "id": record_id, "entry_point": "f", "specification": specification,
        "reference_code": REFERENCE, "code_under_test": mutant,
        "support_context": "",
    }


def test_a_prompt_stating_a_value_the_mutant_fails_is_a_giveaway():
    record = _record("r", "Adds one.\n>>> f(1)\n2\n")
    assert _states_a_killing_value(record, 5.0) is True


def test_a_prompt_whose_stated_value_the_mutant_also_produces_is_not():
    equivalent = "def f(x):\n    y = x\n    return y + 1\n"
    record = _record("r", "Adds one.\n>>> f(1)\n2\n", mutant=equivalent)
    assert _states_a_killing_value(record, 5.0) is False


def test_a_prompt_with_no_example_is_not_a_giveaway():
    assert _states_a_killing_value(_record("r", "Adds one."), 5.0) is False


def test_an_example_that_does_not_run_on_the_reference_is_unverifiable():
    """Not a giveaway and NOT clean; conflating it with clean was the bug."""
    record = _record("r", "Adds one.\n>>> f(undefined_name)\n2\n")
    assert _states_a_killing_value(record, 5.0) is None


def _artifact(tmp_path: Path, rows: list[tuple[str, str, bool]]) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    records = [_record(rid, spec) for rid, spec, _ in rows]
    (corpus / "records.json").write_text(json.dumps(records), encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({
        "evaluation_split": "ablation_dev",
        "final_test_measurement": False,
        "function_results": [
            {"record_id": rid, "dataset_name": "humaneval", "killed": killed}
            for rid, _, killed in rows
        ],
    }), encoding="utf-8")
    return artifact, corpus


def test_unverifiable_records_are_kept_out_of_the_clean_stratum(tmp_path):
    """The exact bug: pooling these two produced a false equality with mbpp."""
    artifact, corpus = _artifact(tmp_path, [
        ("clean1", "Adds one.", True),
        ("clean2", "Adds one.", True),
        ("bad1", "Adds one.\n>>> f(undefined_name)\n2\n", False),
        ("bad2", "Adds one.\n>>> f(undefined_name)\n2\n", False),
    ])

    report = stratify(artifact, corpus, 5.0)
    strata = report["strata"]

    assert strata["humaneval_no_giveaway"]["functions"] == 2
    assert strata["humaneval_no_giveaway"]["kill_rate"] == 1.0, (
        "the clean stratum must not be diluted by records whose examples "
        "could not be verified; pooling them is what produced 0.5833 from a "
        "true 0.7381"
    )
    assert strata["humaneval_unverifiable_example"]["functions"] == 2
    assert strata["humaneval_unverifiable_example"]["kill_rate"] == 0.0


def test_the_giveaway_stratum_is_separated(tmp_path):
    artifact, corpus = _artifact(tmp_path, [
        ("give1", "Adds one.\n>>> f(1)\n2\n", True),
        ("clean1", "Adds one.", False),
    ])

    strata = stratify(artifact, corpus, 5.0)["strata"]

    assert strata["humaneval_prompt_states_a_killing_value"]["functions"] == 1
    assert strata["humaneval_no_giveaway"]["functions"] == 1


def test_sealed_data_is_refused(tmp_path):
    artifact, corpus = _artifact(tmp_path, [("r", "Adds one.", True)])
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["final_test_measurement"] = True
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit):
        stratify(artifact, corpus, 5.0)


COMMITTED = ROOT / "results" / "v4_2_prompt_giveaway_base_adev.json"


def test_the_committed_finding_reports_a_modest_giveaway_advantage():
    """Guards against the overclaim, not just against the undercount."""
    if not COMMITTED.exists():
        return
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert report["giveaway_advantage_points"] < 10.0, (
        "the giveaway advantage has grown past the measured +3.11 points; the "
        "writeup states it is small and must be re-derived before it is cited"
    )
    assert report["strata"]["humaneval_unverifiable_example"]["functions"] > 0, (
        "the unverifiable stratum has vanished, which is how the clean number "
        "was overstated the first time"
    )
