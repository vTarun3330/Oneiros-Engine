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


# --- the stated value must match the reference ----------------------------
#
# The giveaway check compared the reference against the mutant on the
# example's INPUT, and never checked that the example's stated OUTPUT is what
# the reference actually produces. A prompt stating a wrong value was
# therefore counted as handing the model a killing assertion, when copying it
# would fail on correct code and kill nothing.
#
# Re-measured after the fix, ablation_dev humaneval was UNCHANGED: 0 examples
# state a value the reference does not produce, so all 52 giveaway records and
# the +3.11 / +6.50 advantages stand. The defect was real and its impact here
# was nil, which is worth pinning in both directions.

def test_a_prompt_stating_a_wrong_value_is_not_a_giveaway():
    record = _record("r", "Adds one.\n>>> f(0)\n999\n")
    assert _states_a_killing_value(record, 5.0) is False, (
        "asserting the stated pair would FAIL on correct code, so it is a "
        "wrong example rather than a handed-over killing assertion"
    )


def test_a_prompt_stating_the_correct_value_is_still_a_giveaway():
    record = _record("r", "Adds one.\n>>> f(1)\n2\n")
    assert _states_a_killing_value(record, 5.0) is True


def test_stated_values_are_compared_as_values_not_text():
    """"0b11" and '0b11' are the same answer written two ways."""
    from scripts.audit_native_example_leakage import stated_matches_reference

    assert stated_matches_reference('"0b11"', "'0b11'")
    assert stated_matches_reference("[1, 2]", "[1, 2]")
    assert not stated_matches_reference("'0b100'", "'0b11'")


def test_the_committed_audit_reports_the_wrong_value_count():
    """Zero here is a measurement, not an absence of the check."""
    import json
    report = ROOT / "results" / "v4_2_native_example_leakage_adev.json"
    if not report.exists():
        return
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "examples_stating_a_value_the_reference_does_not_produce" in payload, (
        "the corrected audit must report this count even when it is zero, or "
        "a reader cannot tell the check ran"
    )
