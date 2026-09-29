"""CPU dry run of the frozen Choice B v2 analysis on real gate functions with SYNTHETIC
outputs (no model, no GPU, no Choice B outcome)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data/corpus/v4_1_research_hardened_candidate/development_view/train.records.json"


def _synthetic_look(functions, correct: bool, killed: bool):
    from harness import choice_b_evaluation as ev
    outputs = {}
    for f in functions:
        for item in f["items"]:
            value = item["expected_repr"] if correct else "None"
            outputs[f"{item['item_id']}::prefill_assertion"] = value
            outputs[f"{item['item_id']}::answer_schema"] = f"ANSWER: {value}"
    kill = [ev.compact_function({"record_id": f["record_id"],
                                 "diversity": {"exact_unique_ratio": 0.75},
                                 "candidate_outcomes": [
                                     {"rank": r, "parse_valid": True, "execution_valid": True,
                                      "reference_valid": True, "killed": killed and r == 2,
                                      "code": f"assert x == {r}"} for r in range(1, 9)]})
            for f in functions]
    return {"evidence": {"fixed_input_outputs": outputs, "kill_at_8_functions": kill}}


@pytest.mark.skipif(not TRAIN.exists(), reason="development view not materialised")
def test_frozen_analysis_runs_end_to_end_on_synthetic_evidence():
    from scripts.phase4_choice_b_v2 import analysis_from_evidence
    split = json.loads((ROOT / "results/sft_root_cause_phase4_choice_b_split_v3.json")
                       .read_text(encoding="utf-8"))
    functions = split["gate"]["functions"][:6]
    mini = {"gate": {"functions": functions}}
    records = {r["id"]: r for r in json.loads(TRAIN.read_text(encoding="utf-8"))
               if r["id"] in {f["record_id"] for f in functions}}
    looks = {"control": _synthetic_look(functions, correct=False, killed=False),
             "treatment": _synthetic_look(functions, correct=True, killed=True)}
    result = analysis_from_evidence(mini, looks, records, {})
    for schema in ("prefill_assertion", "answer_schema"):
        accuracy = result["primary_fixed_input"][schema]["accuracy"]
        assert accuracy["control"] == 0.0 and accuracy["treatment"] == 100.0
        assert result["primary_fixed_input"][schema]["nonanswer_rate"]["control"] == 0.0
    assert result["downstream_kill_at_8"]["kill_at_8"]["treatment"] == 100.0
    assert result["decision"]["outcome"] in ("promising", "inconclusive_power")
    assert len(result["per_item_evidence"]) == 6 * 4
    assert set(result["descriptive_slices"]) >= {"dataset", "input_kind", "complexity_tier",
                                                 "bug_family", "status"}
    # A missing key refuses instead of analysing a subset.
    broken = json.loads(json.dumps(looks))
    broken["treatment"]["evidence"]["fixed_input_outputs"].popitem()
    with pytest.raises(Exception, match="incomplete"):
        analysis_from_evidence(mini, broken, records, {})
