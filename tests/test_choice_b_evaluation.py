"""Choice B v2 evaluation integrity, statistics language and near-duplicate audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness import choice_b_evaluation as ev
from harness.choice_b_neardup import anonymised_ast, audit

ROOT = Path(__file__).resolve().parent.parent
FUNCTIONS = [{"record_id": f"r{i}", "group_id": f"g{i}",
              "items": [{"input_kind": k, "item_id": f"r{i}::{k}", "call": f"f({i})",
                         "expected_repr": str(i)} for k in ev.INPUT_KINDS]}
             for i in range(3)]


def _line(key, sha="c" * 64, output="1"):
    return json.dumps({"key": key, "contract_sha256": sha, "output": output}) + "\n"


def test_expected_keys_cover_groups_inputs_and_schemas_exactly():
    keys = ev.expected_fixed_keys(FUNCTIONS)
    assert len(keys) == len(set(keys)) == 3 * 2 * 2
    bad = [{**FUNCTIONS[0], "items": FUNCTIONS[0]["items"][:1]}]
    with pytest.raises(ev.IntegrityError):
        ev.expected_fixed_keys(bad)


@pytest.mark.parametrize("content, problem", [
    (lambda k: _line(k[0]) + _line(k[0]), "duplicate key"),
    (lambda k: _line(k[0]) + '{"key": ', "partial final line"),
    (lambda k: _line(k[0]) + "not json\n", "malformed line"),
    (lambda k: _line(k[0], sha="d" * 64), "different contract"),
    (lambda k: _line("r9::upstream::answer_schema"), "unexpected key"),
])
def test_resumable_file_problems_are_detected_not_merged(tmp_path, content, problem):
    keys = ev.expected_fixed_keys(FUNCTIONS)
    path = tmp_path / "fixed_input.jsonl"
    path.write_text(content(keys), encoding="utf-8")
    rows, problems = ev.read_generation_file(path, "c" * 64, keys)
    assert any(problem in p for p in problems)
    assert len(rows) <= 1        # the duplicate never overwrote the first row
    moved = ev.quarantine(path, problems, tmp_path / "resume_log.jsonl")
    assert moved.exists() and not path.exists()
    assert json.loads((tmp_path / "resume_log.jsonl").read_text().splitlines()[0])["reasons"]


def test_complete_key_set_is_required():
    keys = ev.expected_fixed_keys(FUNCTIONS)
    rows = {k: {} for k in keys}
    ev.require_complete(rows, keys)
    with pytest.raises(ev.IntegrityError):
        ev.require_complete({k: {} for k in keys[:-1]}, keys)
    with pytest.raises(ev.IntegrityError):
        ev.require_complete({**rows, "extra::key": {}}, keys)


def test_kill_rows_must_match_exactly_without_intersection():
    rows = [{"record_id": "r0"}, {"record_id": "r1"}]
    assert set(ev.require_function_rows(rows, ["r0", "r1"])) == {"r0", "r1"}
    for bad in ([{"record_id": "r0"}], rows + [{"record_id": "r1"}],
                rows + [{"record_id": "r2"}]):
        with pytest.raises(ev.IntegrityError):
            ev.require_function_rows(bad, ["r0", "r1"])


def test_function_metrics_cover_every_promised_candidate_metric():
    row = {"record_id": "r0", "diversity": {"exact_unique_ratio": 0.5}, "candidate_outcomes": [
        {"rank": 1, "parse_valid": True, "execution_valid": True, "reference_valid": True,
         "killed": False, "code": "assert f(1) == 2"},
        {"rank": 3, "parse_valid": True, "execution_valid": True, "reference_valid": True,
         "killed": True, "code": "assert f(2) in (1, 2)"},
        {"rank": 6, "parse_valid": False, "execution_valid": False, "reference_valid": False,
         "killed": False, "code": None}, {"rank": 8, "parse_valid": True,
                                          "execution_valid": False, "reference_valid": False,
                                          "killed": False, "code": "x"}]}
    m = ev.function_metrics(ev.compact_function(row))
    assert (m["kill_at_1"], m["kill_at_4"], m["kill_at_8"]) == (0.0, 1.0, 1.0)
    assert m["parse_success"] == 0.75 and m["execution_success"] == 0.5
    assert m["reference_validity_per_requested"] == 0.5
    assert m["exact_equality_validity_per_requested"] == 0.25
    assert m["has_valid_killing_candidate"] == 1.0 and m["exact_unique_ratio"] == 0.5
    assert ev.is_exact_equality("assert a == b") and not ev.is_exact_equality("assert a < b")


def _interval(point, low, high):
    return {"point": point, "low": low, "high": high}


def test_outcome_precedence_separates_statistics_from_operational_stops():
    ok = {"answer_rate": ev.guardrail("answer_rate_points", _interval(0.5, -1.0, 2.0))}
    good = {"a": _interval(6, 1, 11), "b": _interval(4, 0.5, 8)}
    assert ev.classify(good, ok)["outcome"] == "promising"
    adverse = ev.classify({"a": _interval(-6, -11, -1), "b": _interval(2, -2, 6)}, ok)
    assert adverse["outcome"] == "statistically_supported_adverse"
    point_only = {"validity": ev.guardrail("reference_validity_points",
                                           _interval(-3.5, -8.0, 1.0))}
    stop = ev.classify(good, point_only)
    assert stop["outcome"] == "operational_guardrail_stop"
    assert stop["legacy_outcome"] == "harm" and "point_beyond_threshold" in stop["reason_codes"][0]
    supported = {"validity": ev.guardrail("reference_validity_points", _interval(-6, -9, -4))}
    assert ev.classify(good, supported)["outcome"] == "statistically_supported_adverse"
    flip = ev.classify({"a": _interval(3, -1, 7), "b": _interval(-1, -5, 3)}, ok)
    assert flip["outcome"] == "inconclusive_power" and flip["reason_codes"] == ["schema_sign_flip"]


def test_paired_and_relative_bootstraps_are_deterministic():
    groups = [f"g{i // 2}" for i in range(100)]
    c = [float(i % 3 == 0) for i in range(100)]
    t = [1.0 if i % 7 == 0 else x for i, x in enumerate(c)]
    assert ev.paired(groups, c, t) == ev.paired(groups, c, t)
    rel = ev.relative_change_bootstrap(groups, [0.5] * 100, [0.45] * 100)
    assert rel["point"] == pytest.approx(-0.1) and rel["low"] <= rel["point"] <= rel["high"]


def test_frozen_spec_v2_states_the_corrected_language():
    spec = json.loads((ROOT / "results/sft_root_cause_phase4_choice_b_evaluation_spec_v2.json")
                      .read_text(encoding="utf-8"))
    assert "OPERATIONAL SAFETY THRESHOLDS" in spec["guardrails"]["status"]
    assert "not a scalar gradient-dose" in spec["estimand"]
    assert spec["outcomes"]["precedence"] == list(ev.OUTCOMES)
    assert "not impossible" in spec["power_at_gate"]["reading"]
    assert spec["frozen_before_any_choice_b_outcome"] is True
    assert set(ev.SLICES) <= set(spec["descriptive_slices"])
    power = ROOT / spec["power_at_gate"]["path"]
    assert hashlib.sha256(power.read_bytes()).hexdigest() == spec["power_at_gate"]["sha256"]


def test_near_duplicate_audit_flags_clones_and_leaves_distinct_code():
    gate = [{"id": "g1", "group_id": "G", "reference_code": "def f(a):\n    return a + 1\n",
             "specification": "Add one to the given integer and return the result please now"}]
    train = [{"id": "t1", "group_id": "T1",
              "reference_code": "def g(x):\n    '''doc'''\n    return x + 1\n",
              "specification": "unrelated text"},
             {"id": "t2", "group_id": "T2",
              "reference_code": "def h(s):\n    return s[::-1]\n", "specification": "reverse"}]
    result = audit(train, gate)
    assert result["excluded_training_groups"] == ["T1"]
    assert anonymised_ast("def f(a): return a") == anonymised_ast("def q(z): return z")


def test_frozen_split_v3_excludes_audited_groups_and_keeps_the_gate():
    split = json.loads((ROOT / "results/sft_root_cause_phase4_choice_b_split_v3.json")
                       .read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / "results/sft_root_cause_phase4_choice_b_split_v2.json")
                    .read_text(encoding="utf-8"))
    audit_file = json.loads((ROOT / "results/sft_root_cause_phase4_choice_b_neardup_audit_v1.json")
                            .read_text(encoding="utf-8"))
    excluded = set(audit_file["excluded_training_groups"])
    assert excluded and not excluded & {r["group_id"] for r in split["rows"]}
    assert split["gate"] == v2["gate"]
    assert len(split["rows"]) == len(v2["rows"]) - audit_file["training_rows_removed"]
    assert audit_file["outcome_blind"] is True
