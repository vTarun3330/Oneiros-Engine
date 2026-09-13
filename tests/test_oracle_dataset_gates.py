"""Each gate defect, one test per refusal condition.

These cover the three chains the dataset build depends on: the receipt binds to
the bytes actually read, the labelling code is the code that produced the
outcomes, and recorded fields are used verbatim rather than reconstructed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.corpus import sha256_file
from harness.oracle_dataset_gates import (
    CONTRACT_SOURCES, EXPECTED_RECEIPT_SCHEMA, OUTCOME_FIELDS,
    assert_no_source_drift, assert_receipt, outcome_fields, verify_raw_output,
)
from harness.oracle_labels import SYNTAX_OR_POLICY_INVALID, classify_candidate
from harness.unique_first_balance import (
    MIN_DENOMINATOR_FOR_SHARE_CAPS, select, weights,
)

ROOT = Path(__file__).resolve().parent.parent


def _files(tmp_path):
    original = tmp_path / "orig.json"
    original.write_text("{}", encoding="utf-8")
    derived = tmp_path / "derived.json"
    derived.write_text('{"x": 1}', encoding="utf-8")
    return derived, original


def _receipt(derived, original, **over):
    body = {
        "schema_version": EXPECTED_RECEIPT_SCHEMA,
        "verified": True, "evaluation_split": "train",
        "artifact_sha256": hashlib.sha256(derived.read_bytes()).hexdigest(),
        "parent_artifact_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "execution_harness_failures": 0, "raw_output_hash_mismatches": 0,
        "prompt_budget_failed_functions": 0,
        "completion_limit_threshold_exceeded": False,
        "ineligible_ceiling_exceeded": False,
        "execution_harness_ceiling_exceeded": False,
    }
    body.update(over)
    return body


# --------------------------------------------------- receipt refusal matrix

@pytest.mark.parametrize("override,fragment", [
    ({"schema_version": "something_else"}, "schema"),
    ({"verified": False}, "not verified"),
    ({"verified": None}, "not verified"),
    ({"evaluation_split": "val"}, "not train"),
    ({"artifact_sha256": None}, "no artifact_sha256"),
    ({"artifact_sha256": "0" * 64}, "describes"),
    ({"execution_harness_failures": 5}, "execution_harness_failures"),
    ({"execution_harness_failures": None}, "does not record"),
    ({"raw_output_hash_mismatches": 1}, "raw_output_hash_mismatches"),
    ({"prompt_budget_failed_functions": 2}, "prompt_budget_failed_functions"),
    ({"completion_limit_threshold_exceeded": True}, "completion_limit"),
    ({"ineligible_ceiling_exceeded": True}, "ineligible_ceiling"),
    ({"execution_harness_ceiling_exceeded": True}, "execution_harness_ceiling"),
    ({"parent_artifact_sha256": None}, "parent_artifact_sha256"),
    ({"parent_artifact_sha256": "0" * 64}, "parent chain"),
])
def test_every_receipt_condition_is_refused(tmp_path, override, fragment):
    derived, original = _files(tmp_path)
    problems = assert_receipt(_receipt(derived, original, **override),
                              derived, original)
    assert any(fragment in p for p in problems), problems


def test_a_clean_receipt_raises_nothing(tmp_path):
    derived, original = _files(tmp_path)
    assert assert_receipt(_receipt(derived, original), derived, original) == []


def test_the_receipt_is_bound_to_bytes_not_a_path(tmp_path):
    """A receipt naming a path cannot detect a different file at that path."""
    derived, original = _files(tmp_path)
    receipt = _receipt(derived, original)
    derived.write_text('{"x": 2}', encoding="utf-8")     # same path, new bytes
    problems = assert_receipt(receipt, derived, original)
    assert any("describes" in p for p in problems)


# ------------------------------------------------------------- source drift

@pytest.mark.parametrize("field,relative", sorted(CONTRACT_SOURCES.items()))
def test_drift_in_any_contract_source_is_refused(field, relative):
    contract = {f: sha256_file(ROOT / r) for f, r in CONTRACT_SOURCES.items()}
    contract[field] = "0" * 64
    assert any(relative in p for p in assert_no_source_drift(contract, ROOT))


def test_matching_sources_do_not_drift():
    contract = {f: sha256_file(ROOT / r) for f, r in CONTRACT_SOURCES.items()}
    assert assert_no_source_drift(contract, ROOT) == []


def test_a_contract_missing_its_hashes_is_refused():
    assert len(assert_no_source_drift({}, ROOT)) == len(CONTRACT_SOURCES)


# ------------------------------------------------- exact recorded outcomes

def test_policy_invalid_with_a_non_null_shape_stays_invalid():
    """The replaced bug: the policy returns an EMPTY shape on rejection, so
    `shape is not None` was true for every rejected candidate."""
    outcome = {"parse_valid": True, "policy_valid": False,
               "candidate_shape": "", "policy_error": "imports not permitted",
               "reference_status": "pass", "reference_valid": True,
               "killed": True}
    fields = outcome_fields(outcome)
    assert fields["policy_valid"] is False
    assert classify_candidate({**outcome, **fields})["label"] \
        == SYNTAX_OR_POLICY_INVALID


def test_even_a_populated_shape_does_not_imply_policy_valid():
    outcome = {"parse_valid": True, "policy_valid": False,
               "candidate_shape": "assertion", "reference_status": "pass",
               "reference_valid": True, "killed": True}
    assert classify_candidate({**outcome, **outcome_fields(outcome)})["label"] \
        == SYNTAX_OR_POLICY_INVALID


def test_all_required_fields_are_carried_verbatim():
    outcome = {field: f"v_{field}" for field in OUTCOME_FIELDS}
    assert outcome_fields(outcome) == outcome


def test_a_missing_field_stays_none_rather_than_false():
    """'the artifact never said' must not become 'the artifact said no'."""
    assert outcome_fields({})["policy_valid"] is None
    assert outcome_fields({})["killed"] is None


def test_a_raw_output_hash_mismatch_is_detected():
    text = "assert f(1) == 1"
    good = {"raw_output": text,
            "raw_output_sha256": hashlib.sha256(text.encode()).hexdigest()}
    assert verify_raw_output(good) is None
    assert "does not match" in verify_raw_output(dict(good, raw_output="assert f(1) == 2"))
    assert "no raw output" in verify_raw_output({"raw_output_sha256": "x"})
    assert "no recorded" in verify_raw_output({"raw_output": "x"})


# --------------------------------------------------------- balance selector

def _row(i, source="mbpp", family="arith", lineage=None, target=None):
    return {"i": i, "source_dataset": source, "bug_family": family,
            "function_lineage": lineage or f"L{i}",
            "complexity_tier": "simple", "supervision_role": "positive_original",
            "target": target or f"t{i}"}


CAPS = {"source_dataset": 0.70, "bug_family": 0.35}


def _select(rows, max_per_lineage=4):
    return select(rows, target_of=lambda r: r["target"],
                  lineage_key="function_lineage",
                  max_per_lineage=max_per_lineage, share_caps=CAPS)


def test_final_shares_actually_satisfy_the_caps():
    """The replaced selector could report a cap beside a share exceeding it."""
    rows = [_row(i, source="mbpp", family="arith") for i in range(200)]
    rows += [_row(1000 + i, source="humaneval", family=f"f{i % 5}")
             for i in range(200)]
    result = _select(rows)
    for dimension, cap in CAPS.items():
        for value, share in result["final_shares"][dimension].items():
            assert share <= cap + 1e-9, (dimension, value, share)
    assert result["caps_satisfied"] is True
    assert result["cap_violations"] == {}


def test_caps_satisfied_is_false_when_a_cap_cannot_be_met():
    """Honest reporting beats a silently relaxed cap."""
    rows = [_row(i, source="mbpp", family="arith") for i in range(10)]
    result = _select(rows)
    # Below the share-cap denominator, so caps cannot bind; the report must not
    # claim they were satisfied by pretending the shares are different.
    assert result["final_shares"]["source_dataset"]["mbpp"] == 1.0
    assert len(result["selected"]) <= MIN_DENOMINATOR_FOR_SHARE_CAPS


def test_nothing_is_ever_duplicated():
    rows = [_row(i) for i in range(60)]
    result = _select(rows)
    targets = [r["target"] for r in result["selected"]]
    assert len(targets) == len(set(targets))


def test_exact_duplicate_targets_collapse_to_one():
    rows = [_row(i, target="same") for i in range(8)]
    result = _select(rows)
    assert len(result["selected"]) == 1
    assert result["dropped"]["exact_duplicate_target"] == 7


def test_the_lineage_cap_binds():
    rows = [_row(i, lineage="ONE") for i in range(30)]
    result = _select(rows, max_per_lineage=3)
    assert result["max_rows_from_one_lineage"] <= 3


def test_a_scarce_stratum_is_preserved_not_padded():
    rows = [_row(i, family="common") for i in range(80)]
    rows += [_row(999, family="rare", target="rare")]
    result = _select(rows)
    families = [r["bug_family"] for r in result["selected"]]
    assert families.count("rare") == 1


def test_weights_are_metadata_across_every_requested_dimension():
    rows = [_row(i, family="common") for i in range(30)]
    rows += [_row(99, family="rare", target="rare")]
    picked = _select(rows)["selected"]
    table = weights(picked, ("source_dataset", "bug_family",
                             "complexity_tier", "supervision_role"))
    assert set(table) == {"source_dataset", "bug_family", "complexity_tier",
                          "supervision_role"}
    if "rare" in table["bug_family"] and "common" in table["bug_family"]:
        assert table["bug_family"]["rare"] > table["bug_family"]["common"]


def test_selection_is_deterministic():
    rows = [_row(i, family=f"f{i % 7}", source=f"s{i % 3}") for i in range(120)]
    first = [r["target"] for r in _select(rows)["selected"]]
    second = [r["target"] for r in _select(list(reversed(rows)))["selected"]]
    assert first == second


def test_the_report_states_what_was_achieved_not_intended():
    rows = [_row(i, family=f"f{i % 4}") for i in range(100)]
    result = _select(rows)
    assert set(result) >= {"final_shares", "cap_violations", "caps_satisfied",
                           "share_caps", "enforcement_passes", "dropped"}
    assert "achieved" in result["method"]
