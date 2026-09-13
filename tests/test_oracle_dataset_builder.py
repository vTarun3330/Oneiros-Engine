"""The replacement O1 builder, held to what the old one got wrong.

The builder this replaces failed four ways at once: canonical corpus access, a
majority label per function, balancing by duplication, and silently dropping
multi-assertion candidates. Each has a test here that fails if it comes back.
"""
from __future__ import annotations

import builtins
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import scripts.build_oracle_dataset as builder
from harness.oracle_labels import (
    FABRICATED_API, HARNESS_ENVIRONMENT_FAILURE, LABELS, NEVER_POSITIVE,
    SEMANTIC_EXECUTION_ERROR, SYNTAX_OR_POLICY_INVALID, UNCERTAIN,
    VALID_KILLING, VALID_NON_KILLING, WRONG_INPUT, WRONG_ORACLE,
    classify_candidate, needs_call_replay,
)
from scripts.build_oracle_dataset import assert_source_artifact, balance

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"


def _ok(**over):
    base = {"parse_valid": True, "policy_valid": True, "reference_valid": False,
            "killed": False, "reference_status": "pass", "reference_error": ""}
    base.update(over)
    return base


# ------------------------------------------------------------ label taxonomy

def test_every_declared_label_is_reachable():
    produced = {
        classify_candidate(_ok(reference_status="worker_error"))["label"],
        classify_candidate(_ok(parse_valid=False))["label"],
        classify_candidate(_ok(reference_valid=True, killed=True))["label"],
        classify_candidate(_ok(reference_valid=True))["label"],
        classify_candidate(_ok(reference_status="assertion_error"),
                           distinguishing_call=True)["label"],
        classify_candidate(_ok(reference_status="assertion_error"),
                           distinguishing_call=False)["label"],
        classify_candidate(_ok(reference_status="error",
                               reference_error="NameError: no such name"))["label"],
        classify_candidate(_ok(reference_status="error",
                               reference_error="ValueError: bad"))["label"],
        classify_candidate(_ok(reference_status="assertion_error"),
                           distinguishing_call=None)["label"],
    }
    assert produced == set(LABELS)


def test_harness_failure_outranks_everything():
    """A candidate that was never scored has no other readable property."""
    verdict = classify_candidate(_ok(reference_status="worker_error",
                                     reference_valid=True, killed=True))
    assert verdict["label"] == HARNESS_ENVIRONMENT_FAILURE


def test_wrong_input_is_primary_and_wrong_oracle_secondary():
    verdict = classify_candidate(_ok(reference_status="assertion_error"),
                                 distinguishing_call=False)
    assert verdict["label"] == WRONG_INPUT
    assert verdict["secondary_label"] == WRONG_ORACLE
    assert "no expected value could make it kill" in verdict["reason"]


def test_a_distinguishing_call_is_wrong_oracle_only():
    verdict = classify_candidate(_ok(reference_status="assertion_error"),
                                 distinguishing_call=True)
    assert verdict["label"] == WRONG_ORACLE
    assert verdict["secondary_label"] is None


def test_an_unreplayable_assertion_failure_is_uncertain_not_guessed():
    verdict = classify_candidate(_ok(reference_status="assertion_error"),
                                 distinguishing_call=None)
    assert verdict["label"] == UNCERTAIN


@pytest.mark.parametrize("error", [
    "NameError: name 'helper' is not defined",
    "AttributeError: module has no attribute 'thing'",
    "TypeError: f() takes 2 positional arguments but 3 were given",
    "TypeError: f() missing 1 required positional argument: 'b'",
    "TypeError: f() got an unexpected keyword argument 'mode'",
])
def test_interface_failures_are_fabricated_api(error):
    assert classify_candidate(
        _ok(reference_status="error", reference_error=error))["label"] == FABRICATED_API


@pytest.mark.parametrize("error", [
    "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
    "ValueError: invalid literal",
    "IndexError: list index out of range",
    "ZeroDivisionError: division by zero",
])
def test_value_failures_are_semantic_not_fabrication(error):
    assert classify_candidate(
        _ok(reference_status="error", reference_error=error))["label"] \
        == SEMANTIC_EXECUTION_ERROR


def test_the_two_typeerror_kinds_are_separated():
    """Both are TypeError; only one is a fabricated interface."""
    arity = classify_candidate(_ok(reference_status="error",
        reference_error="TypeError: f() takes 1 positional argument but 2 were given"))
    value = classify_candidate(_ok(reference_status="error",
        reference_error="TypeError: unsupported operand type(s)"))
    assert arity["label"] != value["label"]


def test_only_assertion_failures_need_the_expensive_replay():
    assert needs_call_replay(_ok(reference_status="assertion_error")) is True
    assert needs_call_replay(_ok(reference_status="pass", reference_valid=True)) is False
    assert needs_call_replay(_ok(reference_status="worker_error")) is False


# ------------------------------------------------------- positive eligibility

def test_no_failure_label_can_ever_be_a_positive_target():
    for label in (WRONG_INPUT, WRONG_ORACLE, SYNTAX_OR_POLICY_INVALID,
                  FABRICATED_API, SEMANTIC_EXECUTION_ERROR,
                  HARNESS_ENVIRONMENT_FAILURE, UNCERTAIN):
        assert label in NEVER_POSITIVE
    assert VALID_KILLING not in NEVER_POSITIVE
    assert VALID_NON_KILLING not in NEVER_POSITIVE


def test_the_builder_only_marks_two_kinds_of_positive():
    import inspect
    source = inspect.getsource(builder.build)
    assert 'row["supervision_role"] = "positive_original"' in source
    assert '"positive_verified_correction"' in source
    # A positive must come from VALID_KILLING or a verified correction.
    assert 'if row["label"] == VALID_KILLING' in source
    assert 'correction.get("verified")' in source


# --------------------------------------------------------- balance behaviour

def _positive(lineage, family, source="mbpp", target=None, i=0):
    return {"sft_target": target or f"assert f({i}) == {i}",
            "function_lineage": lineage, "bug_family": family,
            "source_dataset": source, "complexity_tier": "simple",
            "origin": "synthetic", "record_id": f"r{i}", "rank": 1,
            "candidate_position": 0, "label": VALID_KILLING}


def test_balancing_never_duplicates_a_row():
    rows = [_positive(f"L{i}", "arith", i=i) for i in range(50)]
    result = balance(rows)
    targets = [r["sft_target"] for r in result["selected"]]
    assert len(targets) == len(set(targets)), "a row was repeated to balance"
    assert "unique-first" in result["balancing_method"]


def test_a_scarce_class_is_left_scarce_rather_than_padded():
    rows = [_positive(f"L{i}", "common", i=i) for i in range(60)]
    rows += [_positive("Lrare", "rare", target="assert rare() == 1", i=999)]
    result = balance(rows)
    families = Counter(r["bug_family"] for r in result["selected"])
    assert families["rare"] == 1, "the scarce class was padded"


def test_the_lineage_cap_drops_surplus_instead_of_repeating():
    rows = [_positive("SAME", "arith", target=f"assert f({i}) == {i}", i=i)
            for i in range(20)]
    result = balance(rows)
    assert result["max_rows_from_one_lineage"] <= builder.MAX_PER_FUNCTION_LINEAGE
    assert result["dropped_by_cap"]["function_lineage_cap"] > 0


def test_identical_targets_are_deduplicated():
    rows = [_positive(f"L{i}", "arith", target="assert f(1) == 1", i=i)
            for i in range(5)]
    result = balance(rows)
    assert len(result["selected"]) == 1
    assert result["dropped_by_cap"]["exact_duplicate_target"] == 4


def test_weights_are_metadata_not_repetition():
    rows = [_positive(f"L{i}", "common", i=i) for i in range(30)]
    rows += [_positive("Lr", "rare", target="assert rare() == 1", i=99)]
    result = balance(rows)
    assert result["sampling_weights_by_bug_family"]["rare"] > \
        result["sampling_weights_by_bug_family"]["common"]
    assert len(result["selected"]) == len({r["sft_target"] for r in result["selected"]})


# ------------------------------------------------------------ source refusal

def _artifact(tmp_path, **over):
    body = {"evaluation_split": "train", "final_test_measurement": False,
            "derived_from_sha256": "deadbeef",
            "run_contract": {"candidate_parse_mode": "whole_output",
                             "retain_raw_output": True},
            "function_results": []}
    body.update(over)
    path = tmp_path / "d.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_legacy_protocol_source_is_refused(tmp_path):
    path = _artifact(tmp_path, run_contract={
        "candidate_parse_mode": "first_assertion", "retain_raw_output": False})
    with pytest.raises(SystemExit, match="whole_output"):
        assert_source_artifact(path, tmp_path / "missing.json")


def test_a_source_without_a_parent_chain_is_refused(tmp_path):
    path = _artifact(tmp_path, derived_from_sha256=None)
    with pytest.raises(SystemExit, match="provenance"):
        assert_source_artifact(path, tmp_path / "missing.json")


def test_a_broken_parent_chain_is_refused(tmp_path):
    original = tmp_path / "orig.json"
    original.write_text("{}", encoding="utf-8")
    path = _artifact(tmp_path, derived_from_sha256="0" * 64)
    with pytest.raises(SystemExit, match="parent chain broken"):
        assert_source_artifact(path, original)


def test_a_non_train_source_is_refused(tmp_path):
    path = _artifact(tmp_path, evaluation_split="val")
    with pytest.raises(SystemExit, match="not train"):
        assert_source_artifact(path, tmp_path / "missing.json")


# ------------------------------------------------------------- leakage proof

def test_the_builder_never_opens_the_canonical_corpus(tmp_path, monkeypatch):
    """Intercept the filesystem; a manifest may claim isolation only if proved."""
    opened: list[str] = []
    real_open, real_read = builtins.open, Path.read_text

    def canonical(name):
        p = Path(str(name))
        return (p.name in ("records.json", "splits.json")
                and p.parent.name != "development_view")

    def guarded_open(file, *a, **k):
        if canonical(file):
            opened.append(str(file))
            raise AssertionError("canonical corpus opened: " + str(file))
        return real_open(file, *a, **k)

    def guarded_read(self, *a, **k):
        if canonical(self):
            opened.append(str(self))
            raise AssertionError("canonical corpus opened: " + str(self))
        return real_read(self, *a, **k)

    if not (CORPUS / "development_view" / "train.records.json").exists():
        pytest.skip("development view not materialised")

    original = tmp_path / "orig.json"
    real_open(original, "w").write("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    derived = tmp_path / "d.json"
    real_open(derived, "w").write(json.dumps({
        "evaluation_split": "train", "final_test_measurement": False,
        "derived_from_sha256": digest,
        "run_contract": {"candidate_parse_mode": "whole_output",
                         "retain_raw_output": True},
        "function_results": []}))

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    rows, context = builder.build(derived, original, CORPUS, workers=2)
    assert opened == [], "the builder opened the canonical corpus"
    assert rows == []


def test_the_interceptor_would_actually_fire(monkeypatch):
    """Without this, the isolation test could pass by doing nothing."""
    real_read = Path.read_text

    def guarded(self, *a, **k):
        if Path(str(self)).name == "records.json" \
                and Path(str(self)).parent.name != "development_view":
            raise AssertionError("fired")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(AssertionError, match="fired"):
        (CORPUS / "records.json").read_text(encoding="utf-8")


# -------------------------------------------------- whole-output preservation

def test_multi_assertion_candidates_are_not_silently_dropped():
    """The retired builder returned None unless count_assertions == 1."""
    import inspect
    source = inspect.getsource(builder)
    assert "count_assertions(assertion) != 1" not in source
    assert '"one_assertion_reduction_applied": False' in source
    assert '"whole_output_preserved": True' in source


def test_the_builder_labels_per_candidate_not_per_function():
    import inspect
    source = inspect.getsource(builder.build)
    assert "most_common(1)" not in source, "a majority label reappeared"
    assert "for position, outcome in enumerate" in source
