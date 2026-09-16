"""The admission defect that burned the sealed final test, pinned on real data.

The sealed loader required seven fields non-empty on every record in a split
and raised before returning any. Repository-style records carry ``entry_point``
and usually ``specification`` blank by design, so it refused records the
evaluator was always going to exclude - and it did so after the one-time token
had been spent.

Every test here runs against the **real permitted splits**. That is the whole
point: the old check passed thousands of tests because none of them used real
data, and one that did would have failed instantly and cost nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.evaluation_admission import (  # noqa: E402
    FUNCTION_REQUIRED_FIELDS, REFUSED_SPLITS, UNIVERSAL_REQUIRED_FIELDS,
    AdmissionError, RefusedSplitError, admission_binding_problems,
    admission_source_hashes, execution_mode_of, function_execution_mode,
    is_function_mode, is_repository_mode, refuse_refused_split,
    repository_execution_modes, required_fields_for, scope_split,
)

CORPUS = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"

#: The counts every historical artifact reports. Locked validation measured 757
#: function records on val; the four-arm development evaluation measured 542 on
#: ablation_dev. Admission must reproduce both exactly.
EXPECTED_TARGETS = {"val": 757, "ablation_dev": 542}


@pytest.fixture(scope="module")
def corpus():
    if not (CORPUS / "splits.json").is_file():
        pytest.skip("canonical corpus absent")
    return (
        json.loads((CORPUS / "splits.json").read_text(encoding="utf-8")),
        json.loads((CORPUS / "records.json").read_text(encoding="utf-8")),
    )


def _function_record(**overrides):
    record = {
        "id": "synthetic::function::1",
        "task_type": "hidden_mutation_reproduction",
        "entry_point": "add_two",
        "reference_code": "def add_two(a, b):\n    return a + b\n",
        "code_under_test": "def add_two(a, b):\n    return a - b\n",
        "specification": "Return the sum of two numbers.",
        "tests": [{"code": "assert add_two(1, 2) == 3"}],
        "quality": {"execution_mode": "function_assertion"},
    }
    record.update(overrides)
    return record


def _repository_record(**overrides):
    """A repository record in the shape the corpus actually stores."""
    record = {
        "id": "synthetic::repository::1",
        "task_type": "official_repository_pytest_reproduction",
        "entry_point": "",
        "reference_code": "def test_thing():\n    assert True\n",
        "code_under_test": "def test_thing():\n    assert False\n",
        "specification": "",
        "tests": [{"code": "assert True"}],
        "quality": {"execution_mode": "repository_pytest_fragment"},
    }
    record.update(overrides)
    return record


def _splits(**named):
    return {name: [r["id"] for r in rows] for name, rows in named.items()}


# --------------------------------------------------------------------------
# 1 & 2. The real splits scope to exactly the historical counts.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split,expected", sorted(EXPECTED_TARGETS.items()))
def test_permitted_splits_scope_to_the_historical_target_count(
        corpus, split, expected):
    splits, records = corpus

    scope = scope_split(splits, records, split)

    assert scope.target_count == expected
    assert len(scope.eligible) == expected
    assert all(is_function_mode(execution_mode_of(r)) for r in scope.eligible)


def test_the_counts_match_what_the_historical_evaluator_produced(corpus):
    """Same partition, computed the way train_on_dataset.py computes it."""
    from scripts.train_on_dataset import FUNCTION_EXECUTION_MODE

    splits, records = corpus
    by_id = {r["id"]: r for r in records}
    for split, expected in EXPECTED_TARGETS.items():
        historical = [
            by_id[i] for i in splits[split]
            if (by_id[i].get("quality") or {}).get(
                "execution_mode", FUNCTION_EXECUTION_MODE) == FUNCTION_EXECUTION_MODE
        ]
        assert len(historical) == expected
        assert scope_split(splits, records, split).target_count == len(historical)


def test_scope_order_follows_split_order(corpus):
    """The scope digest is taken over the id sequence, so order is part of it."""
    splits, records = corpus
    by_id = {r["id"]: r for r in records}

    scope = scope_split(splits, records, "ablation_dev")
    in_split_order = [
        i for i in splits["ablation_dev"]
        if is_function_mode(execution_mode_of(by_id[i]))
    ]

    assert [str(r["id"]) for r in scope.eligible] == in_split_order


# --------------------------------------------------------------------------
# 3. Repository records are excluded, never a rejection.
# --------------------------------------------------------------------------

def test_repository_records_do_not_reject_the_split(corpus):
    """The exact failure that spent the authorization, on permitted data."""
    splits, records = corpus

    for split, expected in EXPECTED_TARGETS.items():
        scope = scope_split(splits, records, split)   # must not raise
        assert scope.excluded_repository, \
            f"{split} is expected to contain repository records"
        assert scope.requested == expected + len(scope.excluded_repository)


def test_a_split_of_mixed_modes_is_scoped_not_refused():
    good = _function_record()
    repo = _repository_record()
    splits = _splits(rehearsal=[good, repo])

    scope = scope_split(splits, [good, repo], "rehearsal")

    assert scope.target_count == 1
    assert scope.excluded_repository == [repo["id"]]
    assert scope.requested == 2


def test_a_blank_entry_point_is_fine_on_a_repository_record():
    """entry_point is required only where scoring dereferences it."""
    repo = _repository_record(entry_point="", specification="")
    good = _function_record()
    splits = _splits(rehearsal=[good, repo])

    assert scope_split(splits, [good, repo], "rehearsal").target_count == 1


def test_repository_modes_have_no_function_requirements():
    for mode in repository_execution_modes():
        assert is_repository_mode(mode)
        with pytest.raises(AdmissionError, match="no function-scoring"):
            required_fields_for(mode)


# --------------------------------------------------------------------------
# 4. A blank specification never rejects a function-mode record.
# --------------------------------------------------------------------------

def test_a_function_record_with_a_blank_specification_is_accepted():
    blank = _function_record(id="synthetic::blank-spec", specification="")
    splits = _splits(rehearsal=[blank])

    scope = scope_split(splits, [blank], "rehearsal")

    assert scope.target_count == 1
    assert "specification" not in required_fields_for(function_execution_mode())


def test_the_real_val_split_contains_scored_records_without_a_specification(corpus):
    """Not hypothetical: locked validation measured records like this.

    Three function-mode records on val have a blank specification. Requiring it
    would silently drop scored targets, so this is pinned against real data.
    """
    splits, records = corpus
    scope = scope_split(splits, records, "val")

    blank = [r for r in scope.eligible if not r.get("specification")]

    assert len(blank) == 3
    assert scope.target_count == 757


# --------------------------------------------------------------------------
# 5. A genuinely broken function record fails closed.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", sorted(
    set(UNIVERSAL_REQUIRED_FIELDS + FUNCTION_REQUIRED_FIELDS) - {"id"}))
def test_a_function_record_missing_a_required_field_fails_closed(field):
    broken = _function_record(id="synthetic::broken", **{field: ""})
    splits = _splits(rehearsal=[broken])

    with pytest.raises(AdmissionError, match="function scoring requires"):
        scope_split(splits, [broken], "rehearsal")


def test_one_broken_function_record_fails_the_whole_split():
    """Fail closed, not skip. A silently dropped target changes the denominator."""
    good = _function_record(id="synthetic::ok")
    broken = _function_record(id="synthetic::broken", entry_point="")
    splits = _splits(rehearsal=[good, broken])

    with pytest.raises(AdmissionError):
        scope_split(splits, [good, broken], "rehearsal")


def test_an_empty_or_duplicated_split_is_refused():
    good = _function_record()
    with pytest.raises(AdmissionError, match="empty"):
        scope_split({"rehearsal": []}, [good], "rehearsal")
    with pytest.raises(AdmissionError, match="duplicate"):
        scope_split({"rehearsal": [good["id"], good["id"]]}, [good], "rehearsal")


def test_a_split_of_only_repository_records_is_refused():
    repo = _repository_record()
    splits = _splits(rehearsal=[repo])

    with pytest.raises(AdmissionError, match="no function-mode records"):
        scope_split(splits, [repo], "rehearsal")


# --------------------------------------------------------------------------
# 6. The consumed split is refused everywhere.
# --------------------------------------------------------------------------

def test_the_consumed_split_is_refused_by_name():
    assert "test" in REFUSED_SPLITS
    with pytest.raises(RefusedSplitError, match="consumed"):
        refuse_refused_split("test")


def test_scoping_refuses_the_consumed_split_before_opening_anything():
    """The refusal precedes the corpus, so nothing is read on the way to it."""
    good = _function_record()
    splits = _splits(rehearsal=[good])
    splits["test"] = [good["id"]]

    with pytest.raises(RefusedSplitError):
        scope_split(splits, [good], "test")


def test_the_refusal_names_the_incident_so_it_cannot_be_read_as_a_bug():
    with pytest.raises(RefusedSplitError, match="SEALED_FINAL_INCIDENT"):
        refuse_refused_split("test")


def test_the_rehearsal_command_refuses_the_consumed_split():
    result = subprocess.run(
        [sys.executable, "scripts/run_rehearsal_evaluation.py",
         "--split", "test", "--dry-run"],
        capture_output=True, text=True, cwd=ROOT)

    assert result.returncode == 2
    assert "REFUSED" in result.stdout
    assert "consumed" in result.stdout


def test_the_rehearsal_evaluator_refuses_the_consumed_split():
    from harness.rehearsal_evaluator import RehearsalError, run_rehearsal_evaluation

    with pytest.raises(RehearsalError, match="refusing to rehearse"):
        run_rehearsal_evaluation(
            load_records=lambda: [_function_record()],
            generate_batch=lambda batch: [[]],
            output_dir=ROOT / "results" / "_never_written",
            split_name="test",
            scope_summary={},
            frozen_settings={})
    assert not (ROOT / "results" / "_never_written").exists()


# --------------------------------------------------------------------------
# Source binding.
# --------------------------------------------------------------------------

def test_the_admission_sources_are_bound_by_hash():
    current = admission_source_hashes()

    assert admission_binding_problems(current) == []
    for role in ("admission", "historical_scoping"):
        assert len(current[role]["canonical_sha256"]) == 64


def test_a_changed_admission_source_invalidates_a_binding():
    tampered = json.loads(json.dumps(admission_source_hashes()))
    tampered["admission"]["canonical_sha256"] = "0" * 64

    problems = admission_binding_problems(tampered)

    assert any("admission" in item for item in problems)


def test_mode_semantics_come_from_the_historical_evaluator():
    """Not restated here. Two sources of truth is how this drifts."""
    from scripts.train_on_dataset import (
        FUNCTION_EXECUTION_MODE, REPOSITORY_EXECUTION_MODES,
    )

    assert function_execution_mode() == FUNCTION_EXECUTION_MODE
    assert repository_execution_modes() == frozenset(REPOSITORY_EXECUTION_MODES)

    source = (ROOT / "harness" / "evaluation_admission.py").read_text(encoding="utf-8")
    assert 'FUNCTION_EXECUTION_MODE = "' not in source
    assert '"function_assertion"' not in source


def test_execution_mode_is_read_from_both_record_shapes():
    """Canonical records nest it under quality; adapted pairs hoist it."""
    canonical = _function_record()
    adapted = {"id": "x", "execution_mode": "repository_pytest_fragment"}

    assert execution_mode_of(canonical) == "function_assertion"
    assert execution_mode_of(adapted) == "repository_pytest_fragment"
    assert execution_mode_of({"id": "y"}) == function_execution_mode()


# --------------------------------------------------------------------------
# The sealed path is not resurrected.
# --------------------------------------------------------------------------

def test_the_rehearsal_does_not_call_the_sealed_loader():
    source = (ROOT / "scripts" / "run_rehearsal_evaluation.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "harness" / "rehearsal_evaluator.py").read_text(encoding="utf-8")

    for text in (source, evaluator):
        assert "sealed_final_loader" not in text
        assert "sealed_records" not in text
        assert "authorization_token" not in text
        assert "SealedFinalGuard" not in text


def test_rehearsal_artifacts_are_never_final_test_measurements():
    from harness.rehearsal_evaluator import REHEARSAL_LABEL

    assert "not eligible for model selection" in REHEARSAL_LABEL
    evaluator = (ROOT / "harness" / "rehearsal_evaluator.py").read_text(encoding="utf-8")
    assert '"final_test_measurement": False' in evaluator
    assert '"eligible_for_model_selection": False' in evaluator
