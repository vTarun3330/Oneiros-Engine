"""Re-scoring must repair unscored candidates without inventing answers.

The danger is not that re-scoring fails. It is that it succeeds too broadly:
retrying a candidate that already produced a real answer until a different one
appears is outcome selection, not repair. So most of these tests are about what
the tool REFUSES to touch.
"""
from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

import scripts.rescore_harness_failures as rescore
from scripts.rescore_harness_failures import (
    HARNESS_STATUSES, MAX_RETRIES, SEMANTIC_STATUSES, assert_preconditions,
    build, rescore_one, select_jobs,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"

REFERENCE = "def twice(n):\n    return n * 2\n"
MUTANT = "def twice(n):\n    return n * 3\n"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _outcome(status, code="assert twice(5) == 10", rank=1):
    return {"rank": rank, "code": code, "raw_output": code,
            "raw_output_sha256": _sha(code), "reference_status": status,
            "parse_valid": True}


def _records():
    return {"r1": {"id": "r1", "entry_point": "twice",
                   "reference_code": REFERENCE, "code_under_test": MUTANT,
                   "support_context": ""}}


def _payload(outcomes, split="train"):
    return {"evaluation_split": split, "final_test_measurement": False,
            "run_contract": {"allow_test_function_candidates": True},
            "function_results": [{"record_id": "r1", "bug_family": "arith",
                                  "candidate_outcomes": outcomes}]}


# ---------------------------------------------------------------- selection

@pytest.mark.parametrize("status", HARNESS_STATUSES)
def test_every_harness_status_is_selected(status):
    jobs, _ = select_jobs(_payload([_outcome(status)]), _records(), True)
    assert len(jobs) == 1
    assert jobs[0]["original_status"] == status


@pytest.mark.parametrize("status", SEMANTIC_STATUSES)
def test_no_semantic_outcome_is_ever_retried(status):
    """A real answer must not be re-rolled until it changes."""
    jobs, skipped = select_jobs(_payload([_outcome(status)]), _records(), True)
    assert jobs == []
    assert skipped[status] == 1


def test_a_non_killing_candidate_is_not_retried():
    jobs, _ = select_jobs(_payload([_outcome("pass")]), _records(), True)
    assert jobs == []


def test_selection_is_not_conditioned_on_the_later_outcome():
    """Selection may only look at the ORIGINAL status."""
    import inspect
    source = inspect.getsource(select_jobs)
    for forbidden in ("killed", "kill_rate", "reference_valid"):
        assert forbidden not in source


def test_a_raw_hash_mismatch_is_refused():
    bad = _outcome("worker_error")
    bad["raw_output"] = "assert twice(5) == 999"       # text changed, hash not
    jobs, skipped = select_jobs(_payload([bad]), _records(), True)
    assert jobs == []
    assert skipped["raw_hash_mismatch"] == 1


def test_identity_is_carried_through():
    jobs, _ = select_jobs(_payload([_outcome("worker_error", rank=4)]),
                          _records(), True)
    job = jobs[0]
    assert (job["record_id"], job["rank"], job["position"]) == ("r1", 4, 0)
    assert job["code_sha256"] == _sha(job["code"])


# ------------------------------------------------------------- preconditions

def test_a_non_train_split_is_refused(tmp_path):
    path = tmp_path / "a.json"
    payload = _payload([_outcome("worker_error")], split="val")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="train split"):
        assert_preconditions(payload, path)


def test_drifted_source_is_refused(tmp_path):
    path = tmp_path / "a.json"
    payload = _payload([_outcome("worker_error")])
    payload["run_contract"]["evaluator_source_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="drifted"):
        assert_preconditions(payload, path)


def test_an_artifact_without_a_contract_is_refused(tmp_path):
    path = tmp_path / "a.json"
    payload = _payload([_outcome("worker_error")])
    payload["run_contract"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="run_contract"):
        assert_preconditions(payload, path)


# ------------------------------------------------------------------ scoring

def test_a_repaired_candidate_gets_a_real_outcome():
    job = {"record_id": "r1", "rank": 1, "position": 0, "entry_point": "twice",
           "code": "assert twice(5) == 10", "code_sha256": "x",
           "raw_output_sha256": "y", "original_status": "worker_error",
           "reference": REFERENCE, "mutant": MUTANT, "allow_test_function": True}
    result = rescore_one(job)
    assert result["repaired"] is True
    assert result["final"]["reference_status"] == "pass"
    assert result["final"]["killed"] is True     # 10 != 15 on the mutant
    assert len(result["attempts"]) == 1


def test_every_attempt_is_preserved():
    job = {"record_id": "r1", "rank": 1, "position": 0, "entry_point": "twice",
           "code": "assert twice(5) == 10", "code_sha256": "x",
           "raw_output_sha256": "y", "original_status": "worker_error",
           "reference": REFERENCE, "mutant": MUTANT, "allow_test_function": True}
    result = rescore_one(job)
    for attempt in result["attempts"]:
        assert {"attempt", "reference_status", "seconds", "timestamp_utc"} <= set(attempt)


def test_a_candidate_that_never_scores_stays_unscored(monkeypatch):
    """Exhausting the retries must not manufacture an answer."""
    monkeypatch.setattr(rescore, "classify_assertions",
                        lambda *a, **k: [{"golden": {"status": "worker_error"}}])
    job = {"record_id": "r1", "rank": 1, "position": 0, "entry_point": "twice",
           "code": "assert twice(5) == 10", "code_sha256": "x",
           "raw_output_sha256": "y", "original_status": "worker_error",
           "reference": REFERENCE, "mutant": MUTANT, "allow_test_function": True}
    result = rescore_one(job)
    assert result["repaired"] is False
    assert result["final"] is None
    assert len(result["attempts"]) == MAX_RETRIES


# ------------------------------------------------------------- leakage guard

def test_canonical_records_json_is_never_opened(tmp_path, monkeypatch):
    """Intercept the filesystem rather than grep the source."""
    opened: list[str] = []
    real_open, real_read = builtins.open, Path.read_text

    def canonical(name):
        p = Path(str(name))
        return p.name in ("records.json", "splits.json") and p.parent.name != "development_view"

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

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read)

    if not (CORPUS / "development_view" / "train.records.json").exists():
        pytest.skip("development view not materialised")
    parent = tmp_path / "p.json"
    payload = _payload([_outcome("worker_error")])
    payload["run_contract"] = {"allow_test_function_candidates": True}
    real_open(parent, "w").write(json.dumps(payload))
    overlay, derived = build(parent, CORPUS, None, workers=2)
    assert opened == []
    assert overlay["canonical_records_json_opened"] is False
    assert overlay["sealed_final_test_accessed"] is False


# -------------------------------------------------------------- immutability

@pytest.fixture
def synthetic_shard(monkeypatch):
    """Make build() see the synthetic record.

    Without this the record is absent from the real train shard, select_jobs
    skips it, and every assertion about replacements passes on an empty list -
    three tests were green for exactly that reason.
    """
    monkeypatch.setattr(rescore, "load_development_split",
                        lambda *a, **k: list(_records().values()))


def test_the_parent_artifact_is_not_modified(tmp_path, synthetic_shard):
    parent = tmp_path / "p.json"
    parent.write_text(json.dumps(_payload([_outcome("worker_error")])),
                      encoding="utf-8")
    before = hashlib.sha256(parent.read_bytes()).hexdigest()
    overlay, _ = build(parent, CORPUS, None, workers=2)
    assert overlay["candidates_selected"] == 1, "test would pass vacuously"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == before


def test_the_derived_artifact_records_its_parent(tmp_path, synthetic_shard):
    parent = tmp_path / "p.json"
    parent.write_text(json.dumps(_payload([_outcome("worker_error")])),
                      encoding="utf-8")
    overlay, derived = build(parent, CORPUS, None, workers=2)
    assert overlay["candidates_repaired"] == 1, "test would pass vacuously"
    assert derived["rescored_candidates"] == 1
    assert derived["derived_from_sha256"] == overlay["parent_sha256"]
    assert derived["derived_from"] == overlay["parent_artifact"]
    assert overlay["source_commit"]
    assert overlay["max_retries"] == MAX_RETRIES


def test_the_overlay_keeps_the_original_status_of_each_replacement(tmp_path, synthetic_shard):
    parent = tmp_path / "p.json"
    parent.write_text(json.dumps(_payload([_outcome("worker_error")])),
                      encoding="utf-8")
    overlay, _ = build(parent, CORPUS, None, workers=2)
    assert overlay["replacements"][0]["original_reference_status"] == "worker_error"
    assert overlay["replacements"][0]["attempts"]
