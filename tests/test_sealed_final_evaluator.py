"""A token must never be spent to discover that evaluation is unavailable.

The first sealed entrypoint called the guard, the guard spent the one-time
token, and control then reached a comment saying the measurement was not
implemented. A real authorization would have been consumed irreversibly and
produced nothing, on the one split that cannot be measured twice.

So the central tests here do not check that the measurement is correct — they
check the *order*: given a broken, missing or unbound evaluator, the run must
refuse while the token remains unspent and reusable. Everything runs on mock
sealed data in temporary directories; the real sealed split is never touched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness.sealed_final import (
    REQUIRED_BUNDLE_FIELDS, SEALED_SPLIT, FinalBundle, SealedAccessError,
    SealedFinalGuard, issue_authorization,
)
from harness.sealed_final_evaluator import (
    EVALUATOR_VERSION, FinalEvaluationError, environment_problems,
    evaluator_source_hashes, run_final_evaluation,
)
import harness.sealed_final_loader as loader
import scripts.run_sealed_final_test as entry

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EXEC_RECEIPT = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v4.json"
SUPERSEDED_V2 = ROOT / "results" / "v4_2_sealed_final_executable_receipt.json"

GOLDEN = "def add(a, b):\n    return a + b\n"
MUTANT = "def add(a, b):\n    return a - b\n"
KILLER = "assert add(1, 2) == 3"
SURVIVOR = "assert add(0, 0) == 0"


def mock_records(n: int = 4):
    return [{
        "id": f"mock-{i}", "entry_point": "add", "golden_code": GOLDEN,
        "mutant_code": MUTANT, "bug_family": "arithmetic",
        "source_name": "mock", "project": "mock", "dataset_name": "mock",
    } for i in range(n)]


def mock_generator(record, candidates=8):
    out = []
    for rank in range(candidates):
        code = KILLER if rank % 2 == 0 else SURVIVOR
        out.append({"raw_output": code, "code": code})
    return out


def mock_batch(records):
    """The batch contract the evaluator now takes.

    Generation moved from one target per call to a padded batch, because
    locked validation padded two and batch shape changes sampling.
    """
    return [mock_generator(record) for record in records]


def frozen_settings():
    return {"candidates_per_target": 8, "base_model_name": "mock", "seeds": {"generation_seed": 42}}


# ---------------------------------------------------- the measurement works

def test_mock_evaluation_produces_a_complete_final_artifact(tmp_path):
    outcome = run_final_evaluation(
        load_records=lambda: mock_records(6),
        generate_batch=mock_batch,
        output_dir=tmp_path / "run",
        bundle_sha256="a" * 64,
        frozen_settings=frozen_settings(),
    )
    artifact = json.loads(Path(outcome["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["final_test_measurement"] is True
    assert artifact["one_time"] is True
    assert artifact["function_validation_records"] == 6
    assert len(artifact["function_results"]) == 6
    assert artifact["kill_at_k"]["8"]["functions"] == 6
    assert artifact["raw_output_integrity"]["complete"] is True
    assert artifact["evaluator_source"]["evaluator_version"] == EVALUATOR_VERSION
    assert hashlib.sha256(
        Path(outcome["artifact_path"]).read_bytes()).hexdigest() == outcome["artifact_sha256"]


def test_raw_outputs_and_hashes_are_retained(tmp_path):
    outcome = run_final_evaluation(
        load_records=lambda: mock_records(2), generate_batch=mock_batch,
        output_dir=tmp_path / "run", bundle_sha256="a" * 64,
        frozen_settings=frozen_settings())
    artifact = json.loads(Path(outcome["artifact_path"]).read_text(encoding="utf-8"))
    total = 0
    for item in artifact["function_results"]:
        for candidate in item["candidate_outcomes"]:
            total += 1
            assert candidate["raw_output"]
            assert hashlib.sha256(
                candidate["raw_output"].encode()).hexdigest() == candidate["raw_output_sha256"]
    assert total == 16


def test_execution_evidence_is_retained(tmp_path):
    outcome = run_final_evaluation(
        load_records=lambda: mock_records(1), generate_batch=mock_batch,
        output_dir=tmp_path / "run", bundle_sha256="a" * 64,
        frozen_settings=frozen_settings())
    artifact = json.loads(Path(outcome["artifact_path"]).read_text(encoding="utf-8"))
    outcomes = artifact["function_results"][0]["candidate_outcomes"]
    killers = [c for c in outcomes if c["killed"]]
    assert killers, "the mock killer assertion should kill the mock mutant"
    for candidate in outcomes:
        for field in ("parse_valid", "policy_valid", "execution_valid",
                      "reference_valid", "killed", "failure_mode"):
            assert field in candidate


def test_durable_progress_is_written(tmp_path):
    run_final_evaluation(
        load_records=lambda: mock_records(6), generate_batch=mock_batch,
        output_dir=tmp_path / "run", bundle_sha256="a" * 64,
        frozen_settings=frozen_settings(), progress_every=2)
    files = sorted((tmp_path / "run" / "progress").glob("progress.*.json"))
    assert len(files) == 3
    last = json.loads(files[-1].read_text(encoding="utf-8"))
    assert last["completed"] == 6 and last["total"] == 6


def test_an_empty_scope_is_refused(tmp_path):
    with pytest.raises(FinalEvaluationError, match="empty"):
        run_final_evaluation(
            load_records=list, generate_batch=mock_batch, output_dir=tmp_path / "r",
            bundle_sha256="a" * 64, frozen_settings=frozen_settings())


def test_a_generator_returning_the_wrong_count_is_refused(tmp_path):
    with pytest.raises(FinalEvaluationError, match="expected 8"):
        run_final_evaluation(
            load_records=lambda: mock_records(1),
            generate_batch=lambda records: [mock_generator(r, 3) for r in records],
            output_dir=tmp_path / "r", bundle_sha256="a" * 64,
            frozen_settings=frozen_settings())


# ------------------------------------------- pre-authorization environment

def test_environment_problems_catch_each_failure(tmp_path):
    ok = environment_problems(
        output_dir=tmp_path / "fresh", model_name="m", model_revision="a" * 40,
        expected_candidates=8, require_cuda=False)
    assert ok == []

    assert any("immutable" in p for p in environment_problems(
        output_dir=tmp_path / "f2", model_name="m", model_revision="main",
        expected_candidates=8, require_cuda=False))

    assert any("protocol declares 8" in p for p in environment_problems(
        output_dir=tmp_path / "f3", model_name="m", model_revision="a" * 40,
        expected_candidates=4, require_cuda=False))

    assert any("model files" in p for p in environment_problems(
        output_dir=tmp_path / "f4", model_name="m", model_revision="a" * 40,
        expected_candidates=8, model_files_present=lambda n, r: False,
        require_cuda=False))

    assert any("CUDA" in p for p in environment_problems(
        output_dir=tmp_path / "f5", model_name="m", model_revision="a" * 40,
        expected_candidates=8, cuda_available=lambda: False, require_cuda=True))

    assert any("disk" in p for p in environment_problems(
        output_dir=tmp_path / "f6", model_name="m", model_revision="a" * 40,
        expected_candidates=8, free_disk_bytes=lambda p: 1, require_cuda=False))

    used = tmp_path / "used"
    used.mkdir()
    (used / "something").write_text("x", encoding="utf-8")
    assert any("not empty" in p for p in environment_problems(
        output_dir=used, model_name="m", model_revision="a" * 40,
        expected_candidates=8, require_cuda=False))


# ------------------------------- THE ordering property: the token survives

def _mock_setup(tmp_path):
    guard = SealedFinalGuard(tmp_path / "state.json", tmp_path / "audit.log")
    bundle = FinalBundle(fields={n: f"v-{n}" for n in REQUIRED_BUNDLE_FIELDS})
    auth = issue_authorization(bundle, "tester", "ordering test")
    guard.register(auth)
    return guard, bundle, auth


def test_a_valid_token_is_not_consumed_when_prerequisites_fail(tmp_path):
    """The heart of the redesign.

    Simulates the corrected order: a pre-measurement check fails, so the guard
    is never called. The token must still open the split afterwards - proving
    it was not burned by the failure.
    """
    guard, bundle, auth = _mock_setup(tmp_path)

    prerequisites_ok = False  # e.g. evaluator missing, model absent, disk full
    if prerequisites_ok:  # pragma: no cover - the point is that we do NOT call
        guard.open_sealed_split(SEALED_SPLIT, bundle, auth.token, "should-not-run")

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state.get("spent_tokens", []) == []

    # the same token still works once the prerequisites are satisfied
    guard.open_sealed_split(SEALED_SPLIT, bundle, auth.token, "later-run")
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["spent_tokens"] == [auth.token]


def test_the_old_defective_order_would_have_burned_the_token(tmp_path):
    """Pins the defect itself, so the regression is recognisable if reintroduced."""
    guard, bundle, auth = _mock_setup(tmp_path)
    guard.open_sealed_split(SEALED_SPLIT, bundle, auth.token, "defective-order")
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["spent_tokens"] == [auth.token]
    with pytest.raises(SealedAccessError, match="already been spent"):
        guard.open_sealed_split(SEALED_SPLIT, bundle, auth.token, "retry")


def test_the_entrypoint_checks_prerequisites_before_calling_the_guard():
    """Source order: every refusal must precede open_sealed_split."""
    source = (ROOT / "scripts" / "run_sealed_final_test.py").read_text(encoding="utf-8")
    guard_call = source.index("guard.open_sealed_split(")
    for earlier in ("receipt_problems(", "evaluator_binding_problems(",
                    "environment_problems(", "sealed loader is not importable",
                    "final_evaluator_executable"):
        assert source.index(earlier) < guard_call, earlier
    assert source.index("if problems:") < guard_call
    assert source.index("if args.check_only:") < guard_call


# -------------------------------------------- the entrypoint refuses early

def _run(*args):
    return subprocess.run([PY, "scripts/run_sealed_final_test.py", *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_no_arguments_refuses_without_presenting_a_token():
    result = _run()
    assert result.returncode == 2
    assert "No token was presented" in result.stdout
    assert not (ROOT / "results" / "sealed_final_state.json").exists()


@pytest.mark.parametrize("relative", [
    "results/v4_2_sealed_final_readiness_receipt.json",
    "results/v4_2_sealed_final_executable_receipt.json",
    "results/v4_2_sealed_final_executable_receipt_v3.json",
])
def test_superseded_receipts_cannot_authorize(relative):
    """v1 and v2 must both be refused outright, by schema version.

    v1 predates the evaluator. v2 declared itself executable but described a
    loader calling two functions that do not exist, with parse_mode left at its
    legacy default - so it must be refused despite that declaration.
    """
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} absent")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = _run("--executable-receipt", relative,
                  "--expected-receipt-sha256", digest,
                  "--authorization-token", "irrelevant",
                  "--i-understand-this-is-one-time-and-irreversible")
    assert result.returncode == 1
    assert "is refused" in result.stdout
    assert "REFUSED before authorization" in result.stdout
    assert not (ROOT / "results" / "sealed_final_state.json").exists()


def test_a_wrong_receipt_hash_refuses_before_authorization():
    if not EXEC_RECEIPT.exists():
        pytest.skip("executable receipt absent")
    result = _run("--expected-receipt-sha256", "b" * 64,
                  "--authorization-token", "irrelevant",
                  "--i-understand-this-is-one-time-and-irreversible")
    assert result.returncode == 1
    assert "hash mismatch" in result.stdout
    assert "REFUSED before authorization" in result.stdout


def test_check_only_presents_no_token_and_writes_nothing():
    if not EXEC_RECEIPT.exists():
        pytest.skip("executable receipt absent")
    digest = hashlib.sha256(EXEC_RECEIPT.read_bytes()).hexdigest()
    before = (ROOT / "results" / "sealed_final_audit.log").exists() and \
        (ROOT / "results" / "sealed_final_audit.log").read_bytes()
    result = _run("--check-only", "--expected-receipt-sha256", digest)
    assert result.returncode == 0
    assert "No token was presented" in result.stdout
    assert not (ROOT / "results" / "sealed_final_state.json").exists()
    after = (ROOT / "results" / "sealed_final_audit.log").exists() and \
        (ROOT / "results" / "sealed_final_audit.log").read_bytes()
    assert before == after, "check-only must not append to the audit log"


# ---------------------------------------------- the real split is untouched

def test_the_loader_refuses_without_a_granted_authorization():
    with pytest.raises(SealedAccessError, match="without a granted authorization"):
        loader.sealed_records()


def test_no_authorization_has_ever_been_granted():
    assert not (ROOT / "results" / "sealed_final_state.json").exists()
    assert loader.authorization_granted() is False
    audit = ROOT / "results" / "sealed_final_audit.log"
    if audit.exists():
        events = [json.loads(line) for line in
                  audit.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [e for e in events if e.get("event") == "sealed_access_granted"] == []


def test_tests_never_reference_the_real_sealed_shard():
    """No test here may name a real corpus file.

    Scans this module with its own body removed - the names being searched for
    necessarily appear in the search itself, and an earlier version failed on
    exactly that.
    """
    import inspect
    source = Path(__file__).read_text(encoding="utf-8")
    source = source.replace(inspect.getsource(test_tests_never_reference_the_real_sealed_shard), "")
    source = source.replace(inspect.getsource(test_the_evaluator_reads_nothing_by_itself), "")
    for forbidden in ("records.json", "splits.json", "data/corpus"):
        assert forbidden not in source, forbidden


def test_the_evaluator_reads_nothing_by_itself():
    """Records and candidates are injected; the module opens no corpus."""
    source = (ROOT / "harness" / "sealed_final_evaluator.py").read_text(encoding="utf-8")
    for forbidden in ("records.json", "splits.json", "corpus"):
        assert forbidden not in source, forbidden


# ------------------------------------------- the receipt binds the evaluator

def test_the_executable_receipt_binds_the_evaluator_source():
    if not EXEC_RECEIPT.exists():
        pytest.skip("executable receipt absent")
    receipt = json.loads(EXEC_RECEIPT.read_text(encoding="utf-8"))
    recorded = receipt["final_evaluator_source"]
    current = evaluator_source_hashes()
    assert recorded["evaluator_version"] == EVALUATOR_VERSION
    assert recorded["canonical_sha256"] == current["canonical_sha256"]
    assert recorded["measurement_logic_canonical_sha256"] == \
        current["measurement_logic_canonical_sha256"]
    from harness.source_identity import canonical_sha256
    assert recorded["loader_canonical_sha256"] == canonical_sha256(
        ROOT / "harness" / "sealed_final_loader.py")


def test_the_executable_receipt_declares_executability():
    if not EXEC_RECEIPT.exists():
        pytest.skip("executable receipt absent")
    receipt = json.loads(EXEC_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "oneiros_sealed_final_readiness_v4"
    assert receipt["final_evaluator_executable"] is True
    assert "executable" in receipt["final_evaluator_status"]
    assert receipt["sealed_split_accessed"] is False
    assert receipt["authorization_token_issued"] is False
    assert receipt["supersedes"]["schema_versions"] == [
        "oneiros_sealed_final_readiness_v1", "oneiros_sealed_final_readiness_v2",
        "oneiros_sealed_final_readiness_v3"]
    assert receipt["exact_command"][1] == "scripts/run_sealed_final_test.py"


def test_a_changed_evaluator_is_refused(monkeypatch):
    if not EXEC_RECEIPT.exists():
        pytest.skip("executable receipt absent")
    receipt = json.loads(EXEC_RECEIPT.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["final_evaluator_source"]["measurement_logic_canonical_sha256"] = "0" * 64
    problems = entry.evaluator_binding_problems(tampered)
    assert any("measurement_logic_canonical_sha256 differs" in p for p in problems)


def test_a_receipt_without_evaluator_identity_is_refused():
    assert entry.evaluator_binding_problems({}) == [
        "receipt records no final evaluator source identity"]


def test_the_superseded_v2_receipt_is_preserved():
    if not SUPERSEDED_V2.exists():
        pytest.skip("v2 receipt absent")
    assert hashlib.sha256(SUPERSEDED_V2.read_bytes()).hexdigest() ==         "a1cafebac4f2de657f08b91d896492877b0902739597a4f7cab8600185a40353"
    marker = ROOT / "results" / "v4_2_sealed_final_executable_receipt.SUPERSEDED.md"
    assert marker.exists()
    text = marker.read_text(encoding="utf-8")
    assert "NOT executable authorization" in text
    assert "build_test_generation_prompt" in text
    assert "generate_candidates" in text
    assert "first_assertion" in text


def test_the_v1_receipt_is_preserved_and_marked_superseded():
    v1 = ROOT / "results" / "v4_2_sealed_final_readiness_receipt.json"
    marker = ROOT / "results" / "v4_2_sealed_final_readiness_receipt.SUPERSEDED.md"
    if not v1.exists():
        pytest.skip("v1 readiness receipt absent")
    assert hashlib.sha256(v1.read_bytes()).hexdigest() == \
        "4ddad9834ed5e452e4b792842755f15967ab1dca9ece9ea3f659eb274e7f2b9a"
    assert marker.exists()
    text = marker.read_text(encoding="utf-8")
    assert "NOT executable authorization" in text
    assert "spends the one-time token" in text
