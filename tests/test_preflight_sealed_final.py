"""The sealed split must stay shut, including while preparing to open it.

A preflight for a one-time measurement has an obvious failure mode: it reads
the thing it is preparing to read, "just to check". Then the measurement is
already contaminated and nobody notices, because the preflight reported
success.

So the tests here are mostly about what did NOT happen - no sealed record read,
no sealed id enumerated, no sealed payload hashed, no token minted - plus proof
that the guard's refusals actually refuse.
"""
from __future__ import annotations

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
from scripts import preflight_sealed_final as pf

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
QWEN = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REV = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
RECEIPT = ROOT / "results" / "v4_2_sealed_final_readiness_receipt.json"


@pytest.fixture(scope="module")
def built():
    problems: list[str] = []
    return pf.collect(problems), problems


# ------------------------------------------------------ nothing was opened

def test_the_preflight_is_ready_and_clean(built):
    receipt, problems = built
    assert problems == [], problems
    assert receipt["ready_for_authorization"] is True


def test_the_sealed_split_was_not_touched(built):
    receipt, _ = built
    assert receipt["sealed_split_accessed"] is False
    assert receipt["sealed_records_read"] == 0
    assert receipt["sealed_ids_enumerated"] == 0
    assert receipt["sealed_payload_hashed"] is False


def test_split_identity_is_bound_by_manifest_digests_not_by_enumeration(built):
    receipt, _ = built
    fields = receipt["frozen_bundle"]["fields"]
    view = json.loads((ROOT / "data/corpus/v4_1_research_hardened_candidate"
                       / "development_view/manifest.json").read_text(encoding="utf-8"))
    assert fields["split_ids_sha256"] == view["source_splits_sha256"]
    assert fields["corpus_records_sha256"] == view["source_records_sha256"]
    assert "no sealed record was read" in receipt["split_identity_binding"].lower()


def test_the_preflight_never_reads_the_sealed_shard():
    """Source-level: it must not name a test shard or the canonical records."""
    source = (ROOT / "scripts" / "preflight_sealed_final.py").read_text(encoding="utf-8")
    for forbidden in ("test.records.json", "records.json\"", "splits.json\""):
        assert forbidden not in source, forbidden


def test_no_authorization_token_is_minted_by_the_preflight(built):
    receipt, _ = built
    assert receipt["authorization_token_issued"] is False
    assert "defeat the one-time protection" in receipt["token_policy"]


def test_no_authorization_has_ever_been_spent():
    """No state file means no token was ever spent and the split never opened.

    The audit log is a different matter and may well exist: the guard records
    every attempt, refusals included, and a refused attempt SHOULD leave a
    trace. An earlier version of this test asserted the log's absence, which
    would have made correct auditing look like a failure.
    """
    assert not (ROOT / "results" / "sealed_final_state.json").exists()

    audit = ROOT / "results" / "sealed_final_audit.log"
    if not audit.exists():
        return
    events = [json.loads(line) for line in
              audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    granted = [e for e in events if e.get("event") == "sealed_access_granted"]
    assert granted == [], f"the sealed split was opened: {granted}"
    assert all(e.get("event") in {"sealed_access_refused", "authorization_issued"}
               for e in events), events


# ------------------------------------------------- the frozen final candidate

def test_the_final_candidate_is_the_immutable_base_model(built):
    receipt, _ = built
    candidate = receipt["final_candidate"]
    assert candidate["model"] == QWEN
    assert candidate["model_revision"] == REV
    assert candidate["tokenizer_revision"] == REV
    assert candidate["adapter"] is None


def test_the_base_arm_identity_digest_is_the_evaluators_convention(built):
    receipt, _ = built
    expected = hashlib.sha256(f"{QWEN}@{REV}".encode()).hexdigest()
    assert receipt["final_candidate"]["adapter_identity_digest"] == expected
    assert receipt["frozen_bundle"]["fields"]["adapter_sha256"] == expected


def test_the_bundle_is_fully_frozen(built):
    receipt, _ = built
    assert receipt["frozen_bundle"]["missing_fields"] == []
    assert receipt["frozen_bundle"]["frozen"] is True
    assert len(receipt["bundle_sha256"]) == 64


@pytest.mark.parametrize("field", REQUIRED_BUNDLE_FIELDS)
def test_every_required_bundle_field_is_pinned(built, field):
    receipt, _ = built
    assert receipt["frozen_bundle"]["fields"].get(field) not in (None, "", [], {}), field


def test_the_frozen_protocol_matches_the_successor_protocol(built):
    from harness.successor_protocol import SUCCESSOR_PROTOCOL, protocol_sha256
    receipt, _ = built
    fields = receipt["frozen_bundle"]["fields"]
    assert fields["candidates_per_target"] == SUCCESSOR_PROTOCOL["candidates_per_function"] == 8
    assert fields["seeds"]["generation_seed"] == 42
    assert fields["sampling"]["temperature"] == 0.7
    assert fields["sampling"]["top_p"] == 0.9
    assert fields["sampling"]["candidate_parse_mode"] == "whole_output"
    assert fields["sampling"]["retain_raw_output"] is True
    assert fields["sampling"]["reranking"] == "none"
    assert fields["sampling"]["feedback_rounds"] == 0
    assert fields["prompt_budgets"]["prompt_token_limit"] == 1024
    assert fields["prompt_budgets"]["generation_completion_token_limit"] == 1024
    assert fields["prompt_budgets"]["max_sequence_tokens"] == 3072
    assert fields["evaluator_version"]["protocol_sha256"] == protocol_sha256()


def test_the_selection_rule_records_that_no_checkpoint_was_selected(built):
    receipt, _ = built
    rule = receipt["frozen_bundle"]["fields"]["checkpoint_selection_rule"]
    assert "No checkpoint is selected" in rule
    assert "0.0783" in rule
    assert "base model with no adapter" in rule


def test_baselines_are_pinned_by_hash(built):
    receipt, _ = built
    baselines = receipt["frozen_bundle"]["fields"]["baseline_versions"]
    assert baselines
    for path, digest in baselines.items():
        assert len(digest) == 64
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_the_evidence_the_decision_rests_on_is_hashed(built):
    receipt, _ = built
    evidence = receipt["decision_evidence"]
    assert evidence["results/v4_2_locked_validation_result_receipt.json"]
    for rel, digest in evidence.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest


# ----------------------------------------------------------- output isolation

def test_the_final_run_name_cannot_overwrite_any_prior_run(built):
    receipt, _ = built
    isolation = receipt["output_isolation"]
    assert isolation["final_run_name"] not in receipt["protected_run_names"]
    assert isolation["collides_with_existing_runs"] is False
    for run in ("locked_val_base_qwen_s42", "locked_val_armA_ckpt431_s42",
                "local_sft_armA_baseline_successor_s42"):
        assert run in isolation["may_not_write_into"]


def test_the_final_output_directories_do_not_exist_yet():
    for parent in ("results", "checkpoints"):
        assert not (ROOT / parent / pf.FINAL_RUN_NAME).exists()


# -------------------------------------------------- the mock guard rehearsal

@pytest.mark.parametrize("refusal", [
    "unfrozen_bundle_refused", "missing_token_refused", "unknown_token_refused",
    "spent_token_refused", "bundle_edited_after_authorization_refused",
    "development_command_refused",
])
def test_each_guard_refusal_holds(built, refusal):
    receipt, _ = built
    assert receipt["mock_guard_rehearsal"][refusal] is True, refusal


def test_a_valid_token_is_granted_exactly_once(built):
    receipt, _ = built
    rehearsal = receipt["mock_guard_rehearsal"]
    assert rehearsal["valid_token_granted_once"] is True
    assert rehearsal["spent_token_refused"] is True
    assert rehearsal["all_refusals_hold"] is True


def test_the_rehearsal_is_mock_only(tmp_path):
    """Run it again and prove it leaves nothing behind in the repository."""
    before = {p.name for p in (ROOT / "results").glob("sealed_final_*")}
    pf.rehearse_with_mocks()
    after = {p.name for p in (ROOT / "results").glob("sealed_final_*")}
    assert before == after


def test_a_development_split_needs_no_token(built):
    receipt, _ = built
    assert receipt["mock_guard_rehearsal"]["development_split_needs_no_token"] is True


def test_the_one_time_property_directly(tmp_path):
    """Independently of the preflight: one token, one open."""
    guard = SealedFinalGuard(tmp_path / "s.json", tmp_path / "a.log")
    bundle = FinalBundle(fields={name: f"x-{name}" for name in REQUIRED_BUNDLE_FIELDS})
    auth = issue_authorization(bundle, "tester", "one-time property")
    guard.register(auth)
    guard.open_sealed_split(SEALED_SPLIT, bundle, auth.token, "first")
    with pytest.raises(SealedAccessError, match="already been spent"):
        guard.open_sealed_split(SEALED_SPLIT, bundle, auth.token, "second")


# ------------------------------------------------- the authorized entrypoint

def test_the_entrypoint_is_separate_from_the_development_entrypoint(built):
    receipt, _ = built
    entry = receipt["authorized_entrypoint"]
    assert entry["path"] == "scripts/run_sealed_final_test.py"
    assert entry["development_entrypoint"] == "scripts/train_on_dataset.py"
    assert entry["separate_from_development_entrypoint"] is True
    assert len(entry["canonical_sha256"]) == 64


def test_the_entrypoint_refuses_with_no_arguments():
    """Refusal, and — since the ordering fix — an explicit no-token claim.

    This previously asserted an exact sentence from the entrypoint's help text,
    which broke when that text was rewritten. What matters is that it refuses
    and that no token was presented or spent.
    """
    result = subprocess.run([PY, "scripts/run_sealed_final_test.py"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 2
    assert "REFUSED" in result.stdout
    assert "No token was presented" in result.stdout
    assert not (ROOT / "results" / "sealed_final_state.json").exists()


def test_the_entrypoint_refuses_without_the_acknowledgement():
    result = subprocess.run(
        [PY, "scripts/run_sealed_final_test.py",
         "--expected-receipt-sha256", "a" * 64,
         "--authorization-token", "whatever"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 2
    assert "i-understand-this-is-one-time-and-irreversible" in result.stdout


def test_the_entrypoint_refuses_a_wrong_receipt_hash():
    if not RECEIPT.exists():
        pytest.skip("readiness receipt not generated")
    result = subprocess.run(
        [PY, "scripts/run_sealed_final_test.py",
         "--expected-receipt-sha256", "b" * 64,
         "--authorization-token", "whatever",
         "--i-understand-this-is-one-time-and-irreversible"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1
    assert "hash mismatch" in result.stdout


def test_the_entrypoint_refuses_an_unknown_token_with_the_right_receipt():
    if not RECEIPT.exists():
        pytest.skip("readiness receipt not generated")
    digest = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    result = subprocess.run(
        [PY, "scripts/run_sealed_final_test.py",
         "--expected-receipt-sha256", digest,
         "--authorization-token", "not-a-real-token",
         "--i-understand-this-is-one-time-and-irreversible"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1
    assert "REFUSED" in result.stdout
    # and it must not have created guard state in the repository
    assert not (ROOT / "results" / "sealed_final_state.json").exists()


def test_the_entrypoint_lists_the_post_execution_prohibitions():
    result = subprocess.run([PY, "scripts/run_sealed_final_test.py"],
                            capture_output=True, text=True, cwd=ROOT)
    for prohibited in ("retraining of any kind", "prompt changes",
                       "threshold changes", "re-running the sealed final test"):
        assert prohibited in result.stdout


def test_the_receipt_states_the_prohibitions(built):
    receipt, _ = built
    for prohibited in ("retraining of any kind", "prompt changes",
                       "threshold changes", "re-running the sealed final test"):
        assert prohibited in receipt["prohibited_after_execution"]


# ------------------------------------------------------ claims stay bounded

def test_the_receipt_refuses_to_claim_sft_generalization(built):
    receipt, _ = built
    limits = " ".join(receipt["interpretation_limits"])
    assert "No SFT model was promoted" in limits
    assert "no real native repository result is claimed" in limits.lower()
    assert "Repository records are excluded" in limits


def test_the_published_receipt_is_clean_if_present():
    if not RECEIPT.exists():
        pytest.skip("readiness receipt not generated")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["preflight_problems"] == []
    assert receipt["ready_for_authorization"] is True
    assert receipt["sealed_split_accessed"] is False
    assert receipt["authorization_token_issued"] is False
