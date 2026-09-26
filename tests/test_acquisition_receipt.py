"""The acquisition receipt schema: identity is mandatory and events never contradict gates."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from harness.acquisition_receipt import (
    SCHEMA, ProtectedAccessMonitor, source_tree_identity, validate_receipt,
)

ROOT = Path(__file__).resolve().parent.parent
H = "a" * 64


def _receipt() -> dict:
    return {
        "schema_version": SCHEMA,
        "identity": {
            "git_commit": "c" * 40, "source_tree_sha256": H, "dirty_tree": False,
            "bundle_generation": "g" * 24, "bundle_manifest_sha256": H,
            "reference_universe_receipt_sha256": H, "isolation_version": "v5",
            "diff_policy_id": "A_prime", "candidate_repository_list_sha256": H,
            "tool_source_sha256": {"x.py": H}, "journal_sha256": H,
            "store_verification": {"objects_checked": 3, "problems": 0},
            "command": "python run.py", "configuration": {"cap": 10},
            "start_utc": "2026-09-26T00:00:00Z", "end_utc": "2026-09-26T01:00:00Z",
            "api": {"api_calls": 1}},
        "events": {"protected_data_access": False, "model_called": False,
                   "evaluation_set_created": False},
        "protected_access_evidence": {"installed": True, "protected_paths_opened": []},
        "gate": {"no_protected_data_access": True, "minimum_scale": False},
    }


def test_a_complete_consistent_receipt_is_accepted():
    assert validate_receipt(_receipt()) == []


@pytest.mark.parametrize("edit, expected", [
    (lambda r: r["events"].update(protected_data_access=True),
     "gate.no_protected_data_access contradicts the observed event"),
    (lambda r: r["gate"].update(protected_data_access=True),
     "gate must not carry an event field named protected_data_access"),
    (lambda r: r["gate"].update(no_protected_data_access=False),
     "gate.no_protected_data_access contradicts the observed event"),
    (lambda r: r["events"].update(model_called=True), "events.model_called must be false"),
    (lambda r: r["events"].update(evaluation_set_created=True),
     "events.evaluation_set_created must be false"),
    (lambda r: r["identity"].pop("reference_universe_receipt_sha256"),
     "identity.reference_universe_receipt_sha256 is absent"),
    (lambda r: r["identity"].update(source_tree_sha256=""), "identity.source_tree_sha256 is absent"),
    (lambda r: r["identity"].update(dirty_tree=None), "identity.dirty_tree must be a boolean"),
    (lambda r: r["protected_access_evidence"].update(installed=False),
     "protected-access evidence was not collected"),
    (lambda r: r["protected_access_evidence"].update(protected_paths_opened=["x/sealed.json"]),
     "protected-access evidence contradicts the event"),
    (lambda r: r["gate"].update(minimum_scale="yes"), "gate.minimum_scale must be a boolean"),
])
def test_contradictions_and_missing_identity_are_rejected(edit, expected):
    receipt = copy.deepcopy(_receipt())
    edit(receipt)
    assert expected in validate_receipt(receipt)


def test_publication_is_refused_without_identity(tmp_path):
    from scripts.run_repository_native_acquisition_pilot import publish
    receipt = _receipt()
    receipt["identity"].pop("bundle_generation")
    with pytest.raises(SystemExit):
        publish(receipt, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_the_audit_hook_records_protected_opens_only(tmp_path):
    ProtectedAccessMonitor.install()
    since = ProtectedAccessMonitor.mark()
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text("{}", encoding="utf-8")
    ordinary.read_text()
    assert ProtectedAccessMonitor.evidence(since)["protected_paths_opened"] == []
    protected = tmp_path / "sealed_final_items.json"
    protected.write_text("{}", encoding="utf-8")
    evidence = ProtectedAccessMonitor.evidence(since)
    assert evidence["installed"] and any("sealed_final" in path
                                         for path in evidence["protected_paths_opened"])


def test_source_tree_identity_is_computed_from_git():
    identity = source_tree_identity(ROOT)
    assert len(identity["git_commit"]) == 40 and len(identity["source_tree_sha256"]) == 64
    assert isinstance(identity["dirty_tree"], bool) and identity["source_tree_files"] > 100
