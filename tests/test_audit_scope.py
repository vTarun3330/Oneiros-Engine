"""Protected-location audit: resolved paths, complete process scope, subprocess confinement.

Every protected file used here is a FAKE created under a temporary root; no real
protected dataset is ever opened.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from harness.acquisition_receipt import (
    ProtectedAccessMonitor, ProtectedLocations, confine_to_store,
)
from harness.github_acquisition import IntegrityViolation, LocalGitRepository
from tests.test_acquisition_receipt import _receipt

ROOT = Path(__file__).resolve().parent.parent

PROTECTED = [
    ("data/corpus/v4_1_research_hardened_candidate/development_view/val.records.json",
     "non_train_development_shard"),
    ("data/corpus/v4_1_research_hardened_candidate/development_view/ablation_dev.records.json",
     "non_train_development_shard"),
    ("data/corpus/v4_2_balanced_expansion_candidate/development_view/val.records.json",
     "non_train_development_shard"),
    ("data/corpus/v4_2_balanced_expansion_candidate/development_view/test.records.json",
     "non_train_development_shard"),
    ("data/corpus/v4_1_research_hardened_candidate/records.json", "canonical_corpus_records"),
    ("data/corpus/v1/records.json", "canonical_corpus_records"),
    ("data/corpus/v3_final_candidate/splits.json", "corpus_split_assignment"),
    ("results/sealed_final_state.json", "sealed_final"),
    ("results/sealed_final_run_state.json", "sealed_final"),
    ("results/sealed_final_audit.log", "sealed_final"),
    ("results/sealed_final_base_qwen_s42/rehearsal_result.json", "sealed_final"),
    ("results/v4_3_execution_supervision_v1/unopened_confirmation.ids.json",
     "reserved_confirmation"),
]
NOT_PROTECTED = [
    "data/repository_native/aprime_confirmation_pilot/journal.jsonl",
    "data/repository_native/aprime_confirmation_pilot/objects/git/00/abc",
    "data/repository_native/aprime_confirmation_pilot_retry/journal.jsonl",
    "docs/repository_native_confirmation_repositories.json",
    "docs/SEALED_FINAL_INCIDENT.md",
    "harness/sealed_final_loader.py",
    "scripts/confirmation_helpers.py",
    "results/v4_2_sealed_final_executable_receipt_v6.json",
    "results/v4_3_repository_native_aprime_confirmation_pilot_REFUSED.json",
    "data/corpus/v4_1_research_hardened_candidate/development_view/train.records.json",
    "data/corpus/v4_1_research_hardened_candidate/development_view/manifest.json",
    "data/corpus/v4_1_research_hardened_candidate/manifest.json",
]


@pytest.mark.parametrize("relative, reason", PROTECTED)
def test_every_real_protected_location_is_recognised(relative, reason):
    locations = ProtectedLocations(ROOT)
    assert locations.reason(ROOT / relative) == reason       # resolved; never opened here
    assert locations.reason(str(ROOT / relative).upper()) == (reason if sys.platform == "win32"
                                                              else None)


@pytest.mark.parametrize("relative", NOT_PROTECTED)
def test_harmless_names_are_not_flagged(relative):
    assert ProtectedLocations(ROOT).reason(ROOT / relative) is None


def test_relative_and_dotted_paths_resolve_before_matching(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    locations = ProtectedLocations(ROOT)
    assert locations.reason("data/corpus/v1/../v1/records.json") == "canonical_corpus_records"
    assert locations.reason("docs/../results/sealed_final_state.json") == "sealed_final"


# --- complete-scope integration: a protected open anywhere refuses publication -------------

@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "root"
    shard = root / "data/corpus/demo/development_view/val.records.json"
    shard.parent.mkdir(parents=True)
    shard.write_text("[]", encoding="utf-8")
    listing = tmp_path / "repositories.json"
    listing.write_text(json.dumps({"repositories": []}), encoding="utf-8")
    config = {"label": "audit test", "repositories_file": str(listing),
              "store": str(tmp_path / "store"), "report": str(tmp_path / "report.json")}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    yield root, shard, tmp_path
    ProtectedAccessMonitor.configure(ROOT)


def _patch(monkeypatch, pilot, where: str | None, shard: Path):
    def touch():
        shard.read_text(encoding="utf-8")

    def load_frozen():
        if where == "load_frozen":
            touch()
        return None, {"bundle_generation": "g" * 24, "bundle_manifest_sha256": "a" * 64}

    class Frozen:
        receipt_sha256 = "a" * 64

    def fake_load():
        _, identity = load_frozen()
        return Frozen(), identity

    def run(store_root, config, **kwargs):
        if where == "acquisition":
            touch()
        return 0

    def build_report(store_root, config, gate_spec, *, frozen_identity, label):
        if where == "report":
            touch()
        report = copy.deepcopy(_receipt())
        report["protected_access_evidence"] = None
        report["gate"] = {"no_protected_data_access": None}
        return report

    monkeypatch.setattr(pilot, "load_frozen", fake_load)
    monkeypatch.setattr(pilot, "run", run)
    monkeypatch.setattr(pilot, "build_report", build_report)


def test_without_a_protected_open_the_receipt_publishes(fake_root, monkeypatch):
    from scripts import run_repository_native_acquisition_pilot as pilot
    root, shard, tmp = fake_root
    _patch(monkeypatch, pilot, None, shard)
    assert pilot.main(["--config", str(tmp / "config.json")], audit_root=root) == 0
    published = json.loads((tmp / "report.json").read_text(encoding="utf-8"))
    assert published["events"]["protected_data_access"] is False
    assert published["protected_access_evidence"]["complete"] is True


@pytest.mark.parametrize("where", ["load_frozen", "acquisition", "report"])
def test_a_protected_open_in_any_phase_refuses_publication(fake_root, monkeypatch, where):
    from scripts import run_repository_native_acquisition_pilot as pilot
    root, shard, tmp = fake_root
    _patch(monkeypatch, pilot, where, shard)
    with pytest.raises(SystemExit) as refused:
        pilot.main(["--config", str(tmp / "config.json")], audit_root=root)
    assert "protected" in str(refused.value)
    assert not (tmp / "report.json").exists()


# --- subprocesses ------------------------------------------------------------------------------

def test_a_subprocess_argument_in_a_protected_location_is_recorded(fake_root):
    root, shard, _ = fake_root
    ProtectedAccessMonitor.install(root)
    scope = ProtectedAccessMonitor.scope()
    subprocess.run([sys.executable, "-c", "pass", str(shard)], check=True)
    evidence = scope.close()
    assert any(a["kind"] == "subprocess_argument" and a["reason"] == "non_train_development_shard"
               for a in evidence["protected_accesses"])
    assert any(str(shard) in command["argv"] for command in evidence["subprocess_commands"])


def test_git_directories_are_confined_to_the_store(fake_root):
    root, shard, tmp = fake_root
    ProtectedAccessMonitor.install(root)
    store = tmp / "store"
    store.mkdir(exist_ok=True)
    LocalGitRepository(store / "repos" / "a.git", "https://example.invalid/a.git",
                       allowed_root=store)
    with pytest.raises(IntegrityViolation):
        LocalGitRepository(tmp / "elsewhere.git", "https://example.invalid/a.git",
                           allowed_root=store)
    protected_dir = root / "data/corpus/demo/development_view"
    with pytest.raises(IntegrityViolation):
        LocalGitRepository(protected_dir / "val.records.json", "https://example.invalid/a.git",
                           allowed_root=root)
    with pytest.raises(ValueError):
        confine_to_store(store / ".." / "escape", store)
