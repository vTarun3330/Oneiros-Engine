"""Portable Phase 4 evidence: bundles self-validate on any clone; v2 denominators are honest."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from harness.evidence_bundle import bundle_problems, sha256_text
from scripts import phase4_evidence_bundles as bundles

ROOT = Path(__file__).resolve().parent.parent
NATIVE_V2 = ROOT / "results" / "sft_root_cause_phase4_native_rehearsal_receipt_v2.json"


def _load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_both_bundles_verify():
    assert bundles.verify() == {bundles.NATIVE_OUT: [], bundles.SMOKE_OUT: []}


def test_bundles_verify_without_the_ignored_originals(tmp_path):
    for rel in (bundles.NATIVE_OUT, bundles.SMOKE_OUT):
        bundle = _load(rel)
        # A root holding none of the originals: only the portable checks can run.
        assert bundle_problems(tmp_path, bundle) == []
        for entry in bundle["files"].values():
            assert not (tmp_path / entry["original_path"]).exists()


def test_tampering_and_private_paths_are_detected():
    bundle = _load(bundles.SMOKE_OUT)
    edited = copy.deepcopy(bundle)
    entry = edited["files"]["control_receipt.json"]
    entry["text"] = entry["text"].replace('"seed": 42', '"seed": 43')
    assert any("does not match its hash" in p for p in bundle_problems(ROOT, edited))
    leaked = copy.deepcopy(bundle)
    entry = leaked["files"]["control_run_status.json"]
    entry["text"] += "C:\\\\Users\\\\someone\\\\x"
    entry["text_sha256"] = sha256_text(entry["text"])
    assert any("machine-private" in p for p in bundle_problems(ROOT, leaked))
    swapped = copy.deepcopy(bundle)
    swapped["facts"]["arms"]["treatment"]["per_batch"][0]["input_ids_sha256"] = "0" * 64
    problems = bundles.smoke_problems(swapped)
    assert "per-batch input_ids differ between arms" in problems


def test_native_bundle_matches_the_v1_receipt_record_for_record():
    bundle = _load(bundles.NATIVE_OUT)
    records = bundle["files"]["records.jsonl"]
    assert records["normalizations"] == []
    assert records["text_sha256"] == records["original_sha256"] == _load(
        bundles.NATIVE_RECEIPT)["inputs"][f"{bundles.NATIVE_DIR}/records.jsonl"]
    facts = bundle["facts"]
    assert facts["targets"] == 26 and len(facts["per_target"]) == 26
    assert all(row["wall_seconds"] is not None for row in facts["per_target"])
    assert bundle["dependency_resolution"]["recorded_in_v1"] is False


def test_smoke_bundle_preserves_batches_tokens_and_run_status():
    arms = _load(bundles.SMOKE_OUT)["facts"]["arms"]
    control, treatment = arms["control"], arms["treatment"]
    assert control["dataset"]["supervised_tokens"] == 952
    assert treatment["dataset"]["supervised_tokens"] == 220
    assert len(control["per_batch"]) == len(treatment["per_batch"]) > 0
    for arm in (control, treatment):
        assert (arm["state"], arm["exit_code"], arm["artifact_contract"]) == (
            "completed", 0, "none_declared")
        assert arm["optimizer"]["completed_steps"] == 2
        assert all({"input_ids_sha256", "attention_mask_sha256", "labels_sha256",
                    "supervised_tokens"} <= set(batch) for batch in arm["per_batch"])
    assert control["revision"] == treatment["revision"]
    assert control["seed"] == treatment["seed"] == 42


def test_native_v2_separates_the_three_denominators_and_readiness():
    receipt = json.loads(NATIVE_V2.read_text(encoding="utf-8"))
    d = receipt["denominators"]
    assert (d["environment_success"]["k"], d["environment_success"]["n"]) == (25, 26)
    assert (d["semantic_qualification_among_evaluable"]["k"],
            d["semantic_qualification_among_evaluable"]["n"]) == (24, 25)
    assert (d["operational_end_to_end_qualified_throughput"]["k"],
            d["operational_end_to_end_qualified_throughput"]["n"]) == (24, 26)
    assert receipt["readiness"]["native_environment_feasible"] is True
    assert receipt["readiness"]["fixed_input_ready"] is False
    assert receipt["readiness"]["choice_A_ready"] is False
    assert [t["category"] for t in receipt["infrastructure_failures"]["targets"]] == [
        "test_infrastructure_error"]
    assert receipt["supersedes"]["sha256"] == hashlib.sha256(
        (ROOT / receipt["supersedes"]["path"]).read_bytes()).hexdigest()


def test_v1_native_and_smoke_receipts_are_unchanged():
    for commit, rel in (("94a96be", "results/sft_root_cause_phase4_native_rehearsal_receipt_v1.json"),
                        ("62476ef", "results/sft_root_cause_phase4_objective_smoke_v1.json")):
        committed = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT,
                                   capture_output=True, check=True).stdout
        assert (ROOT / rel).read_bytes().replace(b"\r\n", b"\n") == committed, rel


def test_future_native_runs_sanitise_recorded_paths():
    runner = pytest.importorskip("scripts.native_rehearsal_wsl")
    checkout = runner.WORK / "wt" / "t" / "fixed"
    text = f"pkg @ file://{checkout}\nother==1.0\n{runner.WORK}/x"
    assert runner.sanitize(text, checkout) == "pkg @ file://<checkout>\nother==1.0\n<work>/x"
    assert {"pyproject.toml", "uv.lock", "requirements.txt"} <= set(runner.DEPENDENCY_FILES)
