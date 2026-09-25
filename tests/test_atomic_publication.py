"""Artifacts are published all-or-nothing, and closed-pilot evidence is double-checked."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness import atomic_publish
from harness.atomic_publish import PublicationRefused, publish_atomically
from harness.closed_pilot_evidence import verify_closed_pilots, verify_evaluation

ROOT = Path(__file__).resolve().parent.parent


def _targets(tmp_path):
    first, second = tmp_path / "receipt.json", tmp_path / "design.json"
    first.write_bytes(b"old receipt")
    second.write_bytes(b"old design")
    return first, second


def _leftovers(tmp_path):
    return [path.name for path in tmp_path.iterdir() if path.name.endswith(".staged")]


def test_publication_promotes_every_artifact(tmp_path):
    first, second = _targets(tmp_path)
    seen = {}
    publish_atomically({first: b"new receipt", second: b"new design"},
                       verify=lambda staged: seen.update(staged))
    assert first.read_bytes() == b"new receipt" and second.read_bytes() == b"new design"
    assert set(seen) == {first, second} and _leftovers(tmp_path) == []


def test_a_failed_gate_preserves_the_previous_artifacts(tmp_path):
    first, second = _targets(tmp_path)

    def refuse(staged):
        assert staged[first].read_bytes() == b"new receipt"
        raise PublicationRefused("gate failed")
    with pytest.raises(PublicationRefused):
        publish_atomically({first: b"new receipt", second: b"new design"}, verify=refuse)
    assert first.read_bytes() == b"old receipt" and second.read_bytes() == b"old design"
    assert _leftovers(tmp_path) == []


def test_a_failed_promotion_rolls_back_what_was_already_promoted(tmp_path, monkeypatch):
    first, second = _targets(tmp_path)
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(source, target):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("target locked")
        return real_replace(source, target)
    monkeypatch.setattr(atomic_publish.os, "replace", flaky)
    with pytest.raises(OSError):
        publish_atomically({first: b"new receipt", second: b"new design"})
    assert first.read_bytes() == b"old receipt" and second.read_bytes() == b"old design"
    assert _leftovers(tmp_path) == []


def test_the_design_builder_preserves_artifacts_when_a_gate_fails(tmp_path, monkeypatch):
    from scripts import build_next_direction_design as design
    receipt, output = _targets(tmp_path)

    def failing_build(root, probe=True):
        raise PublicationRefused("coverage fails")
    monkeypatch.setattr(design, "build_artifacts", failing_build)
    assert design.main(["--receipt", str(receipt), "--output", str(output), "--no-probe"]) == 2
    assert receipt.read_bytes() == b"old receipt" and output.read_bytes() == b"old design"

    monkeypatch.setattr(design, "build_artifacts",
                        lambda root, probe=True: (b"new receipt", b"new design",
                                                  {"universe": None, "power_bytes": b""}))

    def failing_verify(*args, **kwargs):
        raise PublicationRefused("staged receipt does not freeze")
    monkeypatch.setattr(design, "verify_staged", failing_verify)
    assert design.main(["--receipt", str(receipt), "--output", str(output), "--no-probe"]) == 2
    assert receipt.read_bytes() == b"old receipt" and output.read_bytes() == b"old design"
    assert _leftovers(tmp_path) == []


# --- closed-pilot evidence ------------------------------------------------------------------

def _evaluation(root: Path, envelope_binds: str | None = None) -> dict:
    import hashlib
    result = root / "results/x/rehearsal_result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"function_results": []}', encoding="utf-8")
    result_sha = hashlib.sha256(result.read_bytes()).hexdigest()
    envelope = root / "results/x.json"
    envelope.write_text(json.dumps({"status": "complete", "rehearsal_result": {
        "path": "results/x/rehearsal_result.json",
        "sha256": envelope_binds or result_sha}}), encoding="utf-8")
    return {"envelope": {"path": "results/x.json",
                         "sha256": hashlib.sha256(envelope.read_bytes()).hexdigest()},
            "rehearsal_result": {"path": "results/x/rehearsal_result.json",
                                 "sha256": result_sha}}


def test_an_evaluation_verifies_by_envelope_and_raw_result(tmp_path):
    entry = _evaluation(tmp_path)
    assert verify_evaluation(tmp_path, entry) == []
    (tmp_path / "results/x/rehearsal_result.json").write_text("{}", encoding="utf-8")
    assert any("rehearsal_result hash differs" in problem
               for problem in verify_evaluation(tmp_path, entry))


def test_an_envelope_that_does_not_bind_its_raw_result_is_refused(tmp_path):
    entry = _evaluation(tmp_path, envelope_binds="0" * 64)
    assert any("does not bind the raw result" in problem
               for problem in verify_evaluation(tmp_path, entry))


def test_tracked_closed_pilots_verify_every_envelope_and_raw_result():
    if not (ROOT / "results/v4_3_tool_assisted_decision_receipt.json").exists():
        pytest.skip("closed-pilot receipts not present")
    report = verify_closed_pilots(ROOT)
    assert report["all_evaluations_verify"], report["evaluations"]
    assert set(report["evaluations"]) == {
        "tool_assisted/A", "tool_assisted/B", "tool_assisted/C", "execution_dose/control",
        "execution_dose/dose_treatment", "execution_dose/base"}
    assert report["tool_assisted"]["verdict"] == "fail"
