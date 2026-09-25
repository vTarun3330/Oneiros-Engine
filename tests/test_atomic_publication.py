"""Crash-safe publication, and double-checked closed-pilot evidence."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from harness.atomic_publish import (
    MANIFEST, BundleUnreadable, PublicationRefused, publish_bundle, publish_file_atomically,
    read_current_bundle,
)
from harness.closed_pilot_evidence import verify_closed_pilots, verify_evaluation

ROOT = Path(__file__).resolve().parent.parent
OLD = {"receipt.json": b"old receipt", "design.json": b"old design", "power.json": b"old power"}
NEW = {"receipt.json": b"new receipt", "design.json": b"new design", "power.json": b"new power"}
POINTS = ["staged_file:design.json", "staged_file:power.json", "staged_file:receipt.json",
          "staged_manifest", "staged_directory_synced", "generation_renamed", "pointer_staged",
          "pointer_replaced"]


def _kill_during_publication(store: Path, point: str) -> int:
    """Publish NEW in a child process that dies (os._exit, no cleanup) at ``point``."""
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        from harness import atomic_publish
        def die(reached):
            if reached == {point!r}:
                os._exit(17)
        atomic_publish._checkpoint = die
        atomic_publish.publish_bundle({str(store)!r}, {NEW!r})
    """)
    return subprocess.run([sys.executable, "-c", script], cwd=ROOT).returncode


@pytest.mark.parametrize("point", POINTS)
def test_a_process_killed_at_any_point_leaves_a_complete_old_or_new_generation(tmp_path, point):
    store = tmp_path / "store"
    old_generation = publish_bundle(store, OLD)
    assert _kill_during_publication(store, point) == 17
    bundle = read_current_bundle(store)
    assert bundle["files"] in (OLD, NEW)            # never a mixture
    if point == "pointer_replaced":
        assert bundle["files"] == NEW and bundle["generation"] != old_generation
    else:
        assert bundle["files"] == OLD and bundle["generation"] == old_generation
    # The old generation is never touched, and a later publication still succeeds.
    from harness.atomic_publish import verify_generation
    old_manifest = (store / "generations" / old_generation / MANIFEST).read_bytes()
    import hashlib
    assert verify_generation(store, old_generation,
                             hashlib.sha256(old_manifest).hexdigest()) == OLD
    later = {**NEW, "design.json": b"later design"}
    publish_bundle(store, later)
    assert read_current_bundle(store)["files"] == later


def test_a_crash_before_the_first_publication_leaves_nothing_readable(tmp_path):
    store = tmp_path / "store"
    assert _kill_during_publication(store, "generation_renamed") == 17
    with pytest.raises(BundleUnreadable):
        read_current_bundle(store)


def test_a_failed_gate_leaves_the_current_generation(tmp_path):
    store = tmp_path / "store"
    publish_bundle(store, OLD)

    def refuse(staged):
        assert staged["receipt.json"].read_bytes() == b"new receipt"
        raise PublicationRefused("gate failed")
    with pytest.raises(PublicationRefused):
        publish_bundle(store, NEW, verify=refuse)
    assert read_current_bundle(store)["files"] == OLD
    assert not [p for p in (store / "generations").iterdir() if p.name.startswith(".partial")]
    assert not [p for p in store.iterdir() if p.name.endswith(".staged")]


def test_readers_reject_modified_extra_or_mispointed_generations(tmp_path):
    store = tmp_path / "store"
    generation = publish_bundle(store, OLD)
    directory = store / "generations" / generation
    (directory / "design.json").write_bytes(b"edited")
    with pytest.raises(BundleUnreadable):
        read_current_bundle(store)
    (directory / "design.json").write_bytes(OLD["design.json"])
    (directory / "extra.json").write_bytes(b"{}")
    with pytest.raises(BundleUnreadable):
        read_current_bundle(store)
    (directory / "extra.json").unlink()
    pointer = json.loads((store / "CURRENT").read_text())
    (store / "CURRENT").write_text(json.dumps({**pointer, "manifest_sha256": "0" * 64}))
    with pytest.raises(BundleUnreadable):
        read_current_bundle(store)


def test_an_accepted_generation_is_never_overwritten(tmp_path):
    store = tmp_path / "store"
    first = publish_bundle(store, OLD)
    before = {p.name: p.read_bytes() for p in (store / "generations" / first).iterdir()}
    second = publish_bundle(store, NEW)
    third = publish_bundle(store, OLD)   # new generation: it records a different predecessor
    assert len({first, second, third}) == 3
    assert {p.name: p.read_bytes() for p in (store / "generations" / first).iterdir()} == before
    with pytest.raises(PublicationRefused):
        publish_bundle(store, {MANIFEST: b"x"})


def test_single_file_publication_is_per_file_atomic(tmp_path):
    target = tmp_path / "power.json"
    target.write_bytes(b"old")

    def refuse(staged):
        raise PublicationRefused("gate failed")
    with pytest.raises(PublicationRefused):
        publish_file_atomically(target, b"new", verify=refuse)
    assert target.read_bytes() == b"old"
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        from harness import atomic_publish
        atomic_publish._checkpoint = lambda point: os._exit(17)
        atomic_publish.publish_file_atomically(__import__("pathlib").Path({str(target)!r}), b"new")
    """)
    assert subprocess.run([sys.executable, "-c", script], cwd=ROOT).returncode == 17
    assert target.read_bytes() == b"old"
    publish_file_atomically(target, b"new")
    assert target.read_bytes() == b"new"


def test_the_design_builder_leaves_the_current_generation_when_a_gate_fails(tmp_path,
                                                                           monkeypatch):
    from scripts import build_next_direction_design as design
    store = tmp_path / "store"
    old = {design.RECEIPT_FILE: b"old receipt", design.DESIGN_FILE: b"old design",
           design.POWER_FILE: b"old power"}
    publish_bundle(store, old)

    def failing_build(root, probe=True):
        raise PublicationRefused("coverage fails")
    monkeypatch.setattr(design, "build_artifacts", failing_build)
    assert design.main(["--store", str(store), "--no-probe"]) == 2
    assert read_current_bundle(store)["files"] == old

    new = {name: b"new " + name.encode() for name in old}
    monkeypatch.setattr(design, "build_artifacts",
                        lambda root, probe=True: (new, {"universe": None, "power_bytes": b""}))

    def failing_verify(*args, **kwargs):
        raise PublicationRefused("staged receipt does not freeze")
    monkeypatch.setattr(design, "verify_staged", failing_verify)
    assert design.main(["--store", str(store), "--no-probe"]) == 2
    assert read_current_bundle(store)["files"] == old


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
