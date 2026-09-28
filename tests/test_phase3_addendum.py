"""Phase 3 correction addendum: deterministic, idempotent, atomic, pointer-correct."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from scripts import correct_phase3_addendum as addendum

ROOT = Path(__file__).resolve().parent.parent


def _copy_inputs(tmp_path: Path) -> Path:
    paths = {addendum.LEDGER, addendum.STATE, addendum.FIRST_CORRECTION}
    for versions in addendum.RECEIPTS.values():
        paths.update(versions.values())
    ledger = json.loads((ROOT / addendum.LEDGER).read_text(encoding="utf-8"))
    paths.update(e["receipt"] for h in ledger["hypotheses"] for e in h["evidence"]
                 if e.get("receipt") and e["receipt"] != addendum.ADDENDUM)
    for rel in paths:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    return tmp_path


def _pre_addendum(root: Path) -> None:
    """Roll the copied ledger/state back to their committed pre-addendum form if needed."""
    for rel in (addendum.LEDGER, addendum.STATE):
        data = json.loads((root / rel).read_text(encoding="utf-8"))
        if rel == addendum.LEDGER:
            for h in data["hypotheses"]:
                h["evidence"] = [e for e in h["evidence"] if e["phase"] != addendum.EVIDENCE_PHASE]
        (root / rel).write_text(json.dumps(data), encoding="utf-8")


def test_running_twice_is_byte_identical_and_adds_no_duplicates(tmp_path):
    root = _copy_inputs(tmp_path)
    _pre_addendum(root)
    first = addendum.run(root)
    snapshot = {rel: (root / rel).read_bytes()
                for rel in (addendum.LEDGER, addendum.STATE, addendum.ADDENDUM)}
    second = addendum.run(root)
    assert first == second
    for rel, data in snapshot.items():
        assert (root / rel).read_bytes() == data, rel
    ledger = json.loads(snapshot[addendum.LEDGER])
    for h in ledger["hypotheses"]:
        phases = [e["phase"] for e in h["evidence"]]
        assert phases.count(addendum.EVIDENCE_PHASE) == 1
        assert len(phases) == len(set(phases))


def test_current_state_and_links(tmp_path):
    root = _copy_inputs(tmp_path)
    _pre_addendum(root)
    result = addendum.run(root)
    ledger = json.loads((root / addendum.LEDGER).read_text(encoding="utf-8"))
    state = json.loads((root / addendum.STATE).read_text(encoding="utf-8"))
    status = {h["id"]: h["status"] for h in ledger["hypotheses"]}
    assert status == {"H1": "open", "H2": "strengthened", "H3": "strengthened", "H4": "open",
                      "H5": "weakened", "H6": "weakened", "H7": "strengthened"}
    h3 = next(h for h in ledger["hypotheses"] if h["id"] == "H3")
    assert h3["subclaims"]["adapter_or_parameter_capacity_causal"]["status"] == "open"
    allowed = set(ledger["status_values"])
    for h in ledger["hypotheses"]:
        for e in h["evidence"]:
            assert e["status_after"] in allowed
            if e.get("superseded_by"):
                assert e["superseded_by_sha256"] == result["sha256"]
                assert (root / e["superseded_by"]).exists()
    current = state["phase3_results"]["current"]
    assert current["3A"]["path"].endswith("_v3.json") and current["3C"]["path"].endswith("_v3.json")
    assert set(state["phase3_results"]["history"]) == {"v1", "v2", "commit_2e7ddba_correction"}
    assert "phase3a_result_receipt" not in state["phases"][3]
    events = [entry.get("event") for entry in state["log"]]
    own = events.index("phase 3 correction addendum v2")
    assert all(entry.get("historical") for entry in state["log"][:own])
    assert not any(entry.get("historical") for entry in state["log"][own:])


def test_later_log_entries_are_not_touched_by_a_rerun(tmp_path):
    root = _copy_inputs(tmp_path)
    _pre_addendum(root)
    addendum.run(root)
    state = json.loads((root / addendum.STATE).read_text(encoding="utf-8"))
    state["log"].append({"event": "a later phase", "utc": "2026-09-29"})
    (root / addendum.STATE).write_bytes(addendum._dump(state))
    before = (root / addendum.STATE).read_bytes()
    addendum.run(root)
    assert (root / addendum.STATE).read_bytes() == before


def test_a_failed_write_leaves_nothing_half_published(tmp_path, monkeypatch):
    root = _copy_inputs(tmp_path)
    _pre_addendum(root)
    before = {rel: (root / rel).read_bytes() for rel in (addendum.LEDGER, addendum.STATE)}
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:                       # ledger replaced, state replacement fails
            raise OSError("simulated failure")
        return real_replace(src, dst)

    monkeypatch.setattr(addendum.os, "replace", flaky)
    with pytest.raises(OSError):
        addendum.run(root)
    for rel, data in before.items():
        assert (root / rel).read_bytes() == data
    assert not (root / addendum.ADDENDUM).exists()
    assert not [p for p in (root / "results").iterdir() if ".json." in p.name]


def test_immutable_receipts_are_never_modified(tmp_path):
    root = _copy_inputs(tmp_path)
    _pre_addendum(root)
    watched = [addendum.FIRST_CORRECTION] + [p for v in addendum.RECEIPTS.values()
                                             for p in v.values()]
    before = {rel: (root / rel).read_bytes() for rel in watched}
    addendum.run(root)
    assert all((root / rel).read_bytes() == data for rel, data in before.items())
