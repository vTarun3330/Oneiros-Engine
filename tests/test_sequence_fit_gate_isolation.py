"""A train-only gate must never open the file that holds the sealed test.

`records.json` is the canonical corpus: all four splits in one file, sealed
final test included. Reading it to slice out the train ids works, produces the
right answer, and holds the sealed records in memory for no reason - an
exposure that nothing in the gate's output would reveal.

These tests do not inspect the source for a forbidden string. They intercept
the filesystem underneath the gate and fail the read, so the gate is proven not
to perform it rather than observed not to mention it.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

import scripts.sequence_fit_gate as gate

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
FORBIDDEN = ("records.json", "splits.json")


class CanonicalCorpusOpened(AssertionError):
    """Raised by the interceptor, so a leak fails loudly rather than passing."""


def _is_canonical(name: str) -> bool:
    path = Path(str(name))
    # Only the canonical corpus root, never the per-split shards inside
    # development_view/ (train.records.json and friends).
    return path.name in FORBIDDEN and path.parent.name != "development_view"


@pytest.fixture
def refuse_canonical_reads(monkeypatch):
    """Make any read of the canonical corpus raise."""
    real_open = builtins.open
    real_read_text = Path.read_text
    opened: list[str] = []

    def guarded_open(file, *args, **kwargs):
        if _is_canonical(file):
            opened.append(str(file))
            raise CanonicalCorpusOpened(str(file))
        return real_open(file, *args, **kwargs)

    def guarded_read_text(self, *args, **kwargs):
        if _is_canonical(self):
            opened.append(str(self))
            raise CanonicalCorpusOpened(str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    return opened


def test_the_interceptor_actually_fires(refuse_canonical_reads):
    """Without this, the isolation test could pass by doing nothing at all."""
    with pytest.raises(CanonicalCorpusOpened):
        (CORPUS / "records.json").read_text(encoding="utf-8")
    with pytest.raises(CanonicalCorpusOpened):
        builtins.open(CORPUS / "splits.json")
    assert len(refuse_canonical_reads) == 2


def test_the_interceptor_still_permits_the_development_shard(refuse_canonical_reads):
    shard = CORPUS / "development_view" / "train.records.json"
    if not shard.exists():
        pytest.skip("development view not materialised in this checkout")
    payload = json.loads(shard.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload
    assert refuse_canonical_reads == []


@pytest.mark.skipif(
    not (CORPUS / "development_view" / "train.records.json").exists(),
    reason="development view not materialised in this checkout")
def test_the_gate_never_opens_the_canonical_corpus(refuse_canonical_reads):
    """The real assertion: run the gate with canonical reads booby-trapped."""
    report = gate.run(
        split="train", corpus_dir=CORPUS,
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", revision="main",
        prompt_budget=1024, completion_budget=1024, sequence_limit=3072,
        information_variant="full", instruction_variant="self_contained",
    )
    assert refuse_canonical_reads == [], (
        "the gate opened the canonical corpus: " + ", ".join(refuse_canonical_reads))
    assert report["canonical_records_json_opened"] is False
    assert report["sealed_final_test_accessed"] is False
    assert report["gate_passed"] is True
    assert report["generation_panel_size"] == 5595


def test_the_gate_refuses_the_sealed_split_outright():
    with pytest.raises(SystemExit, match="sealed"):
        gate.run(split="test", corpus_dir=CORPUS,
                 model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", revision="main",
                 prompt_budget=1024, completion_budget=1024,
                 sequence_limit=3072, information_variant="full",
                 instruction_variant="self_contained")


def test_the_committed_gate_report_declares_its_source():
    report_path = ROOT / "results" / "v4_2_sequence_fit_gate_train_successor.json"
    if not report_path.exists():
        pytest.skip("gate not run in this checkout")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["canonical_records_json_opened"] is False
    assert "development view" in report["record_source"]
    assert report["schema_version"].endswith("_v2")
