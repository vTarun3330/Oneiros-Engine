import json
from pathlib import Path

import pytest

from harness.corpus import record_content_hash, sha256_file, write_json
from harness.corpus_view import (
    load_complexity_index,
    load_development_split,
    materialize_development_view,
    verify_development_view,
)


def _record(record_id: str, group: str, code: str) -> dict:
    record = {
        "schema_version": 1,
        "id": record_id,
        "task_type": "hidden_mutation_reproduction",
        "language": "python",
        "source": {"name": "fixture", "upstream": "fixture"},
        "group_id": group,
        "code_under_test": code,
        "reference_code": code,
        "entry_point": "target",
        "specification": "Return a result.",
        "tests": [{"code": "assert target(1) == 1"}],
        "provenance": {"mutation_type": "fixture"},
        "quality": {"execution_mode": "function_assertion"},
        "task_mode": "function",
        "test_format": "assert_statement",
        "target_symbols": ["target"],
        "support_context": "",
        "prompt_code_under_test": code,
    }
    record["content_hash"] = record_content_hash(record)
    return record


def _fixture_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    corpus = tmp_path / "v4_1_fixture"
    corpus.mkdir()
    records = [
        _record("train-1", "group-train", "def target(x):\n    return x\n"),
        _record(
            "dev-1",
            "group-dev",
            "def target(x):\n    for i in range(x):\n        if i:\n            while x:\n                if x % 2 and x > 2:\n                    x -= 1\n    return x\n",
        ),
        _record("val-1", "group-val", "def target(x):\n    return x\n"),
        _record("sealed-secret", "group-test", "def target(x):\n    return x + 99\n"),
    ]
    splits = {
        "train": ["train-1"],
        "ablation_dev": ["dev-1"],
        "val": ["val-1"],
        "test": ["sealed-secret"],
    }
    write_json(corpus / "records.json", records)
    write_json(corpus / "splits.json", splits)
    write_json(corpus / "training_exclusions.json", [])
    manifest = {
        "schema_version": 1,
        "corpus_id": "fixture-corpus",
        "version": "v4_1_fixture",
        "files": {
            "records.json": {"sha256": sha256_file(corpus / "records.json")},
            "splits.json": {"sha256": sha256_file(corpus / "splits.json")},
        },
        "splits": {
            name: {
                "record_count": len(ids),
                "record_ids_sha256": __import__("hashlib").sha256(
                    json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for name, ids in splits.items()
        },
    }
    write_json(corpus / "manifest.json", manifest)
    monkeypatch.setattr("harness.corpus_view.verify_corpus", lambda _: manifest)
    return corpus


def test_view_excludes_sealed_payload_and_loads_complexity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _fixture_corpus(tmp_path, monkeypatch)

    manifest = materialize_development_view(corpus)

    assert manifest["sealed_splits_excluded"] == ["test"]
    for path in (corpus / "development_view").glob("*.json"):
        assert "sealed-secret" not in path.read_text(encoding="utf-8")
        assert "x + 99" not in path.read_text(encoding="utf-8")
    assert load_development_split(corpus, "train")[0]["id"] == "train-1"
    assert load_complexity_index(corpus)["dev-1"]["tier"] == "complex"


def test_view_refuses_test_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _fixture_corpus(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="sealed"):
        materialize_development_view(corpus, ["train", "test"])
    with pytest.raises(ValueError, match="sealed"):
        verify_development_view(corpus, ["test"])


def test_view_detects_modified_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _fixture_corpus(tmp_path, monkeypatch)
    materialize_development_view(corpus)
    shard = corpus / "development_view" / "train.records.json"
    shard.write_text("[]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash gate"):
        verify_development_view(corpus, ["train"])
