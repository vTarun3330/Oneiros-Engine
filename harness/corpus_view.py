"""Sealed-test-safe split shards for local Oneiros development runs.

The canonical historical corpora store every split in one ``records.json``.
That layout is reproducible but unsuitable for a strict local sealed-test run:
loading ``train`` still deserializes ``test``.  This module materializes and
verifies a development-only view containing train, ablation-dev, and validation
shards.  No test record or test shard is written to the view.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from harness.corpus import (
    record_content_hash,
    sha256_file,
    verify_corpus,
    write_json,
)
from harness.function_complexity import (
    COMPLEXITY_POLICY_VERSION,
    analyze_function_complexity,
)


VIEW_SCHEMA_VERSION = "oneiros_development_corpus_view_v1"
COMPLEXITY_MANIFEST_SCHEMA_VERSION = "oneiros_complex_function_manifest_v1"
DEFAULT_INCLUDED_SPLITS = ("train", "ablation_dev", "val")
SEALED_SPLITS = frozenset({"test"})


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _view_dir(corpus_dir: Path) -> Path:
    return corpus_dir / "development_view"


def _record_shard_name(split: str) -> str:
    return f"{split}.records.json"


def _dataset(record: dict[str, Any]) -> str:
    source = record.get("source", {})
    if isinstance(source, dict):
        return str(source.get("upstream") or source.get("name") or "unknown")
    return str(source or "unknown")


def _complexity_entry(record: dict[str, Any], split: str) -> dict[str, Any] | None:
    mode = record.get("quality", {}).get("execution_mode", "function_assertion")
    if mode != "function_assertion":
        return None
    metrics = analyze_function_complexity(
        record.get("prompt_code_under_test") or record["code_under_test"],
        record["entry_point"],
    ).to_dict()
    return {
        "record_id": record["id"],
        "split": split,
        "dataset": _dataset(record),
        "mutation_family": str(
            record.get("provenance", {}).get("mutation_type")
            or record.get("provenance", {}).get("category")
            or record.get("task_type", "unknown")
        ),
        **metrics,
    }


def materialize_development_view(
    corpus_dir: Path,
    included_splits: Iterable[str] = DEFAULT_INCLUDED_SPLITS,
) -> dict[str, Any]:
    """Create exact, hash-bound shards while writing no sealed-test payload.

    The source corpus is fully verified inside this one-purpose materializer.
    It emits only aggregate counts and hashes; callers must not log discarded
    records.  All subsequent local training and preflight paths use the shards
    and therefore never open the combined canonical records file.
    """
    corpus_dir = corpus_dir.resolve()
    split_names = tuple(dict.fromkeys(str(item) for item in included_splits))
    if not split_names or SEALED_SPLITS & set(split_names):
        raise ValueError("development view cannot be empty or include a sealed split")

    parent = verify_corpus(corpus_dir)
    split_ids = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    if not set(split_names).issubset(split_ids):
        raise ValueError("requested development split is absent from the canonical corpus")
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {record["id"]: record for record in records}
    output_dir = _view_dir(corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    view_splits: dict[str, Any] = {}
    complexity_entries: list[dict[str, Any]] = []
    for split in split_names:
        ids = list(split_ids[split])
        shard = [by_id[record_id] for record_id in ids]
        shard_path = output_dir / _record_shard_name(split)
        write_json(shard_path, shard)
        view_splits[split] = {
            "filename": shard_path.name,
            "record_count": len(shard),
            "record_ids_sha256": _sha256_json(ids),
            "sha256": sha256_file(shard_path),
        }
        if split in {"train", "ablation_dev"}:
            complexity_entries.extend(
                entry
                for record in shard
                if (entry := _complexity_entry(record, split)) is not None
            )

    included_ids = {
        record_id for split in split_names for record_id in split_ids[split]
    }
    source_exclusions_path = corpus_dir / "training_exclusions.json"
    exclusions = (
        json.loads(source_exclusions_path.read_text(encoding="utf-8"))
        if source_exclusions_path.exists()
        else []
    )
    exclusions = [item for item in exclusions if item.get("record_id") in included_ids]
    exclusions_path = output_dir / "training_exclusions.json"
    write_json(exclusions_path, exclusions)

    tier_counts = Counter(item["tier"] for item in complexity_entries)
    split_tier_counts = {
        split: dict(sorted(Counter(
            item["tier"] for item in complexity_entries if item["split"] == split
        ).items()))
        for split in ("train", "ablation_dev")
        if split in split_names
    }
    complexity_manifest = {
        "schema_version": COMPLEXITY_MANIFEST_SCHEMA_VERSION,
        "policy_version": COMPLEXITY_POLICY_VERSION,
        "source_lineage": [
            "buggy_revision:prompt_code_under_test",
            "explicit_public_entry_point:entry_point",
        ],
        "prohibited_sources_used": [],
        "eligible_splits": [
            split for split in ("train", "ablation_dev") if split in split_names
        ],
        "function_record_count": len(complexity_entries),
        "tier_counts": dict(sorted(tier_counts.items())),
        "tier_counts_by_split": split_tier_counts,
        "records": complexity_entries,
    }
    complexity_path = output_dir / "complexity_manifest.json"
    write_json(complexity_path, complexity_manifest)

    view_manifest = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "source_corpus_id": parent["corpus_id"],
        "source_corpus_version": parent.get("version", corpus_dir.name),
        "source_records_sha256": parent["files"]["records.json"]["sha256"],
        "source_splits_sha256": parent["files"]["splits.json"]["sha256"],
        "source_corpus_verified_before_materialization": True,
        "included_splits": list(split_names),
        "sealed_splits_excluded": sorted(SEALED_SPLITS),
        "splits": view_splits,
        "training_exclusions": {
            "filename": exclusions_path.name,
            "record_count": len(exclusions),
            "sha256": sha256_file(exclusions_path),
        },
        "complexity_manifest": {
            "filename": complexity_path.name,
            "policy_version": COMPLEXITY_POLICY_VERSION,
            "sha256": sha256_file(complexity_path),
        },
    }
    write_json(output_dir / "manifest.json", view_manifest)
    verify_development_view(corpus_dir, split_names)
    return view_manifest


def verify_development_view(
    corpus_dir: Path,
    required_splits: Iterable[str],
) -> dict[str, Any]:
    """Verify split shards without opening canonical ``records.json``."""
    corpus_dir = corpus_dir.resolve()
    required = tuple(dict.fromkeys(str(item) for item in required_splits))
    if not required or SEALED_SPLITS & set(required):
        raise ValueError("development verification cannot include a sealed split")

    parent_path = corpus_dir / "manifest.json"
    view_path = _view_dir(corpus_dir) / "manifest.json"
    try:
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        view = json.loads(view_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("development corpus view manifest is missing or unreadable") from exc
    if view.get("schema_version") != VIEW_SCHEMA_VERSION:
        raise RuntimeError("unsupported development corpus view schema")
    if view.get("source_corpus_id") != parent.get("corpus_id"):
        raise RuntimeError("development view belongs to a different corpus")
    if view.get("source_records_sha256") != parent.get("files", {}).get("records.json", {}).get("sha256"):
        raise RuntimeError("development view records identity is stale")
    if view.get("source_splits_sha256") != parent.get("files", {}).get("splits.json", {}).get("sha256"):
        raise RuntimeError("development view split identity is stale")
    if set(view.get("sealed_splits_excluded", [])) != SEALED_SPLITS:
        raise RuntimeError("development view does not declare the sealed split excluded")
    if not set(required).issubset(view.get("included_splits", [])):
        raise RuntimeError("required split is absent from the development view")

    output_dir = _view_dir(corpus_dir)
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for split in required:
        descriptor = view.get("splits", {}).get(split, {})
        shard_path = output_dir / str(descriptor.get("filename", ""))
        if not shard_path.is_file() or sha256_file(shard_path) != descriptor.get("sha256"):
            raise RuntimeError(f"development shard {split!r} failed its hash gate")
        records = json.loads(shard_path.read_text(encoding="utf-8"))
        ids = [record.get("id") for record in records if isinstance(record, dict)]
        if (
            len(records) != descriptor.get("record_count")
            or len(ids) != len(records)
            or _sha256_json(ids) != descriptor.get("record_ids_sha256")
        ):
            raise RuntimeError(f"development shard {split!r} has invalid membership")
        parent_split = parent.get("splits", {}).get(split, {})
        if (
            parent_split.get("record_count") != len(records)
            or parent_split.get("record_ids_sha256") != descriptor.get("record_ids_sha256")
        ):
            raise RuntimeError(f"development shard {split!r} differs from the canonical split")
        for record in records:
            if record["id"] in seen_ids or record_content_hash(record) != record.get("content_hash"):
                raise RuntimeError("development shard contains a duplicate or modified record")
            if record.get("schema_version") != parent.get("schema_version"):
                raise RuntimeError("development shard record schema is invalid")
            if record.get("group_id") in seen_groups:
                raise RuntimeError("development shards are not semantic-group disjoint")
            seen_ids.add(record["id"])
        seen_groups.update(record.get("group_id") for record in records)

    for key in ("training_exclusions", "complexity_manifest"):
        descriptor = view.get(key, {})
        path = output_dir / str(descriptor.get("filename", ""))
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            raise RuntimeError(f"development view {key} failed its hash gate")
    return view


def load_development_split(
    corpus_dir: Path, split: str, *, include_excluded: bool = False,
) -> list[dict[str, Any]]:
    """Load one exact non-test shard after split-only verification."""
    view = verify_development_view(corpus_dir, [split])
    output_dir = _view_dir(corpus_dir)
    records = json.loads(
        (output_dir / view["splits"][split]["filename"]).read_text(encoding="utf-8")
    )
    if split in {"train", "ablation_dev"} and not include_excluded:
        exclusions = json.loads(
            (output_dir / view["training_exclusions"]["filename"]).read_text(encoding="utf-8")
        )
        excluded_ids = {item["record_id"] for item in exclusions}
        records = [record for record in records if record["id"] not in excluded_ids]
    return records


def load_complexity_index(corpus_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the hash-bound buggy-side complexity sidecar."""
    view = verify_development_view(corpus_dir, ["train"])
    descriptor = view["complexity_manifest"]
    payload = json.loads(
        (_view_dir(corpus_dir) / descriptor["filename"]).read_text(encoding="utf-8")
    )
    if (
        payload.get("schema_version") != COMPLEXITY_MANIFEST_SCHEMA_VERSION
        or payload.get("policy_version") != COMPLEXITY_POLICY_VERSION
        or payload.get("prohibited_sources_used") != []
    ):
        raise RuntimeError("complexity manifest failed its provenance gate")
    return {item["record_id"]: item for item in payload["records"]}
