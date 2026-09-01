"""Upstream dataset identity for sampling/reporting, never model prompts."""

from collections.abc import Mapping
from typing import Any


DATASET_IDENTITY_POLICY = "source.upstream_then_source.name_v1"


def dataset_name_from_source(source: Any) -> str:
    if isinstance(source, Mapping):
        for key in ("upstream", "name"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "unknown"
    return source.strip() if isinstance(source, str) and source.strip() else "unknown"


def dataset_name_for_pair(pair: Mapping[str, Any]) -> str:
    """Use canonical source metadata; legacy fixtures may provide flat labels.

    Never infer an upstream benchmark from record ids, code, or test contents.
    A missing label stays unknown instead of being guessed.
    """
    if "source" in pair:
        return dataset_name_from_source(pair["source"])
    return dataset_name_from_source(pair.get("dataset_name") or pair.get("source_name"))
