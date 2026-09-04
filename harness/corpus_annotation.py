"""Top-level origin grouping and defect annotation for corpus records.

The V4.1 corpus records where each item came from, but not in a shape that
supports balanced sampling or per-family reporting:

* every real-repository record carries the single generic mutation family
  ``real_repository_defect``;
* synthetic mutants and repository defects share one flat ``mutation_family``
  vocabulary, so a raw row count silently over-weights whichever source
  produced the most sibling mutants;
* repository records have no complexity tier at all.

This module derives, per record, the fields needed for Parts 2, 3, and 5 of the
research plan.  It reads supervision-side evidence (the fixed revision, the
official traceback, patch paths) because corpus construction is allowed to;
nothing it returns may be rendered into a model prompt.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from harness.function_complexity import (
    COMPLEXITY_POLICY_VERSION,
    analyze_function_complexity,
)
from harness.repository_complexity import (
    REPOSITORY_COMPLEXITY_POLICY_VERSION,
    analyze_repository_complexity,
)
from harness.repository_defect_taxonomy import (
    DEFECT_TAXONOMY_VERSION,
    FALLBACK_FAMILY,
    classify_repository_defect,
)


ANNOTATION_SCHEMA_VERSION = "oneiros_corpus_annotation_v1"

#: The three top-level groups.  HumanEval and MBPP are ONE synthetic group:
#: they are both program-synthesis benchmarks whose records are sibling
#: mutations of a reference function.  Manual curated seeds are tracked
#: separately and never counted as real-world repository evidence.
ORIGIN_GROUPS = ("synthetic_function", "real_repository", "manual_curated")

_SYNTHETIC_UPSTREAMS = frozenset({"humaneval", "mbpp"})
_REPOSITORY_UPSTREAMS = frozenset({"BugsInPy", "SWE-bench Verified"})
_MANUAL_UPSTREAMS = frozenset({"manual_curated_examples"})


@dataclass(frozen=True)
class RecordAnnotation:
    record_id: str
    split: str
    origin_group: str
    source_dataset: str
    project: str | None
    repository: str | None
    issue_id: str | None
    function_lineage: str
    primary_bug_family: str
    secondary_bug_tags: tuple[str, ...]
    test_framework: str
    execution_mode: str
    complexity_tier: str
    complexity_policy_version: str
    classification_method: str
    classification_confidence: str
    complexity_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["secondary_bug_tags"] = list(self.secondary_bug_tags)
        return payload


def _upstream(record: dict[str, Any]) -> str:
    source = record.get("source") or {}
    if isinstance(source, dict):
        return str(source.get("upstream") or source.get("name") or "unknown")
    return str(source or "unknown")


def origin_group(record: dict[str, Any]) -> str:
    """Assign the top-level group, failing loudly on an unknown upstream.

    A silent fallback here would let a new dataset drift into whichever group
    the balancer happens to under-fill, so an unrecognized upstream is an
    error rather than a default.
    """
    upstream = _upstream(record)
    if upstream in _SYNTHETIC_UPSTREAMS:
        return "synthetic_function"
    if upstream in _REPOSITORY_UPSTREAMS:
        return "real_repository"
    if upstream in _MANUAL_UPSTREAMS:
        return "manual_curated"
    raise ValueError(f"unknown corpus upstream {upstream!r}; assign it a group first")


def record_test_framework(record: dict[str, Any]) -> str:
    """Execution metadata - deliberately NOT a defect family."""
    mode = (record.get("quality") or {}).get("execution_mode") or ""
    if "pytest" in str(mode):
        return "pytest"
    fmt = record.get("test_format") or ""
    if fmt == "assert_statement":
        return "assert_statement"
    return str(fmt or mode or "unknown")


def _issue_id(provenance: dict[str, Any]) -> str | None:
    for key in ("bug_id", "instance_id", "official_task_id", "seed_id"):
        value = provenance.get(key)
        if value:
            return str(value)
    return None


def _synthetic_family(record: dict[str, Any]) -> str:
    provenance = record.get("provenance") or {}
    return str(
        provenance.get("mutation_type")
        or provenance.get("category")
        or record.get("task_type")
        or "unknown"
    )


def semantic_target_key(record: dict[str, Any]) -> str:
    """The unit that balancing counts: one semantic target, not one mutant row.

    Synthetic mutants of the same reference function share ``group_id`` and
    therefore collapse to one key.  Repository records are already one verified
    defect each, so their key includes the issue identity - two defects in the
    same project are two targets, while sibling mutants are one.
    """
    group = str(record.get("group_id") or "")
    if origin_group(record) == "real_repository":
        provenance = record.get("provenance") or {}
        issue = _issue_id(provenance) or record.get("id")
        return f"{group}::{issue}"
    return group


def annotate_record(record: dict[str, Any], split: str) -> RecordAnnotation:
    """Derive grouping, defect family, and complexity for one record."""
    group = origin_group(record)
    provenance = record.get("provenance") or {}
    upstream = _upstream(record)
    prompt_code = record.get("prompt_code_under_test") or record.get("code_under_test") or ""

    if group == "real_repository":
        classification = classify_repository_defect(
            record.get("code_under_test") or "",
            record.get("reference_code") or "",
            provenance.get("official_test_evidence"),
            provenance.get("patched_source_paths"),
            provenance.get("test_selector"),
        )
        metrics = analyze_repository_complexity(
            prompt_code, record.get("support_context") or "",
        )
        return RecordAnnotation(
            record_id=str(record["id"]),
            split=split,
            origin_group=group,
            source_dataset=upstream,
            project=provenance.get("project"),
            repository=provenance.get("repository") or provenance.get("repository_url"),
            issue_id=_issue_id(provenance),
            function_lineage=semantic_target_key(record),
            primary_bug_family=classification.primary_bug_family,
            secondary_bug_tags=classification.secondary_bug_tags,
            test_framework=record_test_framework(record),
            execution_mode=str((record.get("quality") or {}).get("execution_mode") or ""),
            complexity_tier=metrics.tier,
            complexity_policy_version=REPOSITORY_COMPLEXITY_POLICY_VERSION,
            classification_method=classification.classification_method,
            classification_confidence=classification.classification_confidence,
            complexity_metrics=metrics.to_dict(),
        )

    # Synthetic and manual records are single localized functions.
    tier = "unknown"
    metrics_payload: dict[str, Any] = {}
    entry_point = record.get("entry_point") or ""
    if entry_point:
        try:
            measured = analyze_function_complexity(prompt_code, entry_point)
            tier = measured.tier
            metrics_payload = measured.to_dict()
        except ValueError:
            tier = "unknown"
    return RecordAnnotation(
        record_id=str(record["id"]),
        split=split,
        origin_group=group,
        source_dataset=upstream,
        project=provenance.get("project"),
        repository=None,
        issue_id=_issue_id(provenance),
        function_lineage=semantic_target_key(record),
        primary_bug_family=_synthetic_family(record),
        secondary_bug_tags=(),
        test_framework=record_test_framework(record),
        execution_mode=str((record.get("quality") or {}).get("execution_mode") or ""),
        complexity_tier=tier,
        complexity_policy_version=COMPLEXITY_POLICY_VERSION,
        classification_method="upstream_mutation_metadata",
        classification_confidence="high" if metrics_payload else "medium",
        complexity_metrics=metrics_payload,
    )


def annotate_split(records: Iterable[dict[str, Any]], split: str) -> list[RecordAnnotation]:
    return [annotate_record(record, split) for record in records]


def annotations_sha256(annotations: Iterable[RecordAnnotation]) -> str:
    """Order-independent content hash over the annotation payload."""
    digest = hashlib.sha256()
    for item in sorted(annotations, key=lambda entry: entry.record_id):
        digest.update(item.record_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(item.origin_group.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(item.primary_bug_family.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(item.complexity_tier.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(item.function_lineage.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()
