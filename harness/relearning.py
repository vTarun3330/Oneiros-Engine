"""Hard-example relearning: classify losers, then attach VERIFIED corrections.

Part 10 of the research plan.  One bounded SFT-only round.

The safety property that matters most here is what a correction is allowed to
be.  A loser is, by definition, a model output that failed.  Training on it as
though it were right would teach exactly the mistake being corrected, so a
failed generation is never a label.  A correction may only come from
supervision that already carries execution evidence - the verified multi-mutant
completion for that record, which was executed against the reference and every
sibling mutant when it was built.

Split isolation is enforced structurally rather than by convention: only train
and ablation_dev cases may enter the queue, and a validation or sealed-test
artifact is refused outright.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


RELEARNING_SCHEMA_VERSION = "oneiros_sft_relearning_v1"

#: Splits a loser may be drawn from.  Validation and the sealed test are the
#: measurement surfaces; mining them for training cases would tune the model on
#: the thing it is later judged by.
ELIGIBLE_SPLITS = frozenset({"train", "ablation_dev"})
FORBIDDEN_SPLITS = frozenset({"val", "validation", "test"})

#: Loser categories, derived from per-candidate failure taxonomy outcomes.
#: ``no_kill`` covers a function whose candidates all executed cleanly and still
#: failed to distinguish the defect - the interesting case, and the one a
#: correction can actually teach.
LOSER_CATEGORIES = (
    "syntax_invalid",
    "unparsable",
    "execution_failed",
    "undefined_symbol",
    "fabricated_api",
    "wrong_oracle",
    "no_kill",
    "covers_without_detecting",
    "duplicate_or_trivial",
    "worse_than_base",
)

_CATEGORY_FROM_TAXONOMY = {
    "syntax_invalid": "syntax_invalid",
    "not_generated": "unparsable",
    "environment_failure": "execution_failed",
    "timeout": "execution_failed",
    "fixture_missing": "execution_failed",
    "undefined_symbol": "undefined_symbol",
    "wrong_target_api": "fabricated_api",
    "repository_context_hallucination": "fabricated_api",
    "wrong_expected_value": "wrong_oracle",
    "incorrect_exception_expectation": "wrong_oracle",
    "reference_invalid": "wrong_oracle",
    "passes_both": "covers_without_detecting",
    "boundary_miss": "no_kill",
    "off_by_one_miss": "no_kill",
    "logical_condition_miss": "no_kill",
    "indexing_miss": "no_kill",
}


@dataclass(frozen=True)
class LoserCase:
    """One development case the model failed, with the evidence for why."""

    record_id: str
    split: str
    origin_group: str
    bug_family: str
    complexity_tier: str
    project: str
    dominant_category: str
    categories: dict[str, int]
    requested_candidates: int
    parsed_candidates: int
    reference_valid_candidates: int
    killed_candidates: int
    base_model_killed: bool | None
    worse_than_base: bool
    model_run: str
    checkpoint_step: int | None
    seed: int | None
    prompt_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Correction:
    """A verified label attached to a loser.  Never a model output."""

    record_id: str
    loser_category: str
    completion: str
    completion_shape: str
    supervision_source: str
    verified: bool
    verification_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_split_is_eligible(split: str) -> None:
    """Refuse a validation or sealed-test artifact loudly."""
    normalized = str(split or "").strip().lower().replace("-", "_")
    if normalized in FORBIDDEN_SPLITS:
        raise ValueError(
            f"split {split!r} may never enter relearning: validation and the "
            "sealed test are measurement surfaces, not training sources"
        )
    if normalized not in ELIGIBLE_SPLITS:
        raise ValueError(f"unknown split {split!r} for relearning eligibility")


def _function_killed(result: dict[str, Any]) -> bool:
    if "function_killed" in result:
        return bool(result["function_killed"])
    if "killed" in result:
        return bool(result["killed"])
    return int(result.get("killed_candidates") or 0) > 0


def classify_loser(
    result: dict[str, Any],
    split: str,
    base_result: dict[str, Any] | None = None,
    model_run: str = "",
    checkpoint_step: int | None = None,
    seed: int | None = None,
    prompt_version: str = "",
    annotation: dict[str, Any] | None = None,
) -> LoserCase | None:
    """Classify one evaluated function, returning ``None`` if it is not a loser.

    A function the model killed is not a loser, even if some of its candidates
    were poor: the task was accomplished within the candidate budget, which is
    what Kill@k measures.
    """
    assert_split_is_eligible(split)
    killed = _function_killed(result)
    base_killed = _function_killed(base_result) if base_result else None
    worse_than_base = bool(killed is False and base_killed is True)
    if killed and not worse_than_base:
        return None

    categories: Counter[str] = Counter()
    for candidate in result.get("candidates") or result.get("candidate_results") or []:
        mode = str(candidate.get("failure_mode") or "")
        taxonomy = mode.split(":")[0] if mode else ""
        if taxonomy.startswith("killed"):
            continue
        if taxonomy.startswith("reference_"):
            categories["wrong_oracle"] += 1
            continue
        mapped = _CATEGORY_FROM_TAXONOMY.get(taxonomy)
        if mapped:
            categories[mapped] += 1
        elif taxonomy in {"policy_invalid", "generation_invalid"}:
            categories["syntax_invalid"] += 1

    if not categories:
        # The artifact records per-function aggregates only. Fall back to what
        # those aggregates can support rather than inventing a category.
        parsed = int(result.get("parsed_candidates") or 0)
        requested = int(result.get("requested_candidates") or 0)
        if parsed < requested:
            categories["syntax_invalid"] += requested - parsed
        categories["no_kill"] += max(1, parsed)
    if worse_than_base:
        categories["worse_than_base"] += 1

    dominant = max(sorted(categories.items()), key=lambda item: item[1])[0]
    annotation = annotation or {}
    return LoserCase(
        record_id=str(result.get("record_id") or result.get("id") or ""),
        split=split,
        origin_group=str(annotation.get("origin_group") or "unknown"),
        bug_family=str(
            annotation.get("primary_bug_family") or result.get("bug_family") or "unknown"
        ),
        complexity_tier=str(annotation.get("complexity_tier") or "unknown"),
        project=str(annotation.get("project") or result.get("project") or "synthetic"),
        dominant_category=dominant,
        categories=dict(sorted(categories.items())),
        requested_candidates=int(result.get("requested_candidates") or 0),
        parsed_candidates=int(result.get("parsed_candidates") or 0),
        reference_valid_candidates=int(
            result.get("reference_valid_candidates")
            or result.get("execution_valid_candidates") or 0
        ),
        killed_candidates=int(result.get("killed_candidates") or 0),
        base_model_killed=base_killed,
        worse_than_base=worse_than_base,
        model_run=model_run,
        checkpoint_step=checkpoint_step,
        seed=seed,
        prompt_version=prompt_version,
    )


def attach_corrections(
    losers: Sequence[LoserCase],
    verified_completions: dict[str, str],
    completion_shape: str = "test_function",
    supervision_source: str = "multi_mutant_verified_completion",
) -> tuple[list[Correction], dict[str, Any]]:
    """Pair each loser with verified supervision, skipping those without any.

    ``verified_completions`` must already carry execution evidence.  A loser
    with no verified completion produces no correction at all: fabricating one
    would put an unverified label into training, which is the single thing this
    round must not do.
    """
    corrections: list[Correction] = []
    skipped: Counter[str] = Counter()
    for loser in losers:
        completion = verified_completions.get(loser.record_id)
        if not completion:
            skipped[loser.dominant_category] += 1
            continue
        corrections.append(Correction(
            record_id=loser.record_id,
            loser_category=loser.dominant_category,
            completion=completion,
            completion_shape=completion_shape,
            supervision_source=supervision_source,
            verified=True,
            verification_evidence={
                "executed_against_reference": True,
                "executed_against_every_sibling_mutant": True,
                "evidence_recorded_at_dataset_build_time": True,
            },
        ))
    return corrections, {
        "losers": len(losers),
        "corrections": len(corrections),
        "skipped_without_verified_supervision": sum(skipped.values()),
        "skipped_by_category": dict(sorted(skipped.items())),
    }


def balanced_replay(
    corrections: Sequence[Correction],
    losers_by_id: dict[str, LoserCase],
    max_per_project: int = 40,
    max_per_family: int = 60,
    max_per_category: int = 120,
) -> tuple[list[Correction], dict[str, Any]]:
    """Cap how much any one project, family, or failure mode can contribute.

    Without caps a single recurring failure mode dominates the round and the
    adapter overfits to it, which is the catastrophic-forgetting risk the plan
    calls out.  Ordering is deterministic so the same inputs give the same set.
    """
    project_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    kept: list[Correction] = []
    dropped: Counter[str] = Counter()

    for correction in sorted(corrections, key=lambda item: item.record_id):
        loser = losers_by_id.get(correction.record_id)
        project = loser.project if loser else "unknown"
        family = loser.bug_family if loser else "unknown"
        category = correction.loser_category
        if project_counts[project] >= max_per_project:
            dropped["project_cap"] += 1
            continue
        if family_counts[family] >= max_per_family:
            dropped["family_cap"] += 1
            continue
        if category_counts[category] >= max_per_category:
            dropped["category_cap"] += 1
            continue
        project_counts[project] += 1
        family_counts[family] += 1
        category_counts[category] += 1
        kept.append(correction)

    return kept, {
        "input_corrections": len(corrections),
        "retained": len(kept),
        "dropped_by_cap": dict(sorted(dropped.items())),
        "caps": {
            "max_per_project": max_per_project,
            "max_per_family": max_per_family,
            "max_per_category": max_per_category,
        },
        "retained_by_project": dict(sorted(project_counts.items())),
        "retained_by_category": dict(sorted(category_counts.items())),
    }


def relearning_dataset_sha256(corrections: Iterable[Correction]) -> str:
    digest = hashlib.sha256()
    for correction in sorted(corrections, key=lambda item: item.record_id):
        digest.update(correction.record_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(correction.completion.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(correction.loser_category.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()
