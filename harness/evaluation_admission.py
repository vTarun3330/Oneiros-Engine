"""Mode-aware record admission and scoping, shared by every evaluation path.

This module exists because of a specific failure. The sealed-final loader
required seven fields to be non-empty on **every** record in a split, and
raised before returning any of them. Repository-style records carry
``entry_point`` and usually ``specification`` blank by design. The established
evaluator has always loaded the whole split and *then* scoped the kill rate to
function-mode records, so it never saw a problem; the sealed loader validated
before it filtered, and refused records the pipeline was always going to
discard. The single authorization was spent on that refusal.

Two rules follow, and they are the whole point of this file:

1. **Scope first, validate second.** Requirements are checked only against the
   records that are actually going to be evaluated. A record that will be
   excluded cannot block the run.
2. **Requirements follow the execution mode.** ``entry_point`` is required
   where the evaluator genuinely dereferences it, and ``specification`` is
   never required - blank specifications exist on scored function-mode records
   in the permitted splits today.

The mode semantics are **imported from the historical evaluator**, not restated
here. Restating them is how two sources of truth drift apart, which is the same
shape of defect one layer up.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent

ADMISSION_VERSION = "oneiros_evaluation_admission_v1"

#: Files that decide which records get evaluated. Bound by hash into any
#: receipt that depends on the scope, for the same reason prompt-defining
#: sources are: a change here silently changes what was measured.
ADMISSION_DEFINING_SOURCES = {
    "admission": "harness/evaluation_admission.py",
    "historical_scoping": "scripts/train_on_dataset.py",
}

#: Splits no rehearsal or future protocol may read, with the reason.
#:
#: ``test`` was opened once under authorization on 2026-09-16. The run failed
#: before generating anything, but a split that has been opened is no longer a
#: held-out split regardless of what it produced. It is named here **only** so
#: that it can be refused.
REFUSED_SPLITS = {
    "test": (
        "the 'test' split was consumed by the failed sealed-final attempt of "
        "2026-09-16 and must never be read again; see "
        "docs/SEALED_FINAL_INCIDENT.md"
    ),
}

#: Fields every evaluated record needs, whatever its mode.
UNIVERSAL_REQUIRED_FIELDS: Tuple[str, ...] = ("id", "task_type")

#: Fields function-mode scoring dereferences. ``entry_point`` is here because
#: the evaluator passes it to the parser and the executor. ``specification`` is
#: deliberately ABSENT: it feeds the prompt, the prompt tolerates it being
#: empty, and permitted splits contain scored function records without one.
FUNCTION_REQUIRED_FIELDS: Tuple[str, ...] = (
    "entry_point", "reference_code", "code_under_test", "tests",
)

#: Why repository-mode records are out of scope. Not a defect, not a data
#: problem - there is no native repository evaluator, so there is nothing to
#: score them with.
REPOSITORY_EXCLUSION_REASON = (
    "excluded from function Kill@k: scoring a repository fragment requires a "
    "native project environment, and no validated native repository evaluator "
    "exists. Excluded, never failed."
)


class AdmissionError(RuntimeError):
    """Raised when a split cannot be scoped into something evaluable."""


class RefusedSplitError(AdmissionError):
    """Raised when a caller asks for a split that may not be read."""


# --------------------------------------------------------------------------
# Mode semantics, imported rather than restated.
# --------------------------------------------------------------------------

def _historical_modes() -> Tuple[str, frozenset]:
    """The function mode and repository modes the real evaluator uses.

    Imported from ``scripts.train_on_dataset`` so this module cannot disagree
    with the code that produced every historical result.
    """
    from scripts.train_on_dataset import (
        FUNCTION_EXECUTION_MODE, REPOSITORY_EXECUTION_MODES,
    )
    return FUNCTION_EXECUTION_MODE, frozenset(REPOSITORY_EXECUTION_MODES)


def function_execution_mode() -> str:
    return _historical_modes()[0]


def repository_execution_modes() -> frozenset:
    return _historical_modes()[1]


def execution_mode_of(record: Mapping[str, Any]) -> str:
    """The record's execution mode, defaulting exactly as the evaluator does.

    Canonical records carry it under ``quality``; records already adapted by
    ``_record_to_pair`` carry it at the top level. Both shapes appear in the
    pipeline, so both are read.
    """
    if "execution_mode" in record:
        mode = record.get("execution_mode")
    else:
        mode = (record.get("quality") or {}).get("execution_mode")
    return str(mode or function_execution_mode())


def is_function_mode(mode: str) -> bool:
    return str(mode) == function_execution_mode()


def is_repository_mode(mode: str) -> bool:
    return str(mode) in repository_execution_modes()


def required_fields_for(mode: str) -> Tuple[str, ...]:
    """Fields a record must carry **given the mode it will be scored under**.

    A repository-mode record has no function-scoring requirements because it is
    not function-scored. Asking this for a mode with no evaluator is a
    programming error, not a data error.
    """
    if is_function_mode(mode):
        return UNIVERSAL_REQUIRED_FIELDS + FUNCTION_REQUIRED_FIELDS
    if is_repository_mode(mode):
        raise AdmissionError(
            f"{mode!r} has no function-scoring requirements; "
            + REPOSITORY_EXCLUSION_REASON)
    raise AdmissionError(f"unknown execution mode {mode!r}")


# --------------------------------------------------------------------------
# Scoping.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationScope:
    """What will be evaluated, what will not, and why."""

    split_name: str
    eligible: List[Dict[str, Any]]
    excluded_repository: List[str] = field(default_factory=list)
    excluded_unknown_mode: List[Tuple[str, str]] = field(default_factory=list)
    requested: int = 0

    @property
    def target_count(self) -> int:
        return len(self.eligible)

    def scope_sha256(self) -> str:
        """Digest over the eligible id sequence, in split order."""
        import json
        return hashlib.sha256(json.dumps(
            [str(r.get("id")) for r in self.eligible], separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "admission_version": ADMISSION_VERSION,
            "split": self.split_name,
            "requested_records": self.requested,
            "function_mode_targets": self.target_count,
            "excluded_repository_records": len(self.excluded_repository),
            "excluded_unknown_mode_records": len(self.excluded_unknown_mode),
            "repository_exclusion_reason": REPOSITORY_EXCLUSION_REASON,
            "evaluation_scope_sha256": self.scope_sha256(),
        }


def refuse_refused_split(split_name: str) -> None:
    """Refuse any split that may not be read. Call before touching the corpus."""
    reason = REFUSED_SPLITS.get(str(split_name))
    if reason:
        raise RefusedSplitError(f"refusing split {split_name!r}: {reason}")


def scope_split(
    splits: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    split_name: str,
    *,
    adapt: Optional[Any] = None,
) -> EvaluationScope:
    """Resolve a split into the records that will actually be evaluated.

    Order is the contract: refuse a refused split, resolve ids, partition by
    execution mode, and only then require fields - of the survivors alone.
    """
    refuse_refused_split(split_name)

    if split_name not in splits:
        raise AdmissionError(
            f"split {split_name!r} is absent; available: "
            f"{sorted(k for k in splits if k not in REFUSED_SPLITS)}")
    ids = [str(item) for item in splits[split_name]]
    if not ids:
        raise AdmissionError(f"split {split_name!r} is empty; refusing to measure nothing")
    if len(set(ids)) != len(ids):
        raise AdmissionError(f"split {split_name!r} contains duplicate ids")

    wanted = set(ids)
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        identifier = str(record.get("id"))
        if identifier in wanted:
            by_id[identifier] = dict(record)

    missing = [identifier for identifier in ids if identifier not in by_id]
    if missing:
        raise AdmissionError(
            f"{len(missing)} id(s) in split {split_name!r} have no record")

    # --- partition by mode, BEFORE any field requirement is applied ---------
    eligible_raw: List[Dict[str, Any]] = []
    excluded_repository: List[str] = []
    excluded_unknown: List[Tuple[str, str]] = []
    for identifier in ids:
        record = by_id[identifier]
        mode = execution_mode_of(record)
        if is_function_mode(mode):
            eligible_raw.append(record)
        elif is_repository_mode(mode):
            excluded_repository.append(identifier)
        else:
            excluded_unknown.append((identifier, mode))

    if not eligible_raw:
        raise AdmissionError(
            f"split {split_name!r} has no function-mode records to evaluate")

    # --- now, and only now, require fields of the survivors -----------------
    required = required_fields_for(function_execution_mode())
    incomplete: List[Tuple[str, List[str]]] = []
    for record in eligible_raw:
        absent = [f for f in required if not record.get(f)]
        if absent:
            incomplete.append((str(record.get("id")), absent))
    if incomplete:
        raise AdmissionError(
            f"{len(incomplete)} function-mode record(s) in split {split_name!r} "
            f"are missing a field function scoring requires: "
            f"{[fields for _, fields in incomplete[:3]]}")

    eligible = [adapt(dict(r)) for r in eligible_raw] if adapt else eligible_raw
    return EvaluationScope(
        split_name=split_name,
        eligible=list(eligible),
        excluded_repository=excluded_repository,
        excluded_unknown_mode=excluded_unknown,
        requested=len(ids),
    )


def admission_source_hashes(root=None) -> Dict[str, Any]:
    """Identity of every source that can change which records get evaluated."""
    from harness.source_identity import canonical_sha256, raw_sha256

    base = Path(root) if root else ROOT
    out: Dict[str, Any] = {"admission_version": ADMISSION_VERSION}
    for role, relative in sorted(ADMISSION_DEFINING_SOURCES.items()):
        path = base / relative
        out[role] = {
            "path": relative,
            "raw_sha256": raw_sha256(path),
            "canonical_sha256": canonical_sha256(path),
        }
    return out


def admission_binding_problems(recorded: Mapping[str, Any], root=None) -> List[str]:
    """Refuse if any scope-defining source differs from the approved receipt."""
    if not recorded:
        return ["receipt records no admission source identity"]
    current = admission_source_hashes(root)
    problems: List[str] = []
    if recorded.get("admission_version") != current["admission_version"]:
        problems.append(
            f"admission_version differs: {recorded.get('admission_version')!r} "
            f"vs {current['admission_version']!r}")
    for role in sorted(ADMISSION_DEFINING_SOURCES):
        was = (recorded.get(role) or {}).get("canonical_sha256")
        now = current[role]["canonical_sha256"]
        if was != now:
            problems.append(
                f"admission source {role} ({ADMISSION_DEFINING_SOURCES[role]}) "
                f"differs from the approved receipt: {was} vs {now}")
    return problems
