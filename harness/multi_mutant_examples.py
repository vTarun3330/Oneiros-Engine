"""Build one supervised example per semantic function, not per mutant row.

The V4.1 corpus stores every sibling mutation of a HumanEval or MBPP function
as its own record.  Training on those rows directly produces thousands of
near-identical prompts that differ only in a single mutated operator, and
teaches the model to emit one narrow assertion aimed at one displayed defect.

This module instead groups the siblings of a semantic function, executes every
candidate assertion against the correct implementation and against every
sibling mutant, and selects a compact assertion set with a deterministic
weighted set cover.  The selected assertions are combined into ONE
self-contained test function.

Leakage contract
----------------
The kill matrix, the sibling mutants, the correct implementation, and the
mutation families are TEACHER-SIDE supervision.  They decide which assertions
to keep.  The model prompt still shows exactly one target and its permitted
specification/context - never a sibling, never the reference, never the matrix.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

from harness.candidate_policy import (
    executable_test_function,
    validate_function_assertion,
    validate_test_function,
)
from harness.safe_execution import classify_assertions


MULTI_MUTANT_BUILDER_VERSION = "oneiros_multi_mutant_set_cover_v1"

#: A completion that distinguishes at least this many siblings is the goal the
#: research plan sets.  It is a PREFERENCE, not a filter: a lineage with fewer
#: killable siblings still yields its best achievable example rather than being
#: dropped, which would silently bias the corpus toward easy functions.
PREFERRED_SIBLING_KILLS = 4
DEFAULT_MAX_ASSERTIONS = 8
DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class KillMatrix:
    """Which candidate assertion distinguishes which sibling mutant."""

    lineage: str
    entry_point: str
    assertions: tuple[str, ...]
    mutant_ids: tuple[str, ...]
    mutant_families: tuple[str, ...]
    #: ``killed[a][m]`` - assertion ``a`` fails on mutant ``m``.
    killed: tuple[tuple[bool, ...], ...]
    #: Assertions that passed on the correct implementation in every run.
    reference_valid: tuple[bool, ...]
    rejected: dict[str, str] = field(default_factory=dict)

    def kills_for(self, index: int) -> set[int]:
        return {m for m, hit in enumerate(self.killed[index]) if hit}


@dataclass(frozen=True)
class MultiMutantExample:
    builder_version: str
    lineage: str
    entry_point: str
    displayed_record_id: str
    source_dataset: str
    completion: str
    assertion_count: int
    assertions: tuple[str, ...]
    primary_mutation_family: str
    covered_mutation_families: tuple[str, ...]
    sibling_mutant_ids: tuple[str, ...]
    mutants_evaluated: int
    mutants_killed: int
    mutants_killed_percent: float
    surviving_mutant_ids: tuple[str, ...]
    kills_displayed_target: bool
    meets_preferred_sibling_kills: bool
    verified: bool
    verification_status: str
    test_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "assertions", "covered_mutation_families",
            "sibling_mutant_ids", "surviving_mutant_ids",
        ):
            payload[key] = list(getattr(self, key))
        return payload


def _mutation_family(record: dict[str, Any]) -> str:
    provenance = record.get("provenance") or {}
    return str(
        provenance.get("mutation_type") or provenance.get("category") or "unknown"
    )


def _candidate_assertions(
    records: Sequence[dict[str, Any]],
    entry_point: str,
    extra_assertions: Iterable[str] = (),
) -> list[str]:
    """Every policy-valid assertion available for this lineage, deduplicated.

    Two sources are permitted.  The siblings' retained tests are all
    *distinguishing* assertions, so on their own they bias the pool toward
    inputs that already break something.  ``extra_assertions`` carries the
    upstream benchmark's own reference suite, which also contains ordinary
    passing cases - the ones that pin down correct behaviour rather than a
    single defect.  Both are public, buggy-side-safe test inputs.

    Deduplication is on normalized text, so two sources supplying the same
    assertion contribute one candidate.
    """
    seen: dict[str, str] = {}

    def offer(code: str) -> None:
        code = str(code or "").strip()
        if not code:
            return
        key = " ".join(code.split())
        if key in seen:
            return
        if not validate_function_assertion(code, entry_point).valid:
            return
        seen[key] = code

    for record in records:
        for test in record.get("tests") or []:
            offer((test or {}).get("code") or "")
    for assertion in extra_assertions:
        offer(assertion)
    return [seen[key] for key in sorted(seen)]


def build_kill_matrix(
    records: Sequence[dict[str, Any]],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    classifier: Callable[..., list[dict[str, Any]]] = classify_assertions,
    extra_assertions: Iterable[str] = (),
) -> KillMatrix:
    """Execute every candidate assertion against the correct code and each mutant.

    An assertion that fails on the correct implementation - in ANY sibling's
    run - is rejected outright.  Requiring agreement across runs is what
    removes flaky and non-deterministic candidates: a genuinely deterministic
    assertion behaves identically every time the reference executes it.
    """
    if not records:
        raise ValueError("a lineage must contain at least one record")
    entry_point = str(records[0].get("entry_point") or "")
    if not entry_point:
        raise ValueError("lineage records must declare an entry point")
    golden = str(records[0].get("reference_code") or "")
    if not golden:
        raise ValueError("lineage records must carry a reference implementation")
    lineage = str(records[0].get("group_id") or "")

    candidates = _candidate_assertions(records, entry_point, extra_assertions)
    rejected: dict[str, str] = {}
    if not candidates:
        return KillMatrix(
            lineage=lineage, entry_point=entry_point, assertions=(),
            mutant_ids=tuple(str(item["id"]) for item in records),
            mutant_families=tuple(_mutation_family(item) for item in records),
            killed=(), reference_valid=(),
            rejected={"__lineage__": "no_policy_valid_candidate_assertion"},
        )

    reference_valid = [True] * len(candidates)
    killed: list[list[bool]] = [[False] * len(records) for _ in candidates]
    for column, record in enumerate(records):
        mutant = str(record.get("prompt_code_under_test") or record.get("code_under_test") or "")
        rows = classifier(candidates, golden, mutant, timeout_seconds)
        by_test = {str(row.get("test")): row for row in rows}
        for index, assertion in enumerate(candidates):
            row = by_test.get(assertion)
            if row is None:
                reference_valid[index] = False
                rejected.setdefault(assertion, "missing_execution_row")
                continue
            if not bool(row.get("valid")):
                reference_valid[index] = False
                status = str((row.get("golden") or {}).get("status") or "invalid")
                rejected.setdefault(assertion, f"reference_invalid:{status}")
                continue
            killed[index][column] = bool(row.get("killed"))

    return KillMatrix(
        lineage=lineage,
        entry_point=entry_point,
        assertions=tuple(candidates),
        mutant_ids=tuple(str(item["id"]) for item in records),
        mutant_families=tuple(_mutation_family(item) for item in records),
        killed=tuple(tuple(row) for row in killed),
        reference_valid=tuple(reference_valid),
        rejected=rejected,
    )


def _assertion_shape(assertion: str) -> str:
    """A coarse signature of what an assertion exercises.

    Used only to spread the pinning assertions across visibly different inputs
    rather than repeating one call shape, so it needs to be cheap and stable,
    not semantically precise.
    """
    import ast as _ast

    try:
        tree = _ast.parse(assertion)
    except SyntaxError:
        return assertion.strip()[:48]
    parts: list[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Constant):
            parts.append(f"{type(node.value).__name__}:{str(node.value)[:12]}")
        elif isinstance(node, (_ast.List, _ast.Tuple, _ast.Set)):
            parts.append(f"seq{len(node.elts)}")
        elif isinstance(node, _ast.Dict):
            parts.append(f"map{len(node.keys)}")
    return "|".join(parts[:8])


def select_assertion_cover(
    matrix: KillMatrix,
    target_index: int,
    max_assertions: int = DEFAULT_MAX_ASSERTIONS,
    min_assertions: int = 1,
) -> list[int]:
    """Deterministic weighted greedy set cover over sibling mutants.

    The displayed target is covered first, because a completion that does not
    kill the bug it was shown is not a bug-revealing test.  Remaining picks
    maximize newly distinguished siblings, then newly covered mutation
    families, so the retained assertions spread across defect kinds instead of
    piling onto one.  All ties break on assertion text, making the selection
    reproducible.
    """
    if max_assertions < 1:
        raise ValueError("max_assertions must be at least one")
    usable = [
        index for index, valid in enumerate(matrix.reference_valid)
        if valid and any(matrix.killed[index])
    ]
    if not usable:
        return []

    def family_of(mutant: int) -> str:
        return matrix.mutant_families[mutant]

    selected: list[int] = []
    covered: set[int] = set()
    covered_families: set[str] = set()

    target_killers = [index for index in usable if matrix.killed[index][target_index]]
    if target_killers:
        first = max(
            target_killers,
            key=lambda index: (
                sum(matrix.killed[index]),
                len({family_of(m) for m in matrix.kills_for(index)}),
                # Negated text keeps the tie-break deterministic and stable.
                tuple(-ord(character) for character in matrix.assertions[index][:64]),
            ),
        )
        selected.append(first)
        covered |= matrix.kills_for(first)
        covered_families |= {family_of(m) for m in covered}

    while len(selected) < max_assertions:
        best_index = None
        best_key = None
        for index in usable:
            if index in selected:
                continue
            gained = matrix.kills_for(index) - covered
            if not gained:
                continue
            new_families = {family_of(m) for m in gained} - covered_families
            key = (
                len(gained),
                len(new_families),
                tuple(-ord(character) for character in matrix.assertions[index][:64]),
            )
            if best_key is None or key > best_key:
                best_key, best_index = key, index
        if best_index is None:
            break
        selected.append(best_index)
        gained = matrix.kills_for(best_index)
        covered |= gained
        covered_families |= {family_of(m) for m in gained}

    # Mutation coverage alone can be satisfied by a single sharp assertion,
    # which produces a technically-correct but unrealistically thin test.  Top
    # up with reference-valid assertions that exercise visibly different input
    # shapes.  These pin correct behaviour rather than distinguish a mutant,
    # which is exactly what a real test suite does around a bug-revealing case.
    if len(selected) < min_assertions:
        chosen_shapes = {_assertion_shape(matrix.assertions[index]) for index in selected}
        remaining = sorted(
            (
                index for index, valid in enumerate(matrix.reference_valid)
                if valid and index not in selected
            ),
            key=lambda index: matrix.assertions[index],
        )
        for index in remaining:
            if len(selected) >= min(min_assertions, max_assertions):
                break
            shape = _assertion_shape(matrix.assertions[index])
            if shape in chosen_shapes:
                continue
            chosen_shapes.add(shape)
            selected.append(index)

    return selected


def _test_function_name(entry_point: str, lineage: str) -> str:
    suffix = hashlib.sha256(lineage.encode("utf-8")).hexdigest()[:8]
    stem = "".join(character if character.isalnum() else "_" for character in entry_point)
    return f"test_{stem or 'target'}_{suffix}"


def render_test_function(name: str, assertions: Sequence[str]) -> str:
    body = "\n".join(f"    {assertion.strip()}" for assertion in assertions)
    return f"def {name}():\n{body}\n"


def build_multi_mutant_example(
    records: Sequence[dict[str, Any]],
    target_index: int = 0,
    max_assertions: int = DEFAULT_MAX_ASSERTIONS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    matrix: KillMatrix | None = None,
    classifier: Callable[..., list[dict[str, Any]]] = classify_assertions,
    extra_assertions: Iterable[str] = (),
    min_assertions: int = 1,
) -> MultiMutantExample | None:
    """Produce one verified multi-assertion example for a function lineage.

    Returns ``None`` when no policy-valid assertion distinguishes the displayed
    target: such a lineage has no honest supervised label, and inventing one
    would mean training on a test that does not reveal the bug.
    """
    matrix = matrix or build_kill_matrix(
        records, timeout_seconds, classifier, extra_assertions,
    )
    if not matrix.assertions:
        return None
    selected = select_assertion_cover(
        matrix, target_index, max_assertions, min_assertions,
    )
    if not selected:
        return None
    if not matrix.killed[selected[0]][target_index]:
        # The cover could not start from a target-killing assertion.
        return None

    target = records[target_index]
    entry_point = matrix.entry_point
    name = _test_function_name(entry_point, f"{matrix.lineage}:{target.get('id')}")
    assertions = tuple(matrix.assertions[index] for index in selected)
    completion = render_test_function(name, assertions)

    policy = validate_test_function(completion, entry_point)
    if not policy.valid:
        return MultiMutantExample(
            builder_version=MULTI_MUTANT_BUILDER_VERSION,
            lineage=matrix.lineage, entry_point=entry_point,
            displayed_record_id=str(target["id"]),
            source_dataset=str((target.get("source") or {}).get("upstream") or "unknown"),
            completion=completion, assertion_count=len(assertions),
            assertions=assertions,
            primary_mutation_family=matrix.mutant_families[target_index],
            covered_mutation_families=(), sibling_mutant_ids=matrix.mutant_ids,
            mutants_evaluated=len(matrix.mutant_ids), mutants_killed=0,
            mutants_killed_percent=0.0, surviving_mutant_ids=matrix.mutant_ids,
            kills_displayed_target=False, meets_preferred_sibling_kills=False,
            verified=False, verification_status=f"policy_rejected:{policy.reason}",
            test_sha256=hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        )

    # Re-execute the ASSEMBLED test.  A set of assertions that individually
    # behave can still fail together - shared setup, ordering, or an earlier
    # assertion short-circuiting the rest - so the combined artifact is what
    # must be verified, not the pieces it was built from.
    golden = str(records[0].get("reference_code") or "")
    executable = executable_test_function(completion)
    combined_kills: list[str] = []
    survivors: list[str] = []
    verification_status = "verified"
    reference_rows = classifier([executable], golden, golden, timeout_seconds)
    if not reference_rows or not bool(reference_rows[0].get("valid")):
        status = str((reference_rows[0].get("golden") or {}).get("status") if reference_rows else "no_row")
        verification_status = f"reference_failed:{status}"
    else:
        for index, record in enumerate(records):
            mutant = str(record.get("prompt_code_under_test") or record.get("code_under_test") or "")
            rows = classifier([executable], golden, mutant, timeout_seconds)
            if rows and bool(rows[0].get("killed")):
                combined_kills.append(matrix.mutant_ids[index])
            else:
                survivors.append(matrix.mutant_ids[index])

    kills_target = str(target["id"]) in combined_kills
    verified = verification_status == "verified" and kills_target
    if verification_status == "verified" and not kills_target:
        verification_status = "assembled_test_does_not_kill_displayed_target"

    covered_families = tuple(sorted({
        matrix.mutant_families[index]
        for index, mutant_id in enumerate(matrix.mutant_ids)
        if mutant_id in set(combined_kills)
    }))
    return MultiMutantExample(
        builder_version=MULTI_MUTANT_BUILDER_VERSION,
        lineage=matrix.lineage,
        entry_point=entry_point,
        displayed_record_id=str(target["id"]),
        source_dataset=str((target.get("source") or {}).get("upstream") or "unknown"),
        completion=completion,
        assertion_count=len(assertions),
        assertions=assertions,
        primary_mutation_family=matrix.mutant_families[target_index],
        covered_mutation_families=covered_families,
        sibling_mutant_ids=matrix.mutant_ids,
        mutants_evaluated=len(matrix.mutant_ids),
        mutants_killed=len(combined_kills),
        mutants_killed_percent=round(
            100.0 * len(combined_kills) / max(1, len(matrix.mutant_ids)), 3
        ),
        surviving_mutant_ids=tuple(survivors),
        kills_displayed_target=kills_target,
        meets_preferred_sibling_kills=len(combined_kills) >= PREFERRED_SIBLING_KILLS,
        verified=verified,
        verification_status=verification_status,
        test_sha256=hashlib.sha256(completion.encode("utf-8")).hexdigest(),
    )


def uncovered_mutants(
    matrix: KillMatrix, example: MultiMutantExample | None,
) -> list[int]:
    """Sibling indexes the broad example does not distinguish.

    These are the rare or hard mutations that still deserve their own targeted
    example, so the hybrid corpus keeps narrow supervision where the broad test
    does not reach.
    """
    if example is None:
        return list(range(len(matrix.mutant_ids)))
    killed = set(example.sibling_mutant_ids) & set(
        example.sibling_mutant_ids
    ) - set(example.surviving_mutant_ids)
    return [
        index for index, mutant_id in enumerate(matrix.mutant_ids)
        if mutant_id not in killed
    ]
