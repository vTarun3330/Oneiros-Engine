"""The measurement the sealed split is opened for.

This exists because of an ordering defect in the first version of the sealed
entrypoint. That version called the one-time guard, the guard spent the token,
and only then did control reach a comment saying the measurement was not
implemented. A real authorization would have been consumed irreversibly and
produced nothing, and the sealed split - the one measurement in this project
that cannot be repeated - would have been marked opened with no result to show
for it.

The fix is not a bigger warning. It is that the measurement must exist, be
tested, and be *proven runnable* before the token is ever presented. So this
module is written to be executed end to end against mock data in a temporary
directory, with the same code path the real run takes. The only difference on
the real run is which loader and which generator are injected.

Nothing here reads the sealed split by itself. Records arrive through an
injected loader; candidates arrive through an injected generator. Tests pass
mocks. The authorized entrypoint passes the real ones, and only after every
non-sealed prerequisite has already been checked.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from harness.source_identity import canonical_sha256, canonical_text_sha256, raw_sha256
from metrics.research_evaluation import (
    candidate_diversity, evaluate_candidate_slots, function_result,
    summarise_function_results,
)

EVALUATOR_VERSION = "oneiros_sealed_final_evaluator_v1"

#: Minimum free bytes required before a one-time run may begin. A final
#: measurement that dies half way because the disk filled is not repeatable.
MIN_FREE_DISK_BYTES = 2 * 1024 * 1024 * 1024


class FinalEvaluationError(RuntimeError):
    """Raised for any condition that must stop a final run."""


def evaluator_source_hashes() -> Dict[str, str]:
    """Identity of the code that will produce the final number.

    Bound into the executable receipt so the thing approved and the thing that
    runs are demonstrably the same. The function-level digest is narrower than
    the file digest and moves only when the measurement logic moves.
    """
    path = Path(__file__).resolve()
    body = "".join(inspect.getsource(fn) for fn in (
        run_final_evaluation, _evaluate_one_record, environment_problems,
    ))
    return {
        "module": "harness/sealed_final_evaluator.py",
        "evaluator_version": EVALUATOR_VERSION,
        "raw_sha256": raw_sha256(path),
        "canonical_sha256": canonical_sha256(path),
        "measurement_logic_canonical_sha256": canonical_text_sha256(body),
    }


# --------------------------------------------------------------------------
# Pre-authorization checks. Every one of these runs BEFORE a token is
# presented, because discovering any of them afterwards would have cost the
# single authorization.
# --------------------------------------------------------------------------

def environment_problems(
    *,
    output_dir: Path,
    model_name: str,
    model_revision: str,
    expected_candidates: int,
    model_files_present: Optional[Callable[[str, str], bool]] = None,
    cuda_available: Optional[Callable[[], bool]] = None,
    free_disk_bytes: Optional[Callable[[Path], int]] = None,
    require_cuda: bool = True,
) -> List[str]:
    """Everything that must hold before the sealed split may be opened.

    The callables are injected so this is testable without a GPU, without the
    model on disk and without touching anything real. The real entrypoint
    passes implementations that actually look.
    """
    problems: List[str] = []

    if not model_name:
        problems.append("no base model name was supplied")
    if not model_revision or len(str(model_revision)) != 40:
        problems.append(
            f"base model revision is not an immutable 40-character SHA: {model_revision!r}")

    if model_files_present is not None and not model_files_present(model_name, model_revision):
        problems.append(
            f"model files for {model_name} at revision {model_revision} are not "
            "available locally; a final run must not download weights mid-measurement")

    if expected_candidates != 8:
        problems.append(
            f"candidates per target is {expected_candidates}, the frozen protocol declares 8")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        problems.append(f"output directory is not empty: {output_dir}")
    parent = output_dir.parent
    if not parent.exists():
        problems.append(f"output parent directory does not exist: {parent}")
    else:
        probe = parent / f".write_probe_{int(time.time() * 1000)}"
        try:
            probe.write_text("probe", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            problems.append(f"output directory is not writable: {parent} ({exc})")

    if free_disk_bytes is not None and parent.exists():
        free = free_disk_bytes(parent)
        if free < MIN_FREE_DISK_BYTES:
            problems.append(
                f"insufficient free disk: {free:,} bytes available, "
                f"{MIN_FREE_DISK_BYTES:,} required")

    if require_cuda and cuda_available is not None and not cuda_available():
        problems.append("CUDA is not available; the final measurement requires the GPU")

    # The evaluator must actually be callable. A broken module is exactly the
    # failure that must not be discovered after the token is spent.
    for name, fn in (("run_final_evaluation", run_final_evaluation),
                     ("_evaluate_one_record", _evaluate_one_record)):
        if not callable(fn):
            problems.append(f"final evaluator symbol {name} is not callable")
    return problems


def default_free_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(Path(path)).free


# --------------------------------------------------------------------------
# The measurement itself.
# --------------------------------------------------------------------------

def _slots_from_outputs(raw_outputs: Sequence[str], parsed: Sequence[Optional[str]]):
    """Ordered generation slots, each retaining its raw output and its hash."""
    slots = []
    for rank, (raw, code) in enumerate(zip(raw_outputs, parsed), start=1):
        text = "" if raw is None else str(raw)
        slots.append({
            "rank": rank,
            "code": code,
            "parse_valid": bool(code),
            "raw_output": text,
            "raw_output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return slots


def _evaluate_one_record(
    record: Mapping[str, Any],
    generate: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]],
    candidates_per_target: int,
    allow_test_function: bool,
) -> Dict[str, Any]:
    """Generate for one target and score it with the frozen evaluator."""
    entry_point = str(record.get("entry_point") or "")
    generated = list(generate(record, candidates_per_target))
    if len(generated) != candidates_per_target:
        raise FinalEvaluationError(
            f"generator returned {len(generated)} candidates for "
            f"{record.get('id')!r}, expected {candidates_per_target}")

    raw_outputs = [item.get("raw_output", "") for item in generated]
    parsed = [item.get("code") for item in generated]
    slots = _slots_from_outputs(raw_outputs, parsed)

    budget_failure = bool(record.get("prompt_budget_failure"))
    if budget_failure:
        outcomes = [dict(slot, parse_valid=False, policy_valid=False,
                         execution_valid=False, reference_valid=False,
                         killed=False, failure_mode="prompt_budget_failure")
                    for slot in slots]
    else:
        outcomes = evaluate_candidate_slots(
            slots,
            golden_code=str(record.get("golden_code") or ""),
            mutant_code=str(record.get("mutant_code") or ""),
            entry_point=entry_point,
            allow_test_function=allow_test_function,
        )
        # evaluate_candidate_slots deep-copies its input; carry the retained
        # raw output and its hash back onto the scored outcome so the artifact
        # keeps the evidence, not just the verdict.
        for slot, outcome in zip(slots, outcomes):
            outcome["raw_output"] = slot["raw_output"]
            outcome["raw_output_sha256"] = slot["raw_output_sha256"]

    result = function_result(
        record_id=str(record.get("id")),
        bug_family=str(record.get("bug_family") or "unknown"),
        entry_point=entry_point,
        outcomes=outcomes,
        source_name=str(record.get("source_name") or "unknown"),
        project=str(record.get("project") or "unknown"),
        dataset_name=str(record.get("dataset_name") or "unknown"),
        prompt_budget_failure=budget_failure,
        prompt_budget_failure_reason=record.get("prompt_budget_failure_reason"),
    )
    result["diversity"] = candidate_diversity(outcomes, entry_point)
    result["candidate_outcomes"] = outcomes
    return result


def run_final_evaluation(
    *,
    load_records: Callable[[], Sequence[Mapping[str, Any]]],
    generate: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]],
    output_dir: Path,
    bundle_sha256: str,
    frozen_settings: Mapping[str, Any],
    candidates_per_target: int = 8,
    allow_test_function: bool = True,
    k_values: Sequence[int] = (1, 2, 4, 8),
    progress_every: int = 2,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the one-time final measurement and persist a complete artifact.

    Durable by construction: a progress file is written every ``progress_every``
    targets, so a crash leaves evidence of exactly how far the single authorized
    run got. There is deliberately no resume - a second attempt at a one-time
    measurement is not a resume, it is a second measurement.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    say = log or (lambda message: None)

    records = list(load_records())
    if not records:
        raise FinalEvaluationError("the sealed evaluation scope is empty")

    started = time.time()
    results: List[Dict[str, Any]] = []
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(exist_ok=True)

    for index, record in enumerate(records, start=1):
        results.append(_evaluate_one_record(
            record, generate, candidates_per_target, allow_test_function))
        if index % progress_every == 0 or index == len(records):
            killed = sum(bool(item.get("killed")) for item in results)
            (progress_dir / f"progress.{index:06d}.json").write_text(
                json.dumps({
                    "completed": index, "total": len(records), "killed": killed,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "bundle_sha256": bundle_sha256,
                }, indent=2) + "\n", encoding="utf-8")
            say(f"final evaluation progress={index}/{len(records)} killed={killed}")

    summary = summarise_function_results(results, k_values=k_values)

    total = missing = mismatched = 0
    for item in results:
        for outcome in item["candidate_outcomes"]:
            total += 1
            raw = outcome.get("raw_output")
            digest = outcome.get("raw_output_sha256")
            if raw is None or digest is None:
                missing += 1
            elif hashlib.sha256(str(raw).encode("utf-8")).hexdigest() != digest:
                mismatched += 1
    if missing or mismatched:
        raise FinalEvaluationError(
            f"raw-output integrity failed: {missing} missing, {mismatched} mismatched")

    scope_sha256 = hashlib.sha256(json.dumps(
        [str(r.get("id")) for r in records], separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    artifact = {
        "schema_version": EVALUATOR_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "final_test_measurement": True,
        "one_time": True,
        "bundle_sha256": bundle_sha256,
        "evaluator_source": evaluator_source_hashes(),
        "frozen_settings": dict(frozen_settings),
        "evaluation_scope_sha256": scope_sha256,
        "function_validation_records": len(records),
        "candidates_per_target": candidates_per_target,
        "raw_output_integrity": {
            "candidates": total, "missing": missing, "mismatched": mismatched,
            "complete": missing == 0 and mismatched == 0,
        },
        "wall_time_seconds": round(time.time() - started, 3),
        "function_results": results,
    }
    artifact.update(summary)

    path = output_dir / "sealed_final_result.json"
    payload = (json.dumps(artifact, indent=2) + "\n").encode("utf-8")
    path.write_bytes(payload)
    artifact_sha = hashlib.sha256(payload).hexdigest()
    (output_dir / "sealed_final_result.sha256").write_text(
        artifact_sha + "\n", encoding="utf-8")
    say(f"final artifact written: {path} sha256={artifact_sha}")

    return {
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha,
        "function_validation_records": len(records),
        "evaluation_scope_sha256": scope_sha256,
        "kill_at_k": summary.get("kill_at_k"),
        "raw_output_integrity": artifact["raw_output_integrity"],
        "wall_time_seconds": artifact["wall_time_seconds"],
    }
