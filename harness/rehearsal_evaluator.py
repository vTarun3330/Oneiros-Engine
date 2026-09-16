"""Operational rehearsal of the final-measurement flow, on permitted splits.

**Every artifact this module writes is an operational rehearsal. None of it is
eligible for model selection or a final-performance claim.** That label is not
decoration: `val` is already spent on selection and `ablation_dev` selected the
checkpoints it scored, so a number produced here is an operational fact about
the pipeline, not evidence about the model.

The point is to exercise the stages a future independent final protocol would
use - admission and scoping, the generation adapter, the candidate parser, the
safe executor, raw-output retention, the progress writer and the scoring flow -
against real data, at full scale, *before* anything irreversible depends on
them. The sealed-final attempt failed because its pre-authorization smoke
proved the generator on two synthetic records and nobody had run the loader
against a real split.

The scoring body is **imported from the sealed-final evaluator**, not copied.
Two implementations of the same measurement is two things to be wrong, and the
rehearsal is worthless if it rehearses different code. The consumed sealed path
itself is not modified, resurrected or called: only its pure scoring helpers
are reused, and this module writes its own artifact envelope so a rehearsal can
never be mistaken for a final measurement.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

REHEARSAL_VERSION = "oneiros_operational_rehearsal_v1"

#: Attached to the artifact, to every progress file, and to the printed
#: summary. Anything that quotes a number from here must carry it too.
REHEARSAL_LABEL = (
    "operational rehearsal - not eligible for model selection or "
    "final-performance claims"
)


class RehearsalError(RuntimeError):
    """Raised when the rehearsal cannot complete honestly."""


def rehearsal_source_hashes() -> Dict[str, Any]:
    """Identity of every module that decides a rehearsal number."""
    from harness.evaluation_admission import admission_source_hashes
    from harness.sealed_final_evaluator import evaluator_source_hashes
    from harness.source_identity import canonical_sha256, raw_sha256

    root = Path(__file__).resolve().parent.parent
    here = root / "harness/rehearsal_evaluator.py"
    return {
        "rehearsal_version": REHEARSAL_VERSION,
        "rehearsal": {
            "path": "harness/rehearsal_evaluator.py",
            "raw_sha256": raw_sha256(here),
            "canonical_sha256": canonical_sha256(here),
        },
        "admission": admission_source_hashes(root),
        "scoring": evaluator_source_hashes(),
    }


def run_rehearsal_evaluation(
    *,
    load_records: Callable[[], Sequence[Mapping[str, Any]]],
    generate_batch: Callable[[Sequence[Mapping[str, Any]]], Sequence[Sequence[Mapping[str, Any]]]],
    output_dir: Path,
    split_name: str,
    scope_summary: Mapping[str, Any],
    frozen_settings: Mapping[str, Any],
    receipt_sha256: str = "",
    candidates_per_target: int = 8,
    generation_batch_size: int = 2,
    allow_test_function: bool = True,
    k_values: Sequence[int] = (1, 2, 4, 8),
    progress_every: int = 10,
    log: Optional[Callable[[str], None]] = None,
    seed_record: Optional[Mapping[str, Any]] = None,
    generator_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full flow on a permitted split and persist a labelled artifact."""
    # Reused, not reimplemented. These are the pure scoring helpers the sealed
    # evaluator uses; importing them is what makes this a rehearsal of the real
    # thing rather than a rehearsal of a lookalike.
    from harness.sealed_final_evaluator import (
        _chunked, _evaluate_one_record, summarise_function_results,
    )
    from harness.evaluation_admission import REFUSED_SPLITS

    if str(split_name) in REFUSED_SPLITS:
        raise RehearsalError(
            f"refusing to rehearse on {split_name!r}: {REFUSED_SPLITS[str(split_name)]}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    say = log or (lambda message: None)

    records = list(load_records())
    if not records:
        raise RehearsalError("the rehearsal scope is empty")

    started = time.time()
    results: List[Dict[str, Any]] = []
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(exist_ok=True)

    index = 0
    batches = 0
    failures: Dict[str, int] = {}
    for batch in _chunked(records, generation_batch_size):
        outputs = list(generate_batch(batch))
        batches += 1
        if len(outputs) != len(batch):
            raise RehearsalError(
                f"batch generation returned {len(outputs)} results for "
                f"{len(batch)} targets")
        for record, generated in zip(batch, outputs):
            index += 1
            results.append(_evaluate_one_record(
                record, generated, candidates_per_target, allow_test_function))
        if index % progress_every == 0 or index == len(records):
            killed = sum(bool(item.get("killed")) for item in results)
            (progress_dir / f"progress.{index:06d}.json").write_text(
                json.dumps({
                    "label": REHEARSAL_LABEL,
                    "completed": index, "total": len(records), "killed": killed,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "batches": batches,
                    "generation_batch_size": generation_batch_size,
                    "split": split_name,
                }, indent=2) + "\n", encoding="utf-8")
            say(f"rehearsal progress={index}/{len(records)} killed={killed} "
                f"elapsed={round(time.time() - started)}s")

    summary = summarise_function_results(results, k_values=k_values)

    # Failure taxonomy, over every candidate slot.
    total = missing = mismatched = 0
    for item in results:
        for outcome in item["candidate_outcomes"]:
            total += 1
            mode = outcome.get("failure_mode")
            if mode:
                failures[str(mode)] = failures.get(str(mode), 0) + 1
            raw = outcome.get("raw_output")
            digest = outcome.get("raw_output_sha256")
            if raw is None or digest is None:
                missing += 1
            elif hashlib.sha256(str(raw).encode("utf-8")).hexdigest() != digest:
                mismatched += 1
    if missing or mismatched:
        raise RehearsalError(
            f"raw-output integrity failed: {missing} missing, {mismatched} mismatched")

    scope_sha256 = hashlib.sha256(json.dumps(
        [str(r.get("id")) for r in records], separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    artifact = {
        "schema_version": REHEARSAL_VERSION,
        "label": REHEARSAL_LABEL,
        # Explicitly false. Any consumer that refuses final-test artifacts must
        # see this and accept the file; any consumer looking for a final result
        # must see it and reject it.
        "final_test_measurement": False,
        "eligible_for_model_selection": False,
        "supports_performance_claim": False,
        "operational_rehearsal": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": split_name,
        "rehearsal_receipt_sha256": receipt_sha256,
        "admission_scope": dict(scope_summary),
        "source_hashes": rehearsal_source_hashes(),
        "frozen_settings": dict(frozen_settings),
        "evaluation_scope_sha256": scope_sha256,
        "function_validation_records": len(records),
        "candidates_per_target": candidates_per_target,
        "generation_batch_size": generation_batch_size,
        "generation_batches": batches,
        "seed_application": dict(seed_record) if seed_record else None,
        "generator_identity": dict(generator_identity) if generator_identity else None,
        "failure_taxonomy": dict(sorted(failures.items())),
        "raw_output_integrity": {
            "candidates": total, "missing": missing, "mismatched": mismatched,
            "complete": missing == 0 and mismatched == 0,
        },
        "wall_time_seconds": round(time.time() - started, 3),
        "function_results": results,
    }
    artifact.update(summary)

    path = output_dir / "rehearsal_result.json"
    payload = (json.dumps(artifact, indent=2) + "\n").encode("utf-8")
    path.write_bytes(payload)
    (output_dir / "rehearsal_result.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8")
    say(f"rehearsal artifact: {path}")
    return artifact
