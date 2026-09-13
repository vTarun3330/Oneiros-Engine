"""Re-score candidates the execution harness never scored. CPU only.

23,538 of 44,760 candidates in the successor generation carry
``reference_status = worker_error``: the worker returned no parseable
response, so "did this candidate kill the mutant" has no recorded answer. The
legacy run over the identical panel carried zero. Replaying a wholly-failed
function reproduced clean results, which makes transient executor failure a
plausible hypothesis - not a proven one, which is why this tool measures rather
than assumes.

The GPU is not involved. Raw outputs are retained and hash-valid, so the
generation is sound; only the CPU-side scoring failed.

WHAT IS AND IS NOT RETRIED. Only INFRASTRUCTURE statuses. A candidate that
raised, timed out, failed its assertion or simply did not kill produced a
real answer, and re-running it until a different answer appears would be
selecting outcomes rather than repairing them. ``assertion_error``, ``error``,
``timeout`` and ``pass`` are never touched.

ONE CANDIDATE PER WORKER CALL. The original scoring passed every candidate of
a function to ``classify_assertions`` in a single invocation, which is why
failures were all-or-nothing per function: one broken invocation loses all
eight answers. Scoring individually is semantically identical - each test is
executed independently inside the worker either way - and removes that
coupling entirely.

IMMUTABILITY. The parent artifact is opened read-only and its SHA-256 is
recorded. Output goes to a separately named derived artifact plus an overlay
carrying every retry attempt. Nothing overwrites the original.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import executable_candidate, validate_generated_test
from harness.corpus import sha256_file, write_json
from harness.corpus_view import load_development_split
from harness.parallel_execution import progress_map
from harness.safe_execution import classify_assertions
from metrics.research_evaluation import function_result, summarise_function_results

#: Statuses that mean the harness failed rather than the candidate. Only these
#: are re-scored.
HARNESS_STATUSES = ("worker_error", "system_exit", "keyboard_interrupt")

#: Statuses that are real answers and must never be retried, listed explicitly
#: so that "we only retry infrastructure" is checkable rather than implied.
SEMANTIC_STATUSES = ("pass", "assertion_error", "error", "timeout",
                     "source_policy_error")

MAX_RETRIES = 3
#: Deliberately far below the 32 available cores. The failures being repaired
#: are consistent with executor contention, so the repair does not recreate it.
WORKER_COUNT = 8
TIMEOUT_SECONDS = 5.0

SOURCE_HASH_FIELDS = {
    "evaluator_source_sha256": "metrics/research_evaluation.py",
    "candidate_policy_source_sha256": "harness/candidate_policy.py",
    "safe_execution_source_sha256": "harness/safe_execution.py",
    "prompt_builder_source_sha256": "engine/test_generation_prompt.py",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def assert_preconditions(payload: dict[str, Any], parent: Path) -> dict[str, Any]:
    """Refuse to re-score anything whose provenance cannot be established."""
    problems: list[str] = []
    if payload.get("evaluation_split") != "train":
        problems.append(
            f"evaluation_split is {payload.get('evaluation_split')!r}; only the "
            "train split may be re-scored")
    if payload.get("final_test_measurement"):
        problems.append("artifact is a sealed final-test measurement")

    contract = payload.get("run_contract") or {}
    if not contract:
        problems.append("artifact carries no run_contract; its generation "
                        "settings cannot be established")
    drifted = []
    for field, relative in SOURCE_HASH_FIELDS.items():
        expected = contract.get(field)
        actual = sha256_file(ROOT / relative)
        if expected and expected != actual:
            drifted.append(relative)
    if drifted:
        problems.append(
            "source has drifted since generation for: " + ", ".join(drifted)
            + ". Re-scoring under different evaluator or policy source would "
            "produce outcomes the original run could never have produced")
    if problems:
        raise SystemExit("re-score refused:\n  - " + "\n  - ".join(problems))
    return {"parent_sha256": sha256_file(parent), "run_contract": contract,
            "source_hashes_match": True}


def select_jobs(payload: dict[str, Any], records: dict[str, dict[str, Any]],
                allow_test_function: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exactly the harness-failed candidates, with their identity preserved."""
    jobs: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for result in payload.get("function_results") or []:
        record_id = str(result.get("record_id"))
        record = records.get(record_id)
        for position, outcome in enumerate(result.get("candidate_outcomes") or []):
            status = str(outcome.get("reference_status") or "")
            if status not in HARNESS_STATUSES:
                skipped[status or "(none)"] = skipped.get(status or "(none)", 0) + 1
                continue
            if record is None:
                skipped["record_not_in_train_shard"] = skipped.get(
                    "record_not_in_train_shard", 0) + 1
                continue
            code = outcome.get("code")
            raw = outcome.get("raw_output")
            if not isinstance(code, str) or not code:
                skipped["no_parsed_code"] = skipped.get("no_parsed_code", 0) + 1
                continue
            recorded_raw_hash = outcome.get("raw_output_sha256")
            if isinstance(raw, str) and recorded_raw_hash \
                    and _sha256_text(raw) != recorded_raw_hash:
                skipped["raw_hash_mismatch"] = skipped.get("raw_hash_mismatch", 0) + 1
                continue
            support = str(record.get("support_context") or "")
            jobs.append({
                "record_id": record_id,
                "rank": int(outcome.get("rank") or 0),
                "position": position,
                "entry_point": str(record.get("entry_point") or ""),
                "code": code,
                "code_sha256": _sha256_text(code),
                "raw_output_sha256": recorded_raw_hash,
                "original_status": status,
                "reference": support + "\n" + str(record.get("reference_code") or ""),
                "mutant": support + "\n" + str(record.get("code_under_test") or ""),
                "allow_test_function": allow_test_function,
            })
    return jobs, skipped


def rescore_one(job: dict[str, Any]) -> dict[str, Any]:
    """Re-score one candidate, keeping every attempt."""
    policy = validate_generated_test(job["code"], job["entry_point"],
                                     job["allow_test_function"])
    attempts: list[dict[str, Any]] = []
    if not policy.valid:
        # The policy verdict does not depend on the executor, so a
        # policy-invalid candidate is resolved without running anything.
        return {**job, "attempts": attempts, "repaired": True,
                "final": {"policy_valid": False, "reference_status": "policy_invalid",
                          "policy_error": str(policy.reason)[:240]}}

    executable = executable_candidate(job["code"], policy.shape)
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.time()
        try:
            rows = classify_assertions([executable], job["reference"],
                                       job["mutant"], TIMEOUT_SECONDS)
            row = rows[0] if rows else {}
            golden = row.get("golden") or {}
            mutant = row.get("mutant") or {}
            status = str(golden.get("status", "unknown"))
            error = str(golden.get("error", ""))[:240]
        except Exception as exc:                       # pragma: no cover
            status, error, row, golden, mutant = "worker_error", str(exc)[:240], {}, {}, {}
        attempts.append({
            "attempt": attempt,
            "reference_status": status,
            "error": error,
            "seconds": round(time.time() - started, 3),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if status not in HARNESS_STATUSES:
            return {**job, "attempts": attempts, "repaired": True, "final": {
                "policy_valid": True,
                "candidate_shape": policy.shape,
                "execution_valid": status in {"pass", "assertion_error"},
                "reference_valid": bool(row.get("valid")),
                "killed": bool(row.get("killed")),
                "reference_status": status,
                "mutant_status": str(mutant.get("status", "unknown")),
                "reference_error": error,
            }}
    # Still failing after every attempt: it stays UNSCORED. A candidate with no
    # answer must not acquire one by exhaustion.
    return {**job, "attempts": attempts, "repaired": False, "final": None}


def build(parent: Path, corpus_dir: Path, limit_ids: set[tuple[str, int, int]] | None,
          workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(parent.read_text(encoding="utf-8"))
    provenance = assert_preconditions(payload, parent)
    contract = provenance["run_contract"]

    # Hash-verified train shard only. The canonical records.json holds all four
    # splits including the sealed final test and is never opened here.
    records = {str(r["id"]): r for r in load_development_split(
        corpus_dir, "train", include_excluded=True)}

    jobs, skipped = select_jobs(
        payload, records, bool(contract.get("allow_test_function_candidates")))
    if limit_ids is not None:
        jobs = [j for j in jobs
                if (j["record_id"], j["rank"], j["position"]) in limit_ids]

    rescored = progress_map(rescore_one, jobs, "re-scoring harness failures",
                            every=500, workers=workers)

    by_key = {(r["record_id"], r["rank"], r["position"]): r for r in rescored}
    repaired = sum(1 for r in rescored if r["repaired"])
    still_failing = len(rescored) - repaired
    status_counts: dict[str, int] = {}
    for row in rescored:
        final = row["final"] or {}
        key = str(final.get("reference_status") or "still_unscored")
        status_counts[key] = status_counts.get(key, 0) + 1

    overlay = {
        "schema_version": "oneiros_rescore_overlay_v1",
        "parent_artifact": parent.as_posix().split("results/", 1)[-1],
        "parent_sha256": provenance["parent_sha256"],
        "source_commit": _source_commit(),
        "source_hashes": {f: sha256_file(ROOT / r)
                          for f, r in SOURCE_HASH_FIELDS.items()},
        "run_contract_sha256": payload.get("run_contract_sha256"),
        "sealed_final_test_accessed": False,
        "record_source": "hash-verified development view train shard",
        "canonical_records_json_opened": False,
        "retried_statuses": list(HARNESS_STATUSES),
        "never_retried_statuses": list(SEMANTIC_STATUSES),
        "max_retries": MAX_RETRIES,
        "workers": workers,
        "timeout_seconds": TIMEOUT_SECONDS,
        "candidates_selected": len(jobs),
        "candidates_repaired": repaired,
        "candidates_still_unscored": still_failing,
        "post_retry_harness_failure_rate": round(
            still_failing / len(jobs), 6) if jobs else 0.0,
        "final_status_distribution": dict(sorted(status_counts.items())),
        "skipped_by_original_status": dict(sorted(skipped.items())),
        "replacements": [{
            "record_id": r["record_id"], "rank": r["rank"],
            "position": r["position"],
            "raw_output_sha256": r["raw_output_sha256"],
            "code_sha256": r["code_sha256"],
            "original_reference_status": r["original_status"],
            "attempts": r["attempts"],
            "repaired": r["repaired"],
            "final": r["final"],
        } for r in rescored],
    }

    # Derived artifact: the parent with repaired outcomes substituted in place.
    derived = json.loads(parent.read_text(encoding="utf-8"))
    substituted = 0
    for result in derived.get("function_results") or []:
        record_id = str(result.get("record_id"))
        outcomes = result.get("candidate_outcomes") or []
        for position, outcome in enumerate(outcomes):
            row = by_key.get((record_id, int(outcome.get("rank") or 0), position))
            if row is None or not row["repaired"] or row["final"] is None:
                continue
            outcome.update(row["final"])
            outcome["rescored"] = True
            substituted += 1
        record = records.get(record_id)
        if record is None:
            continue
        rebuilt = function_result(
            record_id, str(result.get("bug_family") or "unknown"),
            str(record.get("entry_point") or ""), outcomes,
            source_name=str(result.get("source_name") or "unknown"),
            dataset_name=result.get("dataset_name"),
            project=str(result.get("project") or "unknown"),
            prompt_budget_failure=bool(result.get("prompt_budget_failure")),
            prompt_budget_failure_reason=result.get("prompt_budget_failure_reason"),
        )
        result.update(rebuilt)

    derived.update(summarise_function_results(derived["function_results"]))
    derived["mode"] = "base_model_validation_only_rescored"
    derived["derived_from"] = overlay["parent_artifact"]
    derived["derived_from_sha256"] = provenance["parent_sha256"]
    derived["rescore_overlay_schema"] = overlay["schema_version"]
    derived["rescored_candidates"] = substituted
    derived["source_commit"] = overlay["source_commit"]
    overlay["candidates_substituted_into_derived"] = substituted
    return overlay, derived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    parser.add_argument("--probe", type=Path, default=None,
                        help="a probe selection file; re-score only those candidates")
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--derived", type=Path, default=None)
    arguments = parser.parse_args()

    limit_ids = None
    if arguments.probe is not None:
        selection = json.loads(arguments.probe.read_text(encoding="utf-8"))
        limit_ids = {(str(k[0]), int(k[1]), int(k[2]))
                     for k in selection["candidate_keys"]}

    before = sha256_file(arguments.parent)
    overlay, derived = build(arguments.parent, arguments.corpus, limit_ids,
                             arguments.workers)
    write_json(arguments.overlay, overlay)
    if arguments.derived is not None:
        arguments.derived.parent.mkdir(parents=True, exist_ok=True)
        write_json(arguments.derived, derived)
    after = sha256_file(arguments.parent)
    if before != after:
        raise SystemExit("the parent artifact changed during re-scoring")

    print(json.dumps({k: v for k, v in overlay.items() if k != "replacements"},
                     indent=2))
    print("\nparent unchanged: " + before)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
