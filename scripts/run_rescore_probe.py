"""Run the R1 probe: do harness failures re-score, and do controls agree?

This deliberately does NOT go through ``select_jobs``. That function refuses to
touch a candidate with a real answer, and tests hold it to that - weakening it
so a probe could re-run controls would remove the guarantee the repair depends
on. The probe calls ``rescore_one`` directly on an explicit key list instead,
which is a measurement, not a repair, and writes no derived artifact.

The control arm is the whole point. Re-scoring the affected candidates can only
show that answers appear; re-scoring candidates that already had answers shows
whether the executor returns the SAME one. Disagreement there would mean the
executor is non-deterministic rather than merely contended, and repairing the
affected set would launder noise into supervision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.corpus_view import load_development_split
from harness.parallel_execution import progress_map
from scripts.rescore_harness_failures import (
    HARNESS_STATUSES, WORKER_COUNT, rescore_one,
)

#: Predeclared, before the probe ran. The executor is expected to be
#: deterministic for these candidates: identical code, identical reference,
#: identical mutant. Anything above this is not "flaky under load", it is a
#: reproducibility failure, and the repair must not proceed.
MAX_CONTROL_DISAGREEMENT_RATE = 0.01
#: Matches the gate the repaired artifact must later clear.
MAX_POST_RETRY_HARNESS_RATE = 0.01


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jobs_for(payload: dict[str, Any], records: dict[str, dict[str, Any]],
              keys: set[tuple[str, int, int]], allow_test_function: bool):
    jobs = []
    for result in payload.get("function_results") or []:
        record_id = str(result.get("record_id"))
        record = records.get(record_id)
        if record is None:
            continue
        for position, outcome in enumerate(result.get("candidate_outcomes") or []):
            key = (record_id, int(outcome.get("rank") or 0), position)
            if key not in keys:
                continue
            code = outcome.get("code")
            if not isinstance(code, str) or not code:
                continue
            support = str(record.get("support_context") or "")
            jobs.append({
                "record_id": record_id, "rank": key[1], "position": position,
                "entry_point": str(record.get("entry_point") or ""),
                "code": code, "code_sha256": _sha_text(code),
                "raw_output_sha256": outcome.get("raw_output_sha256"),
                "original_status": str(outcome.get("reference_status") or ""),
                "reference": support + "\n" + str(record.get("reference_code") or ""),
                "mutant": support + "\n" + str(record.get("code_under_test") or ""),
                "allow_test_function": allow_test_function,
            })
    return jobs


def run(artifact: Path, selection: Path, corpus_dir: Path,
        workers: int) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("evaluation_split") != "train":
        raise SystemExit("probe may only run on the train split")
    probe = json.loads(selection.read_text(encoding="utf-8"))
    contract = payload.get("run_contract") or {}
    records = {str(r["id"]): r for r in load_development_split(
        corpus_dir, "train", include_excluded=True)}

    affected_keys = {(str(k[0]), int(k[1]), int(k[2]))
                     for k in probe["candidate_keys"]}
    control_keys = {(str(k[0]), int(k[1]), int(k[2]))
                    for k in probe["control_keys"]}
    allow = bool(contract.get("allow_test_function_candidates"))

    started = time.time()
    affected = progress_map(
        rescore_one, _jobs_for(payload, records, affected_keys, allow),
        "probe: affected", every=100, workers=workers)
    control = progress_map(
        rescore_one, _jobs_for(payload, records, control_keys, allow),
        "probe: control", every=100, workers=workers)
    elapsed = time.time() - started

    still_failing = sum(1 for r in affected if not r["repaired"])
    affected_status = Counter(
        str((r["final"] or {}).get("reference_status") or "still_unscored")
        for r in affected)

    expected = probe["control_expected"]
    disagreements: list[dict[str, Any]] = []
    control_checked = 0
    for row in control:
        key = f'{row["record_id"]}|{row["rank"]}|{row["position"]}'
        want = expected.get(key)
        if want is None or row["final"] is None:
            continue
        control_checked += 1
        got = row["final"]
        if (got.get("reference_status") != want["reference_status"]
                or bool(got.get("killed")) != bool(want["killed"])
                or bool(got.get("reference_valid")) != bool(want["reference_valid"])):
            disagreements.append({
                "key": key, "expected": want,
                "observed": {k: got.get(k) for k in
                             ("reference_status", "killed", "reference_valid")}})

    harness_rate = still_failing / len(affected) if affected else 0.0
    disagree_rate = len(disagreements) / control_checked if control_checked else 0.0
    passed = (harness_rate < MAX_POST_RETRY_HARNESS_RATE
              and disagree_rate < MAX_CONTROL_DISAGREEMENT_RATE
              and control_checked > 0)

    return {
        "schema_version": "oneiros_rescore_probe_result_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "selection_hash": probe["selection_hash"],
        "sealed_final_test_accessed": False,
        "record_source": "hash-verified development view train shard",
        "canonical_records_json_opened": False,
        "workers": workers,
        "elapsed_seconds": round(elapsed, 1),
        "affected_probe_size": len(affected),
        "affected_repaired": len(affected) - still_failing,
        "affected_still_unscored": still_failing,
        "post_retry_harness_failure_rate": round(harness_rate, 6),
        "max_post_retry_harness_failure_rate": MAX_POST_RETRY_HARNESS_RATE,
        "affected_status_distribution": dict(affected_status.most_common()),
        "control_probe_size": len(control),
        "control_checked": control_checked,
        "control_disagreements": len(disagreements),
        "control_disagreement_rate": round(disagree_rate, 6),
        "max_control_disagreement_rate": MAX_CONTROL_DISAGREEMENT_RATE,
        "control_disagreement_examples": disagreements[:15],
        "retried_statuses": list(HARNESS_STATUSES),
        "gate_passed": passed,
        "evidence": (
            "affected candidates re-scoring cleanly while controls reproduce "
            "their original outcomes supports transient executor failure. "
            "Control disagreement would indicate non-determinism instead, and "
            "the repair would be laundering noise."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = run(arguments.artifact, arguments.selection, arguments.corpus,
                 arguments.workers)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "control_disagreement_examples"}, indent=2))
    if not report["gate_passed"]:
        print("\nPROBE GATE FAILED - do not proceed to the full re-score.")
        for row in report["control_disagreement_examples"][:5]:
            print("  " + json.dumps(row))
        return 1
    print("\nprobe gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
