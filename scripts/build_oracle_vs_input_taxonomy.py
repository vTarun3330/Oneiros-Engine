"""Split the dominant failure into ORACLE errors and INPUT errors, and count both.

``wrong_expected_value`` is the largest failure category in this project, and
until now it has been reported as one thing. It is two things, and they call
for opposite interventions:

* the candidate probed an input where reference and mutant differ, and only the
  expected value was wrong -> better oracle supervision recovers it;
* the candidate probed an input where they agree -> no oracle fixes it, and the
  lever is input selection.

Every other coarse category is carried through unrefined, so the totals here
reconcile with the taxonomy figures already committed.

Reports per benchmark, because the two benchmarks fail differently and the
panel is 92.6% mbpp: a pooled number is the mbpp number wearing a disguise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from harness.oracle_diagnosis import (
    BENIGN_INPUT, BOTH_ERROR, NO_CALL, ORACLE_ERROR, diagnose,
)
from harness.parallel_execution import progress_map
from metrics.research_evaluation import classify_candidate_failure


def _jobs(payload: dict[str, Any], by_id: dict[str, dict[str, Any]],
          benchmark: str | None, timeout: float):
    """Refinement jobs, plus the coarse counts that need no execution."""
    coarse: Counter = Counter()
    per_benchmark_functions: Counter = Counter()
    jobs: list[dict[str, Any]] = []

    for result in payload.get("function_results") or []:
        dataset = str(result.get("dataset_name") or "unknown")
        if benchmark and dataset != benchmark:
            continue
        per_benchmark_functions[dataset] += 1
        if result.get("killed"):
            continue
        record = by_id.get(str(result.get("record_id")))
        if record is None:
            continue
        support = str(record.get("support_context") or "")
        reference = support + "\n" + str(record.get("reference_code") or "")
        mutant = support + "\n" + str(record.get("code_under_test") or "")
        entry = str(record.get("entry_point") or "")
        family = str(result.get("bug_family") or "unknown")
        if not entry:
            continue

        for outcome in result.get("candidate_outcomes") or []:
            label = classify_candidate_failure(outcome, family)
            if label.startswith("killed"):
                continue
            coarse[(dataset, label)] += 1
            if label != "wrong_expected_value":
                continue
            jobs.append({
                "dataset": dataset,
                "record_id": str(result.get("record_id")),
                "entry_point": entry,
                "code": str(outcome.get("code") or ""),
                "reference": reference,
                "mutant": mutant,
                "timeout": timeout,
            })
    return jobs, coarse, per_benchmark_functions


def _run(job: dict[str, Any]) -> dict[str, Any]:
    result = diagnose(job["code"], job["entry_point"], job["reference"],
                      job["mutant"], job["timeout"])
    return {"dataset": job["dataset"], "record_id": job["record_id"], **result}


def build(artifact: Path, corpus_dir: Path, benchmark: str | None,
          timeout: float) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    jobs, coarse, functions = _jobs(payload, by_id, benchmark, timeout)
    refined = progress_map(_run, jobs, "oracle-vs-input refinement", every=200)

    per_dataset: dict[str, Any] = {}
    for dataset in sorted(functions):
        rows = [r for r in refined if r["dataset"] == dataset]
        labels = Counter(r["refined_label"] for r in rows)
        oracle = labels.get(ORACLE_ERROR, 0)
        benign = labels.get(BENIGN_INPUT, 0)
        decided = oracle + benign
        other = {label: count for (source, label), count in coarse.items()
                 if source == dataset and label != "wrong_expected_value"}
        # A function counts as oracle-recoverable if ANY of its candidates
        # already probes a distinguishing input: one correct value there is a
        # kill, and kill@k needs only one.
        recoverable = {r["record_id"] for r in rows
                       if r["refined_label"] == ORACLE_ERROR}
        per_dataset[dataset] = {
            "functions": functions[dataset],
            "wrong_expected_value_candidates": len(rows),
            ORACLE_ERROR: oracle,
            BENIGN_INPUT: benign,
            BOTH_ERROR: labels.get(BOTH_ERROR, 0),
            NO_CALL: labels.get(NO_CALL, 0),
            "oracle_share_of_decided": round(oracle / decided, 6) if decided else None,
            "input_share_of_decided": round(benign / decided, 6) if decided else None,
            "functions_with_an_oracle_recoverable_candidate": len(recoverable),
            "other_coarse_categories": dict(sorted(other.items())),
        }

    all_labels = Counter(r["refined_label"] for r in refined)
    oracle_total = all_labels.get(ORACLE_ERROR, 0)
    benign_total = all_labels.get(BENIGN_INPUT, 0)
    decided_total = oracle_total + benign_total

    return {
        "schema_version": "oneiros_oracle_vs_input_taxonomy_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "refinement": (
            "wrong_expected_value is re-executed on the candidate's OWN call "
            "against reference and mutant. Every other coarse category is "
            "carried through unrefined so totals reconcile with committed "
            "taxonomy figures."
        ),
        "candidates_refined": len(refined),
        ORACLE_ERROR: oracle_total,
        BENIGN_INPUT: benign_total,
        BOTH_ERROR: all_labels.get(BOTH_ERROR, 0),
        NO_CALL: all_labels.get(NO_CALL, 0),
        "oracle_share_of_decided": round(
            oracle_total / decided_total, 6) if decided_total else None,
        "per_benchmark": per_dataset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--benchmark", default=None,
                        help="restrict to one dataset; default reports every one")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.artifact, arguments.corpus,
                   arguments.benchmark, arguments.timeout)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_benchmark"},
                     indent=2))
    for dataset, block in report["per_benchmark"].items():
        print("")
        print(dataset + ": oracle=" + str(block[ORACLE_ERROR])
              + " input=" + str(block[BENIGN_INPUT])
              + " oracle_share=" + str(block["oracle_share_of_decided"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
