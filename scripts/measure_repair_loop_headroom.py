"""How much could a verifier-guided repair round actually recover?

Running the repair loop costs GPU inference. This costs nothing: it replays the
retained generations through the SAME observation the loop would make, and
reports what the loop would have been told. That fixes the loop's ceiling
before any of it is spent.

The reading that matters is the cross-tabulation:

* ``passes_on_code_under_test`` - the candidate cannot reveal a defect in the
  code it was shown. The loop sees this directly and can ask for a different
  input. This is the repairable bucket.
* ``fails_on_code_under_test`` but still not a kill - the candidate already
  distinguishes the shown code, so the loop has nothing useful to say; the
  asserted value is wrong on the reference and the loop cannot see that.
  This is the ORACLE bucket, and it is the loop's blind spot.
* ``raises`` / ``parse_error`` - executability, which the loop sees and can fix.

A loop that is mostly facing the oracle bucket will not move the number much,
and it is cheaper to learn that here.
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
from harness.parallel_execution import progress_map
from harness.verifier_guided_repair import observe
from metrics.research_evaluation import classify_candidate_failure

REPAIRABLE = {"passes_on_code_under_test", "raises_on_code_under_test",
              "parse_error", "policy_rejected"}
BLIND_SPOT = "fails_on_code_under_test"


def _run(job: dict[str, Any]) -> dict[str, Any]:
    result = observe(job["code"], job["entry_point"], job["code_under_test"],
                     timeout=job["timeout"])
    return {"dataset": job["dataset"], "record_id": job["record_id"],
            "coarse": job["coarse"], "observation": result["observation"]}


def build(artifact: Path, corpus_dir: Path, timeout: float,
          limit: int | None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("final_test_measurement") or payload.get("evaluation_split") == "test":
        raise SystemExit(str(artifact) + " is sealed final-test data; refusing")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    jobs: list[dict[str, Any]] = []
    functions: Counter = Counter()
    for result in payload.get("function_results") or []:
        dataset = str(result.get("dataset_name") or "unknown")
        functions[dataset] += 1
        if result.get("killed"):
            continue
        record = by_id.get(str(result.get("record_id")))
        if record is None:
            continue
        support = str(record.get("support_context") or "")
        # The code the model was SHOWN, which is what the loop may execute
        # against. prompt_code_under_test is the displayed form.
        shown = support + "\n" + str(
            record.get("prompt_code_under_test")
            or record.get("code_under_test") or "")
        entry = str(record.get("entry_point") or "")
        family = str(result.get("bug_family") or "unknown")
        if not entry:
            continue
        for outcome in result.get("candidate_outcomes") or []:
            coarse = classify_candidate_failure(outcome, family)
            if coarse.startswith("killed"):
                continue
            jobs.append({
                "dataset": dataset,
                "record_id": str(result.get("record_id")),
                "entry_point": entry,
                "code": str(outcome.get("code") or ""),
                "code_under_test": shown,
                "coarse": coarse,
                "timeout": timeout,
            })
            if limit is not None and len(jobs) >= limit:
                break
        if limit is not None and len(jobs) >= limit:
            break

    observed = progress_map(_run, jobs, "repair-loop observation", every=250)

    per_dataset: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in observed}):
        rows = [row for row in observed if row["dataset"] == dataset]
        labels = Counter(row["observation"] for row in rows)
        repairable = sum(labels.get(label, 0) for label in REPAIRABLE)
        blind = labels.get(BLIND_SPOT, 0)
        total = len(rows) or 1
        # Functions where EVERY failed candidate is in the blind spot: the
        # repair loop has nothing to offer these at all.
        by_function: dict[str, set[str]] = {}
        for row in rows:
            by_function.setdefault(row["record_id"], set()).add(row["observation"])
        wholly_blind = sum(
            1 for labels_seen in by_function.values()
            if labels_seen == {BLIND_SPOT})
        per_dataset[dataset] = {
            "functions_in_panel": functions[dataset],
            "failed_candidates_observed": len(rows),
            "observations": dict(labels.most_common()),
            "repairable_candidates": repairable,
            "repairable_share": round(repairable / total, 6),
            "blind_spot_candidates": blind,
            "blind_spot_share": round(blind / total, 6),
            "unkilled_functions_observed": len(by_function),
            "functions_where_every_candidate_is_blind_spot": wholly_blind,
            "coarse_by_observation": {
                label: dict(Counter(
                    row["coarse"] for row in rows
                    if row["observation"] == label).most_common())
                for label in sorted(labels)
            },
        }

    labels = Counter(row["observation"] for row in observed)
    total = len(observed) or 1
    return {
        "schema_version": "oneiros_repair_loop_headroom_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": payload.get("evaluation_split"),
        "sealed_final_test_accessed": False,
        "boundary": (
            "every observation is produced by executing the candidate against "
            "the code the model was SHOWN. The reference is never executed and "
            "kill status is never observed."
        ),
        "failed_candidates_observed": len(observed),
        "observations": dict(labels.most_common()),
        "repairable_share": round(
            sum(labels.get(label, 0) for label in REPAIRABLE) / total, 6),
        "blind_spot_share": round(labels.get(BLIND_SPOT, 0) / total, 6),
        "per_benchmark": per_dataset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build(arguments.artifact, arguments.corpus, arguments.timeout,
                   arguments.limit)
    write_json(arguments.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_benchmark"},
                     indent=2))
    for dataset, block in report["per_benchmark"].items():
        print("")
        print(dataset + ":")
        print("  repairable = " + str(block["repairable_candidates"])
              + " (" + str(block["repairable_share"]) + ")")
        print("  blind spot = " + str(block["blind_spot_candidates"])
              + " (" + str(block["blind_spot_share"]) + ")")
        print("  functions with nothing to offer = "
              + str(block["functions_where_every_candidate_is_blind_spot"])
              + " / " + str(block["unkilled_functions_observed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
