"""Supervision that makes the ORACLE explicit, because the oracle is what fails.

Measured: on mbpp, ~73% of the dominant failure is a wrong expected VALUE on an
input that already reveals the bug. Every training view so far has shown the
model a finished assertion and left the value prediction implicit inside it.
This one makes the two steps separate and visible:

    # input: count_divisors(10)
    # expected: 4
    assert count_divisors(10) == 4

The completion still parses to exactly one bounded assertion under the frozen
candidate policy - comments vanish in the AST - so this changes what the model
is taught without changing what the evaluator accepts.

WHAT IS DERIVED AND WHAT IS NOT. The input line is the call the verified
completion already makes. The expected line is obtained by EXECUTING the
reference on that call. Neither adds information the assertion did not already
carry, so the completion is not more revealing than the supervision it
replaces - it is the same fact, stated in the order the model has to think in.

NO GENERATED PROSE. The review that prompted this asked for an "expected
behaviour and rationale" field. Three of its five fields are derivable by
execution and are implemented. A natural-language rationale is not: producing
one would mean an LLM writing unverified text as a training label, which is
precisely the confident-guessing failure this view exists to correct. The
verifier outcome is carried as metadata rather than as completion text,
because it is a training-time fact and not something the model should emit.

BALANCE. Lineages whose base-model failure was an oracle error are oversampled,
because they are the failure mode being targeted - but capped, and mixed with
clean lineages, so the view does not become a monoculture of hard cases.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import count_assertions, validate_generated_test
from harness.corpus import write_json
from harness.oracle_diagnosis import (
    BENIGN_INPUT, ORACLE_ERROR, call_expression, diagnose,
)
from harness.parallel_execution import progress_map
from harness.safe_execution import classify_assertions, execute_code
from metrics.research_evaluation import classify_candidate_failure

#: The label set the review asked for. Every emitted example carries exactly
#: one, derived from the base model's failure on that lineage.
LABELS = (
    "wrong_oracle",       # right input, wrong value - the target of this view
    "wrong_input",        # reference and mutant agree there; no oracle helps
    "syntax",
    "api_misuse",
    "reference_failure",
    "non_kill",
    "clean",              # the base model already killed this lineage
)

_COARSE_TO_LABEL = {
    "syntax_invalid": "syntax",
    "not_generated": "syntax",
    "wrong_target_api": "api_misuse",
    "repository_context_hallucination": "api_misuse",
    "undefined_symbol": "api_misuse",
    "reference_invalid": "reference_failure",
    "environment_failure": "reference_failure",
    "timeout": "reference_failure",
    "fixture_missing": "reference_failure",
    "passes_both": "non_kill",
    "boundary_miss": "non_kill",
    "off_by_one_miss": "non_kill",
    "logical_condition_miss": "non_kill",
    "indexing_miss": "non_kill",
    "incorrect_exception_expectation": "wrong_oracle",
}

#: How many times a targeted lineage may appear. Oversampling the failure mode
#: is the point; letting one lineage dominate is not.
MAX_REPEATS = {"wrong_oracle": 3, "wrong_input": 2}
DEFAULT_REPEATS = 1
#: The view must not become a monoculture of hard cases.
MIN_CLEAN_SHARE = 0.30


def _expected_repr(reference: str, call: str, timeout: float) -> str | None:
    """What the reference actually produces. Executed, never guessed."""
    ok, result, _ = execute_code(reference, "result = repr(" + call + ")",
                                 timeout)
    if not ok:
        return None
    text = str(result if result is not None else "").strip()
    # A value that does not fit on one comment line is not worth stating; the
    # assertion carries it anyway.
    return text if text and len(text) <= 120 and "\n" not in text else None


def structure(completion: str, entry_point: str, reference: str,
              timeout: float) -> dict[str, Any] | None:
    """Turn a verified assertion into an input / expected / assertion triple."""
    assertion = completion.strip()
    if count_assertions(assertion) != 1:
        return None
    call = call_expression(assertion, entry_point)
    if call is None:
        return None
    expected = _expected_repr(reference, call, timeout)
    if expected is None:
        return None
    structured = (
        "# input: " + call + "\n"
        "# expected: " + expected + "\n"
        + assertion
    )
    return {"structured": structured, "call": call, "expected": expected}


def _verify(job: dict[str, Any]) -> dict[str, Any]:
    """The structured completion must still be valid AND still kill."""
    built = structure(job["completion"], job["entry_point"], job["reference"],
                      job["timeout"])
    if built is None:
        return {**job, "usable": False, "reason": "not_structurable"}

    policy = validate_generated_test(built["structured"], job["entry_point"], True)
    if not policy.valid:
        return {**job, "usable": False, "reason": "policy:" + str(policy.reason)}

    rows = classify_assertions([built["structured"]], job["reference"],
                               job["mutant"], job["timeout"])
    row = rows[0] if rows else {}
    if not row.get("valid"):
        return {**job, "usable": False, "reason": "not_reference_valid"}
    if not row.get("killed"):
        return {**job, "usable": False, "reason": "does_not_kill_displayed_target"}
    return {**job, "usable": True, "reason": "", **built}


def _labels_from_artifact(artifact: Path, corpus_dir: Path,
                          timeout: float) -> dict[str, str]:
    """Which failure each train lineage's base-model attempt represents."""
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("evaluation_split") != "train":
        raise SystemExit(
            f"{artifact} is split {payload.get('evaluation_split')!r}; only the "
            "train split may be mined for supervision")

    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    by_id = {str(record["id"]): record for record in records}

    jobs: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for result in payload.get("function_results") or []:
        record_id = str(result.get("record_id"))
        if result.get("killed"):
            labels[record_id] = "clean"
            continue
        record = by_id.get(record_id)
        if record is None:
            continue
        support = str(record.get("support_context") or "")
        family = str(result.get("bug_family") or "unknown")
        counts: Counter = Counter()
        pending: list[dict[str, Any]] = []
        for outcome in result.get("candidate_outcomes") or []:
            coarse = classify_candidate_failure(outcome, family)
            if coarse.startswith("killed"):
                continue
            if coarse == "wrong_expected_value":
                pending.append(outcome)
            else:
                counts[_COARSE_TO_LABEL.get(coarse, "non_kill")] += 1
        if pending:
            jobs.append({
                "record_id": record_id,
                "entry_point": str(record.get("entry_point") or ""),
                "reference": support + "\n" + str(record.get("reference_code") or ""),
                "mutant": support + "\n" + str(record.get("code_under_test") or ""),
                "outcomes": pending,
                "counts": counts,
                "timeout": timeout,
            })
        else:
            labels[record_id] = (counts.most_common(1)[0][0] if counts
                                 else "non_kill")

    def refine_one(job: dict[str, Any]) -> dict[str, Any]:
        counts = Counter(job["counts"])
        for outcome in job["outcomes"]:
            result = diagnose(str(outcome.get("code") or ""), job["entry_point"],
                              job["reference"], job["mutant"], job["timeout"])
            if result["refined_label"] == ORACLE_ERROR:
                counts["wrong_oracle"] += 1
            elif result["refined_label"] == BENIGN_INPUT:
                counts["wrong_input"] += 1
            else:
                counts["non_kill"] += 1
        return {"record_id": job["record_id"],
                "label": counts.most_common(1)[0][0] if counts else "non_kill"}

    refined = progress_map(refine_one, jobs, "labelling train failures", every=100)
    for row in refined:
        labels[row["record_id"]] = row["label"]
    return labels


def build(artifact: Path, source_view: Path, corpus_dir: Path,
          timeout: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = _labels_from_artifact(artifact, corpus_dir, timeout)

    payload = json.loads((source_view / "train.examples.json").read_text(encoding="utf-8"))
    examples = payload["examples"] if isinstance(payload, dict) and "examples" in payload else payload
    records = {
        str(r["id"]): r for r in
        json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    }

    jobs: list[dict[str, Any]] = []
    unlabelled = 0
    for example in examples:
        displayed = str(example.get("displayed_record_id"))
        record = records.get(displayed)
        if record is None:
            continue
        label = labels.get(displayed)
        if label is None:
            # No base-model evidence for this lineage. Kept, labelled honestly,
            # and never counted as a targeted correction.
            unlabelled += 1
            label = "unlabelled"
        support = str(record.get("support_context") or "")
        jobs.append({
            "lineage": str(example.get("lineage")),
            "displayed_record_id": displayed,
            "entry_point": str(example.get("entry_point") or ""),
            "source_dataset": str(example.get("source_dataset") or "unknown"),
            "completion": str(example.get("completion") or ""),
            "reference": support + "\n" + str(record.get("reference_code") or ""),
            "mutant": support + "\n" + str(record.get("code_under_test") or ""),
            "label": label,
            "timeout": timeout,
        })

    verified = progress_map(_verify, jobs, "verifying structured completions",
                            every=200)
    usable = [row for row in verified if row["usable"]]
    rejected = Counter(row["reason"].split(":")[0]
                       for row in verified if not row["usable"])

    emitted: list[dict[str, Any]] = []
    per_lineage: Counter = Counter()
    for row in sorted(usable, key=lambda r: (r["label"] != "wrong_oracle",
                                             r["lineage"])):
        repeats = MAX_REPEATS.get(row["label"], DEFAULT_REPEATS)
        for _ in range(repeats):
            per_lineage[row["lineage"]] += 1
            emitted.append({
                "lineage": row["lineage"],
                "displayed_record_id": row["displayed_record_id"],
                "entry_point": row["entry_point"],
                "source_dataset": row["source_dataset"],
                "completion": row["structured"],
                "completion_shape": "assertion",
                "assertion_count": 1,
                "oracle_label": row["label"],
                "declared_input": row["call"],
                "declared_expected": row["expected"],
                "verifier_outcome": "reference_valid_and_kills_displayed_target",
                "verified": True,
            })

    label_counts = Counter(row["oracle_label"] for row in emitted)
    total = len(emitted) or 1
    clean_share = (label_counts.get("clean", 0)
                   + label_counts.get("unlabelled", 0)) / total

    report = {
        "schema_version": "oneiros_oracle_supervision_view_v1",
        "sealed_final_test_accessed": False,
        "source_artifact": artifact.as_posix().split("results/", 1)[-1],
        "source_view": source_view.name,
        "completion_format": "# input / # expected / assert - one bounded assertion",
        "fields_implemented": [
            "target and allowed context (unchanged prompt)",
            "candidate input (the call the verified completion makes)",
            "expected behaviour (obtained by EXECUTING the reference)",
            "executable assertion",
            "verifier outcome (metadata)",
        ],
        "field_not_implemented": (
            "natural-language rationale - it would require an LLM writing "
            "unverified prose as a training label, which is the confident-"
            "guessing failure this view exists to correct"
        ),
        "lineages_considered": len(jobs),
        "lineages_usable": len(usable),
        "lineages_without_base_model_evidence": unlabelled,
        "rejected_reasons": dict(rejected.most_common()),
        "examples_emitted": total if emitted else 0,
        "label_counts": dict(label_counts.most_common()),
        "oversampling": dict(MAX_REPEATS),
        "clean_share": round(clean_share, 4),
        "meets_min_clean_share": clean_share >= MIN_CLEAN_SHARE,
        "min_clean_share": MIN_CLEAN_SHARE,
        "max_examples_from_one_lineage": max(per_lineage.values()) if per_lineage else 0,
        "source_shares": {
            source: round(count / total, 4) for source, count in
            Counter(row["source_dataset"] for row in emitted).most_common()
        },
    }
    return emitted, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path,
                        help="a TRAIN-split evaluation artifact to mine labels from")
    parser.add_argument("--source-view", type=Path,
                        default=ROOT / "data" / "training_views" / "multi_mutant_v1")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    examples, report = build(arguments.artifact, arguments.source_view,
                             arguments.corpus, arguments.timeout)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output_dir / "train.examples.json", examples)
    write_json(arguments.output_dir / "train.manifest.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
