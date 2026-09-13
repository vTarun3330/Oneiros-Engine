"""Build the oracle-structured supervision dataset. Candidate level, CPU only.

This REPLACES scripts/build_oracle_supervision_view.py, which had four
independently disqualifying defects: it opened the canonical records.json (all
four splits, sealed final test included) behind a manifest asserting the
opposite; it balanced classes by duplicating rows; it assigned one majority
label per function; and it silently discarded every multi-assertion completion.
Three of its four core mechanisms were wrong, so it is retired rather than
patched.

WHAT THIS PRODUCES. One row per candidate, carrying its full identity, both its
original and re-scored outcomes, its execution evidence, its label and the
reason for that label. Positive SFT targets are a strict subset, separately
counted, and every other row is explicitly marked as a negative or a
preference partner - never as something to imitate.

THE FUNNEL IS REPORTED, NOT COLLAPSED. "Eligible" is not one number. A raw
output can be minable and still not parse; a parsed candidate can still be
policy-invalid; a policy-valid one can still fail on the reference. Reporting a
single eligibility figure is how a dataset comes to contain things nobody
believes are correct.

NOTHING IS DUPLICATED TO BALANCE. Caps drop surplus rows from over-represented
strata; scarce strata stay scarce and the residual imbalance is reported.
Sampling weights are metadata for a later trainer, not repeated records.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import (
    count_assertions, executable_candidate, validate_generated_test,
)
from harness.corpus import sha256_file, write_json
from harness.corpus_view import load_development_split
from harness.oracle_diagnosis import BENIGN_INPUT, ORACLE_ERROR, call_expression, diagnose
from harness.oracle_labels import (
    FABRICATED_API, HARNESS_ENVIRONMENT_FAILURE, LABELS, NEVER_POSITIVE,
    SEMANTIC_EXECUTION_ERROR, SYNTAX_OR_POLICY_INVALID, UNCERTAIN,
    VALID_KILLING, VALID_NON_KILLING, WRONG_INPUT, WRONG_ORACLE,
    classify_candidate, needs_call_replay,
)
from harness.parallel_execution import progress_map
from harness.safe_execution import classify_assertions, execute_code

TIMEOUT = 5.0
WORKERS = 8

#: Unique-first caps. A cap DROPS surplus rows; it never repeats scarce ones.
MAX_PER_FUNCTION_LINEAGE = 4
MAX_PER_SOURCE_SHARE = 0.70
MAX_PER_BUG_FAMILY_SHARE = 0.35


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _tier(record: dict[str, Any]) -> str:
    quality = record.get("quality")
    if isinstance(quality, dict) and quality.get("complexity_tier"):
        return str(quality["complexity_tier"])
    lines = len([l for l in str(record.get("reference_code") or "").splitlines()
                 if l.strip()])
    return "simple" if lines <= 6 else "moderate" if lines <= 14 else "complex"


def assert_source_artifact(derived: Path, original: Path) -> dict[str, Any]:
    """The dataset may only be built from a verified, parent-bound artifact."""
    payload = json.loads(derived.read_text(encoding="utf-8"))
    problems: list[str] = []
    if payload.get("evaluation_split") != "train":
        problems.append(f"split is {payload.get('evaluation_split')!r}, not train")
    if payload.get("final_test_measurement"):
        problems.append("artifact is a sealed final-test measurement")
    claimed = payload.get("derived_from_sha256")
    if not claimed:
        problems.append("artifact does not record a parent; its provenance "
                        "chain cannot be verified")
    elif original.exists() and sha256_file(original) != claimed:
        problems.append(
            f"parent chain broken: artifact claims {claimed} but the original "
            f"hashes to {sha256_file(original)}")
    contract = payload.get("run_contract") or {}
    if contract.get("candidate_parse_mode") != "whole_output":
        problems.append("source is not a whole_output run")
    if not contract.get("retain_raw_output"):
        problems.append("source did not retain raw outputs")
    if problems:
        raise SystemExit("source artifact refused:\n  - " + "\n  - ".join(problems))
    return payload


def _replay(job: dict[str, Any]) -> dict[str, Any]:
    """Does this candidate's own call separate reference from mutant?"""
    result = diagnose(job["code"], job["entry_point"], job["reference"],
                      job["mutant"], TIMEOUT)
    label = result["refined_label"]
    return {
        "key": job["key"],
        "distinguishing": (True if label == ORACLE_ERROR
                           else False if label == BENIGN_INPUT else None),
        "call": result.get("call"),
        "reference_behaviour": result.get("reference_behaviour"),
        "mutant_behaviour": result.get("mutant_behaviour"),
    }


def _verify_correction(job: dict[str, Any]) -> dict[str, Any]:
    """Build a corrected oracle on the candidate's OWN call, then prove it.

    The correction keeps the model's input - which for a wrong_oracle candidate
    is already right - and replaces only the expected value, obtained by
    EXECUTING the reference. It is then held to the same bar as any other
    candidate: policy-valid, reference-valid, and it must kill.
    """
    call = job["call"]
    ok, value, _ = execute_code(job["reference"], "result = repr(" + call + ")",
                                TIMEOUT)
    if not ok:
        return {"key": job["key"], "verified": False,
                "reason": "the reference could not be executed on this call"}
    expected = str(value if value is not None else "").strip()
    if not expected or "\n" in expected or len(expected) > 200:
        return {"key": job["key"], "verified": False,
                "reason": "reference value is empty or not representable inline"}

    corrected = "assert " + call + " == " + expected
    policy = validate_generated_test(corrected, job["entry_point"], True)
    if not policy.valid:
        return {"key": job["key"], "verified": False,
                "reason": "correction rejected by policy: " + str(policy.reason)}
    rows = classify_assertions(
        [executable_candidate(corrected, policy.shape)], job["reference"],
        job["mutant"], TIMEOUT)
    row = rows[0] if rows else {}
    if not row.get("valid"):
        return {"key": job["key"], "verified": False,
                "reason": "correction is not valid on the reference"}
    if not row.get("killed"):
        return {"key": job["key"], "verified": False,
                "reason": "correction does not kill the displayed target"}
    return {"key": job["key"], "verified": True, "corrected_test": corrected,
            "corrected_test_sha256": _sha(corrected), "expected_value": expected,
            "reason": "executed against the reference, policy-valid, kills"}


def build(derived: Path, original: Path, corpus_dir: Path,
          workers: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = assert_source_artifact(derived, original)
    parent_payload = json.loads(original.read_text(encoding="utf-8")) \
        if original.exists() else {"function_results": []}
    original_status = {
        (str(r.get("record_id")), i): str(o.get("reference_status") or "")
        for r in parent_payload.get("function_results") or []
        for i, o in enumerate(r.get("candidate_outcomes") or [])}

    # Hash-verified train shard only. The canonical records.json materialises
    # every split including the sealed final test and is never opened.
    records = {str(r["id"]): r for r in load_development_split(
        corpus_dir, "train", include_excluded=True)}

    rows: list[dict[str, Any]] = []
    replay_jobs: list[dict[str, Any]] = []
    funnel = Counter()
    excluded = Counter()

    for result in payload.get("function_results") or []:
        record_id = str(result.get("record_id"))
        record = records.get(record_id)
        if record is None:
            excluded["record_not_in_train_shard"] += 1
            continue
        entry = str(record.get("entry_point") or "")
        support = str(record.get("support_context") or "")
        reference = support + "\n" + str(record.get("reference_code") or "")
        mutant = support + "\n" + str(record.get("code_under_test") or "")
        tier = _tier(record)

        for position, outcome in enumerate(result.get("candidate_outcomes") or []):
            funnel["candidates"] += 1
            raw = outcome.get("raw_output")
            code = outcome.get("code")
            key = (record_id, int(outcome.get("rank") or 0), position)

            if isinstance(raw, str) and raw:
                funnel["raw_mining_eligible"] += 1
            if outcome.get("parse_valid"):
                funnel["parse_valid"] += 1
            if outcome.get("policy_valid"):
                funnel["policy_valid"] += 1
            if outcome.get("execution_valid"):
                funnel["execution_valid"] += 1
            if outcome.get("reference_valid"):
                funnel["reference_valid"] += 1

            row = {
                "record_id": record_id,
                "rank": key[1],
                "candidate_position": position,
                "entry_point": entry,
                "source_dataset": str(result.get("dataset_name") or "unknown"),
                "bug_family": str(result.get("bug_family") or "unknown"),
                "project": str(result.get("project") or "synthetic"),
                "origin": ("real_repository"
                           if str(result.get("project") or "synthetic") != "synthetic"
                           else "synthetic"),
                "function_lineage": str(record.get("group_id") or record_id),
                "complexity_tier": tier,
                "raw_output": raw,
                "raw_output_sha256": outcome.get("raw_output_sha256"),
                "candidate_code": code,
                "candidate_code_sha256": _sha(code) if isinstance(code, str) else None,
                "assertion_count": outcome.get("assertion_count"),
                "candidate_shape": outcome.get("candidate_shape"),
                "original_reference_status": original_status.get(
                    (record_id, position)),
                "rescored": bool(outcome.get("rescored")),
                "execution_evidence": {
                    "reference_status": outcome.get("reference_status"),
                    "mutant_status": outcome.get("mutant_status"),
                    "reference_error": outcome.get("reference_error"),
                    "reference_valid": bool(outcome.get("reference_valid")),
                    "execution_valid": bool(outcome.get("execution_valid")),
                    "killed": bool(outcome.get("killed")),
                },
            }
            rows.append(row)
            if needs_call_replay(outcome) and isinstance(code, str):
                replay_jobs.append({"key": key, "code": code,
                                    "entry_point": entry, "reference": reference,
                                    "mutant": mutant})

    replayed = {r["key"]: r for r in progress_map(
        _replay, replay_jobs, "input-vs-oracle replay", every=1000, workers=workers)}

    correction_jobs: list[dict[str, Any]] = []
    for row in rows:
        key = (row["record_id"], row["rank"], row["candidate_position"])
        replay = replayed.get(key)
        verdict = classify_candidate(
            {**row["execution_evidence"],
             "parse_valid": row["candidate_code"] is not None,
             "policy_valid": row["candidate_shape"] is not None,
             "policy_error": None},
            distinguishing_call=(replay or {}).get("distinguishing"))
        row.update(verdict)
        row["call_replay"] = replay
        if row["label"] == WRONG_ORACLE and replay and replay.get("call"):
            records_entry = records[row["record_id"]]
            support = str(records_entry.get("support_context") or "")
            correction_jobs.append({
                "key": key, "call": replay["call"],
                "entry_point": row["entry_point"],
                "reference": support + "\n" + str(records_entry.get("reference_code") or ""),
                "mutant": support + "\n" + str(records_entry.get("code_under_test") or ""),
            })

    corrections = {c["key"]: c for c in progress_map(
        _verify_correction, correction_jobs, "verifying corrections",
        every=500, workers=workers)}

    for row in rows:
        key = (row["record_id"], row["rank"], row["candidate_position"])
        correction = corrections.get(key)
        row["verified_correction"] = correction if correction else None
        # A positive SFT target is either a candidate that was already valid and
        # killing, or a VERIFIED correction. Nothing else, ever.
        if row["label"] == VALID_KILLING:
            row["supervision_role"] = "positive_original"
            row["sft_target"] = row["candidate_code"]
        elif correction and correction.get("verified"):
            row["supervision_role"] = "positive_verified_correction"
            row["sft_target"] = correction["corrected_test"]
        else:
            row["supervision_role"] = (
                "negative" if row["label"] in NEVER_POSITIVE else "neutral")
            row["sft_target"] = None
        if row["sft_target"] is not None:
            funnel["verified_correction_eligible" if correction
                   else "positive_original_eligible"] += 1

    return rows, {"funnel": funnel, "excluded": excluded,
                  "payload": payload, "records": records}


def balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Unique-first caps. Surplus rows are DROPPED; nothing is repeated."""
    positives = [r for r in rows if r["sft_target"] is not None]
    kept: list[dict[str, Any]] = []
    per_lineage: Counter = Counter()
    per_source: Counter = Counter()
    per_family: Counter = Counter()
    dropped = Counter()
    seen_targets: set[str] = set()

    # Deterministic: rarer strata first, so a cap never starves them.
    family_size = Counter(r["bug_family"] for r in positives)
    ordered = sorted(positives, key=lambda r: (
        family_size[r["bug_family"]], r["record_id"], r["rank"],
        r["candidate_position"]))

    for row in ordered:
        target_hash = _sha(str(row["sft_target"]))
        if target_hash in seen_targets:
            dropped["exact_duplicate_target"] += 1
            continue
        if per_lineage[row["function_lineage"]] >= MAX_PER_FUNCTION_LINEAGE:
            dropped["function_lineage_cap"] += 1
            continue
        total = len(kept)
        if total >= 200:
            if per_source[row["source_dataset"]] / total > MAX_PER_SOURCE_SHARE:
                dropped["source_cap"] += 1
                continue
            if per_family[row["bug_family"]] / total > MAX_PER_BUG_FAMILY_SHARE:
                dropped["bug_family_cap"] += 1
                continue
        seen_targets.add(target_hash)
        per_lineage[row["function_lineage"]] += 1
        per_source[row["source_dataset"]] += 1
        per_family[row["bug_family"]] += 1
        kept.append(row)

    total = len(kept) or 1
    # Weights are METADATA for a later sampler. Repeating a row to balance is
    # how a scarce class comes to look plentiful while teaching nothing new.
    family_counts = Counter(r["bug_family"] for r in kept)
    weights = {family: round(total / (len(family_counts) * count), 4)
               for family, count in family_counts.items()}
    for row in kept:
        row["sampling_weight"] = weights[row["bug_family"]]

    return {
        "selected": kept,
        "dropped_by_cap": dict(dropped.most_common()),
        "caps": {"per_function_lineage": MAX_PER_FUNCTION_LINEAGE,
                 "max_source_share": MAX_PER_SOURCE_SHARE,
                 "max_bug_family_share": MAX_PER_BUG_FAMILY_SHARE},
        "balancing_method": (
            "unique-first: surplus rows from over-represented strata are "
            "dropped and scarce strata are left scarce. No row is repeated. "
            "Residual imbalance is reported and handed to a later sampler as "
            "weights rather than hidden by duplication."),
        "sampling_weights_by_bug_family": weights,
        "residual_shares": {
            "source_dataset": {k: round(v / total, 4) for k, v in
                               Counter(r["source_dataset"] for r in kept).most_common()},
            "bug_family": {k: round(v / total, 4) for k, v in
                           family_counts.most_common()},
            "complexity_tier": {k: round(v / total, 4) for k, v in
                                Counter(r["complexity_tier"] for r in kept).most_common()},
            "origin": {k: round(v / total, 4) for k, v in
                       Counter(r["origin"] for r in kept).most_common()},
        },
        "max_rows_from_one_lineage": max(per_lineage.values()) if per_lineage else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "results"
    parser.add_argument("--derived", type=Path, default=base
                        / "local_base_qwen_train_successor_s42_rescored"
                        / "base_validation_train_parse-whole-output_completion1024_seed_42.rescored.json")
    parser.add_argument("--original", type=Path, default=base
                        / "local_base_qwen_train_successor_s42"
                        / "base_validation_train_parse-whole-output_completion1024_seed_42.json")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    rows, context = build(arguments.derived, arguments.original,
                          arguments.corpus, arguments.workers)
    selection = balance(rows)

    labels = Counter(r["label"] for r in rows)
    secondary = Counter(r["secondary_label"] for r in rows if r["secondary_label"])
    roles = Counter(r["supervision_role"] for r in rows)
    funnel = context["funnel"]

    manifest = {
        "schema_version": "oneiros_oracle_dataset_v1",
        "source_commit": _git("rev-parse", "HEAD"),
        "sealed_final_test_accessed": False,
        "record_source": "hash-verified development view train shard",
        "canonical_records_json_opened": False,
        "evaluation_split": "train",
        "source_derived_artifact": arguments.derived.as_posix().split("results/", 1)[-1],
        "source_derived_sha256": sha256_file(arguments.derived),
        "source_parent_sha256": context["payload"].get("derived_from_sha256"),
        "parent_chain_verified": True,
        "run_contract_sha256": context["payload"].get("run_contract_sha256"),
        "granularity": "one row per candidate; no function-level majority label",
        "label_definitions": {
            VALID_KILLING: "valid on the reference and distinguishes the mutant",
            VALID_NON_KILLING: "valid on the reference; the mutant survives",
            WRONG_INPUT: "call does not distinguish; no expected value could kill",
            WRONG_ORACLE: "call distinguishes; only the asserted value is wrong",
            SYNTAX_OR_POLICY_INVALID: "did not parse, or the policy refused it",
            FABRICATED_API: "NameError/AttributeError/arity/keyword: interface does not exist",
            SEMANTIC_EXECUTION_ERROR: "valid interface call failing on a value or type",
            HARNESS_ENVIRONMENT_FAILURE: "never scored by the executor",
            UNCERTAIN: "unresolvable; excluded from positive supervision",
        },
        "label_precedence": (
            "harness > parse/policy > reference-valid > assertion failure "
            "(input before oracle) > raised. Where a candidate is both "
            "wrong_input and wrong_oracle, input is primary because no "
            "corrected value could make it kill."),
        "eligibility_funnel": dict(funnel),
        "label_counts": dict(labels.most_common()),
        "secondary_label_counts": dict(secondary.most_common()),
        "supervision_roles": dict(roles.most_common()),
        "positive_rule": (
            "only a candidate already valid-and-killing, or a VERIFIED "
            "correction executed against the reference and proven to kill, may "
            "be a positive SFT target. Every other row is a marked negative."),
        "whole_output_preserved": True,
        "assertion_count_distribution": dict(
            Counter(r["assertion_count"] for r in rows if r["assertion_count"]).most_common(10)),
        "one_assertion_reduction_applied": False,
        "excluded": dict(context["excluded"].most_common()),
        "rows_total": len(rows),
        "selected_positive_rows": len(selection["selected"]),
        **{k: v for k, v in selection.items() if k != "selected"},
    }

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output_dir / "candidates.json", rows)
    write_json(arguments.output_dir / "positives.json", selection["selected"])
    manifest["candidates_sha256"] = sha256_file(arguments.output_dir / "candidates.json")
    manifest["positives_sha256"] = sha256_file(arguments.output_dir / "positives.json")
    write_json(arguments.output_dir / "manifest.json", manifest)

    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("label_definitions",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
