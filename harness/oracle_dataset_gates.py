"""Everything the oracle dataset build must prove before it reads a candidate.

Three independent chains have to hold, and each was a real gap:

* THE RECEIPT. A verification receipt is only evidence if it describes the
  bytes actually being read. The first version recorded the artifact's PATH,
  so a consumer handed that receipt and a different file at the same path
  could not tell. Both ends of the chain are now hashed and checked here.

* THE SOURCE. The evaluator, candidate policy, execution harness and prompt
  builder are all recorded in the generation's run contract. If any has
  changed since, the labels this build produces were computed by different
  code than the outcomes it reads, and the two cannot be reconciled.

* THE OUTCOMES. Recorded fields are used, never inferred. Deriving
  ``policy_valid`` from whether a shape string is present looked equivalent
  and was not: the policy returns an EMPTY shape on rejection, so
  ``shape is not None`` is true for rejected candidates and every one of them
  would have been labelled valid.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from harness.corpus import sha256_file

EXPECTED_RECEIPT_SCHEMA = "oneiros_successor_generation_verification_v1"

CONTRACT_SOURCES = {
    "evaluator_source_sha256": "metrics/research_evaluation.py",
    "candidate_policy_source_sha256": "harness/candidate_policy.py",
    "safe_execution_source_sha256": "harness/safe_execution.py",
    "prompt_builder_source_sha256": "engine/test_generation_prompt.py",
}

#: Fields copied verbatim from each recorded outcome. Nothing in this list is
#: ever reconstructed from another field.
OUTCOME_FIELDS = (
    "parse_valid", "policy_valid", "policy_error", "execution_valid",
    "reference_valid", "reference_status", "mutant_status", "killed",
    "failure_mode", "raw_output_sha256",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_receipt(receipt: Mapping[str, Any], derived: Path,
                   original: Path) -> list[str]:
    """Every condition under which a receipt does not license a build."""
    problems: list[str] = []

    schema = str(receipt.get("schema_version") or "")
    if schema != EXPECTED_RECEIPT_SCHEMA:
        problems.append(
            f"receipt schema is {schema!r}, expected {EXPECTED_RECEIPT_SCHEMA!r}")
    if receipt.get("verified") is not True:
        problems.append("receipt is not verified")
    if receipt.get("evaluation_split") != "train":
        problems.append(
            f"receipt split is {receipt.get('evaluation_split')!r}, not train")

    claimed = receipt.get("artifact_sha256")
    if not claimed:
        problems.append(
            "receipt records no artifact_sha256, so it cannot be bound to the "
            "bytes being read")
    elif not derived.exists():
        problems.append(f"derived artifact missing: {derived}")
    elif sha256_file(derived) != claimed:
        problems.append(
            f"receipt describes {claimed} but the derived artifact hashes to "
            f"{sha256_file(derived)}")

    for field in ("execution_harness_failures", "raw_output_hash_mismatches",
                  "prompt_budget_failed_functions"):
        value = receipt.get(field)
        if value is None:
            problems.append(f"receipt does not record {field}")
        elif int(value) != 0:
            problems.append(f"receipt records {field} = {value}, must be 0")

    for field in ("completion_limit_threshold_exceeded",
                  "ineligible_ceiling_exceeded",
                  "execution_harness_ceiling_exceeded"):
        if receipt.get(field) is True:
            problems.append(f"receipt records {field} = true")

    parent_claim = receipt.get("parent_artifact_sha256")
    if not parent_claim:
        problems.append("receipt records no parent_artifact_sha256")
    elif original.exists() and sha256_file(original) != parent_claim:
        problems.append(
            f"receipt's parent chain claims {parent_claim} but the supplied "
            f"original hashes to {sha256_file(original)}")
    return problems


def assert_no_source_drift(contract: Mapping[str, Any], root: Path) -> list[str]:
    """The labelling code must be the code that produced the outcomes."""
    problems: list[str] = []
    for field, relative in CONTRACT_SOURCES.items():
        expected = contract.get(field)
        if not expected:
            problems.append(f"run contract does not record {field}")
            continue
        actual = sha256_file(root / relative)
        if actual != expected:
            problems.append(
                f"{relative} has changed since generation "
                f"({expected[:12]}... -> {actual[:12]}...); labels would be "
                "computed by different code than the outcomes they describe")
    return problems


def outcome_fields(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """The recorded fields, verbatim. Absent means absent, not False.

    A missing field is carried through as None so that "the artifact never
    said" stays distinguishable from "the artifact said no".
    """
    return {field: outcome.get(field) for field in OUTCOME_FIELDS}


def verify_raw_output(outcome: Mapping[str, Any]) -> str | None:
    """None if the raw output matches its recorded hash, else the reason."""
    raw = outcome.get("raw_output")
    recorded = outcome.get("raw_output_sha256")
    if raw is None:
        return "candidate carries no raw output"
    if not recorded:
        return "candidate carries no recorded raw_output_sha256"
    if sha256_text(str(raw)) != recorded:
        return "raw output does not match its recorded sha256"
    return None
