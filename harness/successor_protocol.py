"""One named successor-protocol configuration, consumed by every entry point.

The legacy/successor split has already produced two protocol-noncomparable
numbers in this project - first_assertion Kill@8 0.629133 and whole_output
Kill@8 0.282216 - and the difference between them was a flag, not a model.
Absence of ``candidate_parse_mode`` means legacy, so a command that simply
forgets the flag silently runs the old protocol and reports a number nobody
can compare with anything.

This module makes that impossible to do by omission. The settings live in one
frozen dict with a hash over its own contents; the preflight records it, the
training command derives its flags from it, and Arm B refuses any Arm A whose
recorded contract does not match it field for field.

TWO COMPLETION BUDGETS, DELIBERATELY SEPARATE. ``generation_*`` budgets bound
what the model may EMIT at evaluation time. The SFT training completion budget
bounds how long a supervised target may be. They are different quantities that
happen to share a unit, and collapsing them is how a run ends up training on
128-token targets while claiming a 1024-token protocol. Nothing in this module
touches the SFT budget.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

NAME = "oneiros_successor_generation_protocol_v1"

#: The frozen successor protocol. Every field here describes GENERATION and
#: EVALUATION. Training hyperparameters are not part of it.
SUCCESSOR_PROTOCOL: dict[str, Any] = {
    "protocol_name": NAME,
    "candidate_parse_mode": "whole_output",
    "retain_raw_output": True,
    "allow_test_function_candidates": True,
    "max_assertions": 24,
    "candidates_per_function": 8,
    "temperature": 0.7,
    "top_p": 0.9,
    "generation_seed": 42,
    "function_generation_completion_limit": 1024,
    "repository_generation_completion_limit": 1024,
}

#: Fields Arm B compares between the arms. A difference in any of them makes
#: the comparison uninterpretable, so it is a refusal rather than a warning.
CONTRACT_FIELDS = tuple(
    field for field in SUCCESSOR_PROTOCOL if field != "protocol_name"
)

#: Source files whose bytes define how a candidate is parsed, judged, and
#: timed out. A receipt naming "whole_output" cannot tell whether the parser
#: changed between two arms; the hash of the file can.
CONTRACT_SOURCES = {
    "parser_source_sha256": "scripts/analyze_parser_pilot.py",
    "candidate_policy_source_sha256": "harness/candidate_policy.py",
    "evaluator_source_sha256": "metrics/research_evaluation.py",
    "timeout_policy_source_sha256": "harness/safe_execution.py",
    "evaluation_protocol_source_sha256": "harness/evaluation_protocol.py",
}


def protocol_sha256() -> str:
    """A hash over the settings themselves, so a silent edit is visible."""
    return hashlib.sha256(
        json.dumps(SUCCESSOR_PROTOCOL, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def contract_source_hashes(root) -> dict[str, str]:
    from pathlib import Path
    root = Path(root)
    return {
        field: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for field, relative in CONTRACT_SOURCES.items()
    }


def as_recorded_contract(root) -> dict[str, Any]:
    """The block a preflight or run manifest records verbatim."""
    return {
        **SUCCESSOR_PROTOCOL,
        "protocol_sha256": protocol_sha256(),
        **contract_source_hashes(root),
    }


def training_command_flags() -> list[str]:
    """The exact flags a training command must carry to run this protocol.

    Derived from the settings rather than typed beside them, because a flag
    typed twice is a flag that can disagree with itself - which is how a
    preflight string once gained --max-sequence-tokens that the launch command
    never had.
    """
    flags = [
        "--candidate-parse-mode", SUCCESSOR_PROTOCOL["candidate_parse_mode"],
    ]
    if SUCCESSOR_PROTOCOL["retain_raw_output"]:
        flags.append("--retain-raw-output")
    if SUCCESSOR_PROTOCOL["allow_test_function_candidates"]:
        flags.append("--allow-test-function-candidates")
    flags += ["--seed", str(SUCCESSOR_PROTOCOL["generation_seed"])]
    return flags


def assert_matches(recorded: dict[str, Any] | None, root
                   ) -> list[str]:
    """Every way a recorded contract can fail to be this protocol."""
    problems: list[str] = []
    if not recorded:
        return ["no successor-protocol contract was recorded at all; absence "
                "of candidate_parse_mode means LEGACY first_assertion"]

    for field in CONTRACT_FIELDS:
        expected = SUCCESSOR_PROTOCOL[field]
        actual = recorded.get(field)
        if actual != expected:
            problems.append(
                f"{field} is {actual!r}, the successor protocol requires "
                f"{expected!r}")

    recorded_hash = recorded.get("protocol_sha256")
    if recorded_hash and recorded_hash != protocol_sha256():
        problems.append(
            f"protocol_sha256 is {str(recorded_hash)[:12]}... but this "
            f"protocol hashes to {protocol_sha256()[:12]}...; the settings "
            "changed since that receipt was written")

    for field, actual in contract_source_hashes(root).items():
        claimed = recorded.get(field)
        if not claimed:
            problems.append(f"contract records no {field}")
        elif claimed != actual:
            problems.append(
                f"{CONTRACT_SOURCES[field]} has changed since that receipt "
                f"({str(claimed)[:12]}... -> {actual[:12]}...); the two arms "
                "would not be parsed, judged or timed out by the same code")
    return problems
