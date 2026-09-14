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
    # Raised from the 2048 project default. The launch guard in
    # train_on_dataset.py refuses prompt + completion >= this value, so a
    # 1024 prompt budget beside a 1024 completion budget is rejected at 2048
    # before chat-template overhead is even counted. The completion budget is
    # never the thing that gets reduced.
    "max_sequence_tokens": 3072,
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
    # The PRODUCTION parser. Phi3Generator._parse_output and
    # _parse_whole_output are what actually turn a model output into a
    # candidate at generation time. scripts/analyze_parser_pilot.py is an
    # analysis script that happens to reimplement the same split; hashing it
    # would pin the identity of a tool nothing generates through.
    "parser_source_sha256": "engine/generator.py",
    "generation_runtime_source_sha256": "engine/model_runtime.py",
    "generation_orchestration_source_sha256": "scripts/train_on_dataset.py",
    "prompt_builder_source_sha256": "engine/test_generation_prompt.py",
    "prompt_budget_source_sha256": "engine/prompt_budget.py",
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
    flags += [
        "--seed", str(SUCCESSOR_PROTOCOL["generation_seed"]),
        "--generation-completion-token-limit",
        str(SUCCESSOR_PROTOCOL["function_generation_completion_limit"]),
        "--max-sequence-tokens", str(SUCCESSOR_PROTOCOL["max_sequence_tokens"]),
    ]
    return flags


#: CLI options the protocol owns. Passing one of these beside
#: --successor-protocol is a conflict, not an override: the protocol exists so
#: that a single name fixes every generation setting, and a value typed beside
#: it can silently disagree with the receipts that name the protocol.
OWNED_CLI_OPTIONS = {
    "candidate_parse_mode": "--candidate-parse-mode",
    "retain_raw_output": "--retain-raw-output",
    "allow_test_function_candidates": "--allow-test-function-candidates",
    "seed": "--seed",
    "generation_completion_token_limit": "--generation-completion-token-limit",
    "max_sequence_tokens": "--max-sequence-tokens",
}


def assert_runtime_matches(root) -> list[str]:
    """The protocol must agree with the constants the code actually uses.

    A named protocol that only describes settings is a comment. These are the
    values the generator and policy enforce at runtime, checked against the
    protocol so the two cannot drift apart silently.
    """
    import sys
    from pathlib import Path
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    problems: list[str] = []
    from config import model_config
    from harness.candidate_policy import MAX_TEST_FUNCTION_ASSERTS

    if MAX_TEST_FUNCTION_ASSERTS != SUCCESSOR_PROTOCOL["max_assertions"]:
        problems.append(
            f"candidate policy allows {MAX_TEST_FUNCTION_ASSERTS} assertions, "
            f"the protocol declares {SUCCESSOR_PROTOCOL['max_assertions']}")
    if model_config.temperature != SUCCESSOR_PROTOCOL["temperature"]:
        problems.append(
            f"model_config.temperature is {model_config.temperature}, the "
            f"protocol declares {SUCCESSOR_PROTOCOL['temperature']}")
    if model_config.top_p != SUCCESSOR_PROTOCOL["top_p"]:
        problems.append(
            f"model_config.top_p is {model_config.top_p}, the protocol "
            f"declares {SUCCESSOR_PROTOCOL['top_p']}")

    from scripts import train_on_dataset as trainer
    if trainer.TESTS_PER_PAIR != SUCCESSOR_PROTOCOL["candidates_per_function"]:
        problems.append(
            f"TESTS_PER_PAIR is {trainer.TESTS_PER_PAIR}, the protocol "
            f"declares {SUCCESSOR_PROTOCOL['candidates_per_function']}")

    declared = (SUCCESSOR_PROTOCOL["function_generation_completion_limit"]
                + 1024)
    if declared >= SUCCESSOR_PROTOCOL["max_sequence_tokens"]:
        problems.append(
            f"a 1024 prompt budget plus the "
            f"{SUCCESSOR_PROTOCOL['function_generation_completion_limit']} "
            f"completion budget is {declared}, which the launch guard refuses "
            f"against max_sequence_tokens "
            f"{SUCCESSOR_PROTOCOL['max_sequence_tokens']}")
    return problems


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
