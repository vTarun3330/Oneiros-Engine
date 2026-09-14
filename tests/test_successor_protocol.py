"""The successor protocol must be impossible to run by accident or omission.

Absence of ``candidate_parse_mode`` means LEGACY. That is why a forgotten flag
is not a neutral event here: it silently selects the old protocol and produces
a number that cannot be compared with any successor measurement. The two such
numbers already in this project - first_assertion Kill@8 0.629133 and
whole_output Kill@8 0.282216 - differ by a flag, not by a model.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.successor_protocol import (
    CONTRACT_FIELDS, CONTRACT_SOURCES, NAME, SUCCESSOR_PROTOCOL,
    as_recorded_contract, assert_matches, contract_source_hashes,
    protocol_sha256, training_command_flags,
)

ROOT = Path(__file__).resolve().parent.parent


def _clean():
    return as_recorded_contract(ROOT)


# ------------------------------------------------- the settings themselves

def test_every_required_setting_has_the_required_value():
    """The values the protocol was specified with, asserted literally."""
    assert SUCCESSOR_PROTOCOL["candidate_parse_mode"] == "whole_output"
    assert SUCCESSOR_PROTOCOL["retain_raw_output"] is True
    assert SUCCESSOR_PROTOCOL["allow_test_function_candidates"] is True
    assert SUCCESSOR_PROTOCOL["max_assertions"] == 24
    assert SUCCESSOR_PROTOCOL["candidates_per_function"] == 8
    assert SUCCESSOR_PROTOCOL["temperature"] == 0.7
    assert SUCCESSOR_PROTOCOL["top_p"] == 0.9
    assert SUCCESSOR_PROTOCOL["generation_seed"] == 42
    assert SUCCESSOR_PROTOCOL["function_generation_completion_limit"] == 1024
    assert SUCCESSOR_PROTOCOL["repository_generation_completion_limit"] == 1024


def test_the_protocol_is_never_legacy():
    assert SUCCESSOR_PROTOCOL["candidate_parse_mode"] != "first_assertion"


def test_the_protocol_carries_no_sft_training_budget():
    """Two completion budgets share a unit and are different quantities."""
    for field in SUCCESSOR_PROTOCOL:
        assert "sft" not in field, (
            f"{field} belongs to training, not to the generation protocol")


def test_the_hash_changes_when_a_setting_changes(monkeypatch):
    before = protocol_sha256()
    monkeypatch.setitem(SUCCESSOR_PROTOCOL, "temperature", 0.8)
    assert protocol_sha256() != before


def test_the_command_flags_are_derived_not_retyped():
    """A flag typed twice is a flag that can disagree with itself."""
    flags = training_command_flags()
    assert "--candidate-parse-mode" in flags
    assert flags[flags.index("--candidate-parse-mode") + 1] == "whole_output"
    assert "--retain-raw-output" in flags
    assert "--allow-test-function-candidates" in flags
    assert flags[flags.index("--seed") + 1] == "42"


def test_the_flags_follow_the_settings(monkeypatch):
    monkeypatch.setitem(SUCCESSOR_PROTOCOL, "retain_raw_output", False)
    assert "--retain-raw-output" not in training_command_flags()


# ------------------------------------------------------- the refusal matrix

def test_a_clean_contract_is_accepted():
    assert assert_matches(_clean(), ROOT) == []


def test_an_absent_contract_is_refused():
    problems = assert_matches(None, ROOT)
    assert any("absence of candidate_parse_mode means LEGACY" in p
               for p in problems)


def test_a_first_assertion_contract_is_refused():
    contract = dict(_clean(), candidate_parse_mode="first_assertion")
    assert any("candidate_parse_mode" in p for p in assert_matches(contract, ROOT))


def test_a_raw_output_mismatch_is_refused():
    contract = dict(_clean(), retain_raw_output=False)
    assert any("retain_raw_output" in p for p in assert_matches(contract, ROOT))


@pytest.mark.parametrize("field,wrong", [
    ("max_assertions", 1),
    ("candidates_per_function", 4),
    ("temperature", 1.0),
    ("top_p", 0.95),
    ("generation_seed", 43),
    ("function_generation_completion_limit", 128),
    ("repository_generation_completion_limit", 512),
    ("allow_test_function_candidates", False),
])
def test_every_contract_field_mismatch_is_refused(field, wrong):
    contract = dict(_clean())
    contract[field] = wrong
    assert any(field in problem for problem in assert_matches(contract, ROOT)), \
        f"{field} mismatch was not refused"


def test_the_field_list_covers_everything_but_the_name():
    assert set(CONTRACT_FIELDS) == set(SUCCESSOR_PROTOCOL) - {"protocol_name"}
    assert SUCCESSOR_PROTOCOL["protocol_name"] == NAME


@pytest.mark.parametrize("field", sorted(CONTRACT_SOURCES))
def test_a_changed_contract_source_is_refused(field):
    """The parser can change while the setting naming it stays identical."""
    contract = dict(_clean())
    contract[field] = "0" * 64
    problems = assert_matches(contract, ROOT)
    assert any(CONTRACT_SOURCES[field] in problem for problem in problems)


@pytest.mark.parametrize("field", sorted(CONTRACT_SOURCES))
def test_a_missing_contract_source_hash_is_refused(field):
    contract = dict(_clean())
    contract.pop(field)
    assert any(f"records no {field}" in p for p in assert_matches(contract, ROOT))


def test_a_stale_protocol_hash_is_refused():
    contract = dict(_clean(), protocol_sha256="0" * 64)
    assert any("protocol_sha256" in p for p in assert_matches(contract, ROOT))


def test_the_recorded_contract_hashes_real_files():
    recorded = _clean()
    actual = contract_source_hashes(ROOT)
    for field in CONTRACT_SOURCES:
        assert recorded[field] == actual[field]
        assert len(recorded[field]) == 64


def test_every_named_contract_source_exists():
    for relative in CONTRACT_SOURCES.values():
        assert (ROOT / relative).is_file(), relative


# ------------------------------- the protocol controls the launch path

def test_the_sequence_limit_is_in_the_protocol_and_raised():
    assert SUCCESSOR_PROTOCOL["max_sequence_tokens"] == 3072


def test_the_declared_allocation_clears_the_launch_guard():
    """train_on_dataset rejects prompt + completion >= max_sequence_tokens."""
    prompt_budget = 1024
    total = prompt_budget + SUCCESSOR_PROTOCOL["function_generation_completion_limit"]
    assert total < SUCCESSOR_PROTOCOL["max_sequence_tokens"], (
        f"{total} would be refused by the launch guard")


def test_the_internal_sequence_limit_matches_the_protocol():
    from engine.sft_trainer import MAX_SFT_SEQUENCE_LENGTH
    assert MAX_SFT_SEQUENCE_LENGTH == SUCCESSOR_PROTOCOL["max_sequence_tokens"]


def test_the_flags_carry_every_generation_setting():
    flags = training_command_flags()
    text = " ".join(flags)
    assert "--generation-completion-token-limit 1024" in text
    assert "--max-sequence-tokens 3072" in text
    assert "--candidate-parse-mode whole_output" in text


def test_the_protocol_agrees_with_the_code_it_names():
    """A named protocol that only describes settings is a comment."""
    from harness.successor_protocol import assert_runtime_matches
    assert assert_runtime_matches(ROOT) == []


def test_the_parser_identity_is_the_production_parser():
    """analyze_parser_pilot reimplements the split; nothing generates through it."""
    assert CONTRACT_SOURCES["parser_source_sha256"] == "engine/generator.py"
    assert "analyze_parser_pilot" not in str(CONTRACT_SOURCES)
    generator = (ROOT / "engine" / "generator.py").read_text(encoding="utf-8")
    assert "_parse_whole_output" in generator and "parse_mode" in generator


def test_the_production_generation_path_is_hashed():
    for relative in ("engine/generator.py", "engine/model_runtime.py",
                     "scripts/train_on_dataset.py",
                     "engine/test_generation_prompt.py"):
        assert relative in CONTRACT_SOURCES.values(), relative


def test_the_launch_option_exists_and_owns_its_flags():
    from harness.successor_protocol import OWNED_CLI_OPTIONS
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    assert '"--successor-protocol"' in source
    assert "args.successor_protocol" in source
    for option in OWNED_CLI_OPTIONS.values():
        assert option in source, option


def test_the_launch_option_sets_rather_than_only_records():
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    block = source.split("if args.successor_protocol:", 1)[1][:3000]
    for assignment in ("args.candidate_parse_mode = ",
                       "args.retain_raw_output = ",
                       "args.allow_test_function_candidates = ",
                       "args.seed = ",
                       "args.generation_completion_token_limit = ",
                       "args.max_sequence_tokens = "):
        assert assignment in block, assignment


def test_absence_of_the_flag_still_means_legacy():
    """The frozen contract: no candidate_parse_mode means first_assertion."""
    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    assert 'args.candidate_parse_mode = "first_assertion"' in source
    assert "if args.candidate_parse_mode is None:" in source
