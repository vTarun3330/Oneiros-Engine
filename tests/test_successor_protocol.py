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
