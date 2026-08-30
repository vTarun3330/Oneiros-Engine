"""V4.1 guards: a fail-closed prompt budget must never abort an evaluation.

Section-aware compaction refuses to slice a target function in half.  Before
these guards the generation path let that refusal escape as an uncaught
``PromptBudgetError``, so one oversized record aborted the whole batch and the
preflight reported ``ready`` without ever rendering an evaluation prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.prompt_budget import PromptBudgetError
from metrics.research_evaluation import (
    FAILURE_TAXONOMY_CATEGORIES,
    classify_candidate_failure,
    function_result,
    summarise_failure_taxonomy,
    summarise_function_results,
)
from scripts import train_on_dataset as trainer
from scripts.failure_taxonomy import build_report, render_markdown


class _StubTokenizer:
    """Minimal tokenizer stand-in; the batch must never reach it."""

    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 0

    def pad(self, *args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("Generation must not run without a promptable record")


class _StubGenerator:
    is_loaded = True
    temperature = 0.7
    top_p = 0.9

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.model = SimpleNamespace(device="cpu")


def _pair(record_id: str) -> dict:
    return {
        "id": record_id,
        "entry_point": "f",
        "execution_mode": trainer.FUNCTION_EXECUTION_MODE,
        "bug_family": "boundary",
        "source_name": "mbpp",
        "golden_code": "def f(x):\n    return x + 1\n",
        "mutant_code": "def f(x):\n    return x\n",
    }


def test_unpromptable_record_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(
        trainer,
        "compact_unified_user_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PromptBudgetError("Required prompt sections exceed the declared budget")
        ),
    )
    monkeypatch.setattr(trainer, "build_pair_prompt", lambda pair: "prompt")

    results, accounting = trainer.generate_tests_ai_batched(
        _StubGenerator(), [_pair("a"), _pair("b")], num=4, return_accounting=True
    )

    assert results == {}
    for index in (0, 1):
        assert accounting[index]["prompt_budget_failure"] is True
        assert "budget" in accounting[index]["prompt_budget_failure_reason"]
        # The denominator must survive so Kill@k stays comparable.
        assert accounting[index]["requested_candidates"] == 4
        assert accounting[index]["parsed_candidates"] == 0
        assert accounting[index]["generation_invalid_candidates"] == 4
        assert len(accounting[index]["candidate_slots"]) == 4


def test_budget_failure_is_reported_in_the_summary():
    outcomes = [
        {"rank": rank, "parse_valid": False, "code": None} for rank in range(1, 5)
    ]
    unpromptable = function_result(
        "rec-1", "boundary", "f", outcomes, source_name="mbpp",
        prompt_budget_failure=True,
        prompt_budget_failure_reason="required sections exceed budget",
    )
    promptable = function_result("rec-2", "boundary", "f", outcomes, source_name="mbpp")

    assert unpromptable["prompt_budget_failure"] is True
    assert promptable["prompt_budget_failure"] is False

    summary = summarise_function_results([unpromptable, promptable])
    assert summary["prompt_budget_failed_functions"] == 1
    assert summary["prompt_budget_failure_rate"] == 0.5


def test_generator_returns_invalid_candidates_instead_of_raising(monkeypatch):
    from engine import generator as generator_module

    monkeypatch.setattr(
        generator_module,
        "compact_unified_user_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(PromptBudgetError("too long")),
    )

    instance = generator_module.Phi3Generator.__new__(generator_module.Phi3Generator)
    instance.is_loaded = True
    instance.tokenizer = _StubTokenizer()
    instance.model = SimpleNamespace(device="cpu")
    instance.stats = {"total_generated": 0, "valid_generated": 0, "invalid_generated": 0}
    instance._create_prompt = lambda **kwargs: "prompt"

    tests = generator_module.Phi3Generator.generate(
        instance,
        function_signature="def f(x):",
        docstring="",
        function_id="rec-1",
        num_samples=3,
    )

    assert len(tests) == 3
    assert all(not test.is_valid for test in tests)
    assert all("prompt_budget_failure" in test.parse_error for test in tests)
    assert instance.stats["invalid_generated"] == 3


# --- §43 failure taxonomy -------------------------------------------------


@pytest.mark.parametrize(
    "outcome, family, mode, expected",
    [
        ({"parse_valid": False}, "boundary", "function_assertion", "syntax_invalid"),
        (
            {"parse_valid": True, "policy_valid": False},
            "boundary", "function_assertion", "wrong_target_api",
        ),
        (
            {
                "parse_valid": True, "policy_valid": True,
                "reference_status": "error", "reference_error": "NameError: helper",
            },
            "boundary", "function_assertion", "undefined_symbol",
        ),
        (
            {
                "parse_valid": True, "policy_valid": True,
                "reference_status": "error", "reference_error": "NameError: helper",
            },
            "real_repository_defect", "repository_pytest_fragment",
            "repository_context_hallucination",
        ),
        (
            {
                "parse_valid": True, "policy_valid": True,
                "reference_status": "error",
                "reference_error": "ModuleNotFoundError: numpy",
            },
            "boundary", "function_assertion", "environment_failure",
        ),
        (
            {
                "parse_valid": True, "policy_valid": True,
                "reference_status": "assertion_error",
                "reference_error": "AssertionError: ",
            },
            "boundary", "function_assertion", "wrong_expected_value",
        ),
        (
            {"parse_valid": True, "policy_valid": True, "reference_status": "timeout"},
            "boundary", "function_assertion", "timeout",
        ),
        (
            {
                "parse_valid": True, "policy_valid": True, "reference_status": "pass",
                "reference_valid": True, "killed": True,
            },
            "boundary", "function_assertion", "killed",
        ),
        (
            {
                "parse_valid": True, "policy_valid": True, "reference_status": "pass",
                "reference_valid": True, "killed": False,
            },
            "off_by_one", "function_assertion", "off_by_one_miss",
        ),
        (
            {
                "parse_valid": True, "policy_valid": True, "reference_status": "pass",
                "reference_valid": True, "killed": False,
            },
            "arithmetic", "function_assertion", "passes_both",
        ),
        (
            {"prompt_budget_failure": True, "parse_valid": False},
            "boundary", "function_assertion", "not_generated",
        ),
    ],
)
def test_failure_categories(outcome, family, mode, expected):
    assert classify_candidate_failure(outcome, family, mode) == expected
    assert expected in FAILURE_TAXONOMY_CATEGORIES


def test_taxonomy_totals_match_candidate_counts():
    results = [
        {
            "record_id": "r1",
            "bug_family": "boundary",
            "source_name": "mbpp",
            "candidate_outcomes": [
                {"rank": 1, "parse_valid": False},
                {
                    "rank": 2, "parse_valid": True, "policy_valid": True,
                    "reference_status": "pass", "reference_valid": True, "killed": True,
                },
            ],
        },
        {"record_id": "r2", "bug_family": "index", "source_name": "humaneval"},
    ]
    taxonomy = summarise_failure_taxonomy(results)
    assert taxonomy["classified_functions"] == 1
    assert taxonomy["unclassifiable_functions"] == 1
    assert taxonomy["classified_candidates"] == 2
    assert taxonomy["overall"]["counts"] == {"killed": 1, "syntax_invalid": 1}
    assert set(taxonomy["by_source"]) == {"mbpp"}


def test_taxonomy_report_refuses_sealed_final_test(tmp_path: Path):
    sealed = tmp_path / "final.json"
    sealed.write_text(
        json.dumps({
            "final_test_measurement": True,
            "evaluation_split": "test",
            "function_results": [],
        }),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        build_report([sealed])


def test_taxonomy_report_accepts_validation_artifact(tmp_path: Path):
    artifact = tmp_path / "seed42.json"
    artifact.write_text(
        json.dumps({
            "final_test_measurement": False,
            "evaluation_split": "ablation_dev",
            "seed": 42,
            "function_results": [
                {
                    "record_id": "r1",
                    "bug_family": "boundary",
                    "source_name": "mbpp",
                    "candidate_outcomes": [{"rank": 1, "parse_valid": False}],
                }
            ],
        }),
        encoding="utf-8",
    )
    report = build_report([artifact])
    assert report["sealed_final_test_accessed"] is False
    assert report["pooled"]["overall"]["counts"] == {"syntax_invalid": 1}
    assert "failure taxonomy" in render_markdown(report).lower()
