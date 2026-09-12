"""A GPU run must not be able to start in the configuration that wasted one.

The failure being guarded: an evaluation launched without --base-model-name,
--attention-implementation and --sft-prompt-token-limit silently used the
canonical Phi-3 defaults at a 512-token budget, overflowed every prompt, and
wrote an artifact reporting a kill rate of 0.0. Nothing errored, because every
one of those flags has a default.
"""
from __future__ import annotations

import json

import pytest

from scripts.gpu_run_preflight import (
    EXPECTED, REQUIRED_FLAGS, check, parse_command,
)

GOOD = (
    "python scripts/train_on_dataset.py --run-name r --phase base_eval "
    "--evaluation-split train "
    "--base-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct "
    "--attention-implementation sdpa --sft-prompt-token-limit 1024 --seed 42"
)

#: Verbatim the command that produced the invalid 0.0 artifact.
THE_COMMAND_THAT_FAILED = (
    "python scripts/train_on_dataset.py --run-name x --phase base_eval "
    "--evaluation-split ablation_dev "
    "--output-instruction-variant metamorphic_allowed --seed 42"
)


def test_the_exact_command_that_wasted_a_run_is_refused():
    report = check(THE_COMMAND_THAT_FAILED, require_cuda=False)
    assert report["ok"] is False
    for flag in REQUIRED_FLAGS:
        assert any(flag in problem for problem in report["problems"])


def test_a_fully_specified_command_passes():
    report = check(GOOD, require_cuda=False)
    assert report["ok"] is True, report["problems"]


@pytest.mark.parametrize("flag", REQUIRED_FLAGS)
def test_dropping_any_single_required_flag_is_refused(flag):
    """Each flag individually, so a partial fix cannot pass."""
    tokens = GOOD.split()
    index = tokens.index(flag)
    del tokens[index:index + 2]
    report = check(" ".join(tokens), require_cuda=False)
    assert report["ok"] is False
    assert any(flag in problem for problem in report["problems"])


def test_a_wrong_value_is_refused_not_just_a_missing_flag():
    command = GOOD.replace("--sft-prompt-token-limit 1024",
                           "--sft-prompt-token-limit 512")
    report = check(command, require_cuda=False)
    assert report["ok"] is False
    assert any("512" in problem for problem in report["problems"])


def test_the_sealed_split_is_refused():
    command = GOOD.replace("--evaluation-split train", "--evaluation-split test")
    report = check(command, require_cuda=False)
    assert report["ok"] is False
    assert any("sealed" in problem for problem in report["problems"])


def test_confirm_final_test_is_refused_even_on_a_permitted_split():
    report = check(GOOD + " --confirm-final-test", require_cuda=False)
    assert report["ok"] is False
    assert any("sealed" in problem for problem in report["problems"])


def test_missing_cuda_is_refused_when_required():
    report = check(GOOD, require_cuda=True)
    # On a machine with CUDA this passes; the point is that the check exists
    # and reports the device rather than assuming it.
    assert "available" in report["cuda"]


def test_a_baseline_measured_on_another_model_is_refused(tmp_path):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({
        "evaluation_split": "train",
        "model_runtime_profile": {"model_name": "microsoft/Phi-3-mini-4k-instruct"},
        "dataset_fingerprint": "x:prompt_token_limit=1024",
    }), encoding="utf-8")
    report = check(GOOD, baseline=baseline, require_cuda=False)
    assert report["ok"] is False
    assert any("not be comparable" in problem for problem in report["problems"])


def test_a_baseline_on_another_split_is_refused(tmp_path):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({
        "evaluation_split": "val",
        "model_runtime_profile": {"model_name": EXPECTED["--base-model-name"]},
        "dataset_fingerprint": "x:prompt_token_limit=1024",
    }), encoding="utf-8")
    report = check(GOOD, baseline=baseline, require_cuda=False)
    assert report["ok"] is False
    assert any("split" in problem for problem in report["problems"])


def test_flags_parse_with_both_spellings():
    parsed = parse_command("cmd --a 1 --b=2 --flag")
    assert parsed["--a"] == "1"
    assert parsed["--b"] == "2"
    assert parsed["--flag"] == "true"
