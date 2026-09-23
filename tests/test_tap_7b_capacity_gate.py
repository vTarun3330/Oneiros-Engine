"""Integrity and decision guards for the predeclared 7B TAP capacity gate."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from config import immutable_revision_for
from scripts.analyse_tap_7b_capacity_gate import gate_checks, validate_candidate
from scripts.preflight_tap_7b_capacity_gate import (
    REQUIRED_ITEMS_SHA256,
    acceptance_policy,
    token_audit,
)
from scripts.run_tap_7b_capacity_gate import (
    EXPECTED_KEYS,
    MODEL_NAME,
    MODEL_REVISION,
    load_checkpoint,
    validate_items,
)

ROOT = Path(__file__).resolve().parent.parent


def _raw_row(item_id: str, verdict: str = "correct") -> dict:
    raw = "assert f(1) == 2"
    return {
        "id": item_id,
        "benchmark": "mbpp",
        "verdict": verdict,
        "truncated": False,
        "expected": "2",
        "predicted": "2",
        "raw": raw,
        "raw_chars": len(raw),
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def test_7b_model_is_bound_to_the_official_immutable_snapshot():
    assert immutable_revision_for(MODEL_NAME) == MODEL_REVISION
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)


def test_frozen_tap_input_is_portable_tracked_and_train_only():
    path = ROOT / "results" / "tap_train_items.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REQUIRED_ITEMS_SHA256
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate_items(items)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "results/tap_train_items.jsonl"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0


def test_validate_items_rejects_non_train_or_wrong_panel_size():
    item = {
        "id": "x",
        "split": "train",
        "prompt_ref": "r",
        "prompt_mut": "m",
        "expected_repr": "1",
        "benchmark": "mbpp",
    }
    with pytest.raises(RuntimeError, match="exactly 600"):
        validate_items([item])
    rows = [dict(item, id=str(index)) for index in range(600)]
    rows[-1]["split"] = "val"
    with pytest.raises(RuntimeError, match="non-train"):
        validate_items(rows)


def test_resume_refuses_contract_drift_and_modified_raw_evidence(tmp_path):
    item_ids = ["a"]
    contract = {"model": MODEL_NAME, "revision": MODEL_REVISION}
    checkpoint = tmp_path / "gate.partial.json"
    checkpoint.write_text(json.dumps({
        "run_contract": contract,
        "created_utc": "now",
        "results": {EXPECTED_KEYS[0]: {"detail": [_raw_row("a")]}}
    }), encoding="utf-8")
    results, _ = load_checkpoint(checkpoint, contract, item_ids)
    assert list(results) == [EXPECTED_KEYS[0]]
    with pytest.raises(RuntimeError, match="contract mismatch"):
        load_checkpoint(checkpoint, {**contract, "batch": 4}, item_ids)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["results"][EXPECTED_KEYS[0]]["detail"][0]["raw"] += " changed"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid raw evidence"):
        load_checkpoint(checkpoint, contract, item_ids)


def test_candidate_requires_exact_identity_train_scope_and_raw_rows():
    ids = [str(index) for index in range(600)]
    split = {"items_file_sha256": "items"}
    baseline = {"run_contract": {"items_file_sha256": "items"}}
    candidate = {
        "status": "complete",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "training_performed": False,
        "weights_written": False,
        "run_contract": {"items_file_sha256": "items", "permitted_split": "train"},
        "detail": {key: [_raw_row(item_id) for item_id in ids] for key in EXPECTED_KEYS},
    }
    indexed = validate_candidate(candidate, baseline, split)
    assert all(len(rows) == 600 for rows in indexed.values())
    candidate["model_revision"] = "0" * 40
    with pytest.raises(ValueError, match="model identity"):
        validate_candidate(candidate, baseline, split)


def test_capacity_gate_requires_gain_ci_answer_rate_and_cap_safety():
    passing = {
        "paired_7b_minus_1_5b": {"difference_pp": 5.0, "ci95_low_pp": 0.01},
        "answer_rate_difference_pp": -5.0,
        "base7b": {"cap_hit_rate": 0.02},
    }
    assert all(gate_checks(passing).values())
    for mutation in (
        {"paired_7b_minus_1_5b": {"difference_pp": 4.99, "ci95_low_pp": 0.01}},
        {"paired_7b_minus_1_5b": {"difference_pp": 5.0, "ci95_low_pp": 0.0}},
        {"answer_rate_difference_pp": -5.01},
        {"base7b": {"cap_hit_rate": 0.0201}},
    ):
        payload = {
            "paired_7b_minus_1_5b": dict(passing["paired_7b_minus_1_5b"]),
            "answer_rate_difference_pp": passing["answer_rate_difference_pp"],
            "base7b": dict(passing["base7b"]),
        }
        payload.update(mutation)
        assert not all(gate_checks(payload).values())


def test_preflight_policy_is_exactly_the_analyzer_policy():
    assert acceptance_policy() == {
        "primary_endpoint": "TAP-mut per-requested accuracy, paired 7B minus 1.5B",
        "minimum_gain_pp": 5.0,
        "minimum_ci95_low_pp_exclusive": 0.0,
        "maximum_answer_rate_regression_pp": 5.0,
        "maximum_cap_hit_rate": 0.02,
    }


def test_token_audit_counts_rendered_prompt_plus_completion():
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"] + "!"

        @staticmethod
        def __call__(text, **_kwargs):
            return {"input_ids": list(text)}

    items = [
        {"prompt_ref": "abc", "prompt_mut": "abcdef"},
        {"prompt_ref": "x", "prompt_mut": "yz"},
    ]
    report = token_audit(items, Tokenizer(), 128)
    assert report["TAP-ref"]["maximum_prompt_tokens"] == 4
    assert report["TAP-mut"]["maximum_prompt_plus_completion"] == 135
    assert all(row["fits"] for row in report.values())


def test_analysis_cli_can_start_from_the_repository_root():
    completed = subprocess.run(
        [
            str(ROOT / ".venv-gpu" / "Scripts" / "python.exe"),
            "scripts/analyse_tap_7b_capacity_gate.py",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "7B-versus-1.5B" in completed.stdout
