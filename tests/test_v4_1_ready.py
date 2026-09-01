import copy
import shlex

import pytest
from utils.dataset_identity import DATASET_IDENTITY_POLICY

from scripts.v4_1_ready import (
    CORPUS_VERSION, build_execution_queue, integration_commands,
    integration_preflight_command, integration_preflight_mismatches, local_commands,
)


def _all_command_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_command_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_command_strings(child)


def test_gpu_ready_queue_never_emits_a_final_test_command():
    queue = build_execution_queue()
    commands = list(_all_command_strings(queue["stages"]))

    assert queue["safety"]["gpu_auto_launch"] is False
    assert queue["safety"]["final_test_command_emitted"] is False
    assert not any("--confirm-final-test" in command for command in commands)
    assert not any("--evaluation-split test" in command for command in commands)
    assert not any("--phase dpo_eval" in command for command in commands)


def test_integration_queue_has_distinct_fresh_and_resume_commands():
    commands = integration_commands()

    assert "--fresh" in commands["fresh"]
    assert "--fresh" not in commands["resume"]
    assert "--evaluation-split ablation_dev" in commands["fresh"]
    assert "--sft-min-monitor-checkpoints 1" in commands["fresh"]
    assert "--max-pairs 32" in commands["fresh"]


@pytest.mark.parametrize("budget", [1024, 1280])
def test_every_gpu_queue_command_declares_its_budget(budget):
    queue = build_execution_queue(budget)
    gpu_commands = [
        text for text in _all_command_strings(queue["stages"])
        if text.startswith("py -3.12 scripts/modal_train.py ")
    ]
    assert gpu_commands
    for command in gpu_commands:
        parts = shlex.split(command)
        declared = int(parts[parts.index("--sft-prompt-token-limit") + 1])
        run_name = parts[parts.index("--run-name") + 1]
        assert declared in (1024, 1280)
        assert f"_p{declared}" in run_name
        if "j_budget_" not in run_name:
            assert declared == budget
    assert queue["prompt_budget"]["status"] == "NOT_A_SELECTED_RESEARCH_WINNER"
    assert queue["prompt_budget"]["frozen_runtime_defaults_modified"] is False
    preflight = integration_preflight_command(budget)
    assert f"--prompt-token-limit {budget}" in preflight
    assert "--evaluation-split ablation_dev" in preflight
    assert f"v4_1_integration_32_p{budget}_preflight.json" in preflight
    assert preflight == local_commands(budget)[-1]


def _matching_preflight():
    manifest = {
        "corpus_id": "fixture-corpus",
        "files": {name: {"sha256": name + "-hash"} for name in (
            "records.json", "splits.json", "ablation_dev_manifest.json", "leakage_audit.json",
        )},
    }
    report = {
        "ready": True,
        "local_test_status": {
            "source_tree_sha256": "source-hash", "returncode": 0, "failed": 0,
            "sealed_final_test_accessed": False,
        },
        "corpus": {
            "version": CORPUS_VERSION, "corpus_id": "fixture-corpus",
            "records_sha256": "records.json-hash", "splits_sha256": "splits.json-hash",
            "ablation_dev_sha256": "ablation_dev_manifest.json-hash",
            "leakage_audit_sha256": "leakage_audit.json-hash",
        },
        "selection": {"requested_pairs": 32, "retained_pairs": 32, "execution_mode_filter": None},
        "evaluation_panel": {
            "evaluation_split": "ablation_dev", "prompt_token_limit": 1024,
            "prompt_budget_failures": 0, "function_records": 10,
            "promptable_function_records": 10,
        },
        "sampling": {
            "target_real_fraction": 0.20, "balanced_sampling_enabled": True,
            "synthetic_balance_fraction": 0.0, "synthetic_balance_mode": "none",
            "dataset_identity_policy": DATASET_IDENTITY_POLICY,
            "example_weights": {
                "dataset_identity_policy": DATASET_IDENTITY_POLICY, "unknown_dataset_examples": 0,
            },
        },
        "tokenization": {
            "prompt_token_limit": 1024, "repository_prompt_token_limit": 1024,
            "completion_token_limit": 128, "repository_completion_token_limit": 1024,
            "sequence_token_limit": 2048, "prompt_information_variant": "full",
            "output_instruction_variant": "self_contained",
        },
        "training": {
            "epochs": 1, "batch_size": 1, "learning_rate": 0.00005,
            "lr_scheduler_type": "constant_with_warmup", "min_function_kill_rate": 0.58,
            "planned_validation_checkpoints": [6],
            "optimizer_schedule": {"minimum_monitor_checkpoints": 1},
        },
        "gates": {
            "zero_sequence_overflows": True, "evaluation_panel_fully_promptable": True,
            "terminal_checkpoint_monitor_enabled": True, "minimum_monitor_schedule_reached": True,
        },
    }
    return report, manifest


def test_doctor_accepts_matching_integration_evidence():
    report, manifest = _matching_preflight()
    assert integration_preflight_mismatches(report, "source-hash", manifest, 1024) == []
    assert integration_preflight_mismatches(report, "source-hash", manifest, 1280)
    assert integration_preflight_mismatches({}, "source-hash", manifest, 1024)


@pytest.mark.parametrize("field,value", [
    ("tokenization.prompt_token_limit", 512),
    ("tokenization.completion_token_limit", 256),
    ("tokenization.prompt_information_variant", "code_only"),
    ("tokenization.output_instruction_variant", "legacy_exactly_one"),
    ("evaluation_panel.evaluation_split", "val"),
    ("evaluation_panel.evaluation_split", "test"),
    ("evaluation_panel.prompt_budget_failures", 1),
    ("evaluation_panel.function_records", 0),
    ("evaluation_panel.promptable_function_records", 9),
    ("local_test_status.source_tree_sha256", "old-source"),
    ("local_test_status.returncode", 1),
    ("corpus.splits_sha256", "wrong-splits"),
    ("selection.retained_pairs", 16),
    ("training.learning_rate", 0.001),
    ("training.optimizer_schedule.minimum_monitor_checkpoints", 0),
    ("training.planned_validation_checkpoints", []),
    ("sampling.target_real_fraction", 0.3),
    ("sampling.dataset_identity_policy", "ingestion_source_only"),
    ("sampling.example_weights.unknown_dataset_examples", 1),
    ("gates.zero_sequence_overflows", False),
])
def test_doctor_rejects_ready_but_wrong_scope_evidence(field, value):
    report, manifest = _matching_preflight()
    changed = copy.deepcopy(report)
    target = changed
    keys = field.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    assert changed["ready"] is True
    assert integration_preflight_mismatches(changed, "source-hash", manifest, 1024)


@pytest.mark.parametrize("budget", [0, 512, 768, 2048])
def test_queue_does_not_materialize_rejected_or_unplanned_budgets(budget):
    with pytest.raises(ValueError, match="admissible"):
        build_execution_queue(budget)
