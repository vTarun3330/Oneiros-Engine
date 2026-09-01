import copy
import json

import pytest

from engine.execution_feedback import build_feedback_prompt, collect_execution_feedback
from metrics.research_evaluation import (
    aggregate_seed_results,
    compare_policy_results,
    evaluate_candidate_slots,
    evaluation_profile_sha256,
    function_result,
    prioritise_diverse_slots,
    summarise_function_results,
)
from scripts.research_ablations import build_ablation_plan, run_local_smoke
from scripts import train_on_dataset as trainer
from scripts import evaluate_external_generations as external_eval


def _slot(rank, code):
    return {
        "rank": rank,
        "parse_valid": code is not None,
        "code": code,
        "raw_output_sha256": f"slot-{rank}",
    }


def test_ordered_candidates_produce_monotonic_kill_and_pass_at_k():
    outcomes = evaluate_candidate_slots(
        [
            _slot(1, None),
            _slot(2, "assert f(0) == 0"),
            _slot(3, "assert f(2) == 3"),
            _slot(4, "assert f('bad') == 'bad'"),
        ],
        "def f(x):\n    return x + 1\n",
        "def f(x):\n    return x\n",
        "f",
    )
    result = function_result("record", "arithmetic", "f", outcomes)
    summary = summarise_function_results([result], k_values=(1, 2, 4))

    assert [item["failure_mode"] for item in outcomes] == [
        "generation_invalid",
        "reference_assertion_error",
        "killed_assertion_error",
        "reference_error",
    ]
    assert summary["kill_at_k"]["1"]["rate"] == 0.0
    assert summary["kill_at_k"]["2"]["rate"] == 0.0
    assert summary["kill_at_k"]["4"]["rate"] == 1.0
    assert summary["pass_at_k"]["1"]["rate"] == 0.0
    assert summary["pass_at_k"]["4"]["rate"] == 1.0


def test_diversity_prioritisation_is_equal_budget_and_stable():
    slots = [
        _slot(1, "assert f(1) == 1"),
        _slot(2, "assert f(2) == 2"),
        _slot(3, "assert f([]) == []"),
        _slot(4, None),
    ]
    ordered = prioritise_diverse_slots(slots, "f", "input_shape")

    assert len(ordered) == len(slots)
    assert [item["rank"] for item in ordered] == [1, 2, 3, 4]
    assert ordered[0]["original_rank"] == 1
    assert ordered[1]["original_rank"] == 3
    assert ordered[-1]["parse_valid"] is False


def test_feedback_uses_only_visible_code_execution():
    feedback = collect_execution_feedback(
        ["assert f(1) == 2", "assert f('x') == 2"],
        "def f(x):\n    return x + 1\n",
    )
    prompt = build_feedback_prompt(feedback, require_novel_shape=True)

    assert [item["status"] for item in feedback] == ["pass", "error"]
    assert "different argument structure" in prompt
    assert "reference" not in prompt.lower()


def _policy_payload(seed=42):
    outcomes = evaluate_candidate_slots(
        [_slot(1, "assert f(2) == 3"), _slot(2, "assert f(0) == 1")],
        "def f(x):\n    return x + 1\n",
        "def f(x):\n    return x\n",
        "f",
    )
    item = function_result("record", "arithmetic", "f", outcomes)
    profile_hash = evaluation_profile_sha256({"candidate_budget": 2})
    return {
        "seed": seed,
        "evaluation_split": "val",
        "evaluation_scope_sha256": "scope",
        "tests_per_function": 2,
        "model_artifact_sha256": "adapter",
        "evaluation_profile_sha256": profile_hash,
        **summarise_function_results([item], k_values=(1, 2)),
        "function_results": [item],
    }


def test_seed_aggregation_and_policy_comparison_require_matched_scope():
    seed_42 = _policy_payload(42)
    seed_43 = _policy_payload(43)
    aggregate = aggregate_seed_results([seed_42, seed_43])
    comparison = compare_policy_results(seed_42, seed_43)

    assert aggregate["seed_count"] == 2
    assert aggregate["function_kill_rate_standard_deviation"] == 0.0
    assert comparison["net_unique_function_gain"] == 0

    mismatched = copy.deepcopy(seed_43)
    mismatched["evaluation_scope_sha256"] = "different"
    with pytest.raises(ValueError, match="not comparable"):
        aggregate_seed_results([seed_42, mismatched])


def test_ablation_plan_uses_training_only_dev_and_locked_three_seeds():
    plan = build_ablation_plan("run_name")
    commands = [
        command
        for experiment in plan["experiments"]
        for command in experiment["commands"]
    ]

    assert plan["final_test_sealed"] is True
    assert plan["plan_identity"]["seeds"] == [42, 43, 44]
    assert plan["plan_identity"]["split"] == "ablation_dev"
    assert all("--evaluation-split ablation_dev" in command for command in commands)
    assert all("dpo_eval" not in command for command in commands)
    assert all("confirm-final-test" not in command for command in commands)
    assert any("--phase base_eval" in command for command in commands)
    assert any("--eval-feedback-rounds 1" in command for command in commands)
    assert any("--holdout-bug-family" in command for command in commands)
    assert any("--prompt-information-variant code_only" in command for command in commands)
    assert any("--prompt-information-variant code_specification" in command for command in commands)
    assert any("--output-instruction-variant legacy_exactly_one" in command for command in commands)
    assert any("--sft-real-target-fraction 0.3" in command for command in commands)
    assert any("--execution-mode function_assertion" in command for command in commands)
    assert any("--max-pairs 4000" in command for command in commands)

    prompt_budget_arms = {
        item["prompt_token_limit"]: item
        for item in plan["experiments"]
        if item["id"] in {"J_prompt_budget_1024", "J_prompt_budget_1280"}
    }
    assert set(prompt_budget_arms) == {1024, 1280}
    for arm in prompt_budget_arms.values():
        assert arm["selection_prompt_token_limit"] == 1024
        assert "--selection-prompt-token-limit 1024" in arm["preflight_command"]
        assert all(
            "--sft-selection-prompt-token-limit 1024" in command
            for command in arm["commands"]
        )

    blocked = {item["id"]: item for item in plan["experiments"] if not item["commands"]}
    assert blocked["B_legacy_vs_unified"]["status"] == "blocked_clean_control_unavailable"
    assert blocked["D_head_tail_vs_section_compaction"]["status"] == "local_safety_only"
    assert blocked["E_localization_assumption"]["status"] == "blocked_paired_scope_unavailable"
    for label in ("G0_proportional", "G1_dataset_balanced", "G2_dataset_family_balanced"):
        assert blocked[label]["status"] == "blocked_unmatched_sampling_controls"
        assert blocked[label]["decision"] == "INCONCLUSIVE"


def test_local_research_smoke_covers_all_metric_layers():
    smoke = run_local_smoke()

    assert smoke["status"] == "passed"
    assert smoke["sealed_test_accessed"] is False
    assert smoke["multi_seed_aggregation"]["seed_count"] == 2
    assert "diversity_ablation" in smoke


def test_ablation_budget_is_explicit_in_identity_commands_and_preflight():
    plan = build_ablation_plan("budget_test", screening_prompt_token_limit=1280)
    default = build_ablation_plan("budget_test")
    assert plan["plan_sha256"] != default["plan_sha256"]
    assert plan["experiment_matrix_sha256"] != default["experiment_matrix_sha256"]
    assert "J" in plan["plan_identity"]["primary_groups"]
    assert plan["experiments"][0]["axis"] == "prompt_budget"
    for item in plan["experiments"]:
        if not item["commands"]:
            continue
        budget = item["prompt_token_limit"] if item["axis"] == "prompt_budget" else 1280
        assert all(f"--sft-prompt-token-limit {budget}" in cmd for cmd in item["commands"])
        if item.get("preflight_command"):
            assert f"--prompt-token-limit {budget}" in item["preflight_command"]
            assert "--evaluation-split ablation_dev" in item["preflight_command"]
        if item["axis"] != "prompt_budget":
            assert "group_J_decision_frozen_at_this_exact_prompt_budget" in item["requires_before_execution"]
    assert all("--sft-prompt-token-limit 1280" in cmd for cmd in plan["modal_smoke_commands"])
    for budget in (512, 768):
        rejected = next(item for item in plan["experiments"] if item["id"] == f"J_prompt_budget_{budget}")
        assert rejected["commands"] == []
        assert rejected["decision"] == "REJECT"


def test_evaluation_profiles_use_distinct_result_files(monkeypatch):
    monkeypatch.setattr(trainer, "MAX_VALIDATION_PAIRS", 32)
    monkeypatch.setattr(trainer, "EVAL_FEEDBACK_ROUNDS", 1)
    monkeypatch.setattr(trainer, "EVAL_DIVERSITY_MODE", "ast")
    monkeypatch.setattr(trainer, "HOLDOUT_BUG_FAMILY", None)

    assert trainer.sft_validation_results_filename(42) == (
        "sft_validation_smoke32_feedback1_diversity-ast_seed_42.json"
    )
    assert trainer._candidate_round_sizes(8, 1) == [4, 4]
    assert trainer._candidate_round_sizes(8, 2) == [3, 3, 2]


def test_family_holdout_changes_only_explicit_training_scope(monkeypatch):
    monkeypatch.setattr(trainer, "HOLDOUT_BUG_FAMILY", None)
    standard = trainer.sft_training_scope(None, True, None)
    monkeypatch.setattr(trainer, "HOLDOUT_BUG_FAMILY", "boundary")
    holdout = trainer.sft_training_scope(None, True, None)

    assert "holdout_bug_family" not in standard
    assert ":holdout_bug_family=boundary" in holdout


def test_external_baseline_uses_same_ordered_metric_schema(tmp_path, monkeypatch):
    pair = {
        "id": "external-record",
        "execution_mode": trainer.FUNCTION_EXECUTION_MODE,
        "golden_code": "def f(x):\n    return x + 1\n",
        "mutant_code": "def f(x):\n    return x\n",
        "entry_point": "f",
        "bug_family": "arithmetic",
        "source_name": "fixture",
        "project": "fixture",
    }
    monkeypatch.setattr(
        external_eval, "verify_corpus", lambda _: {"corpus_id": "fixture-corpus"}
    )
    monkeypatch.setattr(external_eval, "load_phase3_pairs", lambda *_: [pair])
    input_path = tmp_path / "external.jsonl"
    input_path.write_text(json.dumps({
        "record_id": "external-record",
        "model": "provider/model-v1",
        "seed": 42,
        "candidates": ["assert f(2) == 3"],
    }) + "\n", encoding="utf-8")

    result = external_eval.evaluate_external_generations(
        tmp_path, input_path, candidate_budget=2
    )

    assert result["final_test_measurement"] is False
    assert result["function_kill_rate"] == 1.0
    assert result["kill_at_k"]["1"]["rate"] == 1.0
    assert result["pass_at_k"]["1"]["rate"] == 1.0


def test_feedback_ablation_preserves_eight_candidate_generation_budget(monkeypatch):
    calls = []

    def fake_generate(
        generator,
        pairs,
        num,
        return_accounting,
        prompt_additions,
        rank_offset,
    ):
        calls.append({
            "num": num,
            "rank_offset": rank_offset,
            "prompt_additions": dict(prompt_additions),
        })
        slots = [
            _slot(rank_offset + index + 1, f"assert f({index}) == {index + 1}")
            for index in range(num)
        ]
        return {0: [slot["code"] for slot in slots]}, {0: {
            "requested_candidates": num,
            "raw_generated_sequences": num,
            "parsed_candidates": num,
            "generation_invalid_candidates": 0,
            "candidate_slots": slots,
        }}

    monkeypatch.setattr(trainer, "generate_tests_ai_batched", fake_generate)
    monkeypatch.setattr(
        trainer,
        "collect_execution_feedback",
        lambda candidates, code: [{"attempt": 1, "test": candidates[0], "status": "pass"}],
    )
    monkeypatch.setattr(trainer, "EVAL_FEEDBACK_ROUNDS", 1)
    monkeypatch.setattr(trainer, "EVAL_DIVERSITY_MODE", "none")

    _, accounting = trainer._generate_evaluation_candidates(object(), [{
        "entry_point": "f",
        "mutant_code": "def f(x): return x",
    }])

    assert [call["num"] for call in calls] == [4, 4]
    assert [call["rank_offset"] for call in calls] == [0, 4]
    assert calls[0]["prompt_additions"] == {}
    assert calls[1]["prompt_additions"][0]
    assert accounting[0]["requested_candidates"] == 8
    assert [slot["rank"] for slot in accounting[0]["candidate_slots"]] == list(range(1, 9))


def test_base_model_ablation_is_declared_without_emitting_modal_gpu_commands():
    """Group L is a local screen, so it must never enter the Modal command set.

    The plan-wide invariant is that every entry in ``commands`` evaluates on
    ablation_dev. A local base-model screen has no such flag, so it is recorded
    under ``local_screen_commands`` instead of being allowed to weaken that
    invariant.
    """
    plan = build_ablation_plan("run_name")
    group_l = [
        experiment for experiment in plan["experiments"]
        if experiment["id"].startswith("L_base_model")
    ]

    assert {experiment["id"] for experiment in group_l} == {
        "L_base_model_L0_phi3_eager",
        "L_base_model_L1_phi3_sdpa",
        "L_base_model_L2_qwen25_coder_sdpa",
    }
    assert all(experiment["commands"] == [] for experiment in group_l)
    assert all(experiment["uses_final_test"] is False for experiment in group_l)

    by_id = {experiment["id"]: experiment for experiment in group_l}

    # The infeasible arm keeps its negative result instead of disappearing.
    infeasible = by_id["L_base_model_L1_phi3_sdpa"]
    assert infeasible["decision"] == "REJECT"
    assert "scaled_dot_product_attention" in infeasible["reason"]
    assert "local_screen_commands" not in infeasible

    # A non-canonical base model must be requested explicitly everywhere it is
    # loaded or tokenized, never inherited from a silently changed default.
    treatment = by_id["L_base_model_L2_qwen25_coder_sdpa"]
    screen = treatment["local_screen_commands"]
    assert treatment["command_kind"] == "local_not_modal"
    assert all(
        command.startswith("python scripts/") for command in screen
    )
    assert any("--base-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct" in command
               for command in screen if "preflight_sft_run.py" in command)
    assert any("--attention-implementation sdpa" in command
               for command in screen if "train_on_dataset.py" in command)
    assert all("--evaluation-split val" not in command for command in screen)
    assert all("confirm-final-test" not in command for command in screen)
