import json
import random
from types import SimpleNamespace

import pytest
import torch

from harness.corpus import valid_corpus_version
from scripts.train_on_dataset import (
    FUNCTION_EXECUTION_MODE,
    MAX_DPO_COMPLETION_TOKENS,
    REPOSITORY_EXECUTION_MODE,
    _dpo_training_scope_sha256,
    _adapter_evaluation_context,
    _length_matched_repository_loser,
    _load_adapter_evaluation_progress,
    _load_sft_validation_baseline,
    _restore_adapter_evaluation_rng,
    _save_adapter_evaluation_progress,
    _sft_monitor_gate_decision,
    _paired_function_diagnostics,
    balanced_repeat_examples,
    deduplicate_sft_examples,
    filter_generation_compatible_sft_examples,
    generate_tests_ai_batched,
    normalized_sft_run_hyperparameters,
    select_bounded_train_pairs,
    sft_monitor_acceptance_passed,
    sft_validation_results_filename,
    summarize_train_pair_selection,
    sft_training_scope,
)
from engine.sft_trainer import SFTDataPoint
from engine.dpo_trainer import DPODataPoint, DPOTrainer


def _pair(index: int, repository: bool = False):
    return {
        "id": f"pair-{index:04d}",
        "execution_mode": (
            REPOSITORY_EXECUTION_MODE if repository else FUNCTION_EXECUTION_MODE
        ),
        "project": f"project-{index % 5}" if repository else "synthetic",
        "bug_family": f"family-{index % 4}",
        "source_name": f"source-{index % 3}",
    }


def test_candidate_corpus_version_is_safe_and_supported():
    assert valid_corpus_version("v3_final_candidate")
    assert valid_corpus_version("v3_1")
    assert not valid_corpus_version("../v3")
    assert not valid_corpus_version("v3/final")
    assert not valid_corpus_version("V3")


def test_bounded_smoke_selection_includes_both_sources_deterministically():
    pairs = [*[_pair(i) for i in range(200)], *[_pair(i + 200, True) for i in range(40)]]
    first = select_bounded_train_pairs(pairs, 64)
    second = select_bounded_train_pairs(pairs, 64)

    assert [pair["id"] for pair in first] == [pair["id"] for pair in second]
    assert len(first) == 64
    assert sum(pair["execution_mode"] == REPOSITORY_EXECUTION_MODE for pair in first) == 6
    assert sum(pair["execution_mode"] == FUNCTION_EXECUTION_MODE for pair in first) == 58
    summary = summarize_train_pair_selection(first)
    assert len(summary["repository_project_counts"]) == 5
    assert len(summary["bug_family_counts"]) == 4


def test_bounded_smoke_prefers_generation_compatible_repository_records():
    pairs = [
        *[_pair(i) for i in range(200)],
        *[_pair(i + 200, True) for i in range(40)],
    ]
    compatible = {f"pair-{index:04d}" for index in range(205, 240, 5)}

    selected = select_bounded_train_pairs(
        pairs, 64, compatible_repository_ids=compatible
    )
    selected_repository_ids = {
        pair["id"] for pair in selected
        if pair["execution_mode"] == REPOSITORY_EXECUTION_MODE
    }

    assert len(selected_repository_ids) == 6
    assert selected_repository_ids <= compatible


def test_bounded_smoke_filters_incompatible_synthetic_and_repository_records():
    pairs = [
        *[_pair(i) for i in range(200)],
        *[_pair(i + 200, True) for i in range(40)],
    ]
    compatible_synthetic = {f"pair-{index:04d}" for index in range(100)}
    compatible_repository = {f"pair-{index:04d}" for index in range(200, 220)}

    selected = select_bounded_train_pairs(
        pairs,
        64,
        compatible_repository_ids=compatible_repository,
        compatible_synthetic_ids=compatible_synthetic,
    )

    selected_ids = {pair["id"] for pair in selected}
    assert len(selected) == 64
    assert selected_ids <= compatible_synthetic | compatible_repository


def test_bounded_smoke_never_falls_back_to_incompatible_records():
    pairs = [
        *[_pair(i) for i in range(20)],
        *[_pair(i + 20, True) for i in range(5)],
    ]

    selected = select_bounded_train_pairs(
        pairs,
        10,
        compatible_repository_ids=set(),
        compatible_synthetic_ids={"pair-0000", "pair-0001"},
    )

    assert [pair["id"] for pair in selected] == ["pair-0000", "pair-0001"]


def test_bounded_dpo_smoke_preserves_full_sft_identity():
    assert sft_training_scope(100, False, None) == (
        "full_train_split:execution_mode=all:repository_completion_limit=2048:"
        "prompt_token_limit=512:repository_prompt_token_limit=1024:"
        "prompt_compaction=section_aware_ast_units_before_chat_v4_1:"
        "prompt_schema=oneiros_unified_test_generation_v2:"
        "prompt_information=full:output_instruction=self_contained:"
        "dataset_identity=source.upstream_then_source.name_v1:"
        "generation_completion_limit=128:repository_generation_completion_limit=1024"
    )
    assert sft_training_scope(100, True, None) == (
        "first_100_train_records:bounded_selection=stratified_generation_compatible_v3:"
        "execution_mode=all:repository_completion_limit=2048:prompt_token_limit=512:"
        "repository_prompt_token_limit=1024:"
        "prompt_compaction=section_aware_ast_units_before_chat_v4_1:"
        "prompt_schema=oneiros_unified_test_generation_v2:"
        "prompt_information=full:output_instruction=self_contained:"
        "dataset_identity=source.upstream_then_source.name_v1:"
        "generation_completion_limit=128:repository_generation_completion_limit=1024"
    )


class _BatchEncodingStub(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _GenerationTokenizerStub:
    padding_side = "right"
    pad_token_id = None
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert [message["role"] for message in messages] == ["system", "user"]
        return messages[-1]["content"]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [11, 12, 13]}

    def pad(self, features, padding=True, return_tensors="pt"):
        return _BatchEncodingStub({
            "input_ids": torch.tensor([item["input_ids"] for item in features]),
            "attention_mask": torch.tensor([
                item["attention_mask"] for item in features
            ]),
        })

    def decode(self, token_ids, skip_special_tokens=True):
        return "invalid" if int(token_ids[-1]) == 2 else "valid"


class _GenerationModelStub:
    device = "cpu"

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        prefix = input_ids[0]
        return torch.stack([
            torch.cat([prefix, torch.tensor([token])]) for token in (1, 2, 3, 4)
        ])


def test_generation_accounting_reports_unparseable_candidates():
    generator = SimpleNamespace(
        is_loaded=True,
        tokenizer=_GenerationTokenizerStub(),
        model=_GenerationModelStub(),
        temperature=0.7,
        top_p=0.9,
        _parse_output=lambda text, entry_point: SimpleNamespace(
            is_valid=text == "valid", input_code="assert f(0) == 1"
        ),
    )
    pair = {
        "id": "record-a",
        "execution_mode": FUNCTION_EXECUTION_MODE,
        "mutant_code": "def f(x): return x",
        "golden_code": "def f(x): return x + 1",
        "entry_point": "f",
        "specification": "",
    }

    generated, accounting = generate_tests_ai_batched(
        generator, [pair], num=4, return_accounting=True
    )

    assert len(generated[0]) == 3
    assert {
        key: value for key, value in accounting[0].items()
        if key != "candidate_slots"
    } == {
        "requested_candidates": 4,
        "raw_generated_sequences": 4,
        "parsed_candidates": 3,
        "generation_invalid_candidates": 1,
        # A promptable record must state that its prompt budget held, so a
        # budget failure can never be confused with an absent field.
        "prompt_budget_failure": False,
        "prompt_budget_failure_reason": None,
    }
    slots = accounting[0]["candidate_slots"]
    assert [slot["rank"] for slot in slots] == [1, 2, 3, 4]
    assert [slot["parse_valid"] for slot in slots] == [True, False, True, True]
    assert all(slot["raw_output_sha256"] for slot in slots)


def test_optimizer_padding_preserves_the_final_accumulation_window():
    unique_examples = 186
    samples_per_optimizer_step = 16
    padding_examples = (-unique_examples) % samples_per_optimizer_step

    assert padding_examples == 6
    assert (unique_examples + padding_examples) % samples_per_optimizer_step == 0


def test_sft_monitor_uses_five_checkpoint_patience_and_resets_on_improvement():
    baseline = {
        "checkpoint_step": 0,
        "function_validation_killed": 100,
        "function_kill_rate": 0.2,
    }
    trend = []
    for index, killed in enumerate((100, 100, 101, 101, 101, 101, 101, 101), start=1):
        evaluation = {
            "checkpoint_step": index * 100,
            "function_validation_killed": killed,
            "function_kill_rate": killed / 500,
        }
        decision = _sft_monitor_gate_decision(baseline, trend, evaluation, patience=5)
        trend.append(decision)

    assert trend[0]["consecutive_non_improving_checkpoints"] == 1
    assert trend[1]["consecutive_non_improving_checkpoints"] == 2
    assert trend[2]["improved"] is True
    assert trend[2]["consecutive_non_improving_checkpoints"] == 0
    assert trend[-2]["should_stop"] is False
    assert trend[-1]["should_stop"] is True
    assert trend[-1]["decision"] == "early_stop"


def _paired_monitor_evaluation(step: int, killed: int, candidate_rate: float = 0.25):
    return {
        "checkpoint_step": step,
        "function_validation_records": 100,
        "function_validation_killed": killed,
        "function_kill_rate": killed / 100,
        "candidate_kill_rate": candidate_rate,
        "end_to_end_candidate_kill_rate": candidate_rate,
        "parse_success_rate": 0.97,
        "function_results": [
            {"record_id": f"record-{index:03d}", "killed": index < killed}
            for index in range(100)
        ],
    }


def test_sft_monitor_rejects_a_one_function_noise_gain():
    baseline = _paired_monitor_evaluation(0, 53)
    evaluation = _paired_monitor_evaluation(50, 54)

    decision = _sft_monitor_gate_decision(baseline, [], evaluation, patience=5)

    assert decision["minimum_practical_function_gain"] == 2
    assert decision["function_gain_over_best"] == 1
    assert decision["improved"] is False
    assert decision["paired_function_diagnostics"]["net_function_gain"] == 1


def test_sft_monitor_requires_candidate_health_for_a_new_best():
    baseline = _paired_monitor_evaluation(0, 53, candidate_rate=0.25)
    healthy = _paired_monitor_evaluation(50, 55, candidate_rate=0.245)
    degraded = _paired_monitor_evaluation(50, 55, candidate_rate=0.23)

    healthy_decision = _sft_monitor_gate_decision(
        baseline, [], healthy, patience=5
    )
    degraded_decision = _sft_monitor_gate_decision(
        baseline, [], degraded, patience=5
    )

    assert healthy_decision["improved"] is True
    assert degraded_decision["improved"] is False
    assert degraded_decision["candidate_health_passed"] is False
    paired = _paired_function_diagnostics(baseline, healthy)
    assert paired["improved_functions"] == 2
    assert paired["regressed_functions"] == 0


def test_sft_monitor_acceptance_enforces_the_explicit_kill_rate_gate():
    assert not sft_monitor_acceptance_passed(
        True,
        False,
        "checkpoint-143/sft_adapter",
        {"function_kill_rate": 0.56},
        0.58,
    )
    assert sft_monitor_acceptance_passed(
        True,
        False,
        "checkpoint-143/sft_adapter",
        {"function_kill_rate": 0.60},
        0.58,
    )
    assert not sft_monitor_acceptance_passed(
        True,
        False,
        None,
        {"function_kill_rate": 0.60},
        0.58,
    )


def test_sft_monitor_acceptance_validates_threshold_and_disabled_runs():
    assert sft_monitor_acceptance_passed(False, False, None, None, 0.58)
    assert not sft_monitor_acceptance_passed(False, True, None, None, 0.58)
    with pytest.raises(ValueError, match="acceptance rate"):
        sft_monitor_acceptance_passed(False, False, None, None, 1.01)


def test_balanced_sampler_preserves_every_example_and_caps_repetition():
    examples = [
        *[
            SFTDataPoint("p", f"d-{index}", f"d-{index}", project="django")
            for index in range(4)
        ],
        SFTDataPoint("p", "f-0", "f-0", project="flask"),
        *[
            SFTDataPoint("p", f"s-{index}", f"s-{index}", project="sympy")
            for index in range(2)
        ],
    ]

    first, stats = balanced_repeat_examples(examples, 14, 3, "project")
    second, second_stats = balanced_repeat_examples(examples, 14, 3, "project")

    assert [item.function_id for item in first] == [item.function_id for item in second]
    assert stats == second_stats
    assert first[: len(examples)] == examples
    assert len(first) == 14
    assert all(first.count(example) <= 3 for example in examples)
    assert stats["raw_group_counts"] == {"django": 4, "flask": 1, "sympy": 2}
    assert stats["effective_group_counts"]["flask"] > stats["raw_group_counts"]["flask"]
    assert stats["target_reached"] is True


def test_balanced_sampler_reports_an_unreachable_target_truthfully():
    examples = [SFTDataPoint("p", "c", "record", project="only-project")]

    retained, stats = balanced_repeat_examples(examples, 10, 3, "project")

    assert len(retained) == 3
    assert stats["target_examples"] == 10
    assert stats["achievable_max_examples"] == 3
    assert stats["target_reached"] is False


def test_contingency_deduplication_only_removes_exact_training_sequences():
    first = SFTDataPoint("same prompt", "assert f(0) == 1", "record-a")
    duplicate = SFTDataPoint("same prompt", "assert f(0) == 1\n", "record-b")
    distinct = SFTDataPoint("same prompt", "assert f(1) == 2", "record-c")

    retained, stats = deduplicate_sft_examples([first, duplicate, distinct])

    assert retained == [first, distinct]
    assert stats == {
        "input_examples": 3,
        "retained_examples": 2,
        "exact_duplicates_excluded": 1,
    }


def test_old_run_config_gets_only_inactive_sampler_defaults():
    older = {"epochs": 3, "learning_rate": 0.0002}
    current = {
        **older,
        "balanced_sampling_enabled": False,
        "synthetic_balance_fraction": 0.0,
        "synthetic_balance_mode": "none",
        "max_synthetic_repeats": 2,
        "lr_scheduler_type": "cosine",
        "min_function_kill_rate": 0.50,
    }

    assert normalized_sft_run_hyperparameters(older) == current
    assert normalized_sft_run_hyperparameters(older) == normalized_sft_run_hyperparameters(current)


class _WhitespaceTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.replace(self.eos_token, f" {self.eos_token}").split()}


def test_sft_generation_gate_preserves_sources_and_excludes_unemittable_targets():
    tokenizer = _WhitespaceTokenizer()
    short = SFTDataPoint(
        "prompt", "assert f(0) == 1", "short", execution_mode=FUNCTION_EXECUTION_MODE,
    )
    long = SFTDataPoint(
        "prompt", " ".join(["token"] * 129), "long",
        project="django", execution_mode=REPOSITORY_EXECUTION_MODE,
    )
    original = [short, long]

    retained, excluded = filter_generation_compatible_sft_examples(
        original, tokenizer, max_completion_tokens=128,
    )

    assert original == [short, long]
    assert retained == [short]
    assert excluded == [{
        "record_id": "long",
        "execution_mode": REPOSITORY_EXECUTION_MODE,
        "project": "django",
        "bug_family": "unknown",
        "completion_tokens": 130,
        "limit_tokens": 128,
        "reason": "completion_exceeds_live_generation_limit",
    }]


def test_sft_generation_gate_retains_long_verified_repository_fragment_only():
    tokenizer = _WhitespaceTokenizer()
    function_example = SFTDataPoint(
        "prompt", " ".join(["token"] * 200), "function-long",
        execution_mode=FUNCTION_EXECUTION_MODE,
    )
    repository_example = SFTDataPoint(
        "prompt", " ".join(["token"] * 200), "repository-long",
        project="flask", execution_mode=REPOSITORY_EXECUTION_MODE,
    )

    retained, excluded = filter_generation_compatible_sft_examples(
        [function_example, repository_example],
        tokenizer,
        max_completion_tokens=128,
        max_repository_completion_tokens=1024,
    )

    assert retained == [repository_example]
    assert excluded[0]["record_id"] == "function-long"
    assert excluded[0]["limit_tokens"] == 128


def test_repository_dpo_loser_is_token_matched_and_below_context_gate():
    tokenizer = _WhitespaceTokenizer()
    winner = " ".join(["verified"] * 300)

    loser = _length_matched_repository_loser(winner, tokenizer)
    loser_tokens = len(tokenizer(loser)["input_ids"])

    compile(loser, "<test-loser>", "exec")
    assert loser_tokens <= 300
    assert loser_tokens >= 290
    assert loser_tokens < MAX_DPO_COMPLETION_TOKENS


def test_repository_dpo_loser_rejects_an_overlong_winner():
    tokenizer = _WhitespaceTokenizer()
    winner = " ".join(["verified"] * MAX_DPO_COMPLETION_TOKENS)

    with pytest.raises(ValueError, match="winner exceeds"):
        _length_matched_repository_loser(winner, tokenizer)


def test_dpo_dataset_preflight_refuses_silent_completion_truncation():
    trainer = object.__new__(DPOTrainer)
    trainer.tokenizer = _WhitespaceTokenizer()
    trainer.last_context_audit = None
    overlong = " ".join(["verified"] * MAX_DPO_COMPLETION_TOKENS)

    with pytest.raises(RuntimeError, match="completion context gate"):
        trainer.prepare_dataset([
            DPODataPoint(
                prompt="short prompt",
                chosen=overlong,
                rejected="assert True",
                function_id="record-overlong",
            )
        ])


def test_dpo_training_scope_hash_covers_order_and_exact_completions():
    first = [{**_pair(1), "test_cases": ["assert f(0) == 1"]}]
    changed = [{**_pair(1), "test_cases": ["assert f(1) == 2"]}]

    assert _dpo_training_scope_sha256(first) == _dpo_training_scope_sha256(first)
    assert _dpo_training_scope_sha256(first) != _dpo_training_scope_sha256(changed)


def test_sft_baseline_must_cover_the_exact_locked_validation_scope(tmp_path):
    baseline = {
        "dataset_fingerprint": "corpus-fingerprint",
        "adapter": "sft_adapter",
        "evaluation_split": "val",
        "final_test_measurement": False,
        "seed": 42,
        "tests_per_function": 8,
        "function_validation_records": 2,
        "function_kill_rate": 0.5,
        "candidate_kill_rate": 0.25,
        "repository_validation_records_held": 1,
    }
    (tmp_path / sft_validation_results_filename(42)).write_text(
        json.dumps(baseline), encoding="utf-8"
    )

    loaded = _load_sft_validation_baseline(
        tmp_path,
        "corpus-fingerprint",
        expected_function_ids=["record-a", "record-b"],
        expected_repository_records=1,
    )
    assert loaded == baseline

    baseline["function_validation_records"] = 1
    (tmp_path / sft_validation_results_filename(42)).write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="complete locked function scope"):
        _load_sft_validation_baseline(
            tmp_path,
            "corpus-fingerprint",
            expected_function_ids=["record-a", "record-b"],
            expected_repository_records=1,
        )


def test_validation_progress_restores_exact_rng_and_rejects_identity_changes(
    tmp_path, monkeypatch,
):
    from scripts import train_on_dataset

    monkeypatch.setattr(train_on_dataset, "RESULTS_DIR", tmp_path)
    context = _adapter_evaluation_context(
        "corpus-fingerprint",
        "sft_adapter",
        "adapter-sha256",
        "val",
        None,
        "selection-sha256",
        4,
    )
    random.seed(123)
    torch.manual_seed(123)
    saved = _save_adapter_evaluation_progress(
        "sft_validation_results", context, 2, 1, 16, 5, 10.5,
        requested_candidates=16,
        parsed_candidates=15,
        generation_invalid_candidates=1,
        function_results=[{"record_id": "record-a", "killed": True}],
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)

    random.seed(999)
    torch.manual_seed(999)
    progress = _load_adapter_evaluation_progress("sft_validation_results", context)
    _restore_adapter_evaluation_rng(progress)

    assert saved.exists()
    assert progress["completed_functions"] == 2
    assert progress["requested_candidates"] == 16
    assert progress["generation_invalid_candidates"] == 1
    assert progress["function_results"][0]["record_id"] == "record-a"
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)

    changed_context = {**context, "adapter_sha256": "different-adapter"}
    assert _load_adapter_evaluation_progress(
        "sft_validation_results", changed_context
    ) is None


def test_validation_progress_accepts_an_odd_final_batch(tmp_path, monkeypatch):
    from scripts import train_on_dataset

    monkeypatch.setattr(train_on_dataset, "RESULTS_DIR", tmp_path)
    context = _adapter_evaluation_context(
        "corpus-fingerprint",
        "sft_adapter",
        "adapter-sha256",
        "val",
        None,
        "selection-sha256",
        3,
    )
    _save_adapter_evaluation_progress(
        "sft_validation_results", context, 3, 2, 24, 8, 11.0
    )

    progress = _load_adapter_evaluation_progress("sft_validation_results", context)
    assert progress["completed_functions"] == 3
