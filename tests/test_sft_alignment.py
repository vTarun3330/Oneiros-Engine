import pytest

from config import training_config
from engine.sft_trainer import (
    OneirosSFTTrainer,
    SFTDataPoint,
    plan_sft_optimizer_schedule,
)
import engine.sft_trainer as sft_module
from engine.test_generation_prompt import build_unified_user_prompt


class _TokenLengthTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        assert [message["role"] for message in messages] == ["system", "user"]
        return messages[-1]["content"]

    def __call__(self, text, add_special_tokens=False):
        normalized = text.replace(self.eos_token, f" {self.eos_token}")
        return {"input_ids": normalized.split()}


def _dataset_only_trainer(
    prompt_limit=5, repository_prompt_limit=7, completion_limit=4,
    repository_completion_limit=8,
):
    trainer = object.__new__(OneirosSFTTrainer)
    trainer.tokenizer = _TokenLengthTokenizer()
    trainer.max_prompt_tokens = prompt_limit
    trainer.max_repository_prompt_tokens = repository_prompt_limit
    trainer.max_completion_tokens = completion_limit
    trainer.max_repository_completion_tokens = repository_completion_limit
    trainer.dataset_stats = {}
    return trainer


class _DatasetStub(list):
    @classmethod
    def from_dict(cls, columns):
        return cls(
            {name: values[index] for name, values in columns.items()}
            for index in range(len(columns["input_ids"]))
        )


def test_sft_prompt_is_capped_to_the_live_inference_budget(monkeypatch):
    monkeypatch.setattr(sft_module, "Dataset", _DatasetStub, raising=False)
    trainer = _dataset_only_trainer(prompt_limit=120, completion_limit=4)
    prompt = build_unified_user_prompt(
        code_under_test="def works(x):\n    return x",
        execution_mode="function_assertion",
        specification="Return the supplied value.",
        support_context="\n".join(f"def helper_{i}(): return {i}" for i in range(80)),
        target_symbols=["works"],
    )
    dataset = trainer.prepare_dataset([
        SFTDataPoint(
            prompt=prompt,
            completion="assert works",
            function_id="record-a",
        )
    ])

    assert dataset[0]["completion_start"] <= 120
    assert trainer.dataset_stats["prompt_truncated_examples"] == 1
    assert trainer.dataset_stats["max_observed_prompt_tokens"] > 120
    assert trainer.dataset_stats["max_observed_completion_tokens"] == 3


def test_sft_preflight_fails_before_training_on_an_unemittable_completion():
    trainer = _dataset_only_trainer(prompt_limit=5, completion_limit=3)

    with pytest.raises(ValueError, match="generation-compatibility preflight"):
        trainer.prepare_dataset([
            SFTDataPoint(
                prompt=build_unified_user_prompt(
                    code_under_test="def f(): return 1",
                    execution_mode="function_assertion",
                    target_symbols=["f"],
                ),
                completion="one two three four",
                function_id="record-overlong",
            )
        ])


def test_sft_repository_fragment_uses_its_separate_completion_budget(monkeypatch):
    monkeypatch.setattr(sft_module, "Dataset", _DatasetStub, raising=False)
    trainer = _dataset_only_trainer(
        prompt_limit=256, repository_prompt_limit=256,
        completion_limit=3, repository_completion_limit=8,
    )

    dataset = trainer.prepare_dataset([
        SFTDataPoint(
            prompt=build_unified_user_prompt(
                code_under_test="def f(): return 1",
                execution_mode="repository_pytest_fragment",
                target_symbols=["f"],
            ),
            completion="one two three four five",
            function_id="repository-record",
            execution_mode="repository_pytest_fragment",
        )
    ])

    assert len(dataset) == 1
    assert trainer.dataset_stats["max_observed_completion_tokens"] == 6


def test_sft_repository_fragment_uses_its_larger_prompt_budget(monkeypatch):
    monkeypatch.setattr(sft_module, "Dataset", _DatasetStub, raising=False)
    trainer = _dataset_only_trainer(
        prompt_limit=80,
        repository_prompt_limit=160,
        completion_limit=3,
        repository_completion_limit=8,
    )

    dataset = trainer.prepare_dataset([
        SFTDataPoint(
            prompt=build_unified_user_prompt(
                code_under_test="def f(): return 1",
                execution_mode="repository_pytest_fragment",
                specification="Return one.",
                target_symbols=["f"],
            ),
            completion="assert works",
            function_id="repository-record",
            execution_mode="repository_pytest_fragment",
        )
    ])

    assert 80 < dataset[0]["completion_start"] <= 160


def test_safer_v3_sft_defaults_bound_optimizer_drift():
    assert training_config.sft_learning_rate == 5e-5
    assert training_config.sft_epochs == 1
    assert training_config.sft_warmup_steps == 25
    assert training_config.sft_checkpoint_steps == 50
    assert training_config.sft_lr_scheduler_type == "cosine"
    assert training_config.sft_min_function_kill_rate == 0.58
    assert training_config.sft_min_monitor_checkpoints == 2
    assert training_config.sft_prompt_token_limit == 512
    assert training_config.sft_repository_prompt_token_limit == 1024
    assert training_config.sft_completion_token_limit == 128
    assert training_config.sft_repository_completion_token_limit == 1024


def test_optimizer_preflight_reports_effective_warmup_and_checkpoint_horizon():
    underpowered = plan_sft_optimizer_schedule(162, 1, 1, 25, 50)
    powered = plan_sft_optimizer_schedule(1600, 1, 1, 25, 50)

    assert underpowered["planned_optimizer_steps"] == 11
    assert underpowered["effective_warmup_steps"] == 10
    assert underpowered["effective_checkpoint_steps"] == 11
    assert powered["planned_optimizer_steps"] == 100
    assert powered["effective_warmup_steps"] == 25
    assert powered["effective_checkpoint_steps"] == 50


def test_sft_scheduler_is_explicit_and_rejects_unknown_values(monkeypatch, tmp_path):
    monkeypatch.setattr(sft_module, "SFT_AVAILABLE", True)
    monkeypatch.setattr(sft_module, "PEFT_AVAILABLE", True)

    trainer = OneirosSFTTrainer(
        output_dir=tmp_path,
        lr_scheduler_type="constant_with_warmup",
    )
    assert trainer.lr_scheduler_type == "constant_with_warmup"

    with pytest.raises(ValueError, match="LR scheduler"):
        OneirosSFTTrainer(output_dir=tmp_path, lr_scheduler_type="adaptive_magic")
