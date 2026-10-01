"""Protocol v2.5 SFT trainer for ``pytest_module_v1`` supervision (arm C).

A subclass of the historical ``OneirosSFTTrainer``. The historical module is NOT modified: its
hash is bound by earlier tracked design artifacts. Everything held constant by addendum 1
(base model and revision, LoRA, 4-bit setup, learning rate, scheduler, warmup, accumulation 16,
3,072 sequence, completion-only collator) is inherited; only the data contract and the
exposure plan differ:

- completion: the complete verified module, byte for byte (never stripped), every completion
  token and the EOS supervised; the prompt (rendered with the inference chat template) is
  masked from the loss;
- prompt: the frozen v2.5 prompt, never compacted or truncated; an over-budget row refuses
  the whole dataset;
- the dataset is homogeneous ``pytest_module_v1``;
- exposure: an explicit, pre-computed plan (addendum 2) fixes epochs and optimizer steps; no
  duplicated row inside an epoch, no cycling beyond the plan.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional

from engine.sft_trainer import (CompletionOnlyDataCollator, MAX_SFT_SEQUENCE_LENGTH,
                                OneirosSFTTrainer, SFT_GRADIENT_ACCUMULATION_STEPS,
                                SFTCheckpointMonitorCallback, SFTDataPoint, _sequence_sha256)
from engine.test_generation_prompt import format_chat_prompt

PYTEST_MODULE_TASK_KIND = "pytest_module_v1"
TRAINER_VERSION = "oneiros_sft_trainer_v25_pytest_module_v1"


class V25SFTTrainer(OneirosSFTTrainer):
    """Arm C trainer: exact v2.5 prompt/completion pairs, explicit exposure plan."""

    def prepare_dataset(self, data_points: List[SFTDataPoint]):
        from datasets import Dataset
        if not data_points:
            raise ValueError("v2.5 SFT dataset is empty")
        input_ids, masks, starts, refused = [], [], [], []
        max_prompt = max_completion = supervised = 0
        for dp in data_points:
            if dp.task_kind != PYTEST_MODULE_TASK_KIND:
                raise ValueError("the v2.5 trainer accepts only pytest_module_v1 rows; it "
                                 "cannot mix task kinds")
            if not dp.completion.strip() or dp.completion != dp.completion.lstrip():
                raise ValueError("pytest_module_v1 completion is empty or starts with "
                                 "whitespace; it must be the verified module byte for byte")
            completion_ids = list(self.tokenizer(dp.completion + self.tokenizer.eos_token,
                                                 add_special_tokens=False)["input_ids"])
            prompt_ids = list(self.tokenizer(format_chat_prompt(self.tokenizer, dp.prompt),
                                             add_special_tokens=False)["input_ids"])
            max_prompt = max(max_prompt, len(prompt_ids))
            max_completion = max(max_completion, len(completion_ids))
            reasons = []
            if len(prompt_ids) > self.max_prompt_tokens:
                reasons.append(f"prompt {len(prompt_ids)} > {self.max_prompt_tokens}; never "
                               "compacted")
            if len(completion_ids) > self.max_completion_tokens:
                reasons.append(f"completion {len(completion_ids)} > "
                               f"{self.max_completion_tokens}; never truncated")
            if len(prompt_ids) + len(completion_ids) > MAX_SFT_SEQUENCE_LENGTH:
                reasons.append("sequence exceeds 3,072")
            if reasons:
                refused.append(f"{dp.function_id}: {reasons}")
                continue
            input_ids.append(prompt_ids + completion_ids)
            masks.append([1] * (len(prompt_ids) + len(completion_ids)))
            starts.append(len(prompt_ids))
            supervised += len(completion_ids)
        if refused:
            raise ValueError(f"v2.5 SFT refused {len(refused)} row(s) over budget: "
                             f"{refused[:3]}")
        self.dataset_stats = {
            "trainer_version": TRAINER_VERSION, "input_examples": len(data_points),
            "retained_examples": len(input_ids), "max_observed_prompt_tokens": max_prompt,
            "max_observed_completion_tokens": max_completion,
            "supervised_target_tokens_per_epoch": supervised,
            "prompt_truncated_examples": 0, "prompt_compacted_examples": 0,
            "input_ids_sha256": _sequence_sha256(input_ids),
            "task_kind_counts": {PYTEST_MODULE_TASK_KIND: len(input_ids)}}
        return Dataset.from_dict({"input_ids": input_ids, "attention_mask": masks,
                                  "completion_start": starts})

    def _sft_config(self, plan: Mapping[str, Any], batch_size: int = 1,
                    use_cpu: bool = False):
        from trl import SFTConfig
        return SFTConfig(
            output_dir=str(self.output_dir / "sft_tmp"),
            num_train_epochs=plan["epochs"], max_steps=plan["optimizer_steps"],
            per_device_train_batch_size=batch_size, learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm, weight_decay=self.weight_decay,
            logging_steps=10, save_steps=min(self.checkpoint_steps, plan["optimizer_steps"]),
            warmup_steps=min(self.warmup_steps, max(0, plan["optimizer_steps"] - 1)),
            lr_scheduler_type=self.lr_scheduler_type,
            fp16=not self.use_bf16 and not use_cpu, bf16=self.use_bf16 and not use_cpu,
            tf32=None if use_cpu else self.use_bf16,
            gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION_STEPS,
            gradient_checkpointing=True, save_total_limit=2, report_to="none",
            max_seq_length=MAX_SFT_SEQUENCE_LENGTH,
            dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
            seed=42, **({"use_cpu": True} if use_cpu else {}))

    def _sft_trainer(self, dataset, training_args, monitor_callback=None):
        from trl import SFTTrainer
        return SFTTrainer(model=self.model, args=training_args, train_dataset=dataset,
                          processing_class=self.tokenizer,
                          data_collator=CompletionOnlyDataCollator(self.tokenizer),
                          callbacks=[monitor_callback] if monitor_callback else None)

    def train(self, data_points: List[SFTDataPoint], *, plan: Mapping[str, Any],
              batch_size: int = 1,
              checkpoint_monitor: Optional[Callable] = None) -> Dict[str, Any]:
        """Train exactly ``plan`` (addendum 2); refuses a plan that does not match the data."""
        if self.model is None:
            self.setup_model()
        dataset = self.prepare_dataset(data_points)
        if plan.get("unique_examples") != len(dataset):
            raise ValueError("exposure plan was computed for a different dataset")
        per_epoch = math.ceil(len(dataset) / (batch_size * SFT_GRADIENT_ACCUMULATION_STEPS))
        if plan["optimizer_steps"] > plan["epochs"] * per_epoch:
            raise ValueError("plan would cycle the data beyond its frozen epochs")
        monitor = (SFTCheckpointMonitorCallback(checkpoint_monitor, self.tokenizer,
                                                plan["optimizer_steps"])
                   if checkpoint_monitor else None)
        trainer = self._sft_trainer(dataset, self._sft_config(plan, batch_size), monitor)
        result = trainer.train(resume_from_checkpoint=str(self.resume_checkpoint)
                               if self.resume_checkpoint else None)
        loss = float(getattr(result, "training_loss", float("nan")))
        if not math.isfinite(loss):
            raise FloatingPointError(f"SFT produced a non-finite loss: {loss}")
        return {"loss": loss, "plan": dict(plan),
                "completed_optimizer_steps": int(trainer.state.global_step),
                "completed_epochs": float(trainer.state.epoch or 0.0), **self.dataset_stats}


HISTORICAL_TARGET_TOKENS = 570_775        # baseline adapter, one epoch (addendum 2 section 1)
MAX_UPDATES = 431
MAX_PASSES = 3


def plan_v25_exposure(unique_examples: int, target_tokens_per_pass: int,
                      historical_target_tokens: int = HISTORICAL_TARGET_TOKENS,
                      max_updates: int = MAX_UPDATES, max_passes: int = MAX_PASSES,
                      batch: int = SFT_GRADIENT_ACCUMULATION_STEPS) -> Dict[str, Any]:
    """Addendum 2 section 2: passes E = min(3, floor(T_hist / T)); updates U = min(431, E*S)."""
    if unique_examples <= 0 or target_tokens_per_pass <= 0:
        raise ValueError("empty corpus")
    passes = min(max_passes, historical_target_tokens // target_tokens_per_pass)
    if passes < 1:
        raise ValueError("one pass exceeds the historical supervised-token budget")
    per_pass = math.ceil(unique_examples / batch)
    updates = min(max_updates, passes * per_pass)
    return {"rule": "oneiros_v25_addendum2_exposure_v1", "unique_examples": unique_examples,
            "updates_per_pass": per_pass, "target_tokens_per_pass": target_tokens_per_pass,
            "epochs": passes, "optimizer_steps": updates,
            "total_target_tokens_upper_bound": passes * target_tokens_per_pass,
            "warmup_steps": min(25, max(0, updates - 1)),
            "max_exposure_per_example": passes, "duplicates_within_epoch": 0,
            "checkpoint": "final update; no monitor; no confirmation/validation selection"}
