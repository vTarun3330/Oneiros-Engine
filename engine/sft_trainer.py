"""
SFT (Supervised Fine-Tuning) Trainer for Oneiros Engine.

Teaches the model what good mutation-killing tests look like BEFORE
DPO preference optimization. Uses existing dataset test_cases as
ground-truth examples.
"""

from __future__ import annotations

import warnings
import logging
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer.*")
warnings.filterwarnings("ignore", message=".*processing_class.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import gc
import math
import json
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Any
from dataclasses import dataclass

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    SFT_AVAILABLE = True
except ImportError:
    SFT_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import model_config, training_config
from engine.model_runtime import (
    build_4bit_quantization_config,
    runtime_profile,
)
from engine.prompt_budget import (
    PROMPT_COMPACTION_STRATEGY,
    PromptBudgetError,
    compact_unified_user_prompt,
)
from engine.test_generation_prompt import format_chat_prompt


MAX_SFT_SEQUENCE_LENGTH = 2048
SFT_GRADIENT_ACCUMULATION_STEPS = 16


def sft_completion_limit_for_execution_mode(
    execution_mode: str,
    function_completion_limit: int,
    repository_completion_limit: int,
) -> int:
    """Return the immutable SFT output budget for one supervision mode."""
    if str(execution_mode).startswith("repository_"):
        return repository_completion_limit
    return function_completion_limit


def sft_prompt_limit_for_execution_mode(
    execution_mode: str,
    function_prompt_limit: int,
    repository_prompt_limit: int,
) -> int:
    """Return the mode-specific prompt budget inside one unified schema."""
    if str(execution_mode).startswith("repository_"):
        return repository_prompt_limit
    return function_prompt_limit


def plan_sft_optimizer_schedule(
    unique_examples: int,
    num_epochs: int,
    batch_size: int,
    warmup_steps: int,
    checkpoint_steps: int,
    gradient_accumulation_steps: int = SFT_GRADIENT_ACCUMULATION_STEPS,
) -> Dict[str, int]:
    """Calculate the effective schedule before model/GPU initialization."""
    if unique_examples <= 0 or num_epochs <= 0 or batch_size <= 0:
        raise ValueError("SFT examples, epochs, and batch size must be positive")
    if warmup_steps < 0 or checkpoint_steps <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("SFT warmup, checkpoint, and accumulation settings are invalid")
    samples_per_optimizer_step = batch_size * gradient_accumulation_steps
    padding_examples = (-unique_examples) % samples_per_optimizer_step
    optimizer_examples = unique_examples + padding_examples
    planned_optimizer_steps = max(
        1, optimizer_examples * num_epochs // samples_per_optimizer_step
    )
    return {
        "unique_examples": unique_examples,
        "optimizer_examples": optimizer_examples,
        "optimizer_padding_examples": padding_examples,
        "samples_per_optimizer_step": samples_per_optimizer_step,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "planned_optimizer_steps": planned_optimizer_steps,
        "effective_checkpoint_steps": min(checkpoint_steps, planned_optimizer_steps),
        "effective_warmup_steps": min(warmup_steps, max(0, planned_optimizer_steps - 1)),
    }


@dataclass
class SFTDataPoint:
    """A single SFT training example: prompt → expected test output."""
    prompt: str
    completion: str
    function_id: str = ""
    project: str = "synthetic"
    bug_family: str = "unknown"
    semantic_group: str = "unknown"
    execution_mode: str = "function_assertion"
    dataset: str = "unknown"
    dataset_family: str = "unknown::unknown"




class CompletionOnlyDataCollator:
    """Pad pre-tokenized SFT records and mask all prompt tokens from loss."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_features = [
            {
                "input_ids": list(feature["input_ids"]),
                "attention_mask": list(feature["attention_mask"]),
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(batch_features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row, feature in enumerate(features):
            labels[row, : int(feature["completion_start"])] = -100
        batch["labels"] = labels
        return batch


class SFTCheckpointMonitorCallback(TrainerCallback):
    """Run a deterministic validation monitor after each persisted checkpoint."""

    def __init__(self, monitor: Callable, tokenizer, planned_steps: int):
        self.monitor = monitor
        self.tokenizer = tokenizer
        self.planned_steps = planned_steps
        self.completed_checkpoint_metrics: Dict[int, Dict[str, Any]] = {
            int(step): dict(metrics)
            for step, metrics in getattr(
                monitor, "completed_checkpoint_metrics", {}
            ).items()
        }
        self.history: List[Dict[str, Any]] = [
            dict(self.completed_checkpoint_metrics[step])
            for step in sorted(self.completed_checkpoint_metrics)
        ]
        self.stopped_early = False
        self.best_adapter_path: str | None = getattr(
            monitor, "initial_best_adapter_path", None
        )
        self.best_metrics: Dict[str, Any] | None = getattr(
            monitor, "initial_best_metrics", None
        )

    def _monitor_checkpoint(self, args, state, control, model):
        step = int(state.global_step)
        if step <= 0 or step in self.completed_checkpoint_metrics:
            return control
        metrics = dict(self.monitor(step, model, self.tokenizer))
        metrics.setdefault("checkpoint_step", step)
        if metrics.get("improved", False):
            # Persist every newly best validation adapter outside Trainer's
            # rolling checkpoint retention.  A later patience stop can then
            # select the genuinely best policy instead of the terminal one.
            best_dir = Path(args.output_dir).parent / "sft_validation_best" / f"checkpoint-{step}"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            self.tokenizer.save_pretrained(str(best_dir))
            best_dir.joinpath("validation_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            self.best_adapter_path = str(best_dir)
            self.best_metrics = dict(metrics)
            metrics["best_adapter_path"] = self.best_adapter_path
        self.completed_checkpoint_metrics[step] = dict(metrics)
        self.history.append(metrics)
        if step < self.planned_steps and metrics.get("should_stop", False):
            self.stopped_early = True
            control.should_training_stop = True
        return control

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Backfill validation when a saved checkpoint outlived its evaluation.

        A cancellation can happen after Trainer atomically saves a checkpoint
        but while the comparatively long kill-rate panel is running. Trainer
        resumes *after* that checkpoint and would never emit ``on_save`` for it
        again, so validate the restored model before the first new optimizer
        step. Completed checkpoint evaluations are reused without rerunning.
        """
        step = int(state.global_step)
        if step <= 0:
            return control
        existing = self.completed_checkpoint_metrics.get(step)
        if existing is not None:
            if step < self.planned_steps and existing.get("should_stop", False):
                self.stopped_early = True
                control.should_training_stop = True
            return control
        print(
            f"[SFT MONITOR] Recovering pending validation for resumed checkpoint {step}",
            flush=True,
        )
        return self._monitor_checkpoint(args, state, control, model)

    def on_save(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step in self.completed_checkpoint_metrics:
            return control
        return self._monitor_checkpoint(args, state, control, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Evaluate the actual terminal optimizer step exactly once.

        Hugging Face only emits ``on_save`` at the configured save interval.
        Consequently, a run ending at step 143 with ``save_steps=50`` would
        otherwise validate 50 and 100 but silently omit 143.  The completed
        metrics map is also restored on resume, so a terminal checkpoint that
        was already evaluated is never executed twice.
        """
        step = int(state.global_step)
        if step <= 0 or step in self.completed_checkpoint_metrics:
            return control
        print(f"[SFT MONITOR] Evaluating terminal optimizer step {step}", flush=True)
        return self._monitor_checkpoint(args, state, control, model)

class OneirosSFTTrainer:
    """
    SFT Trainer for teaching Phi-3 the format and content of
    mutation-killing test assertions.

    Phase 1 of the training pipeline: SFT → DPO
    """

    def __init__(
        self,
        model_name: str = None,
        output_dir: Path = None,
        learning_rate: float = None,
        max_prompt_tokens: int = None,
        max_repository_prompt_tokens: int = None,
        max_completion_tokens: int = None,
        max_repository_completion_tokens: int = None,
        warmup_steps: int = None,
        checkpoint_steps: int = None,
        lr_scheduler_type: str = None,
        model_revision: str = None,
        attention_implementation: str = None,
    ):
        if not SFT_AVAILABLE:
            raise ImportError("trl SFTTrainer required. Install with: pip install trl")
        if not PEFT_AVAILABLE:
            raise ImportError("peft required. Install with: pip install peft")

        self.model_name = model_name or model_config.model_name
        if model_revision is not None:
            self.model_revision = model_revision
        elif self.model_name == model_config.model_name:
            self.model_revision = model_config.model_revision
        else:
            # A non-canonical base model with no explicit pin has no frozen
            # snapshot to fall back to; resolve "main" and let the caller
            # record the actual downloaded snapshot hash for provenance.
            self.model_revision = "main"
        self.attention_implementation = (
            attention_implementation or model_config.attention_implementation
        )
        self.output_dir = Path(output_dir or training_config.checkpoint_dir)
        self.learning_rate = learning_rate or training_config.sft_learning_rate
        self.max_grad_norm = training_config.max_grad_norm
        self.max_prompt_tokens = (
            max_prompt_tokens or training_config.sft_prompt_token_limit
        )
        self.max_repository_prompt_tokens = (
            max_repository_prompt_tokens
            or training_config.sft_repository_prompt_token_limit
        )
        self.max_completion_tokens = (
            max_completion_tokens or training_config.sft_completion_token_limit
        )
        self.max_repository_completion_tokens = (
            max_repository_completion_tokens
            or training_config.sft_repository_completion_token_limit
        )
        self.warmup_steps = (
            training_config.sft_warmup_steps
            if warmup_steps is None else warmup_steps
        )
        self.checkpoint_steps = (
            training_config.sft_checkpoint_steps
            if checkpoint_steps is None else checkpoint_steps
        )
        self.lr_scheduler_type = (
            training_config.sft_lr_scheduler_type
            if lr_scheduler_type is None else lr_scheduler_type
        )
        if self.lr_scheduler_type not in {"cosine", "constant_with_warmup"}:
            raise ValueError(
                "SFT LR scheduler must be cosine or constant_with_warmup"
            )
        if (
            self.max_prompt_tokens <= 0
            or self.max_repository_prompt_tokens <= 0
            or self.max_completion_tokens <= 0
            or self.max_repository_completion_tokens <= 0
        ):
            raise ValueError("SFT prompt and completion token limits must be positive")
        if self.max_repository_completion_tokens >= MAX_SFT_SEQUENCE_LENGTH:
            raise ValueError(
                "SFT repository completion limit must leave room in the sequence"
            )
        if self.warmup_steps < 0 or self.checkpoint_steps <= 0:
            raise ValueError("SFT warmup/checkpoint steps are invalid")

        self.model = None
        self.tokenizer = None
        self.dataset_stats = {
            "input_examples": 0,
            "retained_examples": 0,
            "dropped_overlong_examples": 0,
            "prompt_compacted_examples": 0,
            "prompt_truncated_examples": 0,
            "max_observed_prompt_tokens": 0,
            "max_observed_completion_tokens": 0,
            "prompt_compaction_strategy": PROMPT_COMPACTION_STRATEGY,
            "malformed_prompt_examples": 0,
            "support_units_dropped": 0,
            "code_units_dropped": 0,
        }
        self.resume_checkpoint = None

    def _latest_checkpoint(self):
        """Return the newest complete Trainer checkpoint, if an SFT run stopped."""
        checkpoint_root = self.output_dir / "sft_tmp"
        candidates = []
        for candidate in checkpoint_root.glob("checkpoint-*"):
            try:
                step = int(candidate.name.rsplit("-", 1)[1])
            except ValueError:
                continue
            if candidate.joinpath("trainer_state.json").exists():
                candidates.append((step, candidate))
        return max(candidates, default=(None, None))[1]

    def setup_model(self) -> None:
        """Load and prepare model for SFT training."""
        print(f"Setting up {self.model_name} for SFT training...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config, compute_dtype, dtype_name = build_4bit_quantization_config(
            torch, BitsAndBytesConfig
        )
        self.use_bf16 = dtype_name == "bf16"
        self.runtime_profile = runtime_profile(dtype_name)
        self.runtime_profile["attention_implementation"] = self.attention_implementation
        self.runtime_profile["model_name"] = self.model_name
        self.runtime_profile["model_revision"] = self.model_revision
        print(f"  SFT numerical precision: {dtype_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            attn_implementation=self.attention_implementation,
        )

        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=True
        )
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False

        # Load existing adapter or create new one
        self.resume_checkpoint = self._latest_checkpoint()
        checkpoint_path = self.output_dir / "adapter_model.safetensors"
        if checkpoint_path.exists() and self.resume_checkpoint is None:
            print(f"  Found existing LoRA checkpoint at {self.output_dir}, resuming!")
            self.model = PeftModel.from_pretrained(
                self.model, str(self.output_dir), is_trainable=True
            )
        else:
            lora_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        if self.resume_checkpoint is not None:
            print(f"  Resuming exact SFT Trainer state from {self.resume_checkpoint}")

        print("SFT model setup complete!")
        self.model.print_trainable_parameters()

    def _format_generation_prompt(self, prompt: str) -> str:
        """Render the same user turn used by inference, ending at assistant start."""
        return format_chat_prompt(self.tokenizer, prompt)

    def prepare_dataset(self, data_points: List[SFTDataPoint]) -> Dataset:
        """Pre-tokenize examples while retaining every completion token.

        TRL 0.15 trains full-text examples by default.  Pre-tokenization plus
        CompletionOnlyDataCollator ensures the loss teaches only the assert
        completion, not the prompt that was supplied to the model.
        """
        input_ids, attention_masks, completion_starts = [], [], []
        prompt_truncated = 0
        incompatible = []
        malformed_prompts = []
        support_units_dropped = 0
        code_units_dropped = 0
        max_observed_prompt = 0
        max_observed_completion = 0
        for dp in data_points:
            completion_text = dp.completion.strip() + self.tokenizer.eos_token
            completion_ids = self.tokenizer(completion_text, add_special_tokens=False)["input_ids"]
            completion_limit = sft_completion_limit_for_execution_mode(
                dp.execution_mode,
                self.max_completion_tokens,
                self.max_repository_completion_tokens,
            )

            max_observed_completion = max(max_observed_completion, len(completion_ids))
            if (
                len(completion_ids) > completion_limit
                or len(completion_ids) >= MAX_SFT_SEQUENCE_LENGTH
            ):
                incompatible.append({
                    "function_id": dp.function_id,
                    "execution_mode": dp.execution_mode,
                    "completion_tokens": len(completion_ids),
                    "limit_tokens": completion_limit,
                })
                continue
            mode_prompt_limit = sft_prompt_limit_for_execution_mode(
                dp.execution_mode,
                self.max_prompt_tokens,
                self.max_repository_prompt_tokens,
            )
            max_prompt_tokens = min(
                mode_prompt_limit,
                MAX_SFT_SEQUENCE_LENGTH - len(completion_ids),
            )
            try:
                compaction = compact_unified_user_prompt(
                    self.tokenizer,
                    dp.prompt,
                    max_prompt_tokens,
                    format_chat_prompt,
                )
            except PromptBudgetError as exc:
                malformed_prompts.append({
                    "function_id": dp.function_id,
                    "execution_mode": dp.execution_mode,
                    "reason": str(exc),
                })
                continue
            prompt_ids = compaction.token_ids
            max_observed_prompt = max(
                max_observed_prompt, compaction.original_token_count
            )
            prompt_truncated += int(compaction.compacted)
            support_units_dropped += compaction.support_units_dropped
            code_units_dropped += compaction.code_units_dropped
            token_ids = prompt_ids + completion_ids
            input_ids.append(token_ids)
            attention_masks.append([1] * len(token_ids))
            completion_starts.append(len(prompt_ids))

        self.dataset_stats = {
            "input_examples": len(data_points),
            "retained_examples": len(input_ids),
            "dropped_overlong_examples": len(incompatible),
            # Section-aware compaction removes whole AST/support units and fails
            # closed; it never slices a statement. "truncated" described the
            # behaviour this replaced. The old key is retained as a deprecated
            # alias so existing readers keep working.
            "prompt_compacted_examples": prompt_truncated,
            "prompt_truncated_examples": prompt_truncated,
            "max_observed_prompt_tokens": max_observed_prompt,
            "max_observed_completion_tokens": max_observed_completion,
            "prompt_compaction_strategy": PROMPT_COMPACTION_STRATEGY,
            "malformed_prompt_examples": len(malformed_prompts),
            "support_units_dropped": support_units_dropped,
            "code_units_dropped": code_units_dropped,
        }
        if incompatible:
            sample = ", ".join(
                f"{item['function_id']} ({item['completion_tokens']} tokens; "
                f"limit={item['limit_tokens']})"
                for item in incompatible[:5]
            )
            raise ValueError(
                "SFT generation-compatibility preflight rejected "
                f"{len(incompatible)} completion(s) above their mode-specific "
                f"generation limit: {sample}"
            )
        if malformed_prompts:
            sample = ", ".join(
                f"{item['function_id']}: {item['reason']}"
                for item in malformed_prompts[:3]
            )
            raise ValueError(
                "SFT section-aware prompt preflight rejected "
                f"{len(malformed_prompts)} malformed/over-budget prompt(s): {sample}"
            )
        if not input_ids:
            raise ValueError("SFT dataset contains no completions that fit the sequence limit")

        return Dataset.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "completion_start": completion_starts,
        })

    def train(
        self,
        data_points: List[SFTDataPoint],
        num_epochs: int = 2,
        batch_size: int = 4,
        checkpoint_monitor: Callable | None = None,
    ) -> Dict:
        """Run SFT training on the provided data points."""
        if self.model is None:
            self.setup_model()

        dataset = self.prepare_dataset(data_points)
        unique_examples = len(dataset)
        gradient_accumulation_steps = SFT_GRADIENT_ACCUMULATION_STEPS

        # Older Trainer/TRL releases floor incomplete gradient-accumulation
        # windows at epoch boundaries. Pad deterministically so the tail is
        # trained instead of silently skipped. Padding is accounted for
        # separately from the unique retained-example count.
        schedule = plan_sft_optimizer_schedule(
            unique_examples,
            num_epochs,
            batch_size,
            self.warmup_steps,
            self.checkpoint_steps,
            gradient_accumulation_steps,
        )
        samples_per_optimizer_step = schedule["samples_per_optimizer_step"]
        padding_examples = schedule["optimizer_padding_examples"]
        if padding_examples:
            padding_indices = [
                min(
                    unique_examples - 1,
                    int((index + 0.5) * unique_examples / padding_examples),
                )
                for index in range(padding_examples)
            ]
            dataset = dataset.select(list(range(unique_examples)) + padding_indices)

        optimizer_examples = len(dataset)
        print(
            f"SFT dataset: {unique_examples} unique examples + "
            f"{padding_examples} deterministic optimizer-padding examples"
        )

        optimizer_steps = schedule["planned_optimizer_steps"]
        # Keep cross-account failover rollback bounded. The mounted Modal
        # Volume is committed periodically by the remote launcher.
        checkpoint_save_steps = schedule["effective_checkpoint_steps"]
        warmup_steps = schedule["effective_warmup_steps"]

        training_args = SFTConfig(
            output_dir=str(self.output_dir / "sft_tmp"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            logging_steps=10,
            save_steps=checkpoint_save_steps,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            fp16=not self.use_bf16,
            bf16=self.use_bf16,
            tf32=self.use_bf16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            save_total_limit=2,
            report_to="none",
            max_seq_length=MAX_SFT_SEQUENCE_LENGTH,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
        )

        monitor_callback = (
            SFTCheckpointMonitorCallback(
                checkpoint_monitor, self.tokenizer, optimizer_steps
            )
            if checkpoint_monitor is not None
            else None
        )
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            data_collator=CompletionOnlyDataCollator(self.tokenizer),
            callbacks=[monitor_callback] if monitor_callback is not None else None,
        )

        print(
            f"Starting SFT training: {num_epochs} epochs, "
            f"{unique_examples} unique / {optimizer_examples} optimizer examples..."
        )
        train_result = trainer.train(
            resume_from_checkpoint=str(self.resume_checkpoint) if self.resume_checkpoint else None
        )

        # Extract loss
        loss = 0.0
        if hasattr(trainer, "state") and trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            loss = last_log.get("train_loss", last_log.get("loss", 0.0))
        if loss == 0.0:
            loss = getattr(train_result, "training_loss", 0.0)
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f"SFT produced a non-finite loss: {loss}")

        completed_optimizer_steps = int(trainer.state.global_step)
        completed_epochs = float(trainer.state.epoch or 0.0)

        print(f"SFT training complete! Final loss: {loss:.4f}")

        # Cleanup trainer
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": loss,
            "examples": unique_examples,
            "optimizer_examples": optimizer_examples,
            "optimizer_padding_examples": padding_examples,
            "epochs": num_epochs,
            "resumed_from_checkpoint": str(self.resume_checkpoint) if self.resume_checkpoint else None,
            "trainer_checkpoint": str(self._latest_checkpoint()) if self._latest_checkpoint() else None,
            "checkpoint_save_steps": checkpoint_save_steps,
            "warmup_steps": warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "model_runtime_profile": dict(self.runtime_profile),
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_repository_prompt_tokens": self.max_repository_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "planned_optimizer_steps": optimizer_steps,
            "completed_optimizer_steps": completed_optimizer_steps,
            "completed_epochs": completed_epochs,
            "monitor_stopped_early": bool(
                monitor_callback and monitor_callback.stopped_early
            ),
            "monitor_history": monitor_callback.history if monitor_callback else [],
            "best_validation_adapter_path": (
                monitor_callback.best_adapter_path if monitor_callback else None
            ),
            "best_validation_metrics": (
                monitor_callback.best_metrics if monitor_callback else None
            ),
            **self.dataset_stats,
        }

    def save_adapter(self, path: Path = None) -> Path:
        """Save the SFT-trained LoRA adapter."""
        save_path = path or self.output_dir
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"SFT adapter saved to {save_path}")
        return save_path
