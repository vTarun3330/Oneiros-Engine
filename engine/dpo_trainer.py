"""
DPO Trainer Module for Oneiros Engine.

This module implements Direct Preference Optimization (DPO) training
for fine-tuning Phi-3 based on Winner/Loser test pairs.
"""

import warnings
import logging
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer.*")
warnings.filterwarnings("ignore", message=".*processing_class.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import gc
import hashlib
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from trl import DPOTrainer as TRLDPOTrainer, DPOConfig
    from datasets import Dataset
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("Warning: trl not installed. Install with: pip install trl")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import model_config, training_config
from engine.model_runtime import (
    build_4bit_quantization_config,
    runtime_profile,
)
from engine.prompt_budget import PROMPT_COMPACTION_STRATEGY, compact_prompt_string

DPO_MAX_SEQUENCE_TOKENS = 1536
DPO_MAX_PROMPT_TOKENS = 512
DPO_MAX_COMPLETION_TOKENS = 1024


def require_sft_reference_adapter(adapter_dir: Path) -> Path:
    """Verify the frozen SFT reference and its recorded checksum."""
    adapter_dir = Path(adapter_dir)
    adapter_file = next((
        adapter_dir / name for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (adapter_dir / name).exists()
    ), None)
    if adapter_file is None:
        raise RuntimeError(
            f"DPO requires a frozen SFT reference adapter at {adapter_dir}; "
            "base-model fallback is disabled."
        )
    metadata_file = adapter_dir.parent / "sft_metadata.json"
    if not metadata_file.exists():
        raise RuntimeError(
            f"DPO requires checksum metadata at {metadata_file}; "
            "an unaccounted SFT reference is not accepted."
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected = metadata.get("sft_adapter_sha256")
    digest = hashlib.sha256()
    with adapter_file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if not expected or digest.hexdigest() != expected:
        raise RuntimeError("Frozen SFT reference checksum does not match its metadata")
    return adapter_dir


@dataclass
class DPODataPoint:
    """A single DPO training data point."""
    prompt: str
    chosen: str
    rejected: str
    function_id: str = ""


class DPOTrainer:
    """
    DPO Trainer for fine-tuning Phi-3 on test generation preferences.

    Uses Winner/Loser pairs from the Feedback Oracle to train the model
    to prefer generating bug-finding and novel tests.
    """

    def __init__(
        self,
        model_name: str = None,
        output_dir: Path = None,
        learning_rate: float = None,
        beta: float = None
    ):
        if not TRL_AVAILABLE:
            raise ImportError("trl is required. Install with: pip install trl")
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required. Install with: pip install peft")

        self.model_name = model_name or model_config.model_name
        self.output_dir = Path(output_dir or training_config.checkpoint_dir)
        self.learning_rate = (
            learning_rate if learning_rate is not None
            else getattr(training_config, "dpo_learning_rate", training_config.learning_rate)
        )
        self.beta = beta or training_config.beta
        self.max_grad_norm = getattr(
            training_config, "dpo_max_grad_norm", training_config.max_grad_norm
        )
        self.sft_adapter_dir = self.output_dir / "sft_adapter"
        self.has_reference_adapter = False

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.last_context_audit = None

        self.stats = {
            "iterations_completed": 0,
            "total_pairs_trained": 0,
            "best_loss": float("inf")
        }

    def setup_model(self) -> None:
        """Load and prepare model for training."""
        print(f"Setting up {self.model_name} for DPO training...")
        reference_adapter_path = require_sft_reference_adapter(self.sft_adapter_dir)
        # Ampere A10 GPUs support BF16.  Prefer it for DPO because the policy
        # log-probability differences are substantially more prone to FP16
        # overflow than the preceding SFT completion loss.
        from transformers import BitsAndBytesConfig

        bnb_config, self.compute_dtype, dtype_name = (
            build_4bit_quantization_config(torch, BitsAndBytesConfig)
        )
        self.use_bf16 = dtype_name == "bf16"
        self.runtime_profile = runtime_profile(dtype_name)
        print(
            "  DPO numerical precision: "
            f"{dtype_name}; "
            f"lr={self.learning_rate:g}; max_grad_norm={self.max_grad_norm:g}"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=model_config.model_revision,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # FIX 1: removed undefined `config=`, `quantization_config=` (was wrong var name),
        # and `device_map=self.device_map` (self.device_map never existed)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=model_config.model_revision,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self.compute_dtype,
            attn_implementation=model_config.attention_implementation
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        # CRITICAL FIX for DPO with Gradient Checkpointing
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False

        # Resume the trainable adapter, then keep the completed SFT adapter as
        # DPO's reference policy.  This uses PEFT's adapter switching rather
        # than loading a second 3.8B reference model into VRAM.
        checkpoint_path = self.output_dir / "adapter_model.safetensors"
        train_adapter_path = self.output_dir if checkpoint_path.exists() else self.sft_adapter_dir
        if not any((train_adapter_path / name).exists() for name in (
            "adapter_model.safetensors", "adapter_model.bin",
        )):
            raise RuntimeError(
                f"DPO requires a trainable SFT-derived adapter at {train_adapter_path}"
            )
        print(f"  Loading trainable LoRA adapter from {train_adapter_path}")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model, str(train_adapter_path), is_trainable=True
        )

        self.model.load_adapter(
            str(reference_adapter_path), adapter_name="sft_reference", is_trainable=False
        )
        self.model.set_adapter("default")
        self.has_reference_adapter = True

        # PeftModel.from_pretrained can replace the wrapper created above. Put
        # this hook on the final trainable adapter, otherwise PyTorch gradient
        # checkpointing may warn that its inputs lack gradients even though the
        # LoRA parameters are trainable.
        self.model.enable_input_require_grads()
        print("Model setup complete!")
        # FIX 2: print_trainable_parameters() returns None — don't wrap in print()
        self.model.print_trainable_parameters()

    def prepare_dataset(self, pairs: List[DPODataPoint]) -> 'Dataset':
        """Convert DPO pairs after a fail-closed completion context audit.

        Prompt compaction intentionally matches Oneiros inference's 512-token
        head/tail gate. Chosen and rejected completions must remain
        whole: silently truncating a verified assertion can change whether it
        kills the mutant, so an overlong completion aborts before any update.
        """
        if self.tokenizer is None:
            raise RuntimeError("DPO context audit requires the initialized tokenizer")

        def token_count(text: str) -> int:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

        compacted_prompts = []
        prompt_lengths = []
        compacted_prompt_lengths = []
        compacted_count = 0
        for pair in pairs:
            compacted, changed, before, after = compact_prompt_string(
                self.tokenizer, pair.prompt, DPO_MAX_PROMPT_TOKENS
            )
            compacted_prompts.append(compacted)
            prompt_lengths.append(before)
            compacted_prompt_lengths.append(after)
            compacted_count += int(changed)
        chosen_lengths = [token_count(pair.chosen.strip()) for pair in pairs]
        rejected_lengths = [token_count(pair.rejected.strip()) for pair in pairs]
        violations = []
        for index, pair in enumerate(pairs):
            for label, length in (
                ("chosen", chosen_lengths[index]),
                ("rejected", rejected_lengths[index]),
            ):
                if length >= DPO_MAX_COMPLETION_TOKENS:
                    violations.append(
                        f"{pair.function_id or index}:{label}={length}"
                    )
        if violations:
            raise RuntimeError(
                "DPO completion context gate rejected overlong sequences (limit is "
                f"<{DPO_MAX_COMPLETION_TOKENS} tokens): " + ", ".join(violations)
            )

        self.last_context_audit = {
            "pairs": len(pairs),
            "prompt_limit_tokens": DPO_MAX_PROMPT_TOKENS,
            "completion_limit_tokens": DPO_MAX_COMPLETION_TOKENS,
            "prompts_requiring_intentional_truncation": sum(
                length > DPO_MAX_PROMPT_TOKENS for length in prompt_lengths
            ),
            "prompts_compacted": compacted_count,
            "prompt_compaction_strategy": PROMPT_COMPACTION_STRATEGY,
            "max_prompt_tokens": max(prompt_lengths, default=0),
            "max_compacted_prompt_tokens": max(compacted_prompt_lengths, default=0),
            "max_chosen_tokens": max(chosen_lengths, default=0),
            "max_rejected_tokens": max(rejected_lengths, default=0),
            "completion_truncations_allowed": 0,
        }
        print(f"DPO context audit: {json.dumps(self.last_context_audit, sort_keys=True)}")
        data = {
            "prompt": compacted_prompts,
            "chosen": [p.chosen for p in pairs],
            "rejected": [p.rejected for p in pairs]
        }
        return Dataset.from_dict(data)


    def format_prompt(self, prompt: str) -> str:
        """Render prompts identically for DPO and inference."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    def create_prompt(
        self,
        function_signature: str,
        docstring: str,
        library: str = "unknown"
    ) -> str:
        """Create a prompt for the training data."""
        return f"""You are a test case generator for Python's {library} library.

Function: {function_signature}
Description: {docstring}

Generate a Python test input that could find bugs or edge cases.
Output ONLY the function call.

Your test input:"""

    def train(
        self,
        pairs: List[DPODataPoint],
        num_epochs: int = 1,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Train the model on preference pairs.

        Args:
            pairs: List of DPO training pairs
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.setup_model()

        dataset = self.prepare_dataset(pairs)

        # FIX 3: added max_length + max_prompt_length to suppress warnings
        training_args = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=self.learning_rate,
            beta=self.beta,
            max_grad_norm=self.max_grad_norm,  # from config — prevents gradient explosion
            logging_steps=1,
            save_steps=100,
            warmup_steps=0,
            lr_scheduler_type="constant",
            fp16=not self.use_bf16,
            bf16=self.use_bf16,
            tf32=self.use_bf16,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            # Non-reentrant checkpointing works with LoRA activations without
            # requiring a grad-bearing input tensor, unlike the legacy mode.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            max_length=DPO_MAX_SEQUENCE_TOKENS,
            max_prompt_length=DPO_MAX_PROMPT_TOKENS,
            max_completion_length=DPO_MAX_COMPLETION_TOKENS,
            # The shared head/tail strategy has already produced <=512-token
            # prompts. This is only a fail-safe for trainer internals.
            truncation_mode="keep_end",
            model_adapter_name="default",
            ref_adapter_name="sft_reference",
        )

        # FIX 4: use processing_class= instead of deprecated tokenizer=
        self.trainer = TRLDPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer
        )

        print(f"Training on {len(pairs)} pairs for {num_epochs} epochs...")

        # FIX 5: .train() returns a TrainOutput object, not a generator
        train_result = self.trainer.train()
        # FIX 6: Extract actual loss from the trainer's log history
        loss = 0.0
        if hasattr(self.trainer, 'state') and self.trainer.state.log_history:
            last_log = self.trainer.state.log_history[-1]
            loss = last_log.get('train_loss', last_log.get('loss', 0.0))
        if loss == 0.0:
            loss = getattr(train_result, 'training_loss', 0.0)

        invalid_metric = None
        for metrics in getattr(self.trainer.state, "log_history", []):
            for metric_name in ("loss", "train_loss", "grad_norm"):
                value = metrics.get(metric_name)
                if value is None:
                    continue
                try:
                    if not math.isfinite(float(value)):
                        invalid_metric = f"{metric_name}={value}"
                        break
                except (TypeError, ValueError):
                    continue
            if invalid_metric:
                break
        if invalid_metric:
            del self.trainer
            self.trainer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise FloatingPointError(f"DPO produced a non-finite training metric: {invalid_metric}")

        self.stats["iterations_completed"] += 1
        self.stats["total_pairs_trained"] += len(pairs)
        if loss < self.stats["best_loss"]:
            self.stats["best_loss"] = loss

        # FIX 8: Clean up trainer to prevent memory leak & recursion corruption
        del self.trainer
        self.trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": loss,
            "pairs_trained": len(pairs),
            "epochs": num_epochs
        }

    def save_adapter(self, path: Path = None) -> Path:
        """Save the trained LoRA adapter."""
        # FIX 7: Save directly to self.output_dir so setup_model() can find it for resume
        path = Path(path or self.output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.has_reference_adapter:
            self.model.set_adapter("default")
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(path)

        with open(path / "training_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"Saved adapter to {path}")
        return path

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.stats.copy()


def create_dpo_pairs_from_results(
    winners: List[Dict[str, Any]],
    losers: List[Dict[str, Any]],
    prompt_template: str
) -> List[DPODataPoint]:
    """
    Create DPO training pairs from winner/loser results.

    Args:
        winners: List of winner test dicts with 'input', 'function_id'
        losers: List of loser test dicts
        prompt_template: Template string with {function_id} placeholder

    Returns:
        List of DPODataPoint objects
    """
    pairs = []

    for winner in winners:
        for loser in losers:
            if winner.get("function_id") == loser.get("function_id"):
                prompt = prompt_template.format(
                    function_id=winner.get("function_id", "unknown")
                )
                pairs.append(DPODataPoint(
                    prompt=prompt,
                    chosen=winner.get("input", ""),
                    rejected=loser.get("input", ""),
                    function_id=winner.get("function_id", "")
                ))

    return pairs
