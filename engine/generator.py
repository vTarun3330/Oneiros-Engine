"""
Phi-3 Generator Module for Oneiros Engine.

This module handles test input generation using the Phi-3-mini-4k-instruct model.
It generates test cases based on function signatures and examples from memory.
"""
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. Install with: pip install peft")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import model_config, training_config
from engine.model_runtime import (
    build_4bit_quantization_config,
    runtime_profile,
)
from engine.prompt_budget import compact_prompt_texts
from engine.test_generation_prompt import build_unified_user_prompt, format_chat_prompt
from harness.candidate_policy import validate_function_assertion


@dataclass
class GeneratedTest:
    """Represents a generated test case."""
    id: str
    input_code: str              # The generated test input
    function_id: str             # Target function
    raw_output: str              # Raw model output
    is_valid: bool = True        # Whether syntax is valid
    parse_error: str = ""        # Parse error if invalid


class Phi3Generator:
    """
    Test case generator using Phi-3-mini-4k-instruct.

    Generates test inputs for system-level Python functions based on:
    - Function signature and docstring
    - Examples from FAISS memory
    - Edge cases to explore
    """

    def __init__(
        self,
        model_name: str = None,
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize the Phi-3 generator.

        Args:
            model_name: HuggingFace model name
            load_in_4bit: Whether to use 4-bit quantization (fits on smaller GPUs)
            device_map: Device mapping strategy
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        self.model_name = model_name or model_config.model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map

        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.runtime_profile = None

        # Generation parameters
        # Match the concise function completion contract used by SFT and live
        # validation. Repository fragments use their separate pipeline.
        self.max_new_tokens = training_config.sft_completion_token_limit
        self.temperature = model_config.temperature
        self.top_p = model_config.top_p

        # Statistics
        self.stats = {
            "total_generated": 0,
            "valid_generated": 0,
            "invalid_generated": 0
        }

    def load_model(self) -> None:
        """Load the Phi-3 model and tokenizer."""
        if self.is_loaded:
            return

        print(f"Loading {self.model_name}...")

        # Configure quantization
        if self.load_in_4bit:
            quantization_config, compute_dtype, dtype_name = (
                build_4bit_quantization_config(torch, BitsAndBytesConfig)
            )
        else:
            quantization_config = None
            compute_dtype, dtype_name = (
                (torch.bfloat16, "bf16")
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else (torch.float16, "fp16")
            )
        self.runtime_profile = runtime_profile(dtype_name)
        self.runtime_profile["load_in_4bit"] = bool(self.load_in_4bit)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=model_config.model_revision,
            trust_remote_code=True
        )

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with RoPE scaling fix
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            self.model_name,
            revision=model_config.model_revision,
            trust_remote_code=True,
        )

        # Patch for RoPE scaling compatibility issues
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", ""))
            # Phi-3's custom modeling only accepts "su", "yarn", or no rope_scaling
            # "default" / "linear" / etc. are not recognized — just remove it
            if rope_type in ("default", "linear", ""):
                config.rope_scaling = None
            elif "type" not in config.rope_scaling and "rope_type" in config.rope_scaling:
                config.rope_scaling["type"] = config.rope_scaling["rope_type"]

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=model_config.model_revision,
            config=config,
            quantization_config=quantization_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            attn_implementation=model_config.attention_implementation,
        )

        self.is_loaded = True
        print(f"Model loaded successfully!")

    def load_lora_adapter(self, adapter_path: Path) -> None:
        """Load a LoRA adapter (after DPO training)."""
        if not self.is_loaded:
            self.load_model()

        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print("LoRA adapter loaded!")

    def _create_prompt(
        self,
        function_signature: str,
        docstring: str,
        edge_cases: List[str],
        memory_examples: List[str] = None,
        library: str = "unknown",
        entry_point: str = "",
    ) -> str:
        """Create the canonical model-visible function prompt."""
        context_sections = []
        if memory_examples:
            examples = "\n".join(f"- {ex}" for ex in memory_examples[:3])
            context_sections.append(f"Previous successful test inputs:\n{examples}")

        if edge_cases:
            cases = ", ".join(edge_cases[:5])
            context_sections.append(f"Edge cases to consider: {cases}")

        return build_unified_user_prompt(
            code_under_test=function_signature,
            execution_mode="function_assertion",
            specification=docstring,
            support_context="\n\n".join(context_sections),
            target_symbols=[entry_point] if entry_point else [],
            entry_point=entry_point,
        )

    def _parse_output(
        self, output: str, function_id: str, target_entry_point: str = "",
    ) -> GeneratedTest:
        """Parse model output into a test case, prioritizing assert statements."""
        output = output.strip()
        lines = output.split('\n')
        test_code = None

        # Priority 1: find an assert statement (matches DPO training format)
        for line in lines:
            line = line.strip()
            if line.startswith('assert '):
                test_code = line
                break

        # Priority 2: find result = func(...) and convert to assert
        if not test_code:
            for line in lines:
                line = line.strip()
                if line.startswith('result = ') or line.startswith('output = '):
                    test_code = line
                    break
                elif '(' in line and ')' in line and '=' in line:
                    test_code = line
                    break

        # Priority 3: first non-empty line
        if not test_code:
            for line in lines:
                if line.strip():
                    test_code = line.strip()
                    break

        if not test_code:
            test_code = output[:200]

        # Fail closed: a compilable line is not sufficient. Generated output
        # must be one bounded assertion that actually calls the target.
        policy = validate_function_assertion(
            test_code, target_entry_point or function_id
        )
        is_valid = policy.valid
        parse_error = policy.reason

        test_id = f"gen_{function_id}_{self.stats['total_generated']}"
        self.stats["total_generated"] += 1

        if is_valid:
            self.stats["valid_generated"] += 1
        else:
            self.stats["invalid_generated"] += 1

        return GeneratedTest(
            id=test_id,
            input_code=test_code,
            function_id=function_id,
            raw_output=output,
            is_valid=is_valid,
            parse_error=parse_error
        )

    def generate(
        self,
        function_signature: str,
        docstring: str,
        function_id: str,
        edge_cases: List[str] = None,
        memory_examples: List[str] = None,
        library: str = "unknown",
        num_samples: int = 1
    ) -> List[GeneratedTest]:
        """
        Generate test inputs for a function.

        Args:
            function_signature: The function signature
            docstring: Function docstring
            function_id: Unique function ID
            edge_cases: Known edge cases
            memory_examples: Past successful inputs
            library: Library name
            num_samples: Number of tests to generate

        Returns:
            List of GeneratedTest objects
        """
        if not self.is_loaded:
            self.load_model()

        match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", function_signature)
        target_entry_point = match.group(1) if match else function_id

        user_prompt = self._create_prompt(
            function_signature=function_signature,
            docstring=docstring,
            edge_cases=edge_cases or [],
            memory_examples=memory_examples or [],
            library=library,
            entry_point=target_entry_point,
        )
        prompt = format_chat_prompt(self.tokenizer, user_prompt)

        # Apply the same deterministic token gate used during function SFT.
        bounded_prompts, _ = compact_prompt_texts(
            self.tokenizer, [prompt], training_config.sft_prompt_token_limit
        )
        input_ids = torch.tensor(
            bounded_prompts, dtype=torch.long, device=self.model.device
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,                        # <-- keep True
                past_key_values=None,                  # <-- ADD: force fresh cache
            )

        # Decode
        generated_tests = []
        for output in outputs:
            # Get only the new tokens
            new_tokens = output[inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            test = self._parse_output(text, function_id, target_entry_point)
            generated_tests.append(test)

        return generated_tests

    def generate_batch(
        self,
        functions: List[Dict[str, Any]],
        memory_examples_map: Dict[str, List[str]] = None,
        num_per_function: int = 1
    ) -> Dict[str, List[GeneratedTest]]:
        """
        Generate tests for multiple functions.

        Args:
            functions: List of function configs with signature, docstring, id, etc.
            memory_examples_map: Map of function_id to memory examples
            num_per_function: Tests to generate per function

        Returns:
            Dict mapping function_id to list of GeneratedTest
        """
        memory_examples_map = memory_examples_map or {}
        results = {}

        for func in functions:
            func_id = func["id"]
            tests = self.generate(
                function_signature=func.get("signature", ""),
                docstring=func.get("docstring", ""),
                function_id=func_id,
                edge_cases=func.get("edge_cases", []),
                memory_examples=memory_examples_map.get(func_id, []),
                library=func.get("library", "unknown"),
                num_samples=num_per_function
            )
            results[func_id] = tests

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get generation statistics."""
        return self.stats.copy()


# Mock generator for testing without GPU
class MockGenerator:
    """Mock generator for testing without loading the actual model."""

    def __init__(self):
        self.stats = {"total_generated": 0, "valid_generated": 0, "invalid_generated": 0}

    def load_model(self):
        print("MockGenerator: Model loading skipped")

    def generate(
        self,
        function_signature: str,
        docstring: str,
        function_id: str,
        **kwargs
    ) -> List[GeneratedTest]:
        """Generate mock test cases."""
        mock_tests = [
            f"result = {function_signature.split('(')[0].split()[-1]}({{}})",
            f"result = {function_signature.split('(')[0].split()[-1]}(None)",
            f"result = {function_signature.split('(')[0].split()[-1]}([])",
        ]

        self.stats["total_generated"] += 1
        self.stats["valid_generated"] += 1

        return [GeneratedTest(
            id=f"mock_{function_id}_{i}",
            input_code=test,
            function_id=function_id,
            raw_output=test,
            is_valid=True
        ) for i, test in enumerate(mock_tests[:kwargs.get("num_samples", 1)])]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Phi-3 Generator (Mock Mode)")
    print("=" * 60)

    # Use mock generator for testing
    generator = MockGenerator()

    print("\n1. Generating test for pandas.merge...")
    tests = generator.generate(
        function_signature="def merge_wrapper(left_df, right_df, on=None, how='inner')",
        docstring="Merge two DataFrames on specified column(s).",
        function_id="sys_pandas_merge",
        edge_cases=["Empty DataFrame", "Mismatched keys"],
        num_samples=1
    )

    for test in tests:
        print(f"   Generated: {test.input_code}")
        print(f"   Valid: {test.is_valid}")

    print("\n2. Stats:")
    stats = generator.stats
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("Generator test complete!")
    print("=" * 60)
