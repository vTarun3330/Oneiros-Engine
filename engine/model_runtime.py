"""Shared numerical runtime configuration for generation, SFT, and DPO.

An adapter must be evaluated with the same quantization and compute dtype used
while it was trained.  Keeping these decisions in one helper prevents the
generation engine and the two training phases from silently loading different
effective base models.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from config import model_config


MODEL_RUNTIME_PROFILE_VERSION = "oneiros_phi3_4bit_nf4_aligned_v1"


def resolve_compute_dtype(torch_module) -> Tuple[Any, str]:
    """Return BF16 on a supported CUDA device, otherwise the FP16 fallback."""
    use_bf16 = bool(
        torch_module.cuda.is_available()
        and torch_module.cuda.is_bf16_supported()
    )
    return (
        torch_module.bfloat16 if use_bf16 else torch_module.float16,
        "bf16" if use_bf16 else "fp16",
    )


def build_4bit_quantization_config(
    torch_module, bitsandbytes_config_class
) -> Tuple[Any, Any, str]:
    """Build the one canonical 4-bit configuration used by every phase."""
    compute_dtype, dtype_name = resolve_compute_dtype(torch_module)
    configuration = bitsandbytes_config_class(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )
    return configuration, compute_dtype, dtype_name


def runtime_profile(dtype_name: str) -> Dict[str, Any]:
    """Return a serializable identity for audit and result files."""
    return {
        "profile_version": MODEL_RUNTIME_PROFILE_VERSION,
        "compute_dtype": dtype_name,
        "load_in_4bit": True,
        "quant_type": model_config.bnb_4bit_quant_type,
        "double_quant": model_config.bnb_4bit_use_double_quant,
        "attention_implementation": model_config.attention_implementation,
    }
