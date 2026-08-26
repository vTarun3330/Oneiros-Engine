from config import model_config
from engine.model_runtime import (
    MODEL_RUNTIME_PROFILE_VERSION,
    build_4bit_quantization_config,
    runtime_profile,
)
from scripts.train_on_dataset import normalized_sft_run_hyperparameters


class _CudaStub:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_bf16_supported():
        return True


class _TorchStub:
    cuda = _CudaStub()
    bfloat16 = "bf16-token"
    float16 = "fp16-token"


class _BitsAndBytesConfigStub:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_shared_runtime_profile_uses_stable_non_nested_bf16_quantization():
    configuration, compute_dtype, dtype_name = build_4bit_quantization_config(
        _TorchStub, _BitsAndBytesConfigStub
    )

    assert compute_dtype == "bf16-token"
    assert dtype_name == "bf16"
    assert configuration.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bf16-token",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": False,
    }
    profile = runtime_profile(dtype_name)
    assert profile == {
        "profile_version": MODEL_RUNTIME_PROFILE_VERSION,
        "compute_dtype": "bf16",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quant": False,
        "attention_implementation": "eager",
    }


def test_model_runtime_configuration_is_canonical_for_all_phases():
    assert model_config.bnb_4bit_quant_type == "nf4"
    assert model_config.bnb_4bit_use_double_quant is False
    assert model_config.attention_implementation == "eager"


def test_legacy_sft_run_normalization_preserves_historical_cosine_scheduler():
    normalized = normalized_sft_run_hyperparameters({"epochs": 1})
    assert normalized["lr_scheduler_type"] == "cosine"
    assert normalized["min_function_kill_rate"] == 0.50
