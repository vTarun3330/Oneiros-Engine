from pathlib import Path
import hashlib
import json

import pytest

from engine.dpo_trainer import require_sft_reference_adapter
from engine.prompt_budget import compact_prompt_string


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


def test_dpo_reference_adapter_fails_closed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="base-model fallback is disabled"):
        require_sft_reference_adapter(tmp_path)

    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    (tmp_path.parent / "sft_metadata.json").write_text(json.dumps({
        "sft_adapter_sha256": hashlib.sha256(b"adapter").hexdigest(),
    }), encoding="utf-8")
    assert require_sft_reference_adapter(tmp_path) == tmp_path


def test_dpo_prompt_uses_shared_head_tail_compaction():
    prompt = "A" * 400 + "B" * 400
    compacted, changed, before, after = compact_prompt_string(
        CharacterTokenizer(), prompt, 512
    )
    assert changed is True
    assert before == 800
    assert after == 512
    assert compacted.startswith("A" * 100)
    assert compacted.endswith("B" * 100)


def test_dpo_trainer_follows_the_run_s_base_model_not_the_canonical_default():
    """A LoRA adapter is a delta on specific base weights.

    DPO continues from an SFT adapter. Loading the canonical Phi-3 while the
    adapter was trained on Qwen would silently apply one model's adapter to
    another's weights. This was a real failure: DPO setup loaded
    microsoft/Phi-3-mini-4k-instruct for a Qwen run and died.
    """
    import inspect

    from engine.dpo_trainer import DPOTrainer

    signature = inspect.signature(DPOTrainer.__init__)
    for parameter in ("model_name", "model_revision", "attention_implementation"):
        assert parameter in signature.parameters, f"DPOTrainer must accept {parameter}"

    source = inspect.getsource(DPOTrainer.setup_model)
    # The loader must consume the instance's identity, never the global default.
    assert "self.model_revision" in source
    assert "self.attention_implementation" in source
    assert "model_config.model_revision" not in source
    assert "model_config.attention_implementation" not in source


def test_train_on_dataset_passes_the_override_into_dpo():
    import inspect

    from scripts import train_on_dataset

    source = inspect.getsource(train_on_dataset.run_training)
    start = source.index("dpo_trainer = DPOTrainer(")
    call = source[start:start + 400]
    assert "BASE_MODEL_NAME_OVERRIDE" in call
    assert "BASE_MODEL_REVISION_OVERRIDE" in call
    assert "BASE_MODEL_ATTENTION_IMPLEMENTATION_OVERRIDE" in call
