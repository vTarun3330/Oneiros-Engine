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
