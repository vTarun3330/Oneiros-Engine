"""Data-free GPU check: no models, corpora, checkpoints, or training imports."""
import importlib.metadata
import json
import platform
import sys

import torch
import bitsandbytes as bnb

assert torch.cuda.is_available(), 'CUDA is unavailable'
name = torch.cuda.get_device_name(0)
assert 'RTX 4500 Ada' in name, name
with torch.inference_mode():
    x = torch.arange(4096, device='cuda', dtype=torch.float32).reshape(64, 64) / 4096
    actual = x @ x.T
    expected = x.cpu() @ x.cpu().T
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    packed, state = bnb.functional.quantize_4bit(x.to(torch.bfloat16), quant_type='nf4')
    restored = bnb.functional.dequantize_4bit(packed, state)
    assert restored.is_cuda and restored.shape == x.shape
    assert torch.isfinite(restored).all().item()
    torch.cuda.synchronize()
print(json.dumps({
    'python': platform.python_version(),
    'executable': sys.executable,
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'gpu': name,
    'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    'compute_capability': torch.cuda.get_device_capability(0),
    'bf16_supported': torch.cuda.is_bf16_supported(),
    'cuda_matmul': 'passed',
    'bitsandbytes_nf4_cuda': 'passed',
    'packages': {p: importlib.metadata.version(p) for p in
        ['transformers', 'peft', 'trl', 'accelerate', 'bitsandbytes', 'datasets']},
    'data_accessed': False,
    'training_launched': False,
}, indent=2))
