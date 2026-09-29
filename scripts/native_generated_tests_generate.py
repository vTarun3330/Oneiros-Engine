"""Durable candidate generation for native generated tests (protocol v2, section 2).

    job        build a generation job from sealed, leakage-cleared prompts, applying the
               tokenizer sequence-fit gate (refuse, never truncate)
    run        generate for one arm: --backend mock (tests / dry runs) or hf (GPU; needs a
               green preflight v2 and explicit approval - NOT used in this work block)

Durability: one contract per arm and output directory (exact match or refuse), one JSONL
line per (target, seed) with the contract hash, fsync after every line, strict reading on
resume (partial, malformed, stale, duplicate or unexpected lines are quarantined, never
merged). No reranking; duplicates are kept; every raw output is retained with its hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GENERATOR_VERSION = "oneiros_native_generated_tests_generate_v1"
CONTRACT = {
    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "base_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    "sft_adapter": "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter",
    "seeds": [42, 43, 44], "candidates": 8, "temperature": 0.7, "top_p": 0.9,
    "do_sample": True, "max_new_tokens": 1024, "prompt_token_limit": 2048,
    "sequence_limit": 3072, "batch_size": 2, "extraction": "whole_output_single_fence_strip",
    "reranking": "none", "duplicates": "kept", "raw_output_retained": True,
}
ARMS = ("base", "sft")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contract_sha(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def adapter_manifest(directory: Path) -> Dict[str, str]:
    """Every file in the adapter directory with its SHA-256 (adapter_config.json included)."""
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def extract(raw: str) -> Dict[str, Any]:
    """Whole output; strip only a fence that wraps the ENTIRE output."""
    text = raw.strip("\n")
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```" and \
            sum(l.startswith("```") for l in lines) == 2:
        return {"module": "\n".join(lines[1:-1]) + "\n", "fence_stripped": True}
    return {"module": raw if raw.endswith("\n") else raw + "\n", "fence_stripped": False}


def target_seed(seed: int, target_key: str) -> int:
    return int(hashlib.sha256(f"{seed}:{target_key}".encode()).hexdigest()[:8], 16)


# --- job ------------------------------------------------------------------------------------

def sequence_fit(prompts: Sequence[Mapping[str, Any]], count_tokens: Callable[[str], int],
                 limit: int = CONTRACT["prompt_token_limit"]) -> Dict[str, Any]:
    admitted, refused = [], []
    for sealed in prompts:
        tokens = count_tokens(sealed["prompt"])
        entry = {"target_key": sealed["target_key"], "prompt_tokens": tokens}
        (admitted if tokens <= limit else refused).append(entry)
    return {"admitted": admitted, "refused": refused, "limit": limit}


def build_job(prompts: Sequence[Mapping[str, Any]], scans: Mapping[str, Mapping[str, Any]],
              fit: Mapping[str, Any]) -> Dict[str, Any]:
    """Only prompts that are sealed, leakage-clean and fit enter the job."""
    fits = {e["target_key"] for e in fit["admitted"]}
    items, refused = [], []
    for sealed in prompts:
        key = sealed["target_key"]
        if sha256_text(sealed["prompt"]) != sealed["prompt_sha256"]:
            refused.append({"target_key": key, "reason": "seal_mismatch"})
        elif not scans.get(key, {}).get("ok"):
            refused.append({"target_key": key, "reason": "leakage",
                            "details": scans.get(key, {}).get("reasons")})
        elif key not in fits:
            refused.append({"target_key": key, "reason": "sequence_overflow"})
        else:
            items.append({"target_key": key, "prompt": sealed["prompt"],
                          "prompt_sha256": sealed["prompt_sha256"],
                          "condition": sealed["condition"], "scaffold": sealed.get("scaffold")})
    items.sort(key=lambda i: i["target_key"])
    body = {"items": items, "refused": refused, "sequence_fit": fit}
    return {**body, "job_sha256": contract_sha({"items": items})}


# --- durable run ------------------------------------------------------------------------------

def read_lines(path: Path, contract_hash: str, expected: set) -> tuple[dict, list]:
    rows, problems = {}, []
    if not path.exists():
        return rows, problems
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        problems.append("partial final line")
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"malformed line {number}")
            continue
        key = row.get("key")
        if row.get("contract_sha256") != contract_hash:
            problems.append(f"line {number}: different contract")
        elif key not in expected:
            problems.append(f"line {number}: unexpected key")
        elif key in rows:
            problems.append(f"line {number}: duplicate key")
        else:
            rows[key] = row
    return rows, problems


def quarantine(path: Path, problems: Sequence[str]) -> Path:
    target_dir = path.parent / "quarantine"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / f"{int(time.time() * 1000)}_{path.name}"
    shutil.move(str(path), str(target))
    with (path.parent / "resume_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"quarantined": path.name, "to": target.name,
                                 "reasons": list(problems)[:20]}) + "\n")
    return target


def arm_contract(arm: str, job: Mapping[str, Any], identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {"generator_version": GENERATOR_VERSION, "arm": arm, **CONTRACT,
            "job_sha256": job["job_sha256"], "identity": dict(identity)}


def check_contract(out_dir: Path, contract: Mapping[str, Any], arm: str) -> str:
    path = out_dir / f"contract_{arm}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != dict(contract):
            changed = sorted(k for k in set(stored) | set(contract)
                             if stored.get(k) != contract.get(k))
            raise SystemExit(f"REFUSED: {arm} contract differs in {changed}")
        return "resume"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(contract), indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return "fresh"


def mock_backend(prompt: str, n: int, seed: int) -> List[str]:
    """Deterministic stand-in for tests; never a model."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        value = rng.randint(0, 9)
        body = f"def test_mock_{i}():\n    assert {value} == {value}\n"
        out.append(f"```python\n{body}```" if i % 3 == 0 else body)
    return out


def run(job: Mapping[str, Any], arm: str, out_dir: Path, identity: Mapping[str, Any],
        backend: Callable[[str, int, int], List[str]], crash_after: int | None = None
        ) -> Dict[str, Any]:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm}")
    contract = arm_contract(arm, job, identity)
    chash = contract_sha(contract)
    mode = check_contract(out_dir, contract, arm)
    path = out_dir / f"generations_{arm}.jsonl"
    expected = {f"{item['target_key']}::{seed}" for item in job["items"]
                for seed in CONTRACT["seeds"]}
    rows, problems = read_lines(path, chash, expected)
    if problems:
        quarantine(path, problems)
        rows = {}
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for seed in CONTRACT["seeds"]:
            for item in job["items"]:
                key = f"{item['target_key']}::{seed}"
                if key in rows:
                    continue
                if crash_after is not None and written >= crash_after:
                    raise RuntimeError("deliberate crash for the resume test")
                tseed = target_seed(seed, item["target_key"])
                raws = backend(item["prompt"], CONTRACT["candidates"], tseed)
                if len(raws) != CONTRACT["candidates"]:
                    raise RuntimeError("backend returned the wrong number of candidates")
                extracted = [extract(r) for r in raws]
                handle.write(json.dumps({
                    "key": key, "arm": arm, "seed": seed, "target_seed": tseed,
                    "target_key": item["target_key"], "contract_sha256": chash,
                    "prompt_sha256": item["prompt_sha256"],
                    "raw_outputs": raws, "raw_sha256": [sha256_text(r) for r in raws],
                    "modules": [e["module"] for e in extracted],
                    "module_sha256": [sha256_text(e["module"]) for e in extracted],
                    "fence_stripped": [e["fence_stripped"] for e in extracted]},
                    sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                written += 1
    rows, problems = read_lines(path, chash, expected)
    if problems or set(rows) != expected:
        raise RuntimeError(f"generation file incomplete: {problems[:3]}")
    return {"mode": mode, "contract_sha256": chash, "lines": len(rows),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def hf_backend(arm: str):  # pragma: no cover - GPU path, not executed in this work block
    """Real sampling with the exact base revision (and the verified adapter for sft)."""
    import torch
    import transformers
    from engine.test_generation_prompt import format_chat_prompt
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONTRACT["base_model"], revision=CONTRACT["base_revision"], local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        CONTRACT["base_model"], revision=CONTRACT["base_revision"], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    if arm == "sft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / CONTRACT["sft_adapter"]))
    model.eval()
    tokenizer.padding_side = "left"

    def generate(prompt: str, n: int, seed: int) -> List[str]:
        text = format_chat_prompt(tokenizer, prompt)
        outputs = []
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for start in range(0, n, CONTRACT["batch_size"]):
            batch = [text] * min(CONTRACT["batch_size"], n - start)
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**encoded, do_sample=True, temperature=CONTRACT["temperature"],
                                     top_p=CONTRACT["top_p"],
                                     max_new_tokens=CONTRACT["max_new_tokens"],
                                     pad_token_id=tokenizer.pad_token_id)
            width = encoded["input_ids"].shape[1]
            outputs += [tokenizer.decode(row[width:], skip_special_tokens=True) for row in out]
        return outputs
    return generate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--job", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", required=True, choices=("mock", "hf"))
    args = parser.parse_args(argv)
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    identity = {"backend": args.backend}
    if args.backend == "hf":
        raise SystemExit("REFUSED: GPU generation needs a green preflight v2 and explicit "
                         "approval; it is not enabled in this work block")
    print(json.dumps(run(job, args.arm, Path(args.out), identity, mock_backend)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
