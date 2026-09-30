"""Durable candidate generation for native generated tests (protocol v2 + amendment v2.1).

    run --job FILE --condition primary_whole_module --arm {base,sft} --out DIR
        --backend {mock,hf} [--authorization FILE]

* ``--condition`` selects the nested job inside the job file. Only
  ``primary_whole_module`` exists (v2.1 section C removed the scaffolded diagnostic).
* Every arm run has an IMMUTABLE identity: source commit and canonical source tree,
  generator, prompt-builder and protocol/amendment hashes, the complete job hash, the base
  snapshot, tokenizer and chat-template manifests, the adapter-directory manifest (sft),
  library and CUDA versions, the attention implementation, the FULL generation contract,
  the condition, the arm, the preflight hash and (for GPU runs) the authorisation hash.
  A change to any field refuses to resume and leaves the prior output untouched; a new
  output directory is required.
* The Hugging Face backend is committed source but runs only with a GPU authorisation
  receipt that binds the green preflight, the source commit, the protocol hashes, the job
  hash, the allowed conditions and arms, and the output directory (v2.1 section F). No
  source edit is ever needed to authorise a run.
* One JSONL line per (target, seed) with the identity hash, fsync after each line, strict
  reading on resume (partial, malformed, stale, duplicate or unexpected lines are
  quarantined, never merged). No reranking; duplicates kept; raw outputs retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GENERATOR_VERSION = "oneiros_native_generated_tests_generate_v2"
CONTRACT = {
    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "base_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    "sft_adapter": "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter",
    "seeds": [42, 43, 44], "candidates": 8, "temperature": 0.7, "top_p": 0.9,
    "do_sample": True, "max_new_tokens": 1024, "prompt_token_limit": 2048,
    "sequence_limit": 3072, "batch_size": 2, "extraction": "whole_output_single_fence_strip",
    "reranking": "none", "duplicates": "kept", "raw_output_retained": True,
    "attention_implementation": "sdpa",
}
ARMS = ("base", "sft")
CONDITIONS = ("primary_whole_module",)
REMOVED_CONDITIONS = {"secondary_scaffolded_diagnostic": "removed by amendment v2.1 section C"}
PROTOCOL_FILES = ("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md")
SOURCE_DIRS = ("engine", "harness", "scripts", "config")
AUTH_SCHEMA = "oneiros_native_gpu_authorization_v1"


class Refused(SystemExit):
    """A run that must not start or resume."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_sha(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def adapter_manifest(directory: Path) -> Dict[str, str]:
    """Every file in the adapter directory with its SHA-256 (adapter_config.json included)."""
    return {p.relative_to(directory).as_posix(): sha256_file(p)
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


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
                          "condition": sealed["condition"]})
    items.sort(key=lambda i: i["target_key"])
    body = {"items": items, "refused": refused, "sequence_fit": fit}
    return {**body, "job_sha256": contract_sha({"items": items})}


def select_condition(job_file: Mapping[str, Any], condition: str) -> Dict[str, Any]:
    if condition in REMOVED_CONDITIONS:
        raise Refused(f"REFUSED: condition {condition!r} {REMOVED_CONDITIONS[condition]}")
    if condition not in CONDITIONS or condition not in job_file:
        raise Refused(f"REFUSED: condition {condition!r} not in the job file")
    job = job_file[condition]
    if contract_sha({"items": job["items"]}) != job["job_sha256"]:
        raise Refused("REFUSED: job items do not match their hash")
    return job


# --- identity -------------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def source_tree_identity() -> Dict[str, Any]:
    from harness.source_identity import canonical_sha256
    files = [f for f in _git("ls-files", *SOURCE_DIRS).splitlines()
             if f.endswith((".py", ".sh", ".json"))]
    digest = {f: canonical_sha256(ROOT / f) for f in sorted(files) if (ROOT / f).is_file()}
    return {"files": len(digest), "sha256": contract_sha(digest)}


def model_identity() -> Dict[str, Any]:
    """Base snapshot, tokenizer and chat-template manifests (no model is loaded)."""
    from huggingface_hub import snapshot_download
    import transformers
    snapshot = Path(snapshot_download(CONTRACT["base_model"], revision=CONTRACT["base_revision"],
                                      local_files_only=True))
    files = {p.name: sha256_file(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    tokenizer_files = {k: v for k, v in files.items() if k.startswith(("tokenizer", "vocab",
                                                                       "merges", "special"))}
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    return {"snapshot_manifest_sha256": contract_sha(files),
            "tokenizer_manifest_sha256": contract_sha(tokenizer_files),
            "chat_template_sha256": sha256_text(tokenizer.chat_template or "")}


def library_versions() -> Dict[str, Optional[str]]:
    import importlib.metadata as md
    out = {}
    for name in ("transformers", "torch", "peft", "accelerate", "tokenizers"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
    except ImportError:
        out["cuda"] = None
    return out


def collect_identity(condition: str, arm: str, job_path: Path, job: Mapping[str, Any],
                     backend: str, preflight: Optional[Path], authorization: Optional[Path]
                     ) -> Dict[str, Any]:
    from harness.source_identity import canonical_sha256
    return {
        "backend": backend, "condition": condition, "arm": arm,
        "source_commit": _git("rev-parse", "HEAD"), "source_tree": source_tree_identity(),
        "generator_sha256": canonical_sha256(Path(__file__)),
        "prompt_builder_sha256": canonical_sha256(ROOT / "harness/native_generated_test_prompt.py"),
        "protocol_sha256": {p: sha256_file(ROOT / p) for p in PROTOCOL_FILES},
        "job_file_sha256": sha256_file(job_path), "job_sha256": job["job_sha256"],
        "model": model_identity(),
        "adapter_manifest_sha256": (contract_sha(adapter_manifest(ROOT / CONTRACT["sft_adapter"]))
                                    if arm == "sft" else None),
        "libraries": library_versions(),
        "preflight_sha256": sha256_file(preflight) if preflight else None,
        "authorization_sha256": sha256_file(authorization) if authorization else None,
    }


# --- GPU authorisation (source-stable) --------------------------------------------------------

def verify_authorization(path: Optional[Path], *, job: Mapping[str, Any], condition: str,
                         arm: str, out_dir: Path) -> Dict[str, Any]:
    """Refuse unless a receipt binds this exact preflight, source, protocol, job, arm,
    condition and output directory."""
    if path is None or not Path(path).is_file():
        raise Refused("REFUSED: GPU generation needs a GPU authorisation receipt")
    auth = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    if auth.get("schema_version") != AUTH_SCHEMA:
        problems.append("schema")
    preflight = ROOT / str(auth.get("preflight_path", ""))
    if not preflight.is_file() or sha256_file(preflight) != auth.get("preflight_sha256"):
        problems.append("preflight hash")
    else:
        receipt = json.loads(preflight.read_text(encoding="utf-8"))
        if receipt.get("pipeline_ready") is not True:
            problems.append("preflight not pipeline_ready")
        if receipt.get("source_commit") != auth.get("source_commit"):
            problems.append("preflight source differs")
    if _git("rev-parse", "HEAD") != auth.get("source_commit"):
        problems.append("source commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        problems.append("tracked source is dirty")
    if auth.get("protocol_sha256") != {p: sha256_file(ROOT / p) for p in PROTOCOL_FILES}:
        problems.append("protocol hash")
    if auth.get("job_sha256") != job["job_sha256"]:
        problems.append("job")
    if condition not in auth.get("allowed_conditions", []):
        problems.append("condition")
    if arm not in auth.get("allowed_arms", []):
        problems.append("arm")
    if Path(auth.get("output_dir", "")).resolve() != Path(out_dir).resolve():
        problems.append("output directory")
    if problems:
        raise Refused(f"REFUSED: authorisation does not match: {problems}")
    return auth


# --- durable run ------------------------------------------------------------------------------

def read_lines(path: Path, identity_hash: str, expected: set) -> tuple:
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
        if row.get("identity_sha256") != identity_hash:
            problems.append(f"line {number}: different identity")
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


def arm_contract(identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {"generator_version": GENERATOR_VERSION, "contract": dict(CONTRACT),
            "identity": dict(identity)}


def check_identity(out_dir: Path, contract: Mapping[str, Any], arm: str, condition: str) -> str:
    """Exact match resumes; any difference refuses and leaves the prior output untouched."""
    path = out_dir / f"contract_{condition}_{arm}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != dict(contract):
            changed = sorted(_diff_keys(stored, contract))
            raise Refused(f"REFUSED: {arm} identity differs in {changed}; the prior output is "
                          "preserved; use a new output directory")
        return "resume"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(contract), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return "fresh"


def _diff_keys(a: Mapping, b: Mapping, prefix: str = "") -> List[str]:
    out = []
    for key in set(a) | set(b):
        x, y = a.get(key), b.get(key)
        if isinstance(x, dict) and isinstance(y, dict):
            out += _diff_keys(x, y, f"{prefix}{key}.")
        elif x != y:
            out.append(f"{prefix}{key}")
    return out


def mock_backend(prompt: str, n: int, seed: int) -> List[str]:
    """Deterministic stand-in for tests and dry runs; never a model."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        value = rng.randint(0, 9)
        body = f"def test_mock_{i}():\n    assert {value} == {value}\n"
        out.append(f"```python\n{body}```" if i % 3 == 0 else body)
    return out


def run(job: Mapping[str, Any], arm: str, condition: str, out_dir: Path,
        identity: Mapping[str, Any], backend: Callable[[str, int, int], List[str]],
        crash_after: Optional[int] = None) -> Dict[str, Any]:
    if arm not in ARMS or condition not in CONDITIONS:
        raise Refused(f"REFUSED: unknown arm/condition {arm}/{condition}")
    contract = arm_contract(identity)
    ihash = contract_sha(contract)
    mode = check_identity(out_dir, contract, arm, condition)
    path = out_dir / f"generations_{condition}_{arm}.jsonl"
    expected = {f"{item['target_key']}::{seed}" for item in job["items"]
                for seed in CONTRACT["seeds"]}
    rows, problems = read_lines(path, ihash, expected)
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
                    "key": key, "arm": arm, "condition": condition, "seed": seed,
                    "target_seed": tseed, "target_key": item["target_key"],
                    "identity_sha256": ihash, "prompt_sha256": item["prompt_sha256"],
                    "raw_outputs": raws, "raw_sha256": [sha256_text(r) for r in raws],
                    "modules": [e["module"] for e in extracted],
                    "module_sha256": [sha256_text(e["module"]) for e in extracted],
                    "fence_stripped": [e["fence_stripped"] for e in extracted]},
                    sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                written += 1
    rows, problems = read_lines(path, ihash, expected)
    if problems or set(rows) != expected:
        raise RuntimeError(f"generation file incomplete: {problems[:3]}")
    return {"mode": mode, "identity_sha256": ihash, "lines": len(rows),
            "file_sha256": sha256_file(path)}


def hf_backend(arm: str):  # pragma: no cover - GPU path; only reachable with authorisation
    """Real sampling with the exact base revision (and the verified adapter for sft)."""
    import torch
    import transformers
    from engine.test_generation_prompt import format_chat_prompt
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CONTRACT["base_model"], revision=CONTRACT["base_revision"], local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        CONTRACT["base_model"], revision=CONTRACT["base_revision"], local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=CONTRACT["attention_implementation"]).to("cuda")
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
            if encoded["input_ids"].shape[1] > CONTRACT["prompt_token_limit"]:
                raise RuntimeError("prompt exceeds the frozen limit (never truncated)")
            with torch.no_grad():
                out = model.generate(**encoded, do_sample=CONTRACT["do_sample"],
                                     temperature=CONTRACT["temperature"],
                                     top_p=CONTRACT["top_p"],
                                     max_new_tokens=CONTRACT["max_new_tokens"],
                                     pad_token_id=tokenizer.pad_token_id)
            width = encoded["input_ids"].shape[1]
            outputs += [tokenizer.decode(row[width:], skip_special_tokens=True) for row in out]
        return outputs
    return generate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--job", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", required=True, choices=("mock", "hf"))
    parser.add_argument("--preflight", default=None)
    parser.add_argument("--authorization", default=None)
    args = parser.parse_args(argv)
    job_path = Path(args.job)
    job = select_condition(json.loads(job_path.read_text(encoding="utf-8")), args.condition)
    out = Path(args.out)
    auth = Path(args.authorization) if args.authorization else None
    if args.backend == "hf":
        verify_authorization(auth, job=job, condition=args.condition, arm=args.arm, out_dir=out)
        backend = hf_backend(args.arm)
    else:
        backend = mock_backend
    identity = collect_identity(args.condition, args.arm, job_path, job, args.backend,
                                Path(args.preflight) if args.preflight else None, auth)
    print(json.dumps(run(job, args.arm, args.condition, out, identity, backend)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
