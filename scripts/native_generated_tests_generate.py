"""Durable candidate generation for native generated tests (protocol v2 + amendment v2.1;
generator v5 for protocol v2.5: the CLI runs only a job built by
``harness.native_generated_test_job_v25`` with the current pytest_module_v1 builder hash).

    run --job FILE --condition primary_whole_module --arm {base,sft} --out DIR
        --backend {mock,hf} [--preflight FILE --authorization FILE]

* ``--condition`` selects the nested job inside the job file. Only
  ``primary_whole_module`` exists (v2.1 section C removed the scaffolded diagnostic).
* Every arm run has an IMMUTABLE identity: source commit and canonical source tree,
  generator, prompt-builder and protocol/amendment hashes, the complete job hash, the base
  snapshot, tokenizer and chat-template manifests, the adapter-directory manifest (sft),
  library and CUDA versions, the attention implementation, the FULL generation contract,
  the condition, the arm, the preflight hash and (for GPU runs) the authorisation hash.
  A change to any field refuses to resume and leaves the prior output untouched; a new
  output directory is required.
* The Hugging Face backend is committed source but runs only when the read-only launch
  gate (harness/native_launch_gate.py, amendment v2.2 section D) reports launch_ready: an
  immutable green preflight that still describes the checkout, plus an authorisation
  receipt whose content binds that preflight, the source identity, the job, protocol, model
  and adapter hashes, the condition, the arm and the output directory. Passing
  --authorization with the mock backend runs the same gate. No source edit is ever needed
  to authorise a run.
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

from scripts.native_generation_io import (TELEMETRY_SCHEMA, extract, row_problems,  # noqa: E402
                                          target_seed)

GENERATOR_VERSION = "oneiros_native_generated_tests_generate_v5"   # v2.5: v2.5 jobs only
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
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_2.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_3.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_4.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_1.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_2.md")
SOURCE_DIRS = ("engine", "harness", "scripts", "config")


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


# --- job ------------------------------------------------------------------------------------

def sequence_fit(prompts: Sequence[Mapping[str, Any]], count_tokens: Callable[[str], int],
                 limit: int = CONTRACT["prompt_token_limit"]) -> Dict[str, Any]:
    admitted, refused = [], []
    for sealed in prompts:
        tokens = count_tokens(sealed["prompt"])
        entry = {"target_key": sealed["target_key"], "prompt_tokens": tokens}
        (admitted if tokens <= limit else refused).append(entry)
    return {"admitted": admitted, "refused": refused, "limit": limit}


REFUSAL_REASONS = ("prompt_refused", "seal_mismatch", "leakage", "sequence_overflow")


def build_job(prompts: Sequence[Mapping[str, Any]], scans: Mapping[str, Mapping[str, Any]],
              fit: Mapping[str, Any],
              builder_refused: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """Only prompts that are sealed, leakage-clean and fit enter the job. Every check is
    evaluated for every target and EVERY applicable reason is recorded (v2.2 section B)."""
    fits = {e["target_key"] for e in fit["admitted"]}
    measured = {e["target_key"] for e in fit["admitted"]} | {e["target_key"] for e in fit["refused"]}
    items, refused = [], []
    for b in builder_refused:
        refused.append({"target_key": b["target_key"], "reasons": ["prompt_refused"],
                        "details": {"prompt_refused": b.get("reason")}})
    for sealed in prompts:
        key = sealed["target_key"]
        reasons, details = [], {}
        if sha256_text(sealed["prompt"]) != sealed["prompt_sha256"]:
            reasons.append("seal_mismatch")
        scan = scans.get(key)
        if not scan or not scan.get("ok"):
            reasons.append("leakage")
            details["leakage"] = (scan or {}).get("reasons") or ["not scanned"]
        if key not in fits:
            reasons.append("sequence_overflow")
            if key not in measured:
                details["sequence_overflow"] = ["not measured"]
        if reasons:
            refused.append({"target_key": key, "reasons": reasons, "details": details})
        else:
            items.append({"target_key": key, "prompt": sealed["prompt"],
                          "prompt_sha256": sealed["prompt_sha256"],
                          "condition": sealed["condition"]})
    items.sort(key=lambda i: i["target_key"])
    refused.sort(key=lambda r: r["target_key"])
    body = {"items": items, "refused": refused, "sequence_fit": fit}
    return {**body, "job_sha256": contract_sha({"items": items})}


def refusal_accounting(kept_keys: Sequence[str], job: Mapping[str, Any]) -> Dict[str, Any]:
    """Coverage counts UNIQUE admitted targets, never sums of reason counts."""
    admitted = sorted({i["target_key"] for i in job["items"]})
    refused = {r["target_key"]: tuple(r["reasons"]) for r in job["refused"]}
    per_reason = {reason: sum(reason in rs for rs in refused.values())
                  for reason in REFUSAL_REASONS}
    combos: Dict[str, int] = {}
    for rs in refused.values():
        label = "+".join(sorted(rs))
        combos[label] = combos.get(label, 0) + 1
    kept = set(kept_keys)
    accounted = set(admitted) | set(refused)
    return {"denominator_kept": len(kept), "admitted": len(admitted),
            "admitted_fraction": round(len(admitted) / len(kept), 4) if kept else 0.0,
            "unique_refused": len(refused), "per_reason": per_reason,
            "combinations": dict(sorted(combos.items())),
            "overlapping_targets": sorted(k for k, rs in refused.items() if len(rs) > 1),
            "unaccounted": sorted(kept - accounted), "unexpected": sorted(accounted - kept),
            "admitted_and_refused": sorted(set(admitted) & set(refused))}


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
                     backend: str, preflight: Optional[Path], authorization: Optional[Path],
                     prior_arm: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    from harness.source_identity import canonical_sha256
    return {
        "backend": backend, "condition": condition, "arm": arm,
        "source_commit": _git("rev-parse", "HEAD"), "source_tree": source_tree_identity(),
        "generator_sha256": canonical_sha256(Path(__file__)),
        # v2.5: the pytest_module_v1 builder AND the v2.4 permitted-view code it reuses
        "prompt_builder_sha256": {
            "v25_pytest_module_v1": canonical_sha256(
                ROOT / "harness/native_generated_test_prompt_v25.py"),
            "permitted_view": canonical_sha256(ROOT / "harness/native_generated_test_prompt.py")},
        "protocol_sha256": {p: sha256_file(ROOT / p) for p in PROTOCOL_FILES},
        "job_file_sha256": sha256_file(job_path), "job_sha256": job["job_sha256"],
        "model": model_identity(),
        "adapter_manifest_sha256": (contract_sha(adapter_manifest(ROOT / CONTRACT["sft_adapter"]))
                                    if arm == "sft" else None),
        "libraries": library_versions(),
        "preflight_sha256": sha256_file(preflight) if preflight else None,
        "authorization_sha256": sha256_file(authorization) if authorization else None,
        # sft only: the verified base arm it follows (v2.4 I.2); arm-specific by design
        "prior_arm_verification": dict(prior_arm) if prior_arm else None,
    }


# --- GPU authorisation (amendment v2.2 section D) ---------------------------------------------

def adapter_sha256() -> str:
    return contract_sha(adapter_manifest(ROOT / CONTRACT["sft_adapter"]))


def launch_gate(preflight: Optional[Path], authorization: Optional[Path], *, job_path: Path,
                condition: str, arm: str, out_dir: Path, backend: str = "hf") -> Dict[str, Any]:
    """Refuse unless the read-only launch gate reports launch_ready. Writes nothing."""
    from harness.native_launch_gate import evaluate
    if preflight is None or authorization is None:
        raise Refused("REFUSED: GPU generation needs --preflight and a GPU authorisation receipt")
    result = evaluate(ROOT, Path(preflight), Path(authorization), job_path=Path(job_path),
                      condition=condition, arm=arm, out_dir=Path(out_dir),
                      model_identity=model_identity, adapter_sha256=adapter_sha256,
                      backend=backend, generation_source=source_tree_identity)
    if not result["launch_ready"]:
        raise Refused("REFUSED: launch gate not ready: " + json.dumps(
            {k: result[k] for k in ("pipeline_ready", "gpu_authorized", "pipeline_problems",
                                    "authorization_problems")}))
    return result


# --- durable run ------------------------------------------------------------------------------

def read_lines(path: Path, identity_hash: str, expected: set,
               validate: Optional[Callable[[Mapping[str, Any]], List[str]]] = None) -> tuple:
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
        elif validate is not None and validate(row):
            problems.append(f"line {number}: telemetry {validate(row)[:2]}")
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
    from harness.native_generated_test_job_v25 import JOB_SCHEMA
    return {"generator_version": GENERATOR_VERSION, "job_schema": JOB_SCHEMA,
            "contract": dict(CONTRACT), "telemetry_schema": TELEMETRY_SCHEMA,
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


def _batches(candidates: List[Dict[str, Any]], walls: Sequence[float]) -> List[Dict[str, Any]]:
    size = CONTRACT["batch_size"]
    return [{"wall_seconds": walls[i // size], "candidates": candidates[i:i + size]}
            for i in range(0, len(candidates), size)]


def mock_backend(prompt: str, n: int, seed: int) -> Dict[str, Any]:
    """Deterministic stand-in for tests and dry runs; never a model. Same telemetry schema:
    token counts are whitespace words (mock convention), every fourth candidate is a
    completion-limit hit, timings are deterministic and GPU fields are null."""
    rng = random.Random(seed)
    candidates = []
    for i in range(n):
        value = rng.randint(0, 9)
        body = f"def test_mock_{i}():\n    assert {value} == {value}\n"
        raw = f"```python\n{body}```" if i % 3 == 0 else body
        limit = i % 4 == 3
        candidates.append({"raw": raw, "eos_reached": not limit,
                           "generated_tokens": CONTRACT["max_new_tokens"] if limit
                           else len(raw.split()) + 1})
    walls = [round(0.01 * (b + 1), 6) for b in range(-(-n // CONTRACT["batch_size"]))]
    return {"prompt_tokens": len(prompt.split()), "batches": _batches(candidates, walls),
            "model_load_seconds": 0.0, "peak_allocated_bytes": None, "peak_reserved_bytes": None,
            "process_peak_allocated_bytes": None, "process_peak_reserved_bytes": None}


def texts_backend(texts: Callable[[str, int, int], List[str]]):
    """Wrap a backend that returns raw strings (synthetic pipeline): every candidate is an
    EOS finish of whitespace-word length; deterministic timings."""
    def backend(prompt: str, n: int, seed: int) -> Dict[str, Any]:
        raws = texts(prompt, n, seed)
        candidates = [{"raw": r, "eos_reached": True, "generated_tokens": len(r.split()) + 1}
                      for r in raws]
        walls = [0.01] * (-(-len(raws) // CONTRACT["batch_size"]))
        return {"prompt_tokens": len(prompt.split()), "batches": _batches(candidates, walls),
                "model_load_seconds": 0.0, "peak_allocated_bytes": None,
                "peak_reserved_bytes": None, "process_peak_allocated_bytes": None,
                "process_peak_reserved_bytes": None}
    return backend


def build_row(*, key: str, arm: str, condition: str, seed: int, tseed: int,
              item: Mapping[str, Any], ihash: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    """One target x seed row in telemetry schema v1 (amendment v2.3 section B)."""
    produced = [c for b in result["batches"] for c in b["candidates"]]
    walls = [round(float(b["wall_seconds"]), 6) for b in result["batches"]]
    candidates = []
    for c in produced:
        e = extract(c["raw"])
        candidates.append({
            "raw": c["raw"], "raw_sha256": sha256_text(c["raw"]),
            "module": e["module"], "module_sha256": sha256_text(e["module"]),
            "generated_tokens": c["generated_tokens"], "eos_reached": c["eos_reached"],
            "finish_reason": "eos" if c["eos_reached"] else "length",
            "hit_completion_limit": not c["eos_reached"],
            "fence_stripped": e["fence_stripped"]})
    return {"key": key, "arm": arm, "condition": condition, "seed": seed, "target_seed": tseed,
            "target_key": item["target_key"], "identity_sha256": ihash,
            "prompt_sha256": item["prompt_sha256"], "telemetry_schema": TELEMETRY_SCHEMA,
            "prompt_tokens": result["prompt_tokens"], "batch_wall_seconds": walls,
            "wall_seconds": round(sum(walls), 6),
            "candidates_requested": CONTRACT["candidates"], "candidates_produced": len(produced),
            "model_load_seconds": result["model_load_seconds"],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"],
            "process_peak_allocated_bytes": result["process_peak_allocated_bytes"],
            "process_peak_reserved_bytes": result["process_peak_reserved_bytes"],
            "candidates": candidates}


def run(job: Mapping[str, Any], arm: str, condition: str, out_dir: Path,
        identity: Mapping[str, Any], backend: Callable[[str, int, int], Mapping[str, Any]],
        crash_after: Optional[int] = None) -> Dict[str, Any]:
    if arm not in ARMS or condition not in CONDITIONS:
        raise Refused(f"REFUSED: unknown arm/condition {arm}/{condition}")
    contract = arm_contract(identity)
    ihash = contract_sha(contract)
    gpu = identity.get("backend") == "hf"
    prompts = {item["target_key"]: item["prompt_sha256"] for item in job["items"]}

    def validate(row: Mapping[str, Any]) -> List[str]:
        return row_problems(row, identity_sha256=ihash, arm=arm, condition=condition,
                            prompt_sha256=prompts.get(row.get("target_key")),
                            require_gpu_evidence=gpu, candidates=CONTRACT["candidates"],
                            batch_size=CONTRACT["batch_size"],
                            max_new_tokens=CONTRACT["max_new_tokens"])
    mode = check_identity(out_dir, contract, arm, condition)
    path = out_dir / f"generations_{condition}_{arm}.jsonl"
    expected = {f"{item['target_key']}::{seed}" for item in job["items"]
                for seed in CONTRACT["seeds"]}
    rows, problems = read_lines(path, ihash, expected, validate)
    if problems:
        quarantine(path, problems)
        rows = {}
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for seed in CONTRACT["seeds"]:
            for item in job["items"]:
                key = f"{item['target_key']}::{seed}"
                if key in rows:
                    continue                      # a resumed row keeps its original telemetry
                if crash_after is not None and written >= crash_after:
                    raise RuntimeError("deliberate crash for the resume test")
                tseed = target_seed(seed, item["target_key"])
                row = build_row(key=key, arm=arm, condition=condition, seed=seed, tseed=tseed,
                                item=item, ihash=ihash,
                                result=backend(item["prompt"], CONTRACT["candidates"], tseed))
                bad = validate(row)
                if bad:                           # never written, never accepted
                    raise RuntimeError(f"generation row {key} refused: {bad[:3]}")
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                written += 1
    rows, problems = read_lines(path, ihash, expected, validate)
    if problems or set(rows) != expected:
        raise RuntimeError(f"generation file incomplete: {problems[:3]}")
    return {"mode": mode, "identity_sha256": ihash, "lines": len(rows),
            "file_sha256": sha256_file(path)}


def measured_row(cuda, generate_row: Callable[[], Dict[str, Any]],
                 process: Dict[str, int]) -> Dict[str, Any]:
    """Row-scoped CUDA peaks (v2.4 B.4): synchronise and reset the peak statistics
    immediately before the target-seed row, synchronise after it, then read that row's
    peaks. Process-lifetime peaks are carried separately in ``process`` (updated here)."""
    cuda.synchronize()
    cuda.reset_peak_memory_stats()
    result = generate_row()
    cuda.synchronize()
    allocated, reserved = int(cuda.max_memory_allocated()), int(cuda.max_memory_reserved())
    process["allocated"] = max(process["allocated"], allocated)
    process["reserved"] = max(process["reserved"], reserved)
    return {**result, "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "process_peak_allocated_bytes": process["allocated"],
            "process_peak_reserved_bytes": process["reserved"]}


def eos_ids(model, tokenizer) -> set:  # pragma: no cover - GPU path
    """Every EOS id of the model's generation config plus the tokenizer's EOS."""
    configured = getattr(model.generation_config, "eos_token_id", None)
    ids = set(configured if isinstance(configured, (list, tuple)) else
              ([configured] if configured is not None else []))
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def finish(ids: Sequence[int], stop: set) -> Dict[str, Any]:
    """Exact generated token IDs -> count (through the first EOS, excluding later batch
    padding, even when pad_token_id is an EOS id) and EOS status. Never re-tokenises text."""
    for index, token in enumerate(ids):
        if token in stop:
            return {"generated_tokens": index + 1, "eos_reached": True, "keep": index + 1}
    return {"generated_tokens": len(ids), "eos_reached": False, "keep": len(ids)}


def hf_backend(arm: str):  # pragma: no cover - GPU path; only reachable with authorisation
    """Real sampling with the exact base revision (and the verified adapter for sft)."""
    import torch
    import transformers
    from engine.test_generation_prompt import format_chat_prompt
    torch.cuda.synchronize()
    started = time.perf_counter()
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
    torch.cuda.synchronize()
    load_seconds = round(time.perf_counter() - started, 3)
    stop = eos_ids(model, tokenizer)
    process = {"allocated": int(torch.cuda.max_memory_allocated()),     # includes model load
               "reserved": int(torch.cuda.max_memory_reserved())}

    def generate(prompt: str, n: int, seed: int) -> Dict[str, Any]:
        return measured_row(torch.cuda, lambda: _generate(prompt, n, seed), process)

    def _generate(prompt: str, n: int, seed: int) -> Dict[str, Any]:
        text = format_chat_prompt(tokenizer, prompt)
        prompt_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if prompt_tokens > CONTRACT["prompt_token_limit"]:
            raise RuntimeError("prompt exceeds the frozen limit (never truncated)")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        batches = []
        for start in range(0, n, CONTRACT["batch_size"]):
            batch = [text] * min(CONTRACT["batch_size"], n - start)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**encoded, do_sample=CONTRACT["do_sample"],
                                     temperature=CONTRACT["temperature"],
                                     top_p=CONTRACT["top_p"],
                                     max_new_tokens=CONTRACT["max_new_tokens"],
                                     pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            width = encoded["input_ids"].shape[1]
            candidates = []
            for row in out:
                ids = row[width:].tolist()
                f = finish(ids, stop)
                candidates.append({"raw": tokenizer.decode(ids[:f["keep"]],
                                                           skip_special_tokens=True),
                                   "generated_tokens": f["generated_tokens"],
                                   "eos_reached": f["eos_reached"]})
            batches.append({"wall_seconds": wall, "candidates": candidates})
        return {"prompt_tokens": prompt_tokens, "batches": batches,
                "model_load_seconds": load_seconds}
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
    # v2.5: only a job made by the v2.5 builder with the CURRENT builder hash runs (a v2.4 job,
    # a stale builder or a non-pytest_module_v1 item refuses before anything is written)
    from harness.native_generated_test_job_v25 import validate_job_file
    job_file = json.loads(job_path.read_text(encoding="utf-8"))
    validate_job_file(job_file, args.condition, ROOT)
    job = select_condition(job_file, args.condition)
    out = Path(args.out)
    auth = Path(args.authorization) if args.authorization else None
    preflight = Path(args.preflight) if args.preflight else None
    gate = None
    if args.backend == "hf" or auth is not None:
        # the same read-only gate for the GPU path and for its mock rehearsal
        gate = launch_gate(preflight, auth, job_path=job_path, condition=args.condition,
                           arm=args.arm, out_dir=out, backend=args.backend)
    backend = hf_backend(args.arm) if args.backend == "hf" else mock_backend
    identity = collect_identity(args.condition, args.arm, job_path, job, args.backend,
                                Path(args.preflight) if args.preflight else None, auth,
                                (gate or {}).get("base_arm_verification"))
    print(json.dumps(run(job, args.arm, args.condition, out, identity, backend)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
