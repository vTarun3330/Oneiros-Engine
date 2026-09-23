"""Run the predeclared 7B base-model TAP capacity gate.

This gate changes one factor relative to the frozen 1.5B base evidence: model
capacity within the same Qwen2.5-Coder-Instruct family.  It evaluates only the
7B base model on the exact frozen train-only TAP-ref and TAP-mut items.  The
1.5B LoRA is intentionally absent because it is architecture-specific and
cannot be attached to the 7B model without invalidating the comparison.

Generation is greedy, raw outputs are retained in full, and each completed
condition is checkpointed atomically.  Resume refuses any model, data, source,
or generation-contract drift.  No validation or sealed-final data is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_tap_adapter_compare import report, score  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
SCHEMA_VERSION = "oneiros_tap_7b_capacity_v1"
EXPECTED_KEYS = ("base7b::TAP-ref", "base7b::TAP-mut")
TOKENIZER_MAX_LENGTH = 3072


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_path(output: Path) -> Path:
    return output.with_suffix(".partial.json")


def source_bundle() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "tap_helpers": ROOT / "scripts" / "run_tap_adapter_compare.py",
        "tap_scorer": ROOT / "scripts" / "run_tap_diagnostic.py",
        "generator": ROOT / "engine" / "generator.py",
        "settings": ROOT / "config" / "settings.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def build_contract(*, args, items_path: Path, item_ids: list[str]) -> dict:
    from config.settings import immutable_revision_for, model_config

    pinned = immutable_revision_for(MODEL_NAME)
    if pinned != MODEL_REVISION:
        raise RuntimeError(
            f"REFUSED: configured 7B revision {pinned!r} != gate pin {MODEL_REVISION}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "items_file": str(items_path.relative_to(ROOT)),
        "items_file_sha256": sha256_file(items_path),
        "item_ids_sha256": hashlib.sha256(
            json.dumps(item_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "items": len(item_ids),
        "permitted_split": "train",
        "decoding": "greedy",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "batch": args.batch,
        "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
        "attention_implementation": "sdpa",
        "quantization": {
            "load_in_4bit": True,
            "quant_type": model_config.bnb_4bit_quant_type,
            "double_quant": model_config.bnb_4bit_use_double_quant,
            "compute_dtype": "bf16_if_supported_else_fp16",
        },
        "weights_written": False,
        "source_files_sha256": source_bundle(),
    }


def validate_items(items: list[dict]) -> None:
    if len(items) != 600:
        raise RuntimeError(f"REFUSED: expected exactly 600 frozen TAP items, got {len(items)}")
    ids = [str(item.get("id")) for item in items]
    if len(ids) != len(set(ids)):
        raise RuntimeError("REFUSED: duplicate TAP item IDs")
    bad_splits = sorted({item.get("split") for item in items if item.get("split") != "train"})
    if bad_splits:
        raise RuntimeError(f"REFUSED: non-train TAP rows present: {bad_splits}")
    for item in items:
        for field in ("prompt_ref", "prompt_mut", "expected_repr", "benchmark"):
            if field not in item:
                raise RuntimeError(f"REFUSED: item {item.get('id')} omits {field}")


def validate_rows(results: dict, item_ids: list[str]) -> None:
    expected = set(item_ids)
    for key, result in results.items():
        rows = result.get("detail", [])
        ids = [row.get("id") for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError(f"checkpoint {key} item IDs do not match the run contract")
        for row in rows:
            raw = row.get("raw")
            digest = row.get("raw_sha256")
            if not isinstance(raw, str) or digest != hashlib.sha256(
                    raw.encode("utf-8")).hexdigest():
                raise RuntimeError(f"checkpoint {key}/{row.get('id')} has invalid raw evidence")
            if row.get("raw_chars") != len(raw):
                raise RuntimeError(f"checkpoint {key}/{row.get('id')} has invalid raw length")


def load_checkpoint(path: Path, contract: dict,
                    item_ids: list[str]) -> tuple[dict, str]:
    if not path.exists():
        return {}, datetime.now(timezone.utc).isoformat()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("run_contract") != contract:
        raise RuntimeError(
            f"REFUSED: checkpoint contract mismatch at {path}; preserve it and use "
            "a new output path for a different run"
        )
    results = saved.get("results", {})
    unknown = sorted(set(results) - set(EXPECTED_KEYS))
    if unknown:
        raise RuntimeError(f"checkpoint has unknown conditions: {unknown}")
    validate_rows(results, item_ids)
    return results, saved.get("created_utc", datetime.now(timezone.utc).isoformat())


def checkpoint_payload(contract: dict, results: dict, created_utc: str,
                       *, status: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "label": "train-only capacity diagnostic; not validation or model promotion",
        "status": status,
        "created_utc": created_utc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "run_contract": contract,
        "completed_conditions": sorted(results),
        "results": results,
    }


def final_payload(*, contract: dict, results: dict, created_utc: str,
                  elapsed: float, runtime_profile: dict | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "label": (
            "train-only quantized capacity diagnostic; not generalization, "
            "real-repository, validation, final-test, or promotion evidence"
        ),
        "status": "complete",
        "created_utc": created_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_contract": contract,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "runtime_profile": runtime_profile,
        "wall_time_seconds": round(elapsed, 1),
        "weights_written": False,
        "training_performed": False,
        "arms": {
            key: {
                "per_benchmark": value["per_benchmark"],
                "truncated": value["truncated"],
            }
            for key, value in results.items()
        },
        "detail": {key: value["detail"] for key, value in results.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="results/tap_train_items.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default="results/tap_7b_capacity.json")
    args = parser.parse_args(argv)

    if args.max_new_tokens != 128:
        print("REFUSED: the frozen gate requires max_new_tokens=128")
        return 2
    if args.batch < 1:
        print("REFUSED: batch must be positive")
        return 2

    items_path = (ROOT / args.items).resolve()
    output_path = (ROOT / args.out).resolve()
    partial_path = checkpoint_path(output_path)
    if ROOT not in items_path.parents or ROOT not in output_path.parents:
        print("REFUSED: inputs and outputs must remain inside the repository")
        return 2
    items = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        validate_items(items)
        contract = build_contract(
            args=args,
            items_path=items_path,
            item_ids=[item["id"] for item in items],
        )
        results, created_utc = load_checkpoint(
            partial_path, contract, [item["id"] for item in items]
        )
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2

    if output_path.exists() and set(results) != set(EXPECTED_KEYS):
        print(f"REFUSED: final output already exists without a complete matching checkpoint: {output_path}")
        return 2
    if results:
        print(f"resuming {len(results)}/{len(EXPECTED_KEYS)} conditions", flush=True)

    def write_progress(status: str = "in_progress") -> None:
        atomic_json_write(
            partial_path,
            checkpoint_payload(contract, results, created_utc, status=status),
        )

    if set(results) == set(EXPECTED_KEYS):
        artifact = final_payload(
            contract=contract,
            results=results,
            created_utc=created_utc,
            elapsed=0.0,
            runtime_profile=None,
        )
        atomic_json_write(output_path, artifact)
        write_progress("complete")
        print(f"written: {output_path}")
        return 0

    import torch
    from engine.generator import Phi3Generator

    print(f"loading {MODEL_NAME} @ {MODEL_REVISION} (4-bit base only)", flush=True)
    generator = Phi3Generator(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        attention_implementation="sdpa",
        load_in_4bit=True,
    )
    generator.load_model()
    tokenizer, model = generator.tokenizer, generator.model
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def complete(prompts: list[str]) -> tuple[list[str], list[bool]]:
        outputs: list[str] = []
        truncated: list[bool] = []
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in chunk
            ]
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=TOKENIZER_MAX_LENGTH,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for index in range(len(chunk)):
                new_tokens = generated[index][encoded.input_ids.shape[1]:]
                outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
                retained = [
                    token for token in new_tokens.tolist()
                    if token != tokenizer.pad_token_id
                ]
                truncated.append(len(retained) >= args.max_new_tokens)
            print(f"      {min(start + args.batch, len(prompts))}/{len(prompts)}", flush=True)
        return outputs, truncated

    started = time.time()
    for condition, prompt_key in (("TAP-ref", "prompt_ref"), ("TAP-mut", "prompt_mut")):
        result_key = f"base7b::{condition}"
        if result_key in results:
            print(f"--- {result_key}: checkpoint complete; skipping ---", flush=True)
            continue
        print(f"--- {result_key} ---", flush=True)
        outputs, truncated = complete([item[prompt_key] for item in items])
        per_benchmark, detail = score(items, outputs, truncated)
        results[result_key] = {
            "per_benchmark": per_benchmark,
            "detail": detail,
            "truncated": sum(truncated),
        }
        write_progress()
        report(result_key, per_benchmark)
        print(f"  hit token cap: {sum(truncated)}/{len(items)}", flush=True)

    elapsed = time.time() - started
    artifact = final_payload(
        contract=contract,
        results=results,
        created_utc=created_utc,
        elapsed=elapsed,
        runtime_profile=generator.runtime_profile,
    )
    atomic_json_write(output_path, artifact)
    write_progress("complete")
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
