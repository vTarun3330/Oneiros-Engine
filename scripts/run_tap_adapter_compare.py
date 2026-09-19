"""Blocking gate 1: did ordinary SFT improve, preserve or damage output prediction?

Runs the frozen TAP items through the base model and through the A@431 adapter
under IDENTICAL prompts, decoding and token budget, so the only difference is
the weights.

Two corrections to the first TAP run are applied here:

  * ``--max-new-tokens`` defaults to 128, not 48. Inspection of the first run
    showed ``prediction_not_literal`` was dominated by literals cut off
    mid-value, which is a budget artifact that contaminated the denominator.
  * raw model output is retained, with a truncation flag. The first run stored
    only the extracted right-hand side, which made the 9.8% ``no_assertion``
    bucket impossible to adjudicate after the fact.

Both end-to-end accuracy (over all items) and conditional accuracy (over items
that produced a parseable literal) are reported. Neither replaces the other.

A third arm re-runs A@431 with the Oneiros SFT system prompt prepended. A@431
was tuned on the test-generation prompt, not on bare assertion completion, so a
drop under the bare prompt could be format mismatch rather than damaged output
prediction. This arm separates those two explanations.

Diagnostic only. Permitted splits, greedy decoding, no weights written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_tap_diagnostic import asserted_rhs, values_match  # noqa: E402

ADAPTER = ROOT / "checkpoints" / "local_sft_armA_baseline_successor_s42" / "sft_adapter"
#: byte hash of adapter_model.safetensors, matching the adapter recorded in
#: results/locked_val_armA_ckpt431_s42.log for the locked validation run.
ADAPTER_SHA256 = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"
SCHEMA_VERSION = "oneiros_tap_adapter_compare_v2"
EXPECTED_KEYS = tuple(
    f"{arm}::{condition}"
    for arm in ("base", "a431", "a431_sysprompt")
    for condition in ("TAP-ref", "TAP-mut")
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json_write(path: Path, payload: dict) -> None:
    """Write JSON atomically so an interruption cannot corrupt a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_path(output: Path) -> Path:
    return output.with_suffix(".partial.json")


def source_bundle() -> dict[str, str]:
    """Hashes of the files that define generation and TAP scoring.

    The diagnostic was launched from a dirty worktree, so binding the commit
    alone would be false precision.  These content hashes identify the bytes
    the process actually imported.
    """
    files = {
        "runner": Path(__file__).resolve(),
        "tap_scorer": ROOT / "scripts" / "run_tap_diagnostic.py",
        "generator": ROOT / "engine" / "generator.py",
        "prompt_engine": ROOT / "engine" / "test_generation_prompt.py",
        "settings": ROOT / "config" / "settings.py",
    }
    return {
        name: sha256_file(path)
        for name, path in files.items()
    }


def build_contract(*, args, items_path: Path, item_ids: list[str], model: str,
                   revision: str, system_prompt: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "model_revision": revision,
        "adapter_sha256": ADAPTER_SHA256,
        "items_file": str(items_path),
        "items_file_sha256": sha256_file(items_path),
        "item_ids_sha256": hashlib.sha256(
            json.dumps(item_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "items": len(item_ids),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "batch": args.batch,
        "tokenizer_max_length": 3072,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "source_files_sha256": source_bundle(),
    }


def validate_rows(results: dict, item_ids: list[str]) -> None:
    expected = set(item_ids)
    for key, result in results.items():
        rows = result.get("detail", [])
        ids = [row.get("id") for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"checkpoint {key} contains duplicate item IDs")
        if set(ids) != expected:
            raise RuntimeError(f"checkpoint {key} item IDs do not match the run contract")
        for row in rows:
            raw = row.get("raw")
            digest = row.get("raw_sha256")
            if not isinstance(raw, str) or digest != hashlib.sha256(
                    raw.encode("utf-8")).hexdigest():
                raise RuntimeError(f"checkpoint {key}/{row.get('id')} has invalid raw evidence")


def load_checkpoint(path: Path, contract: dict, item_ids: list[str]) -> tuple[dict, str]:
    if not path.exists():
        return {}, datetime.now(timezone.utc).isoformat()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("run_contract") != contract:
        raise RuntimeError(
            f"REFUSED: checkpoint contract mismatch at {path}; preserve it and "
            "use a new output path for a different run"
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
        "label": "diagnostic only - not model selection or a generalization result",
        "status": status,
        "created_utc": created_utc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "run_contract": contract,
        "completed_conditions": sorted(results),
        "results": results,
    }


def score(items, outs, trunc):
    per = defaultdict(Counter)
    detail = []
    for item, text, cut in zip(items, outs, trunc):
        rhs = asserted_rhs(text)
        verdict = "no_assertion" if rhs is None else values_match(rhs, item["expected_repr"])
        per[item["benchmark"]][verdict] += 1
        per["ALL"][verdict] += 1
        # The complete generation is retained, never a prefix. An earlier
        # version clipped this to 300 characters, which silently discarded most
        # of the adapter's long outputs and left the recorded hashes covering a
        # prefix rather than what the model actually produced.
        detail.append({"id": item["id"], "benchmark": item["benchmark"],
                       "verdict": verdict, "truncated": cut,
                       "expected": item["expected_repr"],
                       "predicted": rhs or "",
                       "raw": text,
                       "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                       "raw_chars": len(text)})
    return {k: dict(v) for k, v in per.items()}, detail


def report(tag, per):
    for bench in ("ALL", "humaneval", "mbpp"):
        counts = per.get(bench)
        if not counts:
            continue
        n = sum(counts.values())
        ok = counts.get("correct", 0)
        parseable = ok + counts.get("wrong_value", 0)
        conditional = f"{ok / parseable:6.1%}" if parseable else "   n/a"
        print(f"  {tag:16} {bench:10} n={n:4d}  end-to-end {ok / n:6.1%}  "
              f"conditional {conditional}  no_assert {counts.get('no_assertion', 0):3d}  "
              f"non_literal {counts.get('prediction_not_literal', 0):3d}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="results/tap_train_items.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default="results/tap_adapter_compare.json")
    args = parser.parse_args(argv)

    digest = hashlib.sha256((ADAPTER / "adapter_model.safetensors").read_bytes()).hexdigest()
    if digest != ADAPTER_SHA256:
        print(f"REFUSED: adapter hash {digest} != recorded {ADAPTER_SHA256}")
        return 2
    print(f"adapter verified: {digest[:16]}... == locked-val A@431")

    items_path = Path(args.items).resolve()
    output_path = Path(args.out).resolve()
    partial_path = checkpoint_path(output_path)
    lines = items_path.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    if args.limit:
        items = items[:args.limit]
    print(f"TAP items: {len(items)}  {dict(Counter(i['benchmark'] for i in items))}")

    import torch
    from engine.generator import Phi3Generator
    from engine.test_generation_prompt import SYSTEM_PROMPT
    from config.settings import immutable_revision_for

    name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    revision = immutable_revision_for(name)
    contract = build_contract(
        args=args,
        items_path=items_path,
        item_ids=[item["id"] for item in items],
        model=name,
        revision=revision,
        system_prompt=SYSTEM_PROMPT,
    )
    try:
        results, created_utc = load_checkpoint(
            partial_path, contract, [item["id"] for item in items]
        )
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2
    if results:
        print(f"resuming {len(results)}/{len(EXPECTED_KEYS)} completed conditions "
              f"from {partial_path}", flush=True)

    def write_progress(status="in_progress"):
        atomic_json_write(
            partial_path,
            checkpoint_payload(contract, results, created_utc, status=status),
        )

    if set(results) == set(EXPECTED_KEYS):
        print("all conditions already checkpointed; rebuilding final artifact", flush=True)
        elapsed = 0.0
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "label": "diagnostic only - not model selection, not a performance claim",
            "status": "complete",
            "run_contract": contract,
            "model": name,
            "model_revision": revision,
            "adapter_path": str(ADAPTER),
            "adapter_sha256": digest,
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "items_file": str(items_path),
            "items": len(items),
            "wall_time_seconds": elapsed,
            "weights_written": False,
            "arms": {k: {"per_benchmark": v["per_benchmark"],
                         "truncated": v["truncated"]}
                     for k, v in results.items()},
            "detail": {k: v["detail"] for k, v in results.items()},
        }
        atomic_json_write(output_path, artifact)
        write_progress("complete")
        print(f"written: {output_path}")
        return 0

    gen = Phi3Generator(model_name=name, model_revision=revision,
                        attention_implementation="sdpa")
    gen.load_model()
    tokenizer = gen.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def complete(prompts, system=None):
        outs, trunc = [], []
        model = gen.model
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            messages = [([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": p}] for p in chunk]
            texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                                   add_generation_prompt=True)
                     for m in messages]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=3072).to(model.device)
            with torch.inference_mode():
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            for j in range(len(chunk)):
                new = ids[j][enc.input_ids.shape[1]:]
                outs.append(tokenizer.decode(new, skip_special_tokens=True))
                kept = [t for t in new.tolist() if t != tokenizer.pad_token_id]
                trunc.append(len(kept) >= args.max_new_tokens)
            if (i // args.batch) % 15 == 0:
                print(f"      {min(i + args.batch, len(prompts))}/{len(prompts)}",
                      flush=True)
        return outs, trunc

    started = time.time()

    def run_arm(arm, system, keys):
        for key in keys:
            condition = "TAP-ref" if key == "prompt_ref" else "TAP-mut"
            result_key = f"{arm}::{condition}"
            if result_key in results:
                print(f"\n--- {result_key}: checkpoint already complete; skipping ---",
                      flush=True)
                continue
            print(f"\n--- {arm} / {condition} ---", flush=True)
            outs, trunc = complete([item[key] for item in items], system)
            per, detail = score(items, outs, trunc)
            results[result_key] = {"per_benchmark": per, "detail": detail,
                                   "truncated": sum(trunc)}
            write_progress()
            report(arm, per)
            print(f"  hit token cap: {sum(trunc)}/{len(items)}")

    run_arm("base", None, ("prompt_ref", "prompt_mut"))

    print(f"\n>>> attaching adapter {ADAPTER.name} ({digest[:12]}...)", flush=True)
    gen.load_lora_adapter(ADAPTER)
    run_arm("a431", None, ("prompt_ref", "prompt_mut"))
    # Both conditions, not TAP-ref alone: without TAP-mut under the adapter's
    # own system prompt there is no way to measure whether SFT preserved
    # sensitivity to an actual code change, and Gate 1 cannot be closed.
    run_arm("a431_sysprompt", SYSTEM_PROMPT, ("prompt_ref", "prompt_mut"))

    elapsed = round(time.time() - started, 1)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "label": "diagnostic only - not model selection, not a performance claim",
        "status": "complete",
        "run_contract": contract,
        "model": name, "model_revision": revision,
        "adapter_path": str(ADAPTER), "adapter_sha256": digest,
        "decoding": "greedy", "max_new_tokens": args.max_new_tokens,
        "items_file": str(items_path), "items": len(items),
        "wall_time_seconds": elapsed, "weights_written": False,
        "arms": {k: {"per_benchmark": v["per_benchmark"], "truncated": v["truncated"]}
                 for k, v in results.items()},
        "detail": {k: v["detail"] for k, v in results.items()},
    }
    atomic_json_write(output_path, artifact)
    write_progress("complete")

    print(f"\n=== TAP-ref: did SFT damage output prediction? ({elapsed}s) ===")
    for arm in ("base", "a431", "a431_sysprompt"):
        key = f"{arm}::TAP-ref"
        if key in results:
            report(arm, results[key]["per_benchmark"])
    print(f"\nwritten: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
