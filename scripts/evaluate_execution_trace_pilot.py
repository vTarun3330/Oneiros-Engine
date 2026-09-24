"""Evaluate one trace arm on the frozen 97-item train-derived mechanism panel."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_supervision_sidecar import sha256_file
from scripts.evaluate_execution_supervision_pilot import (
    BATCH_SIZE, CONDITIONS, MAX_NEW_TOKENS, score_assertion, shown_actual_prompt,
)
from scripts.preflight_execution_supervision_ab import MODEL_NAME, MODEL_REVISION
from scripts.preflight_execution_trace_ab import ARM_FILES


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_FILES))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1")
    parser.add_argument("--source-dataset-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1")
    parser.add_argument("--preflight", type=Path, default=ROOT / "results"
                        / "v4_3_execution_trace_v1" / "preflight.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("ready") is not True:
        raise SystemExit("REFUSED: trace preflight is not ready")
    if _git("rev-parse", "HEAD") != preflight["git"]["commit"] or _git("status", "--short"):
        raise SystemExit("REFUSED: source is dirty or differs from trace preflight")
    for relative, expected in preflight["source_files_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise SystemExit(f"REFUSED: trace preflight source drift: {relative}")
    result = json.loads(args.training_result.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("arm") != args.arm:
        raise SystemExit("REFUSED: training result does not match trace arm")
    if result["run_contract"]["preflight_sha256"] != sha256_file(args.preflight):
        raise SystemExit("REFUSED: adapter and evaluation use different preflights")
    adapter_hash = sha256_file(args.adapter / "adapter_model.safetensors")
    if adapter_hash != result.get("adapter_sha256"):
        raise SystemExit("REFUSED: adapter hash differs from training result")
    pilot_path = args.source_dataset_dir / "pilot_development.execution.json"
    if sha256_file(pilot_path) != preflight["dataset"]["pilot_development_sha256"]:
        raise SystemExit("REFUSED: mechanism pilot hash mismatch")
    items = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not items or any(row.get("evaluation_split") != "train" for row in items):
        raise SystemExit("REFUSED: mechanism pilot is empty or not train-only")

    import torch
    from engine.generator import Phi3Generator
    generator = Phi3Generator(
        model_name=MODEL_NAME, model_revision=MODEL_REVISION,
        attention_implementation="sdpa",
    )
    generator.load_model()
    generator.load_lora_adapter(args.adapter)
    tokenizer, model = generator.tokenizer, generator.model
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    condition_data = {
        "intended_output": (
            [str(row["focused_prompt"]) for row in items],
            [row["execution_evidence"]["intended"] for row in items],
        ),
        "shown_actual_output": (
            [shown_actual_prompt(row) for row in items],
            [row["execution_evidence"]["actual"] for row in items],
        ),
    }
    detail, summary = {}, {}
    for condition in CONDITIONS:
        prompts, expected_values = condition_data[condition]
        rows = []
        for start in range(0, len(items), BATCH_SIZE):
            chunk = prompts[start:start + BATCH_SIZE]
            rendered = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            ) for prompt in chunk]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                                truncation=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for offset, item in enumerate(items[start:start + len(chunk)]):
                new = generated[offset][encoded.input_ids.shape[1]:]
                raw = tokenizer.decode(new, skip_special_tokens=True)
                expected = expected_values[start + offset]
                rows.append({
                    "record_id": item["record_id"],
                    "function_lineage": item["function_lineage"],
                    "source_dataset": item["source_dataset"],
                    "condition": condition, "expected": expected,
                    "strict": score_assertion(raw, item["call_expression"], expected),
                    "lenient": score_assertion(
                        raw, item["call_expression"], expected, lenient=True
                    ),
                    "raw": raw,
                    "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "raw_chars": len(raw),
                    "hit_completion_limit": int(new.numel()) >= MAX_NEW_TOKENS,
                })
            print(f"{args.arm}/{condition}: {min(start+BATCH_SIZE, len(items))}/{len(items)}",
                  flush=True)
        strict = Counter(row["strict"]["verdict"] for row in rows)
        lenient = Counter(row["lenient"]["verdict"] for row in rows)
        detail[condition] = rows
        summary[condition] = {
            "requested": len(rows), "strict_counts": dict(sorted(strict.items())),
            "strict_accuracy_per_requested": strict["correct"] / len(rows),
            "strict_answer_rate": sum(strict[key] for key in (
                "correct", "wrong_value", "wrong_type"
            )) / len(rows),
            "lenient_counts": dict(sorted(lenient.items())),
            "lenient_accuracy_per_requested": lenient["correct"] / len(rows),
            "lenient_answer_rate": sum(lenient[key] for key in (
                "correct", "wrong_value", "wrong_type"
            )) / len(rows),
            "completion_limit_hits": sum(row["hit_completion_limit"] for row in rows),
        }
    artifact = {
        "schema_version": "oneiros_execution_trace_mechanism_eval_v1",
        "label": "train-derived mechanism diagnostic; not generalisation or model selection",
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm, "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "adapter_sha256": adapter_hash, "git_commit": preflight["git"]["commit"],
        "preflight_sha256": sha256_file(args.preflight),
        "pilot_development_sha256": preflight["dataset"]["pilot_development_sha256"],
        "decoding": {"strategy": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
                     "batch_size": BATCH_SIZE, "system_prompt": None},
        "source_sha256": sha256_file(Path(__file__)), "items": len(items),
        "conditions": list(CONDITIONS), "summary": summary, "detail": detail,
        "sealed_final_test_accessed": False, "validation_accessed": False,
        "ablation_dev_accessed": False, "confirmation_opened": False,
        "weights_written": False,
    }
    _atomic(args.output, artifact)
    print(json.dumps({"arm": args.arm, "summary": summary,
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
