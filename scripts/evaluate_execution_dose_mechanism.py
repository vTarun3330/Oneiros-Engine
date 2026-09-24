"""Evaluate the composite execution-intervention treatment on the 97-item panel.

The panel, prompts, scorer, greedy decoding, 128-token limit and batch size
are imported from the evaluator that produced the frozen control's artifact
(eval_control.json), so the two artifacts differ only in the adapter.  The
generation loop is the one the 12% trace arms ran.  No weights are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.source_identity import canonical_sha256
from harness.execution_supervision_sidecar import sha256_file
from scripts.evaluate_execution_supervision_pilot import (
    BATCH_SIZE, CONDITIONS, MAX_NEW_TOKENS, score_assertion, shown_actual_prompt,
)
from scripts.preflight_execution_dose_ab import (
    ARM_NAME, CHECKPOINT, DATASET_DIR, RECEIPT, SOURCE_DIR, verify_receipt_for_launch,
)
from scripts.preflight_execution_supervision_ab import MODEL_NAME, MODEL_REVISION
from scripts.run_execution_dose_pilot import RESULT as TRAINING_RESULT

OUTPUT = DATASET_DIR / "eval_dose_treatment.json"
DECODING = {"strategy": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE, "system_prompt": None}
ANSWER_VERDICTS = ("correct", "wrong_value", "wrong_type")


def summarise(rows: list[dict]) -> dict:
    strict = Counter(row["strict"]["verdict"] for row in rows)
    lenient = Counter(row["lenient"]["verdict"] for row in rows)
    return {
        "requested": len(rows), "strict_counts": dict(sorted(strict.items())),
        "strict_accuracy_per_requested": strict["correct"] / len(rows),
        "strict_answer_rate": sum(strict[key] for key in ANSWER_VERDICTS) / len(rows),
        "lenient_counts": dict(sorted(lenient.items())),
        "lenient_accuracy_per_requested": lenient["correct"] / len(rows),
        "lenient_answer_rate": sum(lenient[key] for key in ANSWER_VERDICTS) / len(rows),
        "completion_limit_hits": sum(row["hit_completion_limit"] for row in rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    receipt = verify_receipt_for_launch()
    if OUTPUT.exists():
        raise SystemExit("REFUSED: evaluation artifact exists; raw artifacts are never "
                         "overwritten")
    result = json.loads(TRAINING_RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("arm") != ARM_NAME:
        raise SystemExit("REFUSED: no completed treatment training result")
    if result["run_contract"]["preflight_sha256"] != sha256_file(RECEIPT):
        raise SystemExit("REFUSED: adapter was trained under a different preflight")
    adapter_hash = sha256_file(CHECKPOINT / "adapter_model.safetensors")
    if adapter_hash != result["adapter_sha256"]:
        raise SystemExit("REFUSED: adapter hash differs from its training result")
    panel_path = SOURCE_DIR / "pilot_development.execution.json"
    if sha256_file(panel_path) != receipt["dataset"]["pilot_development_sha256"]:
        raise SystemExit("REFUSED: mechanism panel hash mismatch")
    items = json.loads(panel_path.read_text(encoding="utf-8"))
    if len(items) != 97 or any(row.get("evaluation_split") != "train" for row in items):
        raise SystemExit("REFUSED: mechanism panel is not the frozen 97 train items")

    import torch
    from engine.generator import Phi3Generator
    generator = Phi3Generator(model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                              attention_implementation="sdpa")
    generator.load_model()
    generator.load_lora_adapter(CHECKPOINT)
    tokenizer, model = generator.tokenizer, generator.model
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    condition_data = {
        "intended_output": ([str(row["focused_prompt"]) for row in items],
                            [row["execution_evidence"]["intended"] for row in items]),
        "shown_actual_output": ([shown_actual_prompt(row) for row in items],
                                [row["execution_evidence"]["actual"] for row in items]),
    }
    detail, summary = {}, {}
    for condition in CONDITIONS:
        prompts, expected_values = condition_data[condition]
        rows = []
        for start in range(0, len(items), BATCH_SIZE):
            chunk = prompts[start:start + BATCH_SIZE]
            rendered = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True) for prompt in chunk]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                                truncation=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
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
                    "lenient": score_assertion(raw, item["call_expression"], expected,
                                               lenient=True),
                    "raw": raw, "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "raw_chars": len(raw),
                    "hit_completion_limit": int(new.numel()) >= MAX_NEW_TOKENS,
                })
            print(f"{ARM_NAME}/{condition}: {min(start + BATCH_SIZE, len(items))}/"
                  f"{len(items)}", flush=True)
        detail[condition], summary[condition] = rows, summarise(rows)
    artifact = {
        "schema_version": "oneiros_execution_dose_mechanism_eval_v1",
        "label": "train-derived mechanism diagnostic; not generalisation or model selection",
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": ARM_NAME, "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "adapter_sha256": adapter_hash, "git_commit": receipt["git"]["commit"],
        "preflight_sha256": sha256_file(RECEIPT),
        "pilot_development_sha256": receipt["dataset"]["pilot_development_sha256"],
        "decoding": DECODING, "source_sha256": canonical_sha256(Path(__file__)),
        "items": len(items), "conditions": list(CONDITIONS),
        "summary": summary, "detail": detail,
        "sealed_final_test_accessed": False, "validation_accessed": False,
        "ablation_dev_accessed": False, "confirmation_opened": False,
        "weights_written": False,
    }
    OUTPUT.write_bytes((json.dumps(artifact, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"arm": ARM_NAME, "summary": summary, "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
