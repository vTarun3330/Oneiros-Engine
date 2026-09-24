"""Frozen train-only mechanism evaluation for the execution-supervision pilot.

This evaluator never executes model output.  It parses one assertion, requires
its left-hand side to be the declared literal call, and compares the literal
right-hand side by both Python value and exact type.  The 97-item panel is
lineage-disjoint from the training arm but still comes from the train corpus;
therefore this is a mechanism diagnostic, not a generalisation result.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_supervision_sidecar import sha256_file
from scripts.preflight_execution_supervision_ab import MODEL_NAME, MODEL_REVISION

SCHEMA = "oneiros_execution_supervision_mechanism_eval_v1"
CONDITIONS = ("intended_output", "shown_actual_output")
MAX_NEW_TOKENS = 128
BATCH_SIZE = 8


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def shown_actual_prompt(item: dict[str, Any]) -> str:
    """Build the frozen code-as-shown condition without intended behaviour."""
    focused = str(item["focused_prompt"])
    function_marker = "### Function (may be defective)\n\n"
    call_marker = "\n\n### Call\n\n"
    assertion_marker = "\n\n### Assertion"
    if function_marker not in focused or call_marker not in focused:
        raise ValueError("focused prompt does not contain frozen function/call sections")
    tail = focused.split(function_marker, 1)[1]
    function_text, tail = tail.split(call_marker, 1)
    call_text = tail.split(assertion_marker, 1)[0].strip()
    if call_text != str(item["call_expression"]).strip():
        raise ValueError("focused prompt call differs from item call_expression")
    return (
        "### SHOWN-CODE EXECUTION TASK\n\n"
        "You are given a Python function and one literal-input call. Write the "
        "single Python `assert` statement that states what the call actually "
        "evaluates to when the function executes exactly as shown.\n\n"
        "Return only that assert statement. Do not return prose, Markdown, a "
        "test function, imports, a repair, or additional assertions.\n\n"
        f"### Function\n\n{function_text.strip()}\n\n"
        f"### Call\n\n{call_text}\n\n### Assertion\n"
    )


def _lenient_code(raw: str) -> str:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    return text


def score_assertion(
    raw: str, call_expression: str, expected: dict[str, str], *, lenient: bool = False,
) -> dict[str, Any]:
    """Score one model response without importing or executing it."""
    code = _lenient_code(raw) if lenient else raw.strip()
    try:
        module = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return {"verdict": "invalid_syntax", "predicted_literal": None}
    if len(module.body) != 1:
        return {"verdict": "multiple_or_missing_statements", "predicted_literal": None}
    statement = module.body[0]
    if not isinstance(statement, ast.Assert):
        return {"verdict": "no_assertion", "predicted_literal": None}
    test = statement.test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
    ):
        return {"verdict": "not_single_equality", "predicted_literal": None}
    try:
        expected_call = ast.parse(call_expression, mode="eval").body
    except SyntaxError as exc:  # dataset invariant, not a model outcome
        raise ValueError(f"invalid frozen call expression: {call_expression}") from exc
    if ast.dump(test.left, include_attributes=False) != ast.dump(
        expected_call, include_attributes=False
    ):
        return {"verdict": "wrong_call", "predicted_literal": None}
    try:
        predicted = ast.literal_eval(test.comparators[0])
        wanted = ast.literal_eval(str(expected["literal"]))
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return {"verdict": "prediction_not_literal", "predicted_literal": None}
    predicted_literal = repr(predicted)
    if type(predicted).__name__ != str(expected["type"]):
        return {"verdict": "wrong_type", "predicted_literal": predicted_literal}
    try:
        equal = predicted == wanted
        if type(predicted) is float and type(wanted) is float:
            equal = abs(predicted - wanted) < 1e-6
    except Exception:  # comparison only; model text is never executed
        equal = False
    return {
        "verdict": "correct" if equal else "wrong_value",
        "predicted_literal": predicted_literal,
    }


def _validate_inputs(
    *, arm: str, adapter: Path | None, training_result: Path | None,
    dataset_dir: Path, preflight_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("ready") is not True:
        raise SystemExit("REFUSED: frozen training preflight is not ready")
    if _git("rev-parse", "HEAD") != (preflight.get("git") or {}).get("commit"):
        raise SystemExit("REFUSED: HEAD differs from frozen preflight")
    if _git("status", "--short"):
        raise SystemExit("REFUSED: working tree is not clean")
    for relative, expected in (preflight.get("source_files_sha256") or {}).items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise SystemExit(f"REFUSED: preflight source drift: {relative}")
    pilot_path = dataset_dir / "pilot_development.execution.json"
    if sha256_file(pilot_path) != preflight["dataset"]["pilot_development_sha256"]:
        raise SystemExit("REFUSED: pilot-development artifact hash mismatch")
    items = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not items or any(row.get("evaluation_split") != "train" for row in items):
        raise SystemExit("REFUSED: mechanism panel is empty or not train-only")
    adapter_hash = None
    if arm == "base":
        if adapter is not None or training_result is not None:
            raise SystemExit("REFUSED: base arm must not load an adapter")
    else:
        if adapter is None or training_result is None:
            raise SystemExit("REFUSED: trained arm requires adapter and result receipt")
        result = json.loads(training_result.read_text(encoding="utf-8"))
        expected_training_arm = "control" if arm == "control" else "treatment"
        if result.get("status") != "complete" or result.get("arm") != expected_training_arm:
            raise SystemExit("REFUSED: training result does not match evaluation arm")
        model_path = adapter / "adapter_model.safetensors"
        adapter_hash = sha256_file(model_path)
        if adapter_hash != result.get("adapter_sha256"):
            raise SystemExit("REFUSED: adapter hash differs from training result")
        contract = result.get("run_contract") or {}
        if contract.get("git_commit") != preflight["git"]["commit"]:
            raise SystemExit("REFUSED: adapter was trained from a different commit")
        if contract.get("preflight_sha256") != sha256_file(preflight_path):
            raise SystemExit("REFUSED: adapter was trained against another preflight")
    return items, preflight, adapter_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["base", "control", "treatment"])
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-result", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1")
    parser.add_argument("--preflight", type=Path, default=ROOT / "results"
                        / "v4_3_execution_supervision_v1" / "preflight.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items, preflight, adapter_hash = _validate_inputs(
        arm=args.arm, adapter=args.adapter, training_result=args.training_result,
        dataset_dir=args.dataset_dir, preflight_path=args.preflight,
    )

    import torch
    from engine.generator import Phi3Generator
    generator = Phi3Generator(
        model_name=MODEL_NAME, model_revision=MODEL_REVISION,
        attention_implementation="sdpa",
    )
    generator.load_model()
    if args.adapter is not None:
        generator.load_lora_adapter(args.adapter)
    tokenizer, model = generator.tokenizer, generator.model
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    conditions = {
        "intended_output": (
            [str(row["focused_prompt"]) for row in items],
            [row["execution_evidence"]["intended"] for row in items],
        ),
        "shown_actual_output": (
            [shown_actual_prompt(row) for row in items],
            [row["execution_evidence"]["actual"] for row in items],
        ),
    }
    detail: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        prompts, expected_values = conditions[condition]
        rows: list[dict[str, Any]] = []
        for start in range(0, len(items), BATCH_SIZE):
            chunk = prompts[start:start + BATCH_SIZE]
            rendered = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            ) for prompt in chunk]
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True, truncation=False,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for offset, item in enumerate(items[start:start + len(chunk)]):
                new_tokens = generated[offset][encoded.input_ids.shape[1]:]
                raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
                strict = score_assertion(
                    raw, str(item["call_expression"]), expected_values[start + offset]
                )
                lenient = score_assertion(
                    raw, str(item["call_expression"]), expected_values[start + offset],
                    lenient=True,
                )
                rows.append({
                    "record_id": item["record_id"],
                    "function_lineage": item["function_lineage"],
                    "source_dataset": item["source_dataset"],
                    "condition": condition,
                    "expected": expected_values[start + offset],
                    "strict": strict,
                    "lenient": lenient,
                    "raw": raw,
                    "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "raw_chars": len(raw),
                    "hit_completion_limit": int(new_tokens.numel()) >= MAX_NEW_TOKENS,
                })
            print(f"{args.arm}/{condition}: {min(start+BATCH_SIZE, len(items))}/{len(items)}",
                  flush=True)
        counts = Counter(row["strict"]["verdict"] for row in rows)
        lenient_counts = Counter(row["lenient"]["verdict"] for row in rows)
        detail[condition] = rows
        summary[condition] = {
            "requested": len(rows),
            "strict_counts": dict(sorted(counts.items())),
            "strict_accuracy_per_requested": counts["correct"] / len(rows),
            "strict_answer_rate": (
                counts["correct"] + counts["wrong_value"] + counts["wrong_type"]
            ) / len(rows),
            "lenient_counts": dict(sorted(lenient_counts.items())),
            "lenient_accuracy_per_requested": lenient_counts["correct"] / len(rows),
            "completion_limit_hits": sum(row["hit_completion_limit"] for row in rows),
        }
    artifact = {
        "schema_version": SCHEMA,
        "label": "train-derived mechanism diagnostic; not generalisation or model selection",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "adapter_sha256": adapter_hash,
        "git_commit": preflight["git"]["commit"],
        "preflight_sha256": sha256_file(args.preflight),
        "pilot_development_sha256": preflight["dataset"]["pilot_development_sha256"],
        "decoding": {"strategy": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
                     "batch_size": BATCH_SIZE, "system_prompt": None},
        "source_sha256": sha256_file(Path(__file__)),
        "items": len(items),
        "conditions": list(CONDITIONS),
        "summary": summary,
        "detail": detail,
        "sealed_final_test_accessed": False,
        "validation_accessed": False,
        "ablation_dev_accessed": False,
        "confirmation_opened": False,
        "weights_written": False,
    }
    _atomic_json(args.output, artifact)
    print(json.dumps({"arm": args.arm, "summary": summary,
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
