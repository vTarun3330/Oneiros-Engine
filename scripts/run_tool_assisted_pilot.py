"""Run the execution-feedback pilot (GPU for generation; scoring is CPU).

Stages, run one at a time and only after explicit approval:

``generate-a``  Arm A, the canonical model-only control: the unchanged
                successor path (8 samples, batch 2, one seed), scored by the
                unchanged rehearsal evaluator.
``generate-bc`` Arms B (sham-feedback, request-budget-matched) and C from
                one shared round 1 through
                :mod:`harness.tool_assisted_generation`.  Writes one lineage
                file per target and an append-only call journal, so a
                disconnect resumes without repeating a completed model call.
                The fixed implementation is never loaded into the loop.
``score-b`` /   Score the FROZEN final slots of B or C with the unchanged
``score-c``     rehearsal evaluator.  No model is loaded; this is the only
                stage that executes the fixed implementation.

Every stage refuses unless the frozen design receipt is ready, HEAD differs
from its commit only by the receipt, and every bound source still has its
LF-canonical hash.  Existing artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_supervision_sidecar import sha256_file
from harness.source_identity import canonical_sha256

OUTPUT_DIR = ROOT / "results" / "v4_3_tool_assisted_v1"
RECEIPT = ROOT / "results" / "v4_3_tool_assisted_design_receipt.json"
PANEL = ROOT / "results" / "v4_3_tool_assisted_panel.json"
CONTROL_ADAPTER = ROOT / "checkpoints" / "v4_3_execsup_a_control_qwen15b_s42"
CONTROL_RESULT = (ROOT / "results" / "v4_3_execution_supervision_v1"
                  / "arm_a_training_result.json")
STAGES = ("generate-a", "generate-bc", "score-b", "score-c")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def verify_design_receipt(receipt_path: Path = RECEIPT) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("ready") is not True:
        raise SystemExit("REFUSED: tool-assisted design receipt is not ready")
    if git("status", "--porcelain"):
        raise SystemExit("REFUSED: working tree is not clean")
    commit = receipt["git"]["commit"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT,
                      capture_output=True).returncode:
        raise SystemExit("REFUSED: design commit is not an ancestor of HEAD")
    changed = set(filter(None, git("diff", "--name-only", commit, "HEAD").splitlines()))
    allowed = {receipt_path.relative_to(ROOT).as_posix()}
    if changed - allowed:
        raise SystemExit(f"REFUSED: files changed since the design receipt: "
                         f"{sorted(changed - allowed)}")
    for relative, expected in receipt["source_files_sha256"].items():
        if canonical_sha256(ROOT / relative) != expected:
            raise SystemExit(f"REFUSED: bound source drift: {relative}")
    if sha256_file(PANEL) != receipt["panel"]["sha256"]:
        raise SystemExit("REFUSED: panel differs from the design receipt")
    return receipt


def load_panel_scope():
    from harness.corpus_view import load_development_split
    from harness.evaluation_admission import scope_split
    from scripts.census_execution_dose_pool import CORPUS
    from scripts.freeze_tool_assisted_panel import PANEL_NAME, ids_sha256
    from scripts.train_on_dataset import _record_to_pair
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    if ids_sha256(panel["record_ids"]) != panel["record_ids_sha256"]:
        raise SystemExit("REFUSED: panel ids do not match their hash")
    scope = scope_split({PANEL_NAME: panel["record_ids"]}, load_development_split(
        CORPUS, "train"), PANEL_NAME, adapt=_record_to_pair)
    if scope.target_count != panel["records"]:
        raise SystemExit("REFUSED: panel did not resolve completely")
    return panel, scope, PANEL_NAME


def load_generator(settings):
    from engine.generator import Phi3Generator
    result = json.loads(CONTROL_RESULT.read_text(encoding="utf-8"))
    if sha256_file(CONTROL_ADAPTER / "adapter_model.safetensors") != result["adapter_sha256"]:
        raise SystemExit("REFUSED: frozen control adapter differs from its training result")
    generator = Phi3Generator(model_name=settings.base_model_name,
                              model_revision=settings.base_model_revision,
                              attention_implementation=settings.attention_implementation)
    generator.temperature, generator.top_p = settings.temperature, settings.top_p
    generator.parse_mode = settings.candidate_parse_mode
    generator.load_model()
    generator.load_lora_adapter(CONTROL_ADAPTER)
    return generator, result["adapter_sha256"]


def make_sampler(generator, settings, build_prompt):
    """One model call: canonical or repair prompts, seeded per call."""
    import torch
    from harness.generation_adapter import build_prompts
    from harness.generation_rng import seed_generation_rngs

    tokenizer = generator.tokenizer

    def sampler(view, additions, per_prompt, seed):
        records = [dict(view) for _ in additions]
        extra = {index: text for index, text in enumerate(additions) if text}
        token_ids, generable, failures = build_prompts(tokenizer, records, settings,
                                                       build_prompt, extra)
        if failures:
            # Matched pairs are pre-checked to fit without compaction, so any
            # budget failure here is an integrity error, not a fallback.
            raise RuntimeError(f"prompt over budget after matching: {failures}")
        seed_generation_rngs(seed)
        original_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        try:
            inputs = tokenizer.pad([{"input_ids": ids, "attention_mask": [1] * len(ids)}
                                    for ids in token_ids], padding=True,
                                   return_tensors="pt").to(generator.model.device)
            generator.model.eval()
            with torch.inference_mode():
                outputs = generator.model.generate(
                    **inputs, max_new_tokens=settings.generation_completion_token_limit,
                    temperature=settings.temperature, top_p=settings.top_p, do_sample=True,
                    num_return_sequences=per_prompt, pad_token_id=tokenizer.pad_token_id,
                    use_cache=True)
        finally:
            tokenizer.padding_side = original_side
        width = inputs.input_ids.shape[1]
        grouped: list[list[dict[str, Any]]] = []
        for prompt_index, ids in enumerate(token_ids):
            group = []
            for offset in range(per_prompt):
                new = outputs[prompt_index * per_prompt + offset][width:].tolist()
                count = 0
                for token in new:
                    count += 1
                    if token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                        break
                group.append({
                    "text": tokenizer.decode(new, skip_special_tokens=True),
                    "input_tokens": len(ids), "output_tokens": min(count, len(new)),
                    "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
            grouped.append(group)
        return grouped
    return sampler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args(argv)
    receipt = verify_design_receipt()
    from harness.generation_adapter import successor_settings
    from harness.prompt_factory import prompt_factory
    from harness.rehearsal_evaluator import run_rehearsal_evaluation
    settings = successor_settings()
    if settings.to_dict() != receipt["generation_settings"]:
        raise SystemExit("REFUSED: generation settings differ from the design receipt")
    panel, scope, panel_name = load_panel_scope()
    build_prompt = prompt_factory(settings.prompt_settings())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate(arm: str, generate_batch, identity: dict[str, Any], seed_record=None):
        out = OUTPUT_DIR / f"eval_{arm}"
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"REFUSED: {out} exists; artifacts are never overwritten")
        artifact = run_rehearsal_evaluation(
            load_records=lambda: scope.eligible, generate_batch=generate_batch,
            output_dir=out, split_name=panel_name, scope_summary=scope.to_dict(),
            frozen_settings=settings.to_dict(), receipt_sha256=sha256_file(RECEIPT),
            candidates_per_target=settings.candidates_per_function,
            generation_batch_size=settings.generation_batch_size,
            allow_test_function=settings.allow_test_function_candidates,
            log=lambda message: print(message, flush=True), seed_record=seed_record,
            generator_identity=identity)
        envelope = {"schema_version": "oneiros_tool_assisted_eval_v1", "arm": arm,
                    "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "panel_record_ids_sha256": panel["record_ids_sha256"],
                    "design_receipt_sha256": sha256_file(RECEIPT),
                    "rehearsal_result": {"path": (out / "rehearsal_result.json").relative_to(
                        ROOT).as_posix(), "sha256": sha256_file(out / "rehearsal_result.json")},
                    "kill_at_k": {k: v["rate"] for k, v in artifact["kill_at_k"].items()},
                    "wall_time_seconds": artifact["wall_time_seconds"],
                    "validation_accessed": False, "ablation_dev_accessed": False,
                    "test_accessed": False, "sealed_final_test_accessed": False,
                    "confirmation_opened": False, "weights_written": False}
        (OUTPUT_DIR / f"eval_{arm}.json").write_bytes(
            (json.dumps(envelope, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({"arm": arm, "kill_at_k": envelope["kill_at_k"]}, indent=2))

    if args.stage == "generate-a":
        from harness.generation_adapter import generate_candidate_slots
        from harness.generation_rng import seed_generation_rngs
        generator, adapter = load_generator(settings)
        seed_record = seed_generation_rngs(settings.seed)
        seed_record["applied_immediately_before_generation"] = True

        def generate_a(batch):
            rows = [dict(record) for record in batch]
            accounting = generate_candidate_slots(generator, rows, settings, build_prompt,
                                                  seed_before_generation=False)
            return [[{"raw_output": slot.get("raw_output", ""), "code": slot.get("code")}
                     for slot in accounting[index]["candidate_slots"]]
                    for index in range(len(rows))]
        evaluate("A", generate_a, {"arm": "A", "adapter_sha256": adapter,
                                   "path": "unchanged successor generation"}, seed_record)
        return 0

    loop_dir = OUTPUT_DIR / "loop"
    if args.stage == "generate-bc":
        from harness.buggy_side_execution import execute_on_code_under_test
        from harness.tool_assisted_generation import (
            Journal, make_frozen_parser, make_token_matcher, run_target)
        generator, adapter = load_generator(settings)
        sampler = make_sampler(generator, settings, build_prompt)
        parse = make_frozen_parser(generator)
        matcher = make_token_matcher(generator.tokenizer, settings, build_prompt)
        journal = Journal(OUTPUT_DIR / "journal.jsonl")
        loop_dir.mkdir(parents=True, exist_ok=True)
        provenance = {"model": settings.base_model_name,
                      "model_revision": settings.base_model_revision,
                      "adapter_sha256": adapter, "seed": settings.seed,
                      "generation_settings": settings.to_dict()}
        for index, record in enumerate(scope.eligible, 1):
            target = loop_dir / f"{hashlib.sha256(str(record['id']).encode()).hexdigest()}.json"
            if target.exists():
                continue
            result = run_target(record, sampler=sampler, parse=parse,
                                execute=execute_on_code_under_test, matcher=matcher,
                                base_seed=settings.seed,
                                journal=journal, timeout=receipt["executor_timeout_seconds"],
                                provenance=provenance)
            target.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
            budget = result["budget"]
            print(f"generate-bc {index}/{len(scope.eligible)} repairs delivered="
                  f"{len(budget['repairs_delivered'])} skipped="
                  f"{len(budget['repairs_skipped_match_infeasible'])}", flush=True)
        return 0

    arm = "B" if args.stage == "score-b" else "C"
    finals: dict[str, list[dict[str, Any]]] = {}
    for record in scope.eligible:
        target = loop_dir / f"{hashlib.sha256(str(record['id']).encode()).hexdigest()}.json"
        if not target.exists():
            raise SystemExit("REFUSED: generation is incomplete; score only frozen slots")
        finals[str(record["id"])] = json.loads(target.read_text(encoding="utf-8"))["final"][arm]

    def replay(batch):
        return [[{"raw_output": slot["raw_output"], "code": slot["code"]}
                 for slot in finals[str(record["id"])]] for record in batch]
    evaluate(arm, replay, {"arm": arm, "path": "frozen loop final slots; no model loaded"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
