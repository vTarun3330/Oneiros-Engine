"""Canonical Kill@8 on the frozen train-derived retention panel, one arm per run.

Runs the project's canonical test-generation path unchanged: the successor
generation settings (8 sampled candidates, frozen temperature/top-p/seed,
batch size 2), the shared prompt factory, the whole-output parser, the safe
executor and the sealed evaluator's pure scoring helpers via the rehearsal
module.  Records come from the train shard of the development view only; the
canonical ``records.json`` and every other split stay closed.

Arms: ``control`` (frozen control adapter), ``dose_treatment`` (the 25% arm)
and ``base`` (no adapter, descriptive).  Run them one at a time, never
concurrently.  No weights are written; existing artifacts are never replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_supervision_sidecar import sha256_file
from scripts.preflight_execution_dose_ab import (
    ARM_NAME, CHECKPOINT, CONTROL_ADAPTER, CONTROL_RESULT, DATASET_DIR, PANEL, RECEIPT,
    verify_receipt_for_launch,
)
from scripts.run_execution_dose_pilot import RESULT as TREATMENT_RESULT

ARMS = ("control", ARM_NAME, "base")


def adapter_for(arm: str) -> tuple[Path | None, str | None]:
    """The adapter directory and its verified hash, or (None, None) for base."""
    if arm == "base":
        return None, None
    directory, result_path = ((CONTROL_ADAPTER, CONTROL_RESULT) if arm == "control"
                              else (CHECKPOINT, TREATMENT_RESULT))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise SystemExit(f"REFUSED: {arm} training result is not complete")
    digest = sha256_file(directory / "adapter_model.safetensors")
    if digest != result["adapter_sha256"]:
        raise SystemExit(f"REFUSED: {arm} adapter differs from its training result")
    return directory, digest


def load_panel_scope():
    from harness.corpus_view import load_development_split
    from harness.evaluation_admission import scope_split
    from scripts.census_execution_dose_pool import CORPUS
    from scripts.freeze_execution_dose_retention_panel import PANEL_NAME, ids_sha256
    from scripts.train_on_dataset import _record_to_pair

    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    if ids_sha256(panel["record_ids"]) != panel["record_ids_sha256"]:
        raise SystemExit("REFUSED: retention panel ids do not match their hash")
    records = load_development_split(CORPUS, "train")
    scope = scope_split({PANEL_NAME: panel["record_ids"]}, records, PANEL_NAME,
                        adapt=_record_to_pair)
    if scope.target_count != panel["records"]:
        raise SystemExit("REFUSED: retention panel did not resolve completely")
    return panel, scope, PANEL_NAME


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    args = parser.parse_args(argv)
    receipt = verify_receipt_for_launch()
    if sha256_file(PANEL) != receipt["dataset"]["retention_panel_sha256"]:
        raise SystemExit("REFUSED: retention panel differs from the preflight's")
    envelope_path = DATASET_DIR / f"retention_{args.arm}.json"
    output_dir = DATASET_DIR / f"retention_{args.arm}"
    if envelope_path.exists() or (output_dir.exists() and any(output_dir.iterdir())):
        raise SystemExit("REFUSED: retention artifacts exist; they are never overwritten")
    adapter_dir, adapter_sha = adapter_for(args.arm)
    panel, scope, panel_name = load_panel_scope()

    from engine.generator import Phi3Generator
    from harness.generation_adapter import generate_candidate_slots, successor_settings
    from harness.generation_rng import seed_generation_rngs
    from harness.prompt_factory import prompt_factory
    from harness.rehearsal_evaluator import run_rehearsal_evaluation

    settings = successor_settings()
    if settings.problems():
        raise SystemExit(f"REFUSED: invalid generation settings: {settings.problems()}")
    generator = Phi3Generator(model_name=settings.base_model_name,
                              model_revision=settings.base_model_revision,
                              attention_implementation=settings.attention_implementation)
    generator.temperature, generator.top_p = settings.temperature, settings.top_p
    generator.parse_mode = settings.candidate_parse_mode
    generator.load_model()
    if adapter_dir is not None:
        generator.load_lora_adapter(adapter_dir)
    build_prompt = prompt_factory(settings.prompt_settings())
    seed_record = seed_generation_rngs(settings.seed)
    seed_record["applied_immediately_before_generation"] = True

    def generate_batch(batch):
        rows = [dict(record) for record in batch]
        accounting = generate_candidate_slots(generator, rows, settings, build_prompt,
                                              seed_before_generation=False)
        return [[{"raw_output": slot.get("raw_output", ""), "code": slot.get("code")}
                 for slot in accounting[index]["candidate_slots"]]
                for index in range(len(rows))]

    artifact = run_rehearsal_evaluation(
        load_records=lambda: scope.eligible, generate_batch=generate_batch,
        output_dir=output_dir, split_name=panel_name, scope_summary=scope.to_dict(),
        frozen_settings=settings.to_dict(), receipt_sha256=sha256_file(RECEIPT),
        candidates_per_target=settings.candidates_per_function,
        generation_batch_size=settings.generation_batch_size,
        allow_test_function=settings.allow_test_function_candidates,
        log=lambda message: print(message, flush=True), seed_record=seed_record,
        generator_identity={"model_name": settings.base_model_name,
                            "model_revision": settings.base_model_revision,
                            "adapter": None if adapter_dir is None
                            else adapter_dir.relative_to(ROOT).as_posix(),
                            "adapter_sha256": adapter_sha,
                            "parse_mode": generator.parse_mode})
    result_path = output_dir / "rehearsal_result.json"
    envelope = {
        "schema_version": "oneiros_execution_dose_retention_eval_v1",
        "label": ("train-derived canonical test-generation retention; not validation, "
                  "not generalisation, not a final-test result"),
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm, "adapter_sha256": adapter_sha,
        "panel_record_ids_sha256": panel["record_ids_sha256"],
        "admission_scope_sha256": scope.scope_sha256(),
        "preflight_sha256": sha256_file(RECEIPT),
        "rehearsal_result": {"path": result_path.relative_to(ROOT).as_posix(),
                             "sha256": sha256_file(result_path)},
        "kill_at_k": {k: value["rate"] for k, value in artifact["kill_at_k"].items()},
        "wall_time_seconds": artifact["wall_time_seconds"],
        "source_sha256": sha256_file(Path(__file__)),
        "sealed_final_test_accessed": False, "validation_accessed": False,
        "ablation_dev_accessed": False, "confirmation_opened": False,
        "weights_written": False,
    }
    envelope_path.write_bytes((json.dumps(envelope, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"arm": args.arm, "kill_at_k": envelope["kill_at_k"],
                      "wall_time_seconds": envelope["wall_time_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
