"""Choice B v2: stage-aware, integrity-checked composite-intervention screen.

Internal, exploratory, Arm-A-exposed, single seed; not untouched, not Phase 6, not
cross-dataset or repository generalisation. Estimand: harness.choice_b_evaluation.ESTIMAND.

CPU commands:
    freeze      split v3 = split v2 minus outcome-blind near-duplicate training groups
    spec        evaluation spec v2 (point-estimate operational guardrails, exact power)
    preflight   versioned preflight v2 (refuses to overwrite)
    status      current lifecycle stage and the next legal action
    validate    re-validate every completed stage artifact (adapters and raw files too)
    analyse-from-evidence   recompute the analysis from tracked evidence only (any clone)

GPU commands (durable; one at a time; each refuses unless it is the next legal stage):
    python scripts/gpu_run.py start --name choice_b_v2_<action> -- \\
        .venv-gpu/Scripts/python.exe scripts/phase4_choice_b_v2.py run <action>
    actions: train_control, train_treatment, evaluate_control, evaluate_treatment, analyse

Integration smoke (disposable directories, non-gate training rows only):
    python scripts/gpu_run.py start --name choice_b_v2_smoke -- \\
        .venv-gpu/Scripts/python.exe scripts/phase4_choice_b_v2.py smoke
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness import choice_b_evaluation as ev
from harness import choice_b_lifecycle as lc

VIEW = "data/corpus/v4_1_research_hardened_candidate/development_view"
TRAIN = f"{VIEW}/train.records.json"
COMPLEXITY = f"{VIEW}/complexity_manifest.json"
SPLIT_V2 = "results/sft_root_cause_phase4_choice_b_split_v2.json"
SPLIT = "results/sft_root_cause_phase4_choice_b_split_v3.json"
NEARDUP = "results/sft_root_cause_phase4_choice_b_neardup_audit_v1.json"
SPEC = "results/sft_root_cause_phase4_choice_b_evaluation_spec_v2.json"
POWER = "results/sft_root_cause_phase4_choice_b_power_200_v1.json"
PREFLIGHT = "results/sft_root_cause_phase4_choice_b_v2_preflight.json"
PREFIX = "results/sft_root_cause_phase4_choice_b_v2_"
RESULTS = {"control_trained": f"{PREFIX}training_control.json",
           "treatment_trained": f"{PREFIX}training_treatment.json",
           "control_evaluated": f"{PREFIX}gate_look_control.json",
           "treatment_evaluated": f"{PREFIX}gate_look_treatment.json",
           "analysed": f"{PREFIX}analysis.json"}
SMOKE_RECEIPT = "results/sft_root_cause_phase4_choice_b_smoke_v2.json"
RAW = "results/sft_root_cause/phase4_choice_b_v2"
SMOKE_RAW = "results/sft_root_cause/phase4_choice_b_v2_SMOKE"
SMOKE_CKPT = "checkpoints/phase4_choice_b_v2_SMOKE_DISPOSABLE"
PYTHON = ".venv-gpu/Scripts/python.exe"
SCRIPT = "scripts/phase4_choice_b_v2.py"

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ARMS = {"control": "full_completion", "treatment": "value_only"}
SEED = 42
LEARNING_RATE = 1e-5
EPOCHS = 2
BATCH_SIZE = 1
TRAINER_KWARGS = {
    "max_prompt_tokens": 1024, "max_repository_prompt_tokens": 2048,
    "max_completion_tokens": 1024, "max_repository_completion_tokens": 1024,
    "warmup_steps": 25, "checkpoint_steps": 50, "lr_scheduler_type": "constant_with_warmup",
}
FIXED_DECODING = {"greedy": True, "max_new_tokens": {"prefill_assertion": 48,
                                                     "answer_schema": 96}, "batch": 8}
SECONDS_PER_STEP = 4.35
SOURCE_FILES = (
    "scripts/phase4_choice_b_v2.py", "scripts/gpu_run.py", "scripts/train_on_dataset.py",
    "harness/choice_b.py", "harness/choice_b_lifecycle.py", "harness/choice_b_evaluation.py",
    "harness/choice_b_neardup.py", "harness/objective_masking.py", "harness/fixed_input_probe.py",
    "harness/safe_execution.py", "harness/rehearsal_evaluator.py",
    "harness/generation_adapter.py", "harness/generation_rng.py", "harness/prompt_factory.py",
    "harness/sealed_final_evaluator.py", "harness/evaluation_admission.py",
    "harness/corpus_view.py", "harness/source_identity.py", "harness/atomic_publish.py",
    "engine/sft_trainer.py", "engine/generator.py", "engine/test_generation_prompt.py",
    "config/settings.py", "requirements-local-gpu.lock.txt",
    SPLIT, NEARDUP, SPEC, POWER, TRAIN, COMPLEXITY,
)


def layout() -> lc.Layout:
    return lc.Layout(root=ROOT, preflight=PREFLIGHT, results=RESULTS, result_prefix=PREFIX,
                     source_files=SOURCE_FILES)


def rel_sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def publish_once(rel: str, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    target = ROOT / rel
    if target.exists() and target.read_bytes() != data:
        raise SystemExit(f"REFUSED: {rel} exists with different bytes; never overwritten")
    publish_file_atomically(target, data)


def checkpoint_dir(arm: str) -> Path:
    return ROOT / "checkpoints" / f"phase4_choice_b_v2_{arm}_qwen15b_s{SEED}"


def train_records() -> Dict[str, dict]:
    return {r["id"]: r for r in load(TRAIN)}


# --- CPU: freeze, spec ---------------------------------------------------------------------

def freeze() -> int:
    from harness.acquisition_receipt import ProtectedAccessMonitor
    from harness.choice_b_neardup import audit
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    v2 = load(SPLIT_V2)
    records = load(TRAIN)
    gate_groups = set(v2["gate"]["group_ids"])
    train_groups = {r["group_id"] for r in v2["rows"]}
    result = audit([r for r in records if r["group_id"] in train_groups],
                   [r for r in records if r["group_id"] in gate_groups])
    excluded = set(result["excluded_training_groups"])
    rows = [r for r in v2["rows"] if r["group_id"] not in excluded]
    evidence = ProtectedAccessMonitor.evidence(mark)
    if evidence["protected_paths_opened"]:
        raise SystemExit(f"REFUSED: protected paths opened {evidence['protected_paths_opened']}")
    publish_once(NEARDUP, {
        "schema_version": "oneiros_choice_b_neardup_audit_v1", "outcome_blind": True,
        "inputs": {SPLIT_V2: rel_sha(SPLIT_V2), TRAIN: rel_sha(TRAIN)},
        "record_disjoint": not ({r["record_id"] for r in v2["rows"]}
                                & {f["record_id"] for f in v2["gate"]["functions"]}),
        "group_disjoint": not (train_groups & gate_groups), **result,
        "training_rows_removed": len(v2["rows"]) - len(rows)})
    publish_once(SPLIT, {
        **{k: v for k, v in v2.items() if k not in ("rows", "training", "supersedes",
                                                    "schema_version", "evaluation_spec",
                                                    "evaluation_spec_sha256")},
        "schema_version": "oneiros_phase4_choice_b_split_v3",
        "supersedes": {"path": SPLIT_V2, "sha256": rel_sha(SPLIT_V2), "modified": False,
                       "reason": "outcome-blind near-duplicate exclusion of training groups"},
        "near_duplicate_audit": {"path": NEARDUP, "sha256": rel_sha(NEARDUP),
                                 "excluded_training_groups": sorted(excluded)},
        "evaluation_spec": {"path": SPEC, "note": "frozen separately (spec v2)"},
        "training": {**v2["training"], "rows": len(rows),
                     "groups_with_rows": len({r["group_id"] for r in rows}),
                     "rows_removed_near_duplicate": len(v2["rows"]) - len(rows),
                     "composition": {key: dict(Counter(str(r[key]) for r in rows).most_common())
                                     for key in ("dataset", "complexity_tier", "bug_family")}},
        "rows": rows,
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"}})
    print(json.dumps({"excluded_groups": len(excluded), "rows": len(rows),
                      "flags": result["flag_reasons"]}, indent=1))
    return 0


def spec() -> int:
    v2 = load(SPLIT_V2)
    publish_once(SPEC, ev.evaluation_spec_v2(
        v2["evaluation_spec"], v2["evaluation_spec_sha256"], load(POWER), rel_sha(POWER)))
    print(rel_sha(SPEC))
    return 0


# --- shared GPU work (used by the real stages AND the smoke) -------------------------------

@dataclass
class TrainPlan:
    arm: str
    rows: List[dict]
    checkpoint_dir: Path
    contract: Dict[str, Any]
    epochs: int
    trainer_kwargs: Dict[str, Any]


def datapoints(rows: Sequence[Mapping[str, Any]], records: Mapping[str, dict]):
    from harness.objective_masking import make_datapoint
    return [make_datapoint(records[r["record_id"]], r["call"], r["value"]) for r in rows]


def seed_everything() -> None:
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def train_arm(plan: TrainPlan, records: Mapping[str, dict]) -> Dict[str, Any]:
    """Train one arm under an exact contract; resume only from a contract-matching dir."""
    import torch
    from engine.sft_trainer import OneirosSFTTrainer
    mode = lc.check_resume(plan.checkpoint_dir, plan.contract)
    seed_everything()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer = OneirosSFTTrainer(model_name=MODEL, model_revision=REVISION,
                                output_dir=plan.checkpoint_dir, learning_rate=LEARNING_RATE,
                                attention_implementation="sdpa",
                                objective_mode=ARMS[plan.arm], **plan.trainer_kwargs)
    metrics = trainer.train(datapoints(plan.rows, records), num_epochs=plan.epochs,
                            batch_size=BATCH_SIZE, checkpoint_monitor=None)
    final = plan.checkpoint_dir / "final_adapter"
    trainer.save_adapter(final)
    adapter_sha = lc.sha256_file(final / "adapter_model.safetensors")
    checkpoints = {p.name: lc.sha256_file(p / "adapter_model.safetensors")
                   for p in sorted(plan.checkpoint_dir.rglob("checkpoint-*"))
                   if (p / "adapter_model.safetensors").exists()}
    peak = (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            if torch.cuda.is_available() else None)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"resume_mode": mode, "metrics": metrics, "adapter_sha256": adapter_sha,
            "adapter_path": final.relative_to(ROOT).as_posix(), "checkpoints": checkpoints,
            "train_seconds": round(time.time() - started, 1), "peak_allocated_mib": peak}


def lora_b_norm(adapter_dir: Path) -> float:
    """Frobenius norm of all LoRA-B weights (zero at initialisation, so > 0 proves an update)."""
    from safetensors.torch import load_file
    tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    return round(sum(float(t.float().pow(2).sum()) for k, t in tensors.items()
                     if "lora_B" in k) ** 0.5, 6)


def load_generator(adapter: Path):
    from engine.generator import Phi3Generator
    generator = Phi3Generator(model_name=MODEL, model_revision=REVISION,
                              attention_implementation="sdpa")
    generator.load_model()
    if generator.model_revision != REVISION:
        raise SystemExit("REFUSED: loaded revision differs from the pinned one")
    generator.load_lora_adapter(adapter)
    generator.model.eval()
    return generator


def generate_fixed(adapter: Path, functions: Sequence[Mapping[str, Any]],
                   records: Mapping[str, dict], raw_dir: Path, contract_sha: str
                   ) -> Dict[str, dict]:
    """Greedy fixed-input generation for every expected key; strict resume."""
    import torch
    from engine.test_generation_prompt import format_chat_prompt
    from harness.fixed_input_probe import build_prompt
    expected = ev.expected_fixed_keys(functions)
    path = raw_dir / "fixed_input.jsonl"
    log = raw_dir / "resume_log.jsonl"
    rows, problems = ev.read_generation_file(path, contract_sha, expected)
    if problems:
        ev.quarantine(path, problems, log)
        rows = {}
    work = []
    for function in functions:
        record = records[function["record_id"]]
        for item in function["items"]:
            for schema, level in ev.SCHEMA_LEVELS.items():
                key = f"{item['item_id']}::{schema}"
                if key not in rows:
                    work.append({"key": key, "schema": schema,
                                 **build_prompt(record, item["call"], item["expected_repr"],
                                                level)})
    if work:
        generator = load_generator(adapter)
        tokenizer = generator.tokenizer
        tokenizer.padding_side = "left"
        with path.open("a", encoding="utf-8") as handle:
            for schema in ev.SCHEMA_LEVELS:
                todo = [w for w in work if w["schema"] == schema]
                limit = FIXED_DECODING["max_new_tokens"][schema]
                for start in range(0, len(todo), FIXED_DECODING["batch"]):
                    chunk = todo[start:start + FIXED_DECODING["batch"]]
                    texts = [format_chat_prompt(tokenizer, w["user"]) + w["prefill"]
                             for w in chunk]
                    encoded = tokenizer(texts, return_tensors="pt", padding=True,
                                        add_special_tokens=False).to(generator.model.device)
                    with torch.no_grad():
                        output = generator.model.generate(**encoded, max_new_tokens=limit,
                                                          do_sample=False,
                                                          pad_token_id=tokenizer.pad_token_id)
                    width = encoded["input_ids"].shape[1]
                    for w, row in zip(chunk, output):
                        handle.write(json.dumps({
                            "key": w["key"], "schema": schema, "contract_sha256": contract_sha,
                            "output": tokenizer.decode(row[width:], skip_special_tokens=True)})
                            + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        del generator
        gc.collect()
        torch.cuda.empty_cache()
    rows, problems = ev.read_generation_file(path, contract_sha, expected)
    if problems:
        raise ev.IntegrityError(f"fixed-input file invalid after generation: {problems[:5]}")
    ev.require_complete(rows, expected)
    return rows


def kill_at_8(adapter: Path, adapter_sha: str, record_ids: Sequence[str], raw_dir: Path,
              contract_sha: str, receipt_sha: str) -> Dict[str, Any]:
    """Canonical Kill@8 on exactly ``record_ids``; an incomplete prior attempt is quarantined."""
    import shutil
    import torch
    out = raw_dir / "kill_at_8"
    done = raw_dir / "kill_at_8_complete.json"
    if done.exists():
        meta = json.loads(done.read_text(encoding="utf-8"))
        result = out / "rehearsal_result.json"
        if meta.get("contract_sha256") != contract_sha or lc.sha256_file(result) != meta[
                "rehearsal_result_sha256"]:
            raise ev.IntegrityError("completed Kill@8 does not match its contract")
        artifact = json.loads(result.read_text(encoding="utf-8"))
    else:
        if out.exists():
            (raw_dir / "quarantine").mkdir(exist_ok=True)
            target = raw_dir / "quarantine" / f"{int(time.time())}_kill_at_8"
            shutil.move(str(out), str(target))
            with (raw_dir / "resume_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"quarantined": "kill_at_8", "to": target.name,
                                         "reasons": ["incomplete prior attempt"]}) + "\n")
        from harness.corpus_view import load_development_split
        from harness.evaluation_admission import scope_split
        from harness.generation_adapter import generate_candidate_slots, successor_settings
        from harness.generation_rng import seed_generation_rngs
        from harness.prompt_factory import prompt_factory
        from harness.rehearsal_evaluator import run_rehearsal_evaluation
        from engine.generator import Phi3Generator
        from scripts.train_on_dataset import _record_to_pair
        panel = "phase4_choice_b_v2_panel"
        scope = scope_split({panel: list(record_ids)}, load_development_split(
            ROOT / "data/corpus/v4_1_research_hardened_candidate", "train"), panel,
            adapt=_record_to_pair)
        if scope.target_count != len(record_ids):
            raise ev.IntegrityError("Kill@8 panel did not resolve completely")
        settings = successor_settings()
        if settings.problems():
            raise SystemExit(f"REFUSED: invalid generation settings {settings.problems()}")
        generator = Phi3Generator(model_name=settings.base_model_name,
                                  model_revision=settings.base_model_revision,
                                  attention_implementation=settings.attention_implementation)
        generator.temperature, generator.top_p = settings.temperature, settings.top_p
        generator.parse_mode = settings.candidate_parse_mode
        generator.load_model()
        generator.load_lora_adapter(adapter)
        build_prompt = prompt_factory(settings.prompt_settings())
        seed_record = seed_generation_rngs(settings.seed)

        def generate_batch(batch):
            accounting = generate_candidate_slots(generator, [dict(r) for r in batch],
                                                  settings, build_prompt,
                                                  seed_before_generation=False)
            return [[{"raw_output": s.get("raw_output", ""), "code": s.get("code")}
                     for s in accounting[i]["candidate_slots"]] for i in range(len(batch))]

        artifact = run_rehearsal_evaluation(
            load_records=lambda: scope.eligible, generate_batch=generate_batch,
            output_dir=out, split_name=panel, scope_summary=scope.to_dict(),
            frozen_settings=settings.to_dict(), receipt_sha256=receipt_sha,
            candidates_per_target=settings.candidates_per_function,
            generation_batch_size=settings.generation_batch_size,
            allow_test_function=settings.allow_test_function_candidates,
            log=lambda m: print(m, flush=True), seed_record=seed_record,
            generator_identity={"model_name": settings.base_model_name,
                                "model_revision": settings.base_model_revision,
                                "adapter_sha256": adapter_sha})
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        done.write_text(json.dumps({"contract_sha256": contract_sha, "rehearsal_result_sha256":
                                    lc.sha256_file(out / "rehearsal_result.json")}) + "\n",
                        encoding="utf-8")
    rows = ev.require_function_rows(artifact["function_results"], record_ids)
    return {"rows": rows, "rehearsal_result_sha256": lc.sha256_file(out / "rehearsal_result.json"),
            "wall_time_seconds": artifact.get("wall_time_seconds")}


def generation_settings_summary() -> Dict[str, Any]:
    from harness.generation_adapter import successor_settings
    settings = successor_settings().to_dict()
    return {"successor_settings": settings,
            "successor_settings_sha256": lc.contract_sha256(settings)}


# --- lifecycle stages ---------------------------------------------------------------------

def stage_train(arm: str) -> int:
    ctx = lc.verify_action(layout(), f"train_{arm}")
    preflight = ctx["preflight"]
    spec_arm = preflight["arms"][arm]
    split = load(SPLIT)
    plan = TrainPlan(arm=arm, rows=split["rows"], checkpoint_dir=ROOT / spec_arm["checkpoint_dir"],
                     contract=spec_arm["run_contract"], epochs=EPOCHS,
                     trainer_kwargs=TRAINER_KWARGS)
    outcome = train_arm(plan, train_records())
    metrics = outcome["metrics"]
    stats = spec_arm["dataset_stats"]
    if (metrics["completed_optimizer_steps"] != preflight["optimizer_schedule"][
            "planned_optimizer_steps"] or metrics["retained_examples"] != len(plan.rows)
            or metrics.get("input_ids_sha256") != stats["input_ids_sha256"]
            or metrics.get("labels_sha256") != stats["labels_sha256"]):
        raise SystemExit("STOPPED: the run violated the frozen plan")
    if outcome["adapter_path"] != spec_arm["adapter_path"]:
        raise SystemExit("STOPPED: adapter written to an unexpected path")
    publish_once(RESULTS[f"{arm}_trained"], {
        **lc.artifact_header(layout(), preflight, f"{arm}_trained", arm),
        "status": "complete", "run_contract": spec_arm["run_contract"],
        "adapter_path": outcome["adapter_path"], "adapter_sha256": outcome["adapter_sha256"],
        "checkpoints_sha256": outcome["checkpoints"], "resume_mode": outcome["resume_mode"],
        "launched_at_head": ctx["head"],
        "metrics": {k: v for k, v in metrics.items() if k not in (
            "model_runtime_profile", "monitor_history", "best_validation_metrics")},
        "train_seconds": outcome["train_seconds"],
        "peak_allocated_mib": outcome["peak_allocated_mib"],
        "loss_is_not_an_efficacy_signal": True, "validation_accessed": False,
        "sealed_final_test_accessed": False})
    print(json.dumps({"stage": f"{arm}_trained", "adapter_sha256": outcome["adapter_sha256"],
                      "steps": metrics["completed_optimizer_steps"]}))
    return 0


def stage_evaluate(arm: str) -> int:
    ctx = lc.verify_action(layout(), f"evaluate_{arm}")
    preflight = ctx["preflight"]
    training_rel = RESULTS[f"{arm}_trained"]
    training = load(training_rel)
    training_sha = rel_sha(training_rel)
    contract = lc.evaluation_contract(preflight, arm, training, training_sha)
    contract_sha = lc.contract_sha256(contract)
    raw_dir = ROOT / preflight["arms"][arm]["raw_dir"]
    lc.check_resume(raw_dir, contract)
    split = load(SPLIT)
    records = train_records()
    adapter = ROOT / training["adapter_path"]
    started = time.time()
    fixed = generate_fixed(adapter, split["gate"]["functions"], records, raw_dir, contract_sha)
    fixed_seconds = round(time.time() - started, 1)
    ids = [f["record_id"] for f in split["gate"]["functions"]]
    kill = kill_at_8(adapter, training["adapter_sha256"], ids, raw_dir, contract_sha,
                     rel_sha(PREFLIGHT))
    publish_once(RESULTS[f"{arm}_evaluated"], {
        **lc.artifact_header(layout(), preflight, f"{arm}_evaluated", arm),
        "status": "complete", "training_result_sha256": training_sha,
        "adapter_sha256": training["adapter_sha256"], "evaluation_contract": contract,
        "evaluation_contract_sha256": contract_sha, "launched_at_head": ctx["head"],
        "raw": {"fixed_input_jsonl_sha256": lc.sha256_file(raw_dir / "fixed_input.jsonl"),
                "kill_at_8_rehearsal_result_sha256": kill["rehearsal_result_sha256"],
                "raw_dir": preflight["arms"][arm]["raw_dir"]},
        "evidence": {"fixed_input_outputs": {k: fixed[k]["output"] for k in sorted(fixed)},
                     "kill_at_8_functions": [ev.compact_function(kill["rows"][i])
                                             for i in sorted(kill["rows"])]},
        "seconds": {"fixed_input": fixed_seconds, "kill_at_8": kill["wall_time_seconds"]},
        "metrics_not_read_at_this_stage": True})
    print(json.dumps({"stage": f"{arm}_evaluated", "fixed_keys": len(fixed),
                      "kill_rows": len(kill["rows"])}))
    return 0


def analysis_from_evidence(split: Mapping[str, Any], looks: Mapping[str, Mapping[str, Any]],
                           records: Mapping[str, dict], tiers: Mapping[str, str]
                           ) -> Dict[str, Any]:
    """Pure: the frozen analysis from tracked evidence (reproducible on any clone)."""
    from harness.fixed_input_probe import extract_answer, score_answers
    functions = split["gate"]["functions"]
    expected = ev.expected_fixed_keys(functions)
    ids = [f["record_id"] for f in functions]
    items = {i["item_id"]: (f, i) for f in functions for i in f["items"]}
    per_arm: Dict[str, Dict[str, dict]] = {}
    kill: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm, look in looks.items():
        outputs = look["evidence"]["fixed_input_outputs"]
        ev.require_complete(outputs, expected)
        scored = {}
        for schema, level in ev.SCHEMA_LEVELS.items():
            by_record: Dict[str, list] = {}
            for item_id, (function, item) in items.items():
                answer = extract_answer(outputs[f"{item_id}::{schema}"], level, item["call"])
                by_record.setdefault(function["record_id"], []).append(
                    {"item_id": item_id, "call": item["call"], "answer": answer["answer"],
                     "group_id": function["group_id"], "input_kind": item["input_kind"]})
            for record_id, entries in by_record.items():
                verdicts = score_answers(entries, str(records[record_id]["reference_code"]))
                for entry, verdict in zip(entries, verdicts):
                    scored[f"{entry['item_id']}::{schema}"] = {
                        **entry, "schema": schema, "record_id": record_id,
                        "answered": entry["answer"] is not None, "correct": verdict["correct"]}
        per_arm[arm] = scored
        rows = ev.require_function_rows(look["evidence"]["kill_at_8_functions"], ids)
        kill[arm] = {rid: ev.function_metrics(row) for rid, row in rows.items()}
    group_of = {f["record_id"]: f["group_id"] for f in functions}

    def fixed_metric(schema, field, keys=None):
        keys = keys or sorted(k for k in expected if k.endswith(f"::{schema}"))
        groups = [per_arm["control"][k]["group_id"] for k in keys]
        c = [float(per_arm["control"][k][field]) for k in keys]
        t = [float(per_arm["treatment"][k][field]) for k in keys]
        return {"control": round(sum(c) / len(c) * 100, 3),
                "treatment": round(sum(t) / len(t) * 100, 3), "items": len(keys),
                "difference_points": ev.paired(groups, c, t)}

    def function_metric(name, record_ids=None):
        record_ids = record_ids or ids
        groups = [group_of[r] for r in record_ids]
        c = [kill["control"][r][name] for r in record_ids]
        t = [kill["treatment"][r][name] for r in record_ids]
        scale = 1 if name == "exact_unique_ratio" else 100
        return {"control": round(sum(c) / len(c) * scale, 4),
                "treatment": round(sum(t) / len(t) * scale, 4), "functions": len(record_ids),
                "difference": ev.paired(groups, c, t) if scale == 100 else None}

    fixed = {schema: {"answer_rate": fixed_metric(schema, "answered"),
                      "accuracy": fixed_metric(schema, "correct")}
             for schema in ev.SCHEMA_LEVELS}
    for schema in fixed:
        fixed[schema]["nonanswer_rate"] = {
            arm: round(100 - fixed[schema]["answer_rate"][arm], 3)
            for arm in ("control", "treatment")}
    names = ("kill_at_1", "kill_at_4", "kill_at_8", "parse_success", "execution_success",
             "reference_validity_per_requested", "exact_equality_validity_per_requested",
             "has_valid_killing_candidate", "exact_unique_ratio")
    downstream = {name: function_metric(name) for name in names}
    diversity = ev.relative_change_bootstrap(
        [group_of[r] for r in ids], [kill["control"][r]["exact_unique_ratio"] for r in ids],
        [kill["treatment"][r]["exact_unique_ratio"] for r in ids])
    guardrails = {f"answer_rate_{s}": {**ev.guardrail("answer_rate_points",
                                                     fixed[s]["answer_rate"]["difference_points"])}
                  for s in ev.SCHEMA_LEVELS}
    guardrails["reference_validity"] = ev.guardrail(
        "reference_validity_points",
        downstream["reference_validity_per_requested"]["difference"])
    guardrails["exact_unique_ratio"] = ev.guardrail("exact_unique_relative", diversity)
    primary = {s: fixed[s]["accuracy"]["difference_points"] for s in ev.SCHEMA_LEVELS}
    decision = ev.classify(primary, guardrails)
    record_meta = {f["record_id"]: records[f["record_id"]] for f in functions}
    slice_value = {
        "dataset": lambda e: str(record_meta[e["record_id"]]["source"]["upstream"]),
        "input_kind": lambda e: e["input_kind"],
        "complexity_tier": lambda e: str(tiers.get(e["record_id"], "none")),
        "bug_family": lambda e: str((record_meta[e["record_id"]].get("provenance") or {})
                                    .get("mutation_type")),
    }
    slices = {}
    for name, value in slice_value.items():
        slices[name] = {}
        for schema in ev.SCHEMA_LEVELS:
            keys = sorted(k for k in expected if k.endswith(f"::{schema}"))
            levels = sorted({value(per_arm["control"][k]) for k in keys})
            slices[name][schema] = {
                level: {"items": len(sel), **{arm: round(sum(per_arm[arm][k]["correct"]
                                                             for k in sel) / len(sel) * 100, 2)
                                              for arm in ("control", "treatment")}}
                for level in levels
                for sel in [[k for k in keys if value(per_arm["control"][k]) == level]]}
    item_evidence = [{"key": k, **{arm: {"answered": per_arm[arm][k]["answered"],
                                         "correct": per_arm[arm][k]["correct"]}
                                   for arm in ("control", "treatment")}} for k in expected]
    return {"primary_fixed_input": fixed, "downstream_kill_at_8": downstream,
            "exact_unique_ratio_relative_change": diversity, "guardrails": guardrails,
            "decision": decision,
            "descriptive_slices": {"status": "DESCRIPTIVE ONLY; likely underpowered; never "
                                             "cross-dataset generalisation", **slices},
            "per_item_evidence": item_evidence}


def stage_analyse() -> int:
    ctx = lc.verify_action(layout(), "analyse")
    preflight = ctx["preflight"]
    looks = {arm: load(RESULTS[f"{arm}_evaluated"]) for arm in ARMS}
    for arm, look in looks.items():
        raw_dir = ROOT / look["raw"]["raw_dir"]
        if lc.sha256_file(raw_dir / "fixed_input.jsonl") != look["raw"]["fixed_input_jsonl_sha256"]:
            raise SystemExit(f"REFUSED: {arm} raw fixed-input file changed after its look")
    tiers = {r["record_id"]: r["tier"] for r in load(COMPLEXITY)["records"]}
    result = analysis_from_evidence(load(SPLIT), looks, train_records(), tiers)
    publish_once(RESULTS["analysed"], {
        **lc.artifact_header(layout(), preflight, "analysed", None),
        "status": "complete", "launched_at_head": ctx["head"],
        "gate_looks": {s: rel_sha(RESULTS[s]) for s in ("control_evaluated",
                                                       "treatment_evaluated")},
        "estimand": ev.ESTIMAND, "labels": load(SPEC)["labels"], **result,
        "root_cause_established": False, "generalization_established": False,
        "efficacy_established": False})
    print(json.dumps(result["decision"], indent=1))
    return 0


def analyse_from_evidence() -> int:
    """Recompute the analysis from tracked files only and compare with the recorded one."""
    recorded = load(RESULTS["analysed"])
    looks = {arm: load(RESULTS[f"{arm}_evaluated"]) for arm in ARMS}
    tiers = {r["record_id"]: r["tier"] for r in load(COMPLEXITY)["records"]}
    again = analysis_from_evidence(load(SPLIT), looks, train_records(), tiers)
    same = all(recorded[k] == json.loads(json.dumps(v)) for k, v in again.items())
    print(json.dumps({"reproduced": same}))
    return 0 if same else 1


# --- preflight v2 -------------------------------------------------------------------------

def package_versions() -> Dict[str, str]:
    import importlib.metadata as md
    return {name: md.version(name) for name in ("torch", "transformers", "peft", "trl",
                                                "accelerate", "datasets", "tokenizers")}


def preflight() -> int:
    from engine.sft_trainer import SFT_GRADIENT_ACCUMULATION_STEPS, plan_sft_optimizer_schedule
    from config.settings import model_config, training_config
    from harness.acquisition_receipt import ProtectedAccessMonitor
    from harness.objective_masking import dataset_trainer
    import transformers
    if (ROOT / PREFLIGHT).exists():
        raise SystemExit("REFUSED: preflight v2 exists; it is never overwritten")
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    problems: List[str] = []
    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        problems.append("working tree is not clean")
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", "origin/experiment/research-eval-ablations"):
        problems.append("HEAD is not equal to origin")
    split = load(SPLIT)
    neardup = load(NEARDUP)
    gate = set(split["gate"]["group_ids"])
    rows = split["rows"]
    excluded = set(neardup["excluded_training_groups"])
    isolation = {
        "no_gate_group_in_training": not (gate & {r["group_id"] for r in rows}),
        "no_gate_record_in_training": not ({f["record_id"] for f in split["gate"]["functions"]}
                                           & {r["record_id"] for r in rows}),
        "near_duplicate_groups_excluded": not (excluded & {r["group_id"] for r in rows}),
        "train_split_only": split["splits_read"] == ["train"],
        "unique_rows": len({(r["record_id"], r["call"], r["value"]) for r in rows}) == len(rows),
        "gate_items_expected": len(ev.expected_fixed_keys(split["gate"]["functions"]))
        == 4 * len(gate),
    }
    problems += [f"isolation: {k}" for k, ok in isolation.items() if not ok]
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                           local_files_only=True)
    points = datapoints(rows, train_records())
    arms_stats = {}
    for arm, mode in ARMS.items():
        trainer = dataset_trainer(tokenizer, mode)
        for key in ("max_prompt_tokens", "max_repository_prompt_tokens",
                    "max_completion_tokens", "max_repository_completion_tokens"):
            setattr(trainer, key, TRAINER_KWARGS[key])
        trainer.prepare_dataset(points)
        arms_stats[arm] = dict(trainer.dataset_stats)
    c, t = arms_stats["control"], arms_stats["treatment"]
    equality = {"input_ids": c["input_ids_sha256"] == t["input_ids_sha256"],
                "attention_mask": c["attention_mask_sha256"] == t["attention_mask_sha256"],
                "labels_differ": c["labels_sha256"] != t["labels_sha256"],
                "retained_all": c["retained_examples"] == t["retained_examples"] == len(rows),
                "no_compaction": not c["prompt_truncated_examples"]
                and not t["prompt_truncated_examples"]}
    problems += [f"arm equality: {k}" for k, ok in equality.items() if not ok]
    schedule = plan_sft_optimizer_schedule(len(rows), EPOCHS, BATCH_SIZE,
                                           TRAINER_KWARGS["warmup_steps"],
                                           TRAINER_KWARGS["checkpoint_steps"],
                                           SFT_GRADIENT_ACCUMULATION_STEPS)
    identity = lc.source_identity(layout())
    optimisation = {"learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                    "gradient_accumulation_steps": SFT_GRADIENT_ACCUMULATION_STEPS,
                    "trainer_kwargs": TRAINER_KWARGS, "max_grad_norm":
                        training_config.max_grad_norm, "weight_decay": 0.0,
                    "precision": "bf16 when supported (engine.sft_trainer.use_bf16), else fp16",
                    "gradient_checkpointing": True, "optimizer": "transformers default (AdamW)",
                    "attention_implementation": "sdpa", "seed": SEED,
                    "row_order_sha256": hashlib.sha256(json.dumps(
                        [[r["record_id"], r["call"]] for r in rows]).encode()).hexdigest()}
    lora = {"r": model_config.lora_r, "alpha": model_config.lora_alpha,
            "dropout": model_config.lora_dropout}
    generation = generation_settings_summary()
    arms = {}
    for arm, mode in ARMS.items():
        ckpt = f"checkpoints/phase4_choice_b_v2_{arm}_qwen15b_s{SEED}"
        run_contract = {"schema_version": "oneiros_choice_b_v2_run_contract", "arm": arm,
                        "objective_mode": mode, "checkpoint_dir": ckpt,
                        "source_identity": identity["sha256"], "split_sha256": rel_sha(SPLIT),
                        "model": MODEL, "revision": REVISION, "lora": lora, **optimisation}
        arms[arm] = {
            "objective_mode": mode, "checkpoint_dir": ckpt,
            "adapter_path": f"{ckpt}/final_adapter", "raw_dir": f"{RAW}/{arm}",
            "run_contract": run_contract,
            "dataset_stats": {k: arms_stats[arm][k] for k in (
                "objective_mode", "input_ids_sha256", "attention_mask_sha256", "labels_sha256",
                "supervised_tokens", "retained_examples", "prompt_truncated_examples")},
            "evaluation_contract_template": {
                "schema_version": "oneiros_choice_b_v2_evaluation_contract", "arm": arm,
                "model": MODEL, "revision": REVISION, "source_identity": identity["sha256"],
                "split_sha256": rel_sha(SPLIT), "evaluation_spec_sha256": rel_sha(SPEC),
                "fixed_input_decoding": FIXED_DECODING, "prompt_schemas": ev.SCHEMA_LEVELS,
                "prompt_builder": "harness.fixed_input_probe.build_prompt + format_chat_prompt",
                "answer_parser": "harness.fixed_input_probe.extract_answer",
                "candidate_parser": generation["successor_settings"].get("candidate_parse_mode"),
                "candidates_per_function": generation["successor_settings"].get(
                    "candidates_per_function"),
                "generation_seed": generation["successor_settings"].get("seed"),
                "kill_at_8_settings_sha256": generation["successor_settings_sha256"]}}
    contract_diff = sorted(k for k in arms["control"]["run_contract"]
                           if arms["control"]["run_contract"][k] !=
                           arms["treatment"]["run_contract"][k])
    if contract_diff != ["arm", "checkpoint_dir", "objective_mode"]:
        problems.append(f"run contracts differ beyond arm/objective/output: {contract_diff}")
    eval_diff = sorted(k for k in arms["control"]["evaluation_contract_template"]
                       if arms["control"]["evaluation_contract_template"][k] !=
                       arms["treatment"]["evaluation_contract_template"][k])
    if eval_diff != ["arm"]:
        problems.append(f"evaluation contracts differ beyond the arm: {eval_diff}")
    commands = lc.stage_commands(PYTHON, SCRIPT)
    pairs = {"train": lc.command_difference(commands["train_control"],
                                            commands["train_treatment"]),
             "evaluate": lc.command_difference(commands["evaluate_control"],
                                               commands["evaluate_treatment"])}
    for name, diff in pairs.items():
        if [d[1:] for d in diff] != [(f"choice_b_v2_{name}_control",
                                      f"choice_b_v2_{name}_treatment"),
                                     (f"{name}_control", f"{name}_treatment")]:
            problems.append(f"{name} commands differ beyond the arm: {diff}")
    for rel in RESULTS.values():
        if subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode == 0:
            problems.append(f"stage result would be ignored: {rel}")
    for arm in ARMS:
        for key in ("checkpoint_dir", "raw_dir"):
            probe = f"{arms[arm][key]}/contract.json"
            if subprocess.run(["git", "check-ignore", "-q", probe], cwd=ROOT).returncode != 0:
                problems.append(f"{key} for {arm} is not ignored")
            if (ROOT / arms[arm][key]).exists():
                problems.append(f"{arms[arm][key]} already exists")
    steps = schedule["planned_optimizer_steps"]
    evidence = ProtectedAccessMonitor.evidence(mark)
    if evidence["protected_paths_opened"]:
        problems.append(f"protected paths opened {evidence['protected_paths_opened']}")
    receipt = {
        "schema_version": lc.PREFLIGHT_SCHEMA, "ready": not problems, "problems": problems,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"commit": head, "clean": not status, "equals_origin": True if not [
            p for p in problems if "origin" in p] else False},
        "supersedes": {"path": "results/sft_root_cause_phase4_choice_b_preflight_receipt.json",
                       "sha256": rel_sha("results/sft_root_cause_phase4_choice_b_preflight_receipt.json"),
                       "modified": False},
        "source_identity": identity, "split": SPLIT, "split_sha256": rel_sha(SPLIT),
        "near_duplicate_audit_sha256": rel_sha(NEARDUP),
        "evaluation_spec": SPEC, "evaluation_spec_sha256": rel_sha(SPEC),
        "model": MODEL, "revision": REVISION, "seed": SEED,
        "tokenizer": {"name": MODEL, "revision": REVISION, "vocab_size": len(tokenizer),
                      "eos_token": tokenizer.eos_token},
        "lora": lora, "optimisation": optimisation, "package_versions": package_versions(),
        "optimizer_schedule": schedule, "arms": arms, "arm_equality": equality,
        "run_contract_differences": contract_diff, "evaluation_contract_differences": eval_diff,
        "isolation": isolation, "generation": generation,
        "lifecycle": {"stages": list(lc.STAGES), "actions": {a: list(v) for a, v in
                                                            lc.ACTIONS.items()},
                      "results": RESULTS, "result_prefix": PREFIX},
        "stage_commands": commands, "command_differences": pairs,
        "estimand": ev.ESTIMAND, "labels": load(SPEC)["labels"],
        "runtime_and_storage_estimate": {
            "planned_optimizer_steps_per_arm": steps,
            "training_minutes_per_arm": round(steps * SECONDS_PER_STEP / 60, 1),
            "evaluation_minutes_per_arm": "~5 fixed-input + ~20-30 Kill@8 (200 x 8)",
            "gpu_hours_total_estimate": round((2 * steps * SECONDS_PER_STEP + 2 * 35 * 60)
                                              / 3600, 2),
            "storage_gb_estimate": "~0.3 per arm (LoRA checkpoints + final adapter) + <0.2 raw"},
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
        "training_launched": False, "evaluation_launched": False,
    }
    publish_once(PREFLIGHT, receipt)
    print(json.dumps({k: receipt[k] for k in ("ready", "problems", "arm_equality",
                                              "runtime_and_storage_estimate")}, indent=1))
    return 0 if receipt["ready"] else 2


# --- integration smoke (disposable; non-gate data) -----------------------------------------

def smoke() -> int:
    """Operational evidence only. Uses the first training rows (never the gate)."""
    import shutil
    import torch
    split = load(SPLIT)
    gate = set(split["gate"]["group_ids"])
    rows = split["rows"][:32]
    if gate & {r["group_id"] for r in rows}:
        raise SystemExit("REFUSED: smoke rows touch the gate")
    records = train_records()
    ckpt_root, raw_root = ROOT / SMOKE_CKPT, ROOT / SMOKE_RAW
    for path in (ckpt_root, raw_root):
        if path.exists():
            quarantine = path.parent / f"{path.name}_QUARANTINE_{int(time.time())}"
            shutil.move(str(path), str(quarantine))
    kwargs = {**TRAINER_KWARGS, "checkpoint_steps": 1, "warmup_steps": 1}
    report: Dict[str, Any] = {"schema_version": "oneiros_choice_b_v2_smoke",
                              "purpose": "operational integration evidence only; not efficacy",
                              "rows": len(rows), "gate_touched": False, "arms": {}}
    fn = {}
    for row in rows:
        if row["record_id"] not in fn and len(fn) < 2:
            fn[row["record_id"]] = row
    functions = []
    for record_id, row in fn.items():
        record = records[record_id]
        functions.append({"record_id": record_id, "group_id": record["group_id"], "items": [
            {"input_kind": kind, "call": row["call"], "expected_repr": row["value"],
             "item_id": f"{record_id}::{kind}"} for kind in ev.INPUT_KINDS]})
    for arm in ARMS:
        contract = {"schema_version": "oneiros_choice_b_v2_smoke_contract", "arm": arm,
                    "objective_mode": ARMS[arm], "rows": len(rows), "epochs": 1,
                    "trainer_kwargs": kwargs}
        plan = TrainPlan(arm=arm, rows=rows, checkpoint_dir=ckpt_root / arm, contract=contract,
                         epochs=1, trainer_kwargs=kwargs)
        first = train_arm(plan, records)
        again = train_arm(plan, records)          # exact-contract resume from the checkpoint
        try:
            lc.check_resume(plan.checkpoint_dir, {**contract, "epochs": 2})
            refused = False
        except lc.LifecycleError:
            refused = True
        adapter = ROOT / first["adapter_path"]
        raw = raw_root / arm
        eval_contract = {"arm": arm, "adapter_sha256": again["adapter_sha256"], "smoke": True}
        lc.check_resume(raw, eval_contract)
        c_sha = lc.contract_sha256(eval_contract)
        fixed = generate_fixed(adapter, functions, records, raw, c_sha)
        resumed = generate_fixed(adapter, functions, records, raw, c_sha)   # no regeneration
        kill = kill_at_8(adapter, again["adapter_sha256"], list(fn), raw, c_sha, "smoke")
        report["arms"][arm] = {
            "steps_first": first["metrics"]["completed_optimizer_steps"],
            "steps_after_resume": again["metrics"]["completed_optimizer_steps"],
            "resume_mode_second_run": again["resume_mode"],
            "resumed_from_checkpoint": again["metrics"].get("resumed_from_checkpoint")
            is not None,
            "changed_contract_refused": refused,
            "checkpoints": first["checkpoints"], "adapter_sha256_first": first["adapter_sha256"],
            "adapter_sha256_after_resume": again["adapter_sha256"],
            "labels_sha256": first["metrics"].get("labels_sha256"),
            "input_ids_sha256": first["metrics"].get("input_ids_sha256"),
            "supervised_tokens": first["metrics"].get("supervised_tokens"),
            "lora_b_norm": lora_b_norm(adapter),
            "train_seconds": first["train_seconds"],
            "peak_allocated_mib": first["peak_allocated_mib"],
            "fixed_keys": len(fixed), "fixed_resume_identical": resumed == fixed,
            "kill_rows": len(kill["rows"]), "kill_seconds": kill["wall_time_seconds"]}
        torch.cuda.empty_cache()
    a, b = report["arms"]["control"], report["arms"]["treatment"]
    report["checks"] = {
        "input_ids_identical": a["input_ids_sha256"] == b["input_ids_sha256"],
        "labels_differ": a["labels_sha256"] != b["labels_sha256"],
        "optimizer_updates": a["steps_first"] >= 1 and b["steps_first"] >= 1
        and a["lora_b_norm"] > 0 and b["lora_b_norm"] > 0,
        "adapter_unchanged_by_exact_resume": all(x["adapter_sha256_first"] ==
                                                 x["adapter_sha256_after_resume"]
                                                 for x in (a, b)),
        "checkpoints_saved": bool(a["checkpoints"]) and bool(b["checkpoints"]),
        "exact_contract_resume": all(x["resume_mode_second_run"] == "resume"
                                     and x["resumed_from_checkpoint"] for x in (a, b)),
        "changed_contract_refused": a["changed_contract_refused"]
        and b["changed_contract_refused"],
        "fixed_generation_complete_and_resumable": all(x["fixed_keys"] == 8 and
                                                       x["fixed_resume_identical"]
                                                       for x in (a, b)),
        "kill_at_8_rows_exact": a["kill_rows"] == b["kill_rows"] == 2}
    report["passed"] = all(report["checks"].values())
    report["disposable_dirs"] = [SMOKE_CKPT, SMOKE_RAW]
    report["cleanup"] = "kept in ignored disposable directories; quarantined on the next smoke"
    raw_path = ROOT / SMOKE_RAW / "smoke_report.json"
    raw_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=1))
    return 0 if report["passed"] else 3


def status() -> int:
    lay = layout()
    done = lc.completed_stages(lay)
    nxt = next((a for a, (req, _, _) in lc.ACTIONS.items() if req == done[-1]), None)
    print(json.dumps({"completed": done, "next_action": nxt}))
    try:
        lc.verify_action(lay, nxt) if nxt else None
        print("launch verifier: PERMITS", nxt)
    except lc.LifecycleError as error:
        print("launch verifier: REFUSES", error)
    return 0


def validate() -> int:
    lay = layout()
    preflight = lc.load_preflight(lay)
    for stage in lc.completed_stages(lay)[1:]:
        artifact = lc.validate_artifact(lay, preflight, stage)
        if stage.endswith("_evaluated"):
            raw = ROOT / artifact["raw"]["raw_dir"]
            if lc.sha256_file(raw / "fixed_input.jsonl") != artifact["raw"][
                    "fixed_input_jsonl_sha256"]:
                raise SystemExit(f"{stage}: raw fixed-input file differs")
        print("valid:", stage)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("freeze", "spec", "preflight", "status", "validate",
                                            "run", "smoke", "analyse-from-evidence"))
    parser.add_argument("action", nargs="?", choices=tuple(lc.ACTIONS))
    args = parser.parse_args(argv)
    if args.command == "run":
        if not args.action:
            parser.error("run needs an action")
        if args.action.startswith("train_"):
            return stage_train(args.action.split("_", 1)[1])
        if args.action.startswith("evaluate_"):
            return stage_evaluate(args.action.split("_", 1)[1])
        return stage_analyse()
    return {"freeze": freeze, "spec": spec, "preflight": preflight, "status": status,
            "validate": validate, "smoke": smoke,
            "analyse-from-evidence": analyse_from_evidence}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
