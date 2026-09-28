"""Phase 1 of the SFT root-cause loop: the candidate-level causal ledger.

Reuses retained raw outputs only; no model is loaded and no GPU is used.

* locked validation (spent): base and arm A checkpoint 431, 757 function records each;
* the arm A training monitor on ablation_dev (spent; it selected checkpoint 431):
  step 0 (the base) and checkpoints 50-431, 500 function records each.

Every requested candidate becomes one ledger row (``harness/causal_ledger.py``).
The ledgers are large and stay local under ``results/sft_root_cause/``; the
committed receipt carries their hashes, the input hashes and the counts.

Refuses any input that is not one of the frozen artifacts below, any artifact
that is a final-test measurement, and any record shard other than val and
ablation_dev.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.causal_ledger import LEDGER_SCHEMA_VERSION, TERMINAL_CATEGORIES, measure_function
from harness.parallel_execution import map_jobs

VIEW = "data/corpus/v4_1_research_hardened_candidate/development_view"
SHARDS = {"val": f"{VIEW}/val.records.json", "ablation_dev": f"{VIEW}/ablation_dev.records.json"}
COMPLEXITY = f"{VIEW}/complexity_manifest.json"
MONITOR = "results/local_sft_armA_baseline_successor_s42"
ARMS = [
    {"arm": "locked_val_base", "split": "val", "model": "base", "step": 0,
     "artifact": "results/locked_val_base_qwen_s42/"
                 "base_validation_parse-whole-output_completion1024_seed_42.json",
     "expected_sha256": "64beb7d83ea234c46813553abe9d590a19558edd24cbe9d161a36923d3541b46"},
    {"arm": "locked_val_armA_431", "split": "val", "model": "arm_a_sft", "step": 431,
     "artifact": "results/locked_val_armA_ckpt431_s42/"
                 "sft_validation_parse-whole-output_completion1024_seed_42.json",
     "expected_sha256": "700a233b509ad2dc22b11b1849a568de5849c7d2f9cea0e484013367e405bfc2"},
] + [
    {"arm": f"monitor_step_{step:03d}", "split": "ablation_dev",
     "model": "base" if step == 0 else "arm_a_sft", "step": step,
     "artifact": f"{MONITOR}/sft_monitor_{'baseline' if step == 0 else f'checkpoint_{step}'}.json",
     "expected_sha256": None}
    for step in (0, 50, 100, 150, 200, 250, 300, 350, 400, 431)
]
OUTPUT_DIR = "results/sft_root_cause"
RECEIPT = "results/sft_root_cause_phase1_ledger_receipt.json"
FORBIDDEN = ("test.records.json", "sealed_final", "unopened_confirmation", "/records.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_shard(split: str) -> dict[str, dict]:
    path = SHARDS[split]
    assert not any(token in "/" + path for token in FORBIDDEN[:3]), path
    return {r["id"]: r for r in json.loads((ROOT / path).read_text(encoding="utf-8"))}


def _job(job: dict) -> list[dict]:
    rows = measure_function(job["result"], job["record"])
    context = job["context"]
    return [{**context, **row} for row in rows]


def main() -> int:
    started = time.time()
    shards = {split: _load_shard(split) for split in SHARDS}
    complexity = {r["record_id"]: r for r in json.loads(
        (ROOT / COMPLEXITY).read_text(encoding="utf-8"))["records"]}
    inputs, jobs = [], []
    for arm in ARMS:
        path = ROOT / arm["artifact"]
        digest = sha256(path)
        if arm["expected_sha256"] and digest != arm["expected_sha256"]:
            raise SystemExit(f"REFUSED: {arm['artifact']} changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("final_test_measurement") or payload.get("evaluation_split") != arm["split"]:
            raise SystemExit(f"REFUSED: {arm['artifact']} is not a {arm['split']} artifact")
        inputs.append({"arm": arm["arm"], "artifact": arm["artifact"], "sha256": digest,
                       "split": arm["split"], "model": arm["model"], "checkpoint_step": arm["step"],
                       "seed": payload.get("seed"),
                       "adapter_sha256": payload.get("adapter_sha256"),
                       "run_contract_sha256": payload.get("run_contract_sha256"),
                       "function_results": len(payload["function_results"])})
        for result in payload["function_results"]:
            record = shards[arm["split"]][result["record_id"]]
            tier = complexity.get(result["record_id"], {})
            jobs.append({"result": result, "record": record, "context": {
                "arm": arm["arm"], "split": arm["split"], "model": arm["model"],
                "checkpoint_step": arm["step"], "seed": payload.get("seed"),
                "record_id": result["record_id"], "group_id": record.get("group_id"),
                "upstream_task": str((record.get("provenance") or {}).get(
                    "upstream_record_id", "")).rsplit("_mut_", 1)[0],
                "dataset": result.get("dataset_name"), "bug_family": result.get("bug_family"),
                "source_name": result.get("source_name"), "project": result.get("project"),
                "record_content_hash": record.get("content_hash"),
                "complexity_tier": tier.get("tier"),
                "cyclomatic_complexity": tier.get("cyclomatic_complexity"),
                "function_killed": bool(result.get("killed"))}})
    print(f"{len(jobs)} function evaluations across {len(ARMS)} arms", flush=True)
    results = map_jobs(_job, jobs)
    out = ROOT / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    by_arm: dict[str, list[dict]] = {}
    for rows in results:
        for row in rows:
            by_arm.setdefault(row["arm"], []).append(row)
    summary = {}
    ledgers = {}
    for arm in ARMS:
        rows = by_arm[arm["arm"]]
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        path = out / f"ledger_{arm['arm']}.jsonl"
        publish_file_atomically(path, text)
        ledgers[arm["arm"]] = {"path": path.relative_to(ROOT).as_posix(),
                               "sha256": hashlib.sha256(text).hexdigest(), "rows": len(rows)}
        categories = Counter(row["terminal_category"] for row in rows)
        reproduced = [row["reproduction"]["agrees"] for row in rows if row["reproduction"]]
        executed = [row for row in rows if row["executed"] and not row["duplicate"]]
        consistency = [row["instrumented_oracle_consistent"] for row in rows
                       if row["instrumented_oracle_consistent"] is not None]
        summary[arm["arm"]] = {
            "candidates": len(rows),
            "terminal_categories": {c: categories.get(c, 0) for c in TERMINAL_CATEGORIES},
            "categories_sum_to_candidates": sum(categories.values()) == len(rows),
            "evaluator_verdict_reproduction": {"checked": len(reproduced),
                                               "agree": sum(reproduced)},
            "instrumented_oracle_consistency": {"checked": len(consistency),
                                                "agree": sum(consistency)},
            "executed_non_duplicate_without_discrimination_measurement": sum(
                1 for row in executed if not row["discrimination"]["measured"]),
            "mutant_run_failures_counted_as_discriminating": sum(
                1 for row in executed if str(row["discrimination"].get("reason", ""))
                .startswith("mutant_run_")),
        }
    receipt = {
        "schema_version": "oneiros_sft_root_cause_phase1_receipt_v1",
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "phase": 1,
        "objective": "candidate-level causal ledger over retained raw outputs",
        "command": "python scripts/build_sft_causal_ledger.py",
        "protocol": "docs/SFT_ROOT_CAUSE_PROTOCOL.md",
        "model_calls": 0, "gpu_used": False,
        "data_accessed": {"record_shards": {s: {"path": p, "sha256": sha256(ROOT / p)}
                                            for s, p in SHARDS.items()},
                          "complexity_manifest": {"path": COMPLEXITY,
                                                  "sha256": sha256(ROOT / COMPLEXITY)},
                          "test_or_sealed_or_confirmation_read": False},
        "inputs": inputs,
        "ledgers": ledgers,
        "summary": summary,
        "duration_seconds": round(time.time() - started, 1),
    }
    publish_file_atomically(ROOT / RECEIPT, (json.dumps(receipt, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({arm: {"categories": s["terminal_categories"],
                            "repro": s["evaluator_verdict_reproduction"]}
                      for arm, s in summary.items()}, indent=1))
    print(f"duration {receipt['duration_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
