"""v3 re-scoring of the retained Phase 3 fixed-input generations (CPU only).

Every scored row carries an exact item identity:
``item_key = "<record_id>::<input_kind>::<sha256(call)[:16]>"``, so paired
comparisons can be validated item by item (``harness.probe_statistics``).

Reads only: the frozen Phase 3A cohort (hash-checked against its design
receipt), the train record shard it names (the reference implementation is
used for scoring only), and generation files hash-checked against their
completion receipts.  No model is called.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.fixed_input_probe import CONTROL_LEVEL, LEVELS, extract_answer, score_answers

ROOT = Path(__file__).resolve().parent.parent
DESIGN_3A = "results/sft_root_cause_phase3a_design_receipt.json"
TRAIN_SHARD = "data/corpus/v4_1_research_hardened_candidate/development_view/train.records.json"
ALL_LEVELS = (*LEVELS, CONTROL_LEVEL)
SOURCES = {"base": "results/sft_root_cause/phase3a",
           "arm_a_431": "results/sft_root_cause/phase3a",
           "qwen7b_base": "results/sft_root_cause/phase3c"}
ROW_HASH_FIELDS = ("arm", "level", "item_key", "group_id", "answer", "method", "correct",
                   "status", "hit_token_limit")


def sha_file(path: str | Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def item_key(record_id: str, input_kind: str, call: str) -> str:
    return f"{record_id}::{input_kind}::{hashlib.sha256(call.encode('utf-8')).hexdigest()[:16]}"


def load_inputs(arms: Sequence[str]) -> dict[str, Any]:
    design = json.loads((ROOT / DESIGN_3A).read_text(encoding="utf-8"))
    if design["record_shard"]["path"] != TRAIN_SHARD:
        raise SystemExit("REFUSED: Phase 3 scoring may read only the train shard")
    cohort_path = ROOT / design["cohort"]["path"]
    if sha_file(cohort_path) != design["cohort"]["sha256"]:
        raise SystemExit("REFUSED: cohort changed")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    ids = {f["record_id"] for f in cohort["functions"]}
    records = {r["id"]: r for r in json.loads((ROOT / TRAIN_SHARD).read_text(encoding="utf-8"))
               if r["id"] in ids}
    if set(records) != ids:
        raise SystemExit("REFUSED: cohort records missing from the train shard")
    generations, completions = {}, {}
    for arm in arms:
        completion_path = f"{SOURCES[arm]}/completion_{arm}.json"
        completion = json.loads((ROOT / completion_path).read_text(encoding="utf-8"))
        path = ROOT / completion["output"]
        if sha_file(path) != completion["output_sha256"]:
            raise SystemExit(f"REFUSED: {arm} generations changed after completion")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        keys = [r["key"] for r in rows]
        if len(keys) != len(set(keys)):
            raise SystemExit(f"REFUSED: duplicate generation keys for {arm}")
        generations[arm] = {r["key"]: r for r in rows}
        completions[arm] = {"completion_receipt": completion_path,
                            "completion_receipt_sha256": sha_file(completion_path),
                            "output": completion["output"],
                            "output_sha256": completion["output_sha256"],
                            "model": completion["model"], "revision": completion["revision"],
                            "adapter_sha256": completion.get("adapter_sha256"),
                            "generations": len(rows)}
    return {"design": design, "cohort": cohort, "records": records,
            "generations": generations, "completions": completions}


def score_function(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    function, record, generations = job["function"], job["record"], job["generations"]
    rows = []
    for item in function["items"]:
        key = item_key(function["record_id"], item["input_kind"], item["call"])
        for arm, gens in generations.items():
            for level in ALL_LEVELS:
                generation = gens.get(f"{function['record_id']}::{item['input_kind']}::{level}")
                answer = (extract_answer(generation["output"], level, item["call"])
                          if generation else {"answer": None, "method": "missing"})
                rows.append({
                    "arm": arm, "level": level, "item_key": key,
                    "record_id": function["record_id"], "input_kind": item["input_kind"],
                    "call": item["call"], "group_id": function["group_id"],
                    "cohort": function["cohort"], "complexity_tier": function["complexity_tier"],
                    "bug_family": function["bug_family"],
                    "sft_target": item["call_in_arm_a_sft_target"],
                    "answer": answer["answer"], "method": answer["method"],
                    "hit_token_limit": bool(generation and generation["hit_token_limit"])})
    for row, verdict in zip(rows, score_answers(rows, record["reference_code"])):
        row.update(verdict)
    return rows


def row_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical keyed row hash plus count checks."""
    keys = [(r["arm"], r["level"], r["item_key"]) for r in rows]
    canonical = sorted(json.dumps({f: r[f] for f in ROW_HASH_FIELDS}, sort_keys=True,
                                  ensure_ascii=True) for r in rows)
    return {"rows": len(rows), "unique_keys": len(set(keys)),
            "duplicates": len(keys) - len(set(keys)),
            "fields": list(ROW_HASH_FIELDS),
            "canonical_scored_rows_sha256": hashlib.sha256(
                "\n".join(canonical).encode("utf-8")).hexdigest()}
