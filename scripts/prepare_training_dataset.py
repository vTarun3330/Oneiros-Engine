"""Create and verify the canonical, leak-free Phase 3 dataset.

This is the only supported preparation path before SFT/DPO.  It preserves the
raw generator output and writes a normalized, de-duplicated canonical dataset,
group-disjoint splits, and a hash manifest consumed by the training launcher.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import queue
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.corpus import write_json as atomic_write_json
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "mutation_pairs.json"
CLEAN_FILE = DATA_DIR / "mutation_pairs_clean.json"
SPLITS_DIR = DATA_DIR / "splits"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.json"
SCHEMA_VERSION = 1
REQUIRED_FIELDS = {
    "id", "source", "golden_code", "mutant_code", "entry_point",
    "test_cases", "mutation_type", "mutation_description",
}


def normalized_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_key(pair: Dict[str, Any]) -> str:
    """Function identity intentionally excludes source to prevent cross-source leaks."""
    return hashlib.sha256(pair["golden_code"].encode("utf-8")).hexdigest()


def mutation_key(pair: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        function_key(pair),
        pair["entry_point"],
        hashlib.sha256(pair["mutant_code"].encode("utf-8")).hexdigest(),
    )


def callable_names(code: str) -> set[str]:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(code)
    return {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def normalize_pair(raw: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str | None]:
    """Perform non-executing schema, syntax, and test normalization checks."""
    if not isinstance(raw, dict) or REQUIRED_FIELDS - raw.keys():
        return None, "missing_required_field"
    if not all(isinstance(raw[name], str) and raw[name].strip() for name in REQUIRED_FIELDS - {"test_cases"}):
        return None, "empty_or_nonstring_field"
    if not isinstance(raw["test_cases"], list) or not all(isinstance(item, str) for item in raw["test_cases"]):
        return None, "invalid_test_cases"
    if not raw["entry_point"].isidentifier():
        return None, "invalid_entry_point"

    pair = dict(raw)
    pair["golden_code"] = normalized_code(raw["golden_code"])
    pair["mutant_code"] = normalized_code(raw["mutant_code"])
    if pair["golden_code"] == pair["mutant_code"]:
        return None, "identical_mutation"
    try:
        golden_names = callable_names(pair["golden_code"])
        callable_names(pair["mutant_code"])
    except (SyntaxError, ValueError, TypeError):
        return None, "invalid_python"
    if pair["entry_point"] not in golden_names:
        return None, "entry_point_not_defined"

    from harness.training_data import extract_dataset_assertions
    assertions = extract_dataset_assertions(raw["test_cases"], pair["entry_point"])
    if not assertions:
        return None, "no_usable_assertions"
    pair["test_cases"] = assertions
    return pair, None


def _behavioral_check(pair: Dict[str, Any]) -> Tuple[bool, str]:
    """Ensure fixtures accept the golden code and expose this mutant."""
    from baseline.benchmark_runner import safe_exec

    killers = 0
    for assertion in pair["test_cases"]:
        golden_ok, _, _ = safe_exec(pair["golden_code"], assertion)
        if not golden_ok:
            return False, "golden_fixture_failure"
        mutant_ok, _, _ = safe_exec(pair["mutant_code"], assertion)
        if not mutant_ok:
            killers += 1
    return (True, "ok") if killers else (False, "surviving_mutant")


def _behavior_worker(tasks: mp.Queue, results: mp.Queue) -> None:
    while True:
        item = tasks.get()
        if item is None:
            return
        index, pair = item
        try:
            valid, reason = _behavioral_check(pair)
        except BaseException as exc:  # never let one fixture kill the preparation job
            valid, reason = False, f"worker_exception:{type(exc).__name__}"
        results.put((index, valid, reason))


def behavioral_filter(pairs: List[Dict[str, Any]], workers: int, timeout: float) -> Tuple[List[Dict[str, Any]], Counter]:
    """Run behavior checks in killable worker processes with per-pair timeouts."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    task_queues: Dict[int, mp.Queue] = {}
    processes: Dict[int, mp.Process] = {}
    active: Dict[int, Tuple[int, float]] = {}
    accepted: Dict[int, Dict[str, Any]] = {}
    rejected: Counter = Counter()
    next_index = 0

    def start_worker(slot: int) -> None:
        task_queue = context.Queue(maxsize=1)
        process = context.Process(target=_behavior_worker, args=(task_queue, result_queue))
        process.start()
        task_queues[slot] = task_queue
        processes[slot] = process

    def dispatch(slot: int) -> None:
        nonlocal next_index
        if next_index >= len(pairs):
            return
        task_queues[slot].put((next_index, pairs[next_index]))
        active[slot] = (next_index, time.monotonic())
        next_index += 1

    for slot in range(max(1, workers)):
        start_worker(slot)
        dispatch(slot)

    try:
        while active:
            try:
                pair_index, valid, reason = result_queue.get(timeout=0.05)
                slot = next(slot for slot, state in active.items() if state[0] == pair_index)
                active.pop(slot)
                if valid:
                    accepted[pair_index] = pairs[pair_index]
                else:
                    rejected[reason] += 1
                dispatch(slot)
            except queue.Empty:
                pass

            now = time.monotonic()
            for slot, (pair_index, started) in list(active.items()):
                if now - started <= timeout:
                    continue
                rejected["behavior_timeout"] += 1
                active.pop(slot)
                processes[slot].terminate()
                processes[slot].join(timeout=1)
                start_worker(slot)
                dispatch(slot)
            if len(accepted) + sum(rejected.values()) and (len(accepted) + sum(rejected.values())) % 500 == 0:
                print(f"  behavior checked: {len(accepted) + sum(rejected.values()):,}/{len(pairs):,}", flush=True)
    finally:
        for slot, process in processes.items():
            if process.is_alive():
                try:
                    task_queues[slot].put_nowait(None)
                except queue.Full:
                    process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
    return [accepted[index] for index in sorted(accepted)], rejected


def split_group_disjoint(pairs: List[Dict[str, Any]], seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    import random

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pairs:
        groups.setdefault(function_key(pair), []).append(pair)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    targets = {"train": int(len(pairs) * 0.8), "val": int(len(pairs) * 0.1)}
    targets["test"] = len(pairs) - targets["train"] - targets["val"]
    output = {"train": [], "val": [], "test": []}
    counts = {name: 0 for name in output}
    for key in keys:
        destination = max(counts, key=lambda name: (targets[name] - counts[name], -counts[name]))
        output[destination].extend(groups[key])
        counts[destination] += len(groups[key])
    return output


def write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def prepare(skip_behavioral: bool = False, workers: int = 8, timeout: float = 2.0) -> Dict[str, Any]:
    raw_pairs = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    clean: List[Dict[str, Any]] = []
    rejected: Counter = Counter()
    seen_ids = set()
    seen_mutations = set()

    for raw in raw_pairs:
        pair, reason = normalize_pair(raw)
        if reason:
            rejected[reason] += 1
            continue
        if pair["id"] in seen_ids:
            rejected["duplicate_id"] += 1
            continue
        key = mutation_key(pair)
        if key in seen_mutations:
            rejected["duplicate_mutation"] += 1
            continue
        seen_ids.add(pair["id"])
        seen_mutations.add(key)
        clean.append(pair)

    if skip_behavioral:
        raise ValueError("Refusing to create a training manifest without behavioral validation.")
    print(f"Running behavioral validation for {len(clean):,} structurally clean pairs...")
    clean, behavior_rejected = behavioral_filter(clean, workers=workers, timeout=timeout)
    rejected.update(behavior_rejected)
    if not clean:
        raise RuntimeError("No pairs passed the dataset quality gate.")

    write_json(CLEAN_FILE, clean)
    splits = split_group_disjoint(clean)
    for name, records in splits.items():
        write_json(SPLITS_DIR / f"{name}_pairs.json", records)

    split_groups = {name: {function_key(pair) for pair in records} for name, records in splits.items()}
    if split_groups["train"] & split_groups["val"] or split_groups["train"] & split_groups["test"] or split_groups["val"] & split_groups["test"]:
        raise RuntimeError("Internal error: function overlap after split.")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "raw_input": {"path": RAW_FILE.name, "sha256": sha256_file(RAW_FILE), "pair_count": len(raw_pairs)},
        "clean_dataset": {"path": CLEAN_FILE.name, "sha256": sha256_file(CLEAN_FILE), "pair_count": len(clean)},
        "splits": {
            name: {
                "path": f"splits/{name}_pairs.json",
                "sha256": sha256_file(SPLITS_DIR / f"{name}_pairs.json"),
                "pair_count": len(records),
                "function_count": len(split_groups[name]),
            }
            for name, records in splits.items()
        },
        "quality_gate": {
            "all_retained_pairs_behaviorally_verified": True,
            "per_pair_timeout_seconds": timeout,
            "exact_duplicate_mutations_removed": rejected.get("duplicate_mutation", 0),
            "rejected": dict(sorted(rejected.items())),
        },
    }
    write_json(MANIFEST_FILE, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the canonical Oneiros SFT/DPO dataset")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent behavioral validation workers")
    parser.add_argument("--timeout", type=float, default=2.0, help="Per-pair behavioral validation timeout")
    args = parser.parse_args()
    manifest = prepare(workers=args.workers, timeout=args.timeout)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
