"""Portable evidence bundles for the Phase 4 native rehearsal and GPU objective smoke.

Build (on the GPU machine, where the ignored originals exist):
    python scripts/phase4_evidence_bundles.py build
Verify (on any clone; the originals are checked too when present):
    python scripts/phase4_evidence_bundles.py verify

Each bundle is one tracked JSON file that embeds the ignored originals as text with
their hashes (see harness/evidence_bundle.py) and derives the headline facts from the
embedded text, cross-checked against the already-tracked v1 receipts. Model weights,
checkpoints, virtual environments and large logs are never embedded. A bundle is
refused, not overwritten, if it already exists with different bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.evidence_bundle import (
    SCHEMA_VERSION, bundle_problems, embed, git_canonical_sha256, load_json_text,
    load_jsonl_text,
)

NATIVE_OUT = "results/sft_root_cause_phase4_native_evidence_v1.json"
SMOKE_OUT = "results/sft_root_cause_phase4_smoke_evidence_v1.json"
NATIVE_RECEIPT = "results/sft_root_cause_phase4_native_rehearsal_receipt_v1.json"
NATIVE_MANIFEST = "results/sft_root_cause_phase4_native_rehearsal_manifest_v1.json"
SMOKE_RECEIPT = "results/sft_root_cause_phase4_objective_smoke_v1.json"
NATIVE_DIR = "results/sft_root_cause/phase4_native_rehearsal"
SMOKE_DIR = "results/sft_root_cause/phase4_smoke"
RUNS = {"control": "runs/20260929-114416-phase4_smoke_control",
        "treatment": "runs/20260929-114445-phase4_smoke_treatment"}
NATIVE_PUBLISH_COMMIT = "94a96be8413a67defef23f4836bd3b8c46347b04"   # commit that published the rehearsal and its runner
NATIVE_SOURCES = ("scripts/native_rehearsal_wsl.py", "scripts/phase4_native_rehearsal_manifest.py",
                  "scripts/phase4_native_rehearsal_summary.py", "scripts/wsl_install_uv.sh")
SMOKE_SOURCES = ("scripts/phase4_objective_smoke.py", "engine/sft_trainer.py",
                 "harness/objective_masking.py", "scripts/gpu_run.py")
DEPENDENCY_LIMITATION = (
    "The v1 native rehearsal did NOT record per-repository dependency resolution (no "
    "uv pip freeze, no lock or requirements hashes). It is not reconstructed "
    "retroactively: re-resolving today could pick different versions. What is known is "
    "the uv version, the Python used per target, and the install-command KIND (extras, "
    "groups, requirement files). scripts/native_rehearsal_wsl.py now records Python and "
    "uv versions, the sanitized install command, dependency-file hashes and a sanitized "
    "freeze with its SHA-256 for every future run.")


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def native_facts(files: Dict[str, dict]) -> dict:
    records = load_jsonl_text(files["records.jsonl"])
    per_target = []
    for r in sorted(records, key=lambda r: r["key"]):
        env = r.get("steps", {}).get("environment", {})
        probe = r.get("fixed_call_probe") or {}
        per_target.append({
            "key": r["key"], "repository": r["repository"], "category": r["category"],
            "environment_failure": r["environment_failure"], "wall_seconds": r["wall_seconds"],
            "python": env.get("python"), "environment_seconds": env.get("seconds"),
            "install_command_kind": [a.get("command_kind") for a in env.get("attempts", [])],
            "tests": r.get("tests"),
            "difference_exposing_count": r.get("difference_exposing_count"),
            "literal_calls_found": probe.get("literal_calls_found"),
            "safe_fixed_call_constructed": probe.get("safe_fixed_call_constructed")})
    categories: Dict[str, int] = {}
    for row in per_target:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    return {"targets": len(per_target), "categories": dict(sorted(categories.items())),
            "per_target": per_target,
            "run_summary": {k: v for k, v in load_json_text(files["run_summary.json"]).items()
                            if k in ("wall_seconds", "uv", "cpus", "host")}}


def smoke_facts(files: Dict[str, dict]) -> dict:
    arms = {}
    for arm in RUNS:
        receipt = load_json_text(files[f"{arm}_receipt.json"])
        status = load_json_text(files[f"{arm}_run_status.json"])
        manifest = load_json_text(files[f"{arm}_run_manifest.json"])
        arms[arm] = {
            "run_id": status["run_id"], "state": status["state"],
            "exit_code": status["exit_code"], "duration_seconds": status["duration_seconds"],
            "artifact_contract": status["artifact_validation"]["artifact_contract"],
            "run_git_commit": manifest["git"]["commit"], "run_git_dirty": manifest["git"]["dirty"],
            "versions": manifest["versions"],
            "objective_mode": receipt["objective_mode"], "model": receipt["model"],
            "revision": receipt["revision"], "seed": receipt["seed"],
            "examples": receipt["examples"], "example_keys_sha256": receipt["example_keys_sha256"],
            "optimizer": receipt["optimizer"], "dataset": receipt["dataset"],
            "trainer_batch_sequence": receipt["trainer_batch_sequence"],
            "per_batch": receipt["trainer_batches"],
            "lora_b_norm": receipt["lora_b_norm"], "timing": receipt["timing"],
            "gpu": receipt["gpu"], "evaluation_run": receipt["evaluation_run"],
            "promoted": receipt["promoted"]}
    return arms


def build_native() -> dict:
    files = {"records.jsonl": embed(ROOT, f"{NATIVE_DIR}/records.jsonl", []),
             "run_summary.json": embed(ROOT, f"{NATIVE_DIR}/run_summary.json", ["wsl_host"]),
             "console.log": embed(ROOT, f"{NATIVE_DIR}/console.log", [])}
    return {
        "schema_version": SCHEMA_VERSION, "bundle": "phase4_native_rehearsal_v1",
        "purpose": ("portable copy of the ignored v1 native-rehearsal evidence; the ignored "
                    "originals stay on the GPU machine"),
        "files": files,
        "links": {NATIVE_RECEIPT: sha(NATIVE_RECEIPT), NATIVE_MANIFEST: sha(NATIVE_MANIFEST)},
        "sources_canonical_sha256_at_publish_commit": {
            "commit": NATIVE_PUBLISH_COMMIT,
            "note": ("the runner ran from the working tree and was committed, unchanged in "
                     "role, at this commit; the runner was later extended (dependency "
                     "recording), which does not affect v1"),
            "files": {rel: git_canonical_sha256(ROOT, NATIVE_PUBLISH_COMMIT, rel)
                      for rel in NATIVE_SOURCES}},
        "dependency_resolution": {"recorded_in_v1": False, "limitation": DEPENDENCY_LIMITATION},
        "facts": native_facts(files),
    }


def build_smoke() -> dict:
    files = {}
    for arm, run in RUNS.items():
        files[f"{arm}_receipt.json"] = embed(ROOT, f"{SMOKE_DIR}/{arm}_receipt.json",
                                             ["repository_root"])
        files[f"{arm}_run_status.json"] = embed(ROOT, f"{run}/status.json", ["repository_root"])
        files[f"{arm}_run_manifest.json"] = embed(ROOT, f"{run}/manifest.json",
                                                  ["repository_root"])
    arms = smoke_facts(files)
    commit = arms["control"]["run_git_commit"]
    return {
        "schema_version": SCHEMA_VERSION, "bundle": "phase4_objective_smoke_v1",
        "purpose": ("portable copy of the ignored GPU-smoke receipts and durable-run status; "
                    "checkpoints (disposable) and logs are not embedded"),
        "files": files,
        "links": {SMOKE_RECEIPT: sha(SMOKE_RECEIPT)},
        "sources_canonical_sha256_at_run_commit": {
            "commit": commit,
            "files": {rel: git_canonical_sha256(ROOT, commit, rel) for rel in SMOKE_SOURCES}},
        "durable_runner_note": ("scripts/gpu_run.py ran both arms with artifact_contract="
                                "none_declared (no --run-name), so the runner validated each "
                                "run on exit code alone; the per-arm receipt, not the runner, "
                                "carries the scientific checks"),
        "facts": {"arms": arms},
    }


def native_problems(bundle: dict) -> List[str]:
    problems = bundle_problems(ROOT, bundle)
    files, facts = bundle["files"], bundle["facts"]
    receipt = load(NATIVE_RECEIPT)
    records_key = f"{NATIVE_DIR}/records.jsonl"
    if files["records.jsonl"]["text_sha256"] != receipt["inputs"][records_key]:
        problems.append("embedded records differ from the records hashed by the v1 receipt")
    if native_facts(files) != facts:
        problems.append("native facts do not re-derive from the embedded records")
    by_key = {row["key"]: row for row in receipt["per_target"]}
    for row in facts["per_target"]:
        tracked = by_key.get(row["key"])
        if tracked is None or (tracked["category"], tracked["wall_seconds"]) != (
                row["category"], row["wall_seconds"]):
            problems.append(f"per-target mismatch with the v1 receipt: {row['key']}")
    if facts["categories"] != receipt["categories"] or facts["targets"] != receipt["targets"]:
        problems.append("category totals differ from the v1 receipt")
    for rel, digest in bundle["links"].items():
        if sha(rel) != digest:
            problems.append(f"linked receipt changed: {rel}")
    return problems


def smoke_problems(bundle: dict) -> List[str]:
    problems = bundle_problems(ROOT, bundle)
    files, arms = bundle["files"], bundle["facts"]["arms"]
    compare = load(SMOKE_RECEIPT)
    for arm in RUNS:
        if files[f"{arm}_receipt.json"]["original_sha256"] != compare["local_receipts_sha256"][arm]:
            problems.append(f"{arm} receipt original differs from the one the v1 smoke hashed")
    if smoke_facts(files) != arms:
        problems.append("smoke facts do not re-derive from the embedded files")
    c, t = arms["control"], arms["treatment"]
    if [b["input_ids_sha256"] for b in c["per_batch"]] != [
            b["input_ids_sha256"] for b in t["per_batch"]]:
        problems.append("per-batch input_ids differ between arms")
    if [b["attention_mask_sha256"] for b in c["per_batch"]] != [
            b["attention_mask_sha256"] for b in t["per_batch"]]:
        problems.append("per-batch attention masks differ between arms")
    if c["trainer_batch_sequence"]["labels_sha256"] == t["trainer_batch_sequence"]["labels_sha256"]:
        problems.append("labels do not differ between arms")
    for key in ("model", "revision", "seed", "examples", "example_keys_sha256", "optimizer",
                "run_git_commit", "versions"):
        if c[key] != t[key]:
            problems.append(f"arms differ in {key}")
    for arm, facts in arms.items():
        if (facts["state"], facts["exit_code"]) != ("completed", 0):
            problems.append(f"{arm} run did not complete with exit 0")
        if facts["optimizer"]["completed_steps"] != facts["optimizer"]["planned_steps"]:
            problems.append(f"{arm} did not complete its optimiser steps")
        if facts["artifact_contract"] != "none_declared":
            problems.append(f"{arm}: unexpected artifact contract")
        if facts["evaluation_run"] or facts["promoted"]:
            problems.append(f"{arm}: evaluation or promotion recorded")
        if compare["per_arm"][arm]["dataset_labels_sha256"] != facts["dataset"]["labels_sha256"]:
            problems.append(f"{arm}: labels hash differs from the v1 smoke receipt")
    for rel, digest in bundle["links"].items():
        if sha(rel) != digest:
            problems.append(f"linked receipt changed: {rel}")
    return problems


def verify() -> Dict[str, List[str]]:
    return {NATIVE_OUT: native_problems(load(NATIVE_OUT)),
            SMOKE_OUT: smoke_problems(load(SMOKE_OUT))}


def publish(rel: str, bundle: dict) -> None:
    data = (json.dumps(bundle, indent=1, sort_keys=True) + "\n").encode("utf-8")
    target = ROOT / rel
    if target.exists() and target.read_bytes() != data:
        raise SystemExit(f"REFUSED: {rel} exists with different bytes; write a successor")
    publish_file_atomically(target, data)


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["verify"])[0]
    if mode == "build":
        native, smoke = build_native(), build_smoke()
        for name, bundle, check in ((NATIVE_OUT, native, native_problems),
                                    (SMOKE_OUT, smoke, smoke_problems)):
            problems = check(bundle)
            if problems:
                raise SystemExit(f"REFUSED {name}: {problems}")
            publish(name, bundle)
    result = verify()
    print(json.dumps({name: problems or "OK" for name, problems in result.items()}, indent=1))
    return 1 if any(result.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
