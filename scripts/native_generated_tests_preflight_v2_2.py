"""Immutable, fail-closed pipeline preflight (protocol v2 + amendments v2.1 and v2.2).

Every check verifies behaviour or a hash, never mere presence. It reports ONLY
``pipeline_ready``: the whole CPU/WSL pipeline is proven for the current canonical executable
source and cohort. Authorisation is never evaluated here (a file at the authorisation path
means nothing); the separate read-only launch gate (scripts/native_generated_tests_launch_gate.py)
reports pipeline_ready, gpu_authorized and launch_ready as distinct states.

The receipt is written once: an existing receipt is never overwritten or re-timestamped, so an
authorisation that names its hash stays valid. v2.1 and its red receipt are unchanged.
Launches nothing.

    python scripts/native_generated_tests_preflight_v2_2.py [--out PATH]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import publish_file_atomically
from harness.source_identity import canonical_sha256

OUTPUT = "results/sft_root_cause_native_generated_tests_preflight_v2_2.json"
AUTHORIZATION = "results/sft_root_cause_native_gpu_authorization_v2.json"
SOURCE_MANIFEST = "results/sft_root_cause_phase4_receiver_capture_manifest_v2.json"
MANIFEST = "results/sft_root_cause_native_v22_rehearsal_manifest_v5.json"
JOB = "results/sft_root_cause_native_v22_rehearsal_job_v3.json"
PROMPTS = "results/sft_root_cause_native_v22_prompt_records.json"
ISOLATION = "results/sft_root_cause_native_v21_isolation_v6.json"
SUITE = "results/sft_root_cause/native_v22_full_suite.json"
CANARY_DIR = "results/sft_root_cause/native_v22_canaries"
GENERATIONS = "results/sft_root_cause/native_v22_generations"
CLIS = ("scripts/native_generated_tests_generate.py", "scripts/native_generated_tests_execute_wsl.py",
        "scripts/native_generated_tests_atheris_wsl.py", "scripts/native_generated_tests_analyse.py",
        "scripts/native_generated_tests_launch_gate.py", "scripts/native_rehearsal_rebuild_v22.py")
FOCUSED_TESTS = ("tests/test_native_generated_generate.py", "tests/test_native_generated_execute.py",
                 "tests/test_native_generated_analyse.py", "tests/test_native_generated_prompt.py",
                 "tests/test_native_generated_prompt_v2.py",
                 "tests/test_native_generated_leakage_v2.py",
                 "tests/test_native_generated_atheris.py", "tests/test_native_launch_gate.py")
FROZEN_CONTRACT = {
    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "base_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    "sft_adapter": "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter",
    "seeds": [42, 43, 44], "candidates": 8, "temperature": 0.7, "top_p": 0.9,
    "do_sample": True, "max_new_tokens": 1024, "prompt_token_limit": 2048,
    "sequence_limit": 3072, "batch_size": 2, "extraction": "whole_output_single_fence_strip",
    "reranking": "none", "duplicates": "kept", "raw_output_retained": True,
    "attention_implementation": "sdpa"}
ADAPTER_SAFETENSORS_SHA256 = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main(argv=None) -> int:
    from harness import native_launch_gate as gate
    from harness.acquisition_receipt import ProtectedAccessMonitor
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=OUTPUT)
    out_rel = parser.parse_args(argv).out
    if (ROOT / out_rel).exists():
        raise SystemExit(f"REFUSED: {out_rel} exists; a preflight receipt is immutable - "
                         "write a successor path")
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    blockers: list = []
    checks: dict = {}

    def require(name: str, ok: bool, detail=None) -> None:
        checks[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        if not ok:
            blockers.append(name if detail is None else
                            f"{name}: {json.dumps(detail, default=str)[:300]}")

    # 1. checkout: fresh fetch, synced, completely clean
    fetch = gate.fresh_fetch(ROOT)
    state = gate.checkout_state(ROOT, fetch)
    head = state["head"]
    require("fresh_fetch_succeeded", fetch["rc"] == 0, fetch)
    require("head_equals_fetched_remote_sha", state["synced"],
            {"head": head, "remote_sha": fetch["remote_sha"]})
    require("clean_tree", not git("status", "--porcelain", "--untracked-files=all"))
    identity = gate.source_identity(ROOT)
    # 2. full suite alone at this executable source
    suite = load(SUITE) if (ROOT / SUITE).exists() else {}
    relation = gate.receipt_only_descendant(ROOT, suite.get("source_commit", ""), head) \
        if suite.get("source_commit") else {"ok": False}
    require("full_suite_green_alone_at_this_source",
            suite.get("exit") == 0 and suite.get("failed") == 0 and suite.get("passed", 0) > 0
            and suite.get("tree_clean_at_start") is True and relation["ok"]
            and suite.get("executable_tree_sha256") == identity["executable_tree_sha256"],
            {k: suite.get(k) for k in ("source_commit", "passed", "failed", "skipped",
                                       "tree_clean_at_start")})
    # 3. CLIs and focused tests
    helps = {}
    for script in CLIS:
        done = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=180)
        helps[script] = done.returncode == 0 and "usage" in done.stdout.lower()
    require("cli_help", all(helps.values()), {k: v for k, v in helps.items() if not v} or None)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            *FOCUSED_TESTS], cwd=ROOT, capture_output=True, text=True,
                           timeout=1800)
    require("focused_unit_scanner_and_lifecycle_tests", tests.returncode == 0,
            tests.stdout.strip().splitlines()[-1:] or None)
    # 4. contract and protocols
    from scripts.native_generated_tests_generate import (CONTRACT, adapter_manifest,
                                                         adapter_sha256, model_identity)
    require("generation_contract_complete_equality", CONTRACT == FROZEN_CONTRACT)
    require("protocol_and_amendments_present",
            len(identity["protocol_sha256"]) == len(gate.PROTOCOL_FILES))
    # 5. canaries bound to the current source
    receipts = {
        "synthetic_pipeline": f"{CANARY_DIR}/pipeline_receipt.json",
        "sandbox_canaries": f"{CANARY_DIR}/canary_receipt_v2.json",
        "atheris_canaries": f"{CANARY_DIR}/atheris_canary_receipt_v3.json"}
    for name, rel in receipts.items():
        receipt = load(rel) if (ROOT / rel).exists() else {}
        if name == "synthetic_pipeline":
            bound = bool(receipt.get("components_sha256")) and all(
                sha(c) == h for c, h in receipt["components_sha256"].items())
        elif name == "sandbox_canaries":
            bound = receipt.get("executor_sha256") == sha("scripts/native_generated_tests_execute_wsl.py") \
                and receipt.get("inner_sha256") == sha("scripts/native_sandbox_inner.sh") \
                and receipt.get("prepare_sha256") == sha("scripts/native_rehearsal_prepare_wsl.py")
        else:
            bound = receipt.get("script_sha256") == sha("scripts/native_generated_tests_atheris_wsl.py") \
                and receipt.get("inner_sha256") == sha("scripts/native_sandbox_inner.sh")
        require(f"{name}_passed_for_current_source", receipt.get("passed") is True and bound,
                None if receipt.get("passed") else
                [k for k, v in (receipt.get("checks") or {}).items() if not v])
    # 6. cohort, prompts and job: an independent rebuild must reproduce them byte for byte
    rebuild = subprocess.run([sys.executable, str(ROOT / "scripts/native_rehearsal_rebuild_v22.py")],
                             cwd=ROOT, capture_output=True, text=True, timeout=1800)
    try:
        rebuilt = json.loads(rebuild.stdout[rebuild.stdout.index("{"):])
    except ValueError:
        rebuilt = {}
    require("job_manifest_and_prompts_reproduce_exactly", rebuild.returncode == 0 and set(
        (rebuilt.get("status") or {}).values()) == {"verified_reproduction"},
        rebuilt.get("status") or rebuild.stderr[-300:])
    manifest = load(MANIFEST) if (ROOT / MANIFEST).exists() else {}
    prompts = load(PROMPTS) if (ROOT / PROMPTS).exists() else {}
    job_file = load(JOB) if (ROOT / JOB).exists() else {}
    require("manifest_present", bool(manifest and prompts and job_file))
    job = (job_file.get("primary_whole_module") or {})
    if manifest and prompts and job_file:
        from harness.native_generated_test_leakage import SCANNER_VERSION
        from harness.native_generated_test_prompt import BUILDER_VERSION
        require("source_manifest_and_target_set",
                manifest["source_manifest"]["sha256"] == sha(SOURCE_MANIFEST)
                and set(manifest["kept_targets"]) <= {t["key"] for t in load(SOURCE_MANIFEST)["targets"]})
        require("protocol_hashes_exact", manifest["protocols"] == {p: sha(p) for p in gate.PROTOCOL_FILES})
        iso = load(ISOLATION)
        require("isolation_bound_to_current_implementation",
                iso["isolation_source_sha256"] == canonical_sha256(ROOT / "harness/repository_isolation.py")
                and manifest["isolation"]["receipt_sha256"] == sha(ISOLATION)
                and manifest["isolation"]["isolation_version"] == "oneiros_repository_isolation_v6")
        require("every_kept_target_v6_and_formally_requalified", all(
            p["isolation_v6_admissible"] and p["requalification"] == "requalified"
            for p in manifest["per_target"] if p["kept"]))
        require("rehearsal_rule", manifest["rehearsal_rule"]["passed"])
        acc = manifest["job"]["accounting"]
        require("job_coverage_unique_admitted_targets",
                manifest["job"]["coverage_passed"] and manifest["job"]["sha256"] == sha(JOB)
                and acc["admitted"] >= 20 and acc["admitted"] >= 0.9 * acc["denominator_kept"]
                and not acc["unaccounted"] and not acc["unexpected"],
                {k: acc[k] for k in ("admitted", "denominator_kept", "per_reason",
                                     "combinations")})
        admitted = [r for r in prompts["rows"] if r.get("admitted")]
        require("admitted_prompts_fit_clean_untruncated",
                len(admitted) == len(job.get("items", [])) and all(
                    r["v22_prompt_tokens"] <= FROZEN_CONTRACT["prompt_token_limit"]
                    and r["leakage_ok"] and r["truncated"] is False for r in admitted),
                {"max_prompt_tokens": max((r["v22_prompt_tokens"] for r in admitted), default=None)})
        require("builder_and_scanner_versions_current",
                prompts["builder_version"] == BUILDER_VERSION
                and prompts["scanner_version"] == SCANNER_VERSION
                and prompts["builder_sha256"] == canonical_sha256(ROOT / "harness/native_generated_test_prompt.py")
                and prompts["scanner_sha256"] == canonical_sha256(ROOT / "harness/native_generated_test_leakage.py"))
        require("manifest_protected_audit_clean",
                not manifest["protected_access_audit"]["protected_paths_opened"])
    # 7. model and adapter identities (no model load)
    adapter = adapter_manifest(ROOT / FROZEN_CONTRACT["sft_adapter"])
    require("adapter_manifest_complete", "adapter_config.json" in adapter
            and adapter.get("adapter_model.safetensors") == ADAPTER_SAFETENSORS_SHA256)
    model = model_identity()
    evidence = ProtectedAccessMonitor.evidence(mark)
    require("no_protected_access", not evidence["protected_paths_opened"],
            evidence["protected_paths_opened"] or None)
    pipeline_ready = not blockers
    items = len(job.get("items", []))
    candidates = 2 * items * len(FROZEN_CONTRACT["seeds"]) * FROZEN_CONTRACT["candidates"]
    receipt = {
        "schema_version": gate.PREFLIGHT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_ready": pipeline_ready, "blockers": blockers, "checks": checks,
        "authorization": "NOT EVALUATED HERE: the separate read-only launch gate reports "
                         "pipeline_ready, gpu_authorized and launch_ready",
        "source": {"commit": head, **identity},
        "fetch": {"rc": fetch["rc"], "remote_sha": fetch["remote_sha"]},
        "job": {"path": JOB, "file_sha256": sha(JOB) if (ROOT / JOB).exists() else None,
                "job_sha256": job.get("job_sha256"), "items": items,
                "targets": [i["target_key"] for i in job.get("items", [])]},
        "manifest": {"path": MANIFEST, "sha256": sha(MANIFEST) if manifest else None},
        "prompt_records": {"path": PROMPTS, "sha256": sha(PROMPTS) if prompts else None},
        "model": model, "adapter_manifest_sha256": adapter_sha256(),
        "generation_contract": FROZEN_CONTRACT,
        "inputs": {rel: sha(rel) for rel in (MANIFEST, JOB, PROMPTS, ISOLATION, SUITE,
                                             *(f"{CANARY_DIR}/{n}" for n in (
                                                 "pipeline_receipt.json", "canary_receipt_v2.json",
                                                 "atheris_canary_receipt_v3.json")))
                   if (ROOT / rel).exists()},
        "proposed_gpu_commands_NOT_EXECUTED": [
            f".venv-gpu/Scripts/python.exe scripts/gpu_run.py start --name native_v22_generate_{arm} "
            f"-- .venv-gpu/Scripts/python.exe scripts/native_generated_tests_generate.py run "
            f"--job {JOB} --condition primary_whole_module --arm {arm} "
            f"--out {GENERATIONS}/{arm} --backend hf --preflight {out_rel} "
            f"--authorization {AUTHORIZATION}" for arm in ("base", "sft")],
        "proposed_launch_gate_check_NOT_EXECUTED": [
            f".venv-gpu/Scripts/python.exe scripts/native_generated_tests_launch_gate.py "
            f"--preflight {out_rel} --authorization {AUTHORIZATION} --job {JOB} --arm {arm} "
            f"--out {GENERATIONS}/{arm}" for arm in ("base", "sft")],
        "estimates_not_executed": {
            "candidates": candidates,
            "gpu_hours_range": [round(candidates * 200 / (2 * 60) / 3600, 2),
                                round(candidates * 1024 / (2 * 60) / 3600, 2)],
            "basis": "sequential arms; batch 2 at ~60 generated tokens/s per sequence; 200 "
                     "(typical) to 1024 (maximum) new tokens per candidate; model load ~1 min "
                     "per arm",
            "gpu_memory_gib": "~4-6 PyTorch allocated (1.5B bf16 weights ~3.1 GB + LoRA + KV "
                              "cache ~0.2 GB for 2 x 3072 tokens) on the 24 GB RTX 4500",
            "storage_mb": "<50 raw generations; sandbox execution ~1-3 CPU-hours in WSL"},
        "confirmation_acquisition": "NOT AUTHORISED",
        "root_cause_established": False, "generalization_established": False,
        "sft_benefit_established": False, "atheris_superiority_established": False,
        "gpu_generation_launched": False,
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
    }
    publish_file_atomically(ROOT / out_rel, (json.dumps(receipt, indent=1, sort_keys=True)
                                             + "\n").encode("utf-8"))
    print(json.dumps({"pipeline_ready": pipeline_ready, "blockers": blockers,
                      "source_commit": head, "items": items}, indent=1))
    return 0 if pipeline_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
