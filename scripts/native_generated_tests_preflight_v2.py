"""Fail-closed, source-bound preflight (protocol v2 + amendment v2.1) for the dress rehearsal.

Every check verifies behaviour or a hash, never mere presence. Reports three separate states:
  pipeline_ready  the whole CPU/WSL pipeline is proven for the current source and cohort;
  gpu_authorized  a GPU authorisation receipt exists for this preflight (never created here);
  launch_ready    pipeline_ready AND gpu_authorized.
Exit status is non-zero unless pipeline_ready. v1 and its receipt are unchanged. Launches
nothing.

    python scripts/native_generated_tests_preflight_v2.py
"""
from __future__ import annotations

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

OUTPUT = "results/sft_root_cause_native_generated_tests_preflight_v2_1.json"
PROTOCOLS = ("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md",
             "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md")
SOURCE_MANIFEST = "results/sft_root_cause_phase4_receiver_capture_manifest_v2.json"
MANIFEST = "results/sft_root_cause_native_v21_rehearsal_manifest_v4.json"
ISOLATION = "results/sft_root_cause_native_v21_isolation_v6.json"
JOB = "results/sft_root_cause_native_v21_rehearsal_job_v2.json"
SUITE = "results/sft_root_cause/native_v2_full_suite.json"
CANARY_DIR = "results/sft_root_cause/native_v21_canaries"
AUTHORIZATION = "results/sft_root_cause_native_gpu_authorization_v1.json"
CLIS = {"scripts/native_generated_tests_generate.py": ["--help"],
        "scripts/native_generated_tests_execute_wsl.py": ["--help"],
        "scripts/native_generated_tests_atheris_wsl.py": ["--help"],
        "scripts/native_generated_tests_analyse.py": ["--help"]}
FROZEN_CONTRACT = {
    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "base_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    "sft_adapter": "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter",
    "seeds": [42, 43, 44], "candidates": 8, "temperature": 0.7, "top_p": 0.9,
    "do_sample": True, "max_new_tokens": 1024, "prompt_token_limit": 2048,
    "sequence_limit": 3072, "batch_size": 2, "extraction": "whole_output_single_fence_strip",
    "reranking": "none", "duplicates": "kept", "raw_output_retained": True,
    "attention_implementation": "sdpa"}
ADAPTER_SHA256 = "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> int:
    from harness.acquisition_receipt import ProtectedAccessMonitor
    ProtectedAccessMonitor.install(ROOT)
    mark = ProtectedAccessMonitor.mark()
    blockers: list = []
    checks: dict = {}

    def require(name: str, ok: bool, detail=None) -> None:
        checks[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        if not ok:
            blockers.append(name if detail is None else f"{name}: {json.dumps(detail, default=str)[:300]}")

    subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT)
    head = git("rev-parse", "HEAD")
    require("clean_tree", not git("status", "--porcelain", "--untracked-files=all"))
    require("head_equals_fetched_origin",
            head == git("rev-parse", "origin/experiment/research-eval-ablations"))
    suite = load(SUITE) if (ROOT / SUITE).exists() else {}
    require("full_suite_green_clean_at_source",
            suite.get("source_commit") == head and suite.get("tree_clean_at_start") is True
            and suite.get("exit") == 0 and suite.get("failed") == 0 and suite.get("passed", 0) > 0,
            {k: suite.get(k) for k in ("source_commit", "tree_clean_at_start", "passed", "failed")})
    helps = {}
    for script, args in CLIS.items():
        done = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
        helps[script] = done.returncode == 0 and "usage" in done.stdout.lower()
    require("cli_help", all(helps.values()), {k: v for k, v in helps.items() if not v} or None)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests/test_native_generated_generate.py",
                            "tests/test_native_generated_execute.py",
                            "tests/test_native_generated_analyse.py",
                            "tests/test_native_generated_prompt.py",
                            "tests/test_native_generated_atheris.py"],
                           cwd=ROOT, capture_output=True, text=True, timeout=900)
    require("identity_authorization_and_unit_tests", tests.returncode == 0,
            tests.stdout.strip().splitlines()[-1:] or None)
    from scripts.native_generated_tests_generate import CONTRACT
    require("generation_contract_complete_equality", CONTRACT == FROZEN_CONTRACT)
    require("protocol_and_amendment_present", all((ROOT / p).is_file() for p in PROTOCOLS))
    receipts = {
        "synthetic_pipeline": (f"{CANARY_DIR}/pipeline_receipt.json", "components_sha256"),
        "sandbox_canaries": (f"{CANARY_DIR}/canary_receipt_v2.json", None),
        "atheris_canaries": (f"{CANARY_DIR}/atheris_canary_receipt_v2.json", None)}
    for name, (rel, _) in receipts.items():
        receipt = load(rel) if (ROOT / rel).exists() else {}
        bound = True
        if name == "synthetic_pipeline":
            bound = all(sha(c) == h for c, h in receipt.get("components_sha256", {}).items()) \
                and bool(receipt.get("components_sha256"))
        elif name == "sandbox_canaries":
            bound = receipt.get("executor_sha256") == sha("scripts/native_generated_tests_execute_wsl.py") \
                and receipt.get("inner_sha256") == sha("scripts/native_sandbox_inner.sh") \
                and receipt.get("prepare_sha256") == sha("scripts/native_rehearsal_prepare_wsl.py")
        else:
            bound = receipt.get("script_sha256") == sha("scripts/native_generated_tests_atheris_wsl.py")
        require(f"{name}_passed_for_current_source", receipt.get("passed") is True and bound,
                None if receipt.get("passed") else receipt.get("checks"))
    manifest = load(MANIFEST) if (ROOT / MANIFEST).exists() else {}
    require("manifest_present", bool(manifest))
    if manifest:
        require("source_manifest_and_target_set",
                manifest["source_manifest"]["sha256"] == sha(SOURCE_MANIFEST)
                and set(manifest["kept_targets"]) <= {t["key"] for t in load(SOURCE_MANIFEST)["targets"]})
        require("protocol_hashes_exact", manifest["protocols"] == {p: sha(p) for p in PROTOCOLS})
        iso = load(ISOLATION)
        require("isolation_bound_to_current_implementation",
                iso["isolation_source_sha256"] == canonical_sha256(ROOT / "harness/repository_isolation.py")
                and manifest["isolation"]["receipt_sha256"] == sha(ISOLATION)
                and manifest["isolation"]["isolation_version"] == "oneiros_repository_isolation_v6",
                {"reference_universe_sha256": manifest["isolation"]["reference_universe_sha256"]})
        require("every_kept_target_v6_and_formally_requalified", all(
            p["isolation_v6_admissible"] and p["requalification"] == "requalified"
            for p in manifest["per_target"] if p["kept"]))
        require("rehearsal_rule", manifest["rehearsal_rule"]["passed"])
        require("job_coverage_and_no_unrecorded_loss", manifest["job"]["coverage_passed"]
                and manifest["job"]["sha256"] == sha(JOB),
                {"items": manifest["job"]["items"], "kept": manifest["kept"],
                 "refused": manifest["job"]["refused"]})
        require("sequence_fit_within_limit",
                (manifest["job"]["max_prompt_tokens"] or 0) <= FROZEN_CONTRACT["prompt_token_limit"])
        require("manifest_protected_audit_clean",
                not manifest["protected_access_audit"]["protected_paths_opened"])
    from scripts.native_generated_tests_generate import adapter_manifest
    adapter = adapter_manifest(ROOT / FROZEN_CONTRACT["sft_adapter"])
    require("adapter_manifest_complete", "adapter_config.json" in adapter
            and adapter.get("adapter_model.safetensors") == ADAPTER_SHA256)
    evidence = ProtectedAccessMonitor.evidence(mark)
    require("no_protected_access", not evidence["protected_paths_opened"],
            evidence["protected_paths_opened"] or None)
    pipeline_ready = not blockers
    gpu_authorized = (ROOT / AUTHORIZATION).exists()
    items = manifest.get("job", {}).get("items", 0)
    candidates = 2 * items * 3 * 8
    receipt = {
        "schema_version": "oneiros_native_generated_tests_preflight_v2_1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": head,
        "receipt_commit": ("this receipt is committed separately; its commit's parent must be "
                           "source_commit"),
        "protocols": {p: sha(p) for p in PROTOCOLS},
        "pipeline_ready": pipeline_ready, "gpu_authorized": gpu_authorized,
        "launch_ready": pipeline_ready and gpu_authorized,
        "blockers": blockers, "checks": checks,
        "inputs": {rel: sha(rel) for rel in (MANIFEST, ISOLATION, JOB, SUITE,
                                             f"{CANARY_DIR}/pipeline_receipt.json",
                                             f"{CANARY_DIR}/canary_receipt_v2.json",
                                             f"{CANARY_DIR}/atheris_canary_receipt_v2.json")
                   if (ROOT / rel).exists()},
        "generation_contract": FROZEN_CONTRACT,
        "proposed_gpu_commands_NOT_EXECUTED": [
            f".venv-gpu/Scripts/python.exe scripts/gpu_run.py start --name native_v21_generate_{arm} "
            f"-- .venv-gpu/Scripts/python.exe scripts/native_generated_tests_generate.py run "
            f"--job {JOB} --condition primary_whole_module --arm {arm} "
            f"--out results/sft_root_cause/native_v21_generations --backend hf "
            f"--preflight {OUTPUT} --authorization {AUTHORIZATION}" for arm in ("base", "sft")],
        "estimates_not_executed": {
            "candidates": candidates,
            "gpu_hours_range": [round(candidates * 200 / (2 * 60) / 3600, 2),
                                round(candidates * 1024 / (2 * 60) / 3600, 2)],
            "basis": "batch 2 at ~60 generated tokens/s per sequence; 200 (typical) to 1024 "
                     "(maximum) new tokens per candidate; model load ~1 min per arm",
            "gpu_memory_gib": "~4-6 PyTorch allocated (1.5B bf16 + LoRA, batch 2, <=3072 tokens)",
            "storage_mb": "<50 raw generations; sandbox execution ~1-3 CPU-hours in WSL"},
        "confirmation_acquisition": "NOT AUTHORISED",
        "root_cause_established": False, "generalization_established": False,
        "gpu_generation_launched": False,
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
    }
    publish_file_atomically(ROOT / OUTPUT, (json.dumps(receipt, indent=1, sort_keys=True)
                                            + "\n").encode("utf-8"))
    print(json.dumps({"pipeline_ready": pipeline_ready, "gpu_authorized": gpu_authorized,
                      "launch_ready": receipt["launch_ready"], "blockers": blockers}, indent=1))
    return 0 if pipeline_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
