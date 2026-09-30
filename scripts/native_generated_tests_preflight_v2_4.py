"""Immutable, fail-closed pipeline preflight (protocol v2 + amendments v2.1 to v2.4).

Every check verifies behaviour or a hash, never mere presence. It reports ONLY
``pipeline_ready``: the whole CPU/WSL pipeline is proven for the current canonical executable
source and the explicit cohort (24 qualified, 23 generated, one pre-generation exclusion,
1,104 expected candidates). Amendment v2.4 adds: telemetry v2, the exact preparation file,
auditable execution evidence, Atheris design v4, the engineering-only study mode and coverage
gate, the sequential/exclusive launch contract, and tracked, sanitised receipts. Authorisation is never evaluated here (a file at the authorisation
path means nothing); the separate read-only launch gate
(scripts/native_generated_tests_launch_gate.py) reports pipeline_ready, gpu_authorized and
launch_ready as distinct states and recomputes every hash in this receipt's ``inputs``.

The receipt is written once: an existing receipt is never overwritten or re-timestamped, so an
authorisation that names its hash stays valid. Earlier preflights are unchanged.
Launches nothing.

    python scripts/native_generated_tests_preflight_v2_4.py [--out PATH] [--suite PATH]
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

OUTPUT = "results/sft_root_cause_native_generated_tests_preflight_v2_4.json"
AUTHORIZATION = "results/sft_root_cause_native_gpu_authorization_v2.json"
SOURCE_MANIFEST = "results/sft_root_cause_phase4_receiver_capture_manifest_v2.json"
MANIFEST = "results/sft_root_cause_native_v24_rehearsal_manifest_v7.json"
JOB = "results/sft_root_cause_native_v24_rehearsal_job_v5.json"
V23_MANIFEST = "results/sft_root_cause_native_v23_rehearsal_manifest_v6.json"
V23_JOB = "results/sft_root_cause_native_v23_rehearsal_job_v4.json"
V22_MANIFEST = "results/sft_root_cause_native_v22_rehearsal_manifest_v5.json"
V22_JOB = "results/sft_root_cause_native_v22_rehearsal_job_v3.json"
PREP = "results/sft_root_cause/native_v21_rehearsal/records.jsonl"
EXCLUSIVE_KEY = "native_v24_generation"
PROMPTS = "results/sft_root_cause_native_v22_prompt_records.json"
ISOLATION = "results/sft_root_cause_native_v21_isolation_v6.json"
SUITE = "results/sft_root_cause/native_v24_full_suite.json"
CANARY_DIR = "results/sft_root_cause/native_v24_canaries"
GENERATIONS = "results/sft_root_cause/native_v24_generations"
LEDGER = "results/sft_root_cause_native_v24_quarantine_ledger.json"
TRACKED_RECEIPTS = (SUITE, f"{CANARY_DIR}/pipeline_receipt.json",
                    f"{CANARY_DIR}/canary_receipt_v2.json",
                    f"{CANARY_DIR}/atheris_canary_receipt_v3.json", LEDGER)
CONDITION = "primary_whole_module"
EXPECTED_COUNTS = {"qualified": 24, "generation": 23, "pre_generation_excluded": 1}
EXPECTED_CANDIDATES = 1104
CLIS = ("scripts/native_generated_tests_generate.py", "scripts/native_generated_tests_execute_wsl.py",
        "scripts/native_generated_tests_atheris_wsl.py", "scripts/native_generated_tests_analyse.py",
        "scripts/native_generated_tests_launch_gate.py", "scripts/native_rehearsal_rebuild_v22.py",
        "scripts/native_rehearsal_rebuild_v23.py", "scripts/native_rehearsal_rebuild_v24.py",
        "scripts/gpu_run.py", "scripts/native_quarantine_ledger_v24.py")
FOCUSED_TESTS = ("tests/test_native_generated_generate.py", "tests/test_native_generated_execute.py",
                 "tests/test_native_generated_analyse.py", "tests/test_native_generated_prompt.py",
                 "tests/test_native_generated_prompt_v2.py",
                 "tests/test_native_generated_leakage_v2.py",
                 "tests/test_native_generated_atheris.py", "tests/test_native_launch_gate.py",
                 "tests/test_native_v23_cohort_pipeline.py", "tests/test_gpu_run_exclusive.py",
                 "tests/test_gpu_run.py", "tests/test_native_historical_scripts.py",
                 "tests/test_native_v24_conformance.py", "tests/test_gpu_run_atomic_lock.py",
                 "tests/test_full_suite_receipt_lf.py")
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


def _rebuild(script: str) -> dict:
    done = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True,
                          text=True, timeout=1800)
    try:
        payload = json.loads(done.stdout[done.stdout.index("{"):])
    except ValueError:
        payload = {}
    return {"rc": done.returncode, "status": payload.get("status") or done.stderr[-300:]}


def main(argv=None) -> int:
    from harness import native_launch_gate as gate
    from harness.acquisition_receipt import ProtectedAccessMonitor
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=OUTPUT)
    parser.add_argument("--suite", default=SUITE, help="full-suite receipt for this source")
    args = parser.parse_args(argv)
    out_rel, suite_rel = args.out, args.suite
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
    suite = load(suite_rel) if (ROOT / suite_rel).exists() else {}
    relation = gate.receipt_only_descendant(ROOT, suite.get("source_commit", ""), head) \
        if suite.get("source_commit") else {"ok": False}
    require("full_suite_green_alone_at_this_source",
            suite.get("exit") == 0 and suite.get("failed") == 0 and suite.get("passed", 0) > 0
            and suite.get("tree_clean_at_start") is True and relation["ok"]
            and suite.get("executable_tree_sha256") == identity["executable_tree_sha256"],
            {k: suite.get(k) for k in ("source_commit", "passed", "failed", "skipped",
                                       "tree_clean_at_start")})
    # 3. CLIs and focused tests (unit, scanner, lifecycle, 24/23 cohort pipeline)
    helps = {}
    for script in CLIS:
        done = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=180)
        helps[script] = done.returncode == 0 and "usage" in done.stdout.lower()
    require("cli_help", all(helps.values()), {k: v for k, v in helps.items() if not v} or None)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            *FOCUSED_TESTS], cwd=ROOT, capture_output=True, text=True,
                           timeout=1800)
    require("focused_unit_scanner_lifecycle_and_cohort_tests", tests.returncode == 0,
            tests.stdout.strip().splitlines()[-1:] or None)
    # 4. contract, telemetry and protocols
    from scripts.native_generated_tests_generate import (CONTRACT, GENERATOR_VERSION,
                                                         adapter_manifest, adapter_sha256,
                                                         model_identity)
    from scripts.native_generation_io import TELEMETRY_SCHEMA, resolve_cohort
    require("generation_contract_complete_equality", CONTRACT == FROZEN_CONTRACT)
    from scripts.native_atheris_results import DESIGN_VERSION as ATHERIS_DESIGN
    from scripts.native_generation_io import resolve_prep
    from scripts.receipt_sanitize import check_file
    require("generator_and_telemetry_versions",
            GENERATOR_VERSION == "oneiros_native_generated_tests_generate_v4"
            and TELEMETRY_SCHEMA == "oneiros_native_generation_telemetry_v2"
            and ATHERIS_DESIGN == "oneiros_native_generated_tests_atheris_v4")
    require("protocol_and_amendments_present",
            len(identity["protocol_sha256"]) == len(gate.PROTOCOL_FILES) == 5)
    tracked = {rel: bool(git("ls-files", rel)) for rel in TRACKED_RECEIPTS}
    require("reproducibility_receipts_tracked_and_pushed", all(tracked.values())
            and not git("status", "--porcelain", "--", *TRACKED_RECEIPTS), tracked)
    exposed = {rel: check_file(ROOT / rel) for rel in TRACKED_RECEIPTS if (ROOT / rel).is_file()}
    require("tracked_receipts_contain_no_paths_secrets_or_protected_data",
            len(exposed) == len(TRACKED_RECEIPTS) and not any(exposed.values()),
            {k: v for k, v in exposed.items() if v} or None)
    # 5. every gate receipt and the quarantine ledger, validated by the SAME validators the
    #    analysis applies (amendment v2.4 G); the preflight binds them as gate_evidence
    from scripts import native_generated_tests_analyse as an
    gate_receipts = {"full_suite": suite_rel,
                     "synthetic_pipeline": f"{CANARY_DIR}/pipeline_receipt.json",
                     "sandbox_canaries": f"{CANARY_DIR}/canary_receipt_v2.json",
                     "atheris_canaries": f"{CANARY_DIR}/atheris_canary_receipt_v3.json"}
    gate_evidence = {
        "receipts": {k: {"path": rel, "sha256": sha(rel)} for k, rel in gate_receipts.items()
                     if (ROOT / rel).is_file()},
        "quarantine_ledger": ({"path": LEDGER, "sha256": sha(LEDGER)}
                              if (ROOT / LEDGER).is_file() else None)}
    for kind in gate_receipts:
        found = an.receipt_problems(kind, gate_evidence["receipts"].get(kind), ROOT)
        require(f"{kind}_receipt_valid_for_current_source", not found, found or None)
    found = an.ledger_problems(gate_evidence["quarantine_ledger"], ROOT, {})
    require("quarantine_ledger_complete_and_bound", not found, found or None)
    # 6. prompts, v2.2 job and v2.3 successors reproduce byte for byte
    for script in ("scripts/native_rehearsal_rebuild_v22.py", "scripts/native_rehearsal_rebuild_v23.py",
                   "scripts/native_rehearsal_rebuild_v24.py"):
        rebuilt = _rebuild(script)
        require(f"reproduces_exactly:{Path(script).stem}", rebuilt["rc"] == 0 and isinstance(
            rebuilt["status"], dict) and set(rebuilt["status"].values()) == {"verified_reproduction"},
            rebuilt["status"])
    present = all((ROOT / p).is_file() for p in (MANIFEST, JOB, V23_MANIFEST, V23_JOB,
                                                   V22_MANIFEST, V22_JOB, PROMPTS, PREP))
    require("artifacts_present", present)
    cohort, manifest, job = {}, {}, {}
    if present:
        from harness.native_generated_test_leakage import SCANNER_VERSION
        from harness.native_generated_test_prompt import BUILDER_VERSION
        manifest, v5, prompts = load(MANIFEST), load(V22_MANIFEST), load(PROMPTS)
        job = load(JOB)[CONDITION]
        try:
            cohort = resolve_cohort(ROOT / JOB, ROOT / MANIFEST, CONDITION)
        except SystemExit as exc:
            require("generation_cohort_resolves", False, str(exc))
        if cohort:
            require("generation_cohort_resolves", True)
            require("cohort_24_qualified_23_generated_1_excluded",
                    manifest["counts"] == EXPECTED_COUNTS
                    and len(cohort["qualified"]) == 24 and len(cohort["generation"]) == 23
                    and [e["reasons"] for e in cohort["pre_generation_exclusions"]]
                    == [["sequence_overflow"]]
                    and cohort["expected"]["candidates_total"] == EXPECTED_CANDIDATES,
                    manifest["counts"])
            require("coverage_rule", len(cohort["generation"]) >= 20
                    and len(cohort["generation"]) >= 0.9 * len(cohort["qualified"]))
        require("successor_bound_to_frozen_v23_and_v22",
                manifest["supersedes"]["manifest_sha256"] == sha(V23_MANIFEST)
                and load(JOB)["reused_from"]["file_sha256"] == sha(V23_JOB)
                and load(V23_JOB)["reused_from"]["file_sha256"] == sha(V22_JOB)
                and load(V22_JOB)[CONDITION]["job_sha256"] == job.get("job_sha256")
                and v5["job"]["sha256"] == sha(V22_JOB))
        try:
            prepared = resolve_prep(ROOT / PREP, ROOT / MANIFEST, ROOT)
            require("exact_preparation_file", prepared["path"] == PREP and len(prepared["rows"]) == 24,
                    {"path": prepared["path"], "sha256": prepared["sha256"]})
        except SystemExit as exc:
            require("exact_preparation_file", False, str(exc))
        require("engineering_only_study_mode",
                manifest.get("study_mode") == "engineering_dress_rehearsal"
                and not any(manifest.get("claims", {}).values())
                and "confirmation_authorization" not in manifest)
        require("coverage_gate_declared", manifest.get("coverage_gate", {}) .get("required_eligible") == 22
                and manifest["coverage_gate"].get("min_repositories") == 5)
        from scripts import native_rehearsal_rebuild_v24 as v24
        require("atomic_sequential_launch_contract",
                manifest.get("launch_contract") == v24.LAUNCH_CONTRACT
                and v24.LAUNCH_CONTRACT["exclusive_key"] == EXCLUSIVE_KEY)
        require("live_view_policy_and_analysis_contract",
                manifest.get("live_view_policy") == v24.LIVE_VIEW_POLICY
                and manifest.get("engineering_gate_requirements")
                == v24.ENGINEERING_GATE_REQUIREMENTS
                and manifest.get("analysis_contract") == v24.ANALYSIS_CONTRACT
                and manifest.get("telemetry") == {"schema": TELEMETRY_SCHEMA,
                                                  "generator_version": GENERATOR_VERSION})
        require("generation_output_layout", manifest["generation_outputs"] == {
            "base": f"{GENERATIONS}/base", "sft": f"{GENERATIONS}/sft"})
        require("source_manifest_and_target_set",
                manifest["source_manifest"]["sha256"] == sha(SOURCE_MANIFEST)
                and set(manifest["qualified_targets"]) <= {t["key"] for t in load(SOURCE_MANIFEST)["targets"]})
        require("protocol_hashes_exact", manifest["protocols"] == {p: sha(p) for p in gate.PROTOCOL_FILES})
        iso = load(ISOLATION)
        require("isolation_bound_to_current_implementation",
                iso["isolation_source_sha256"] == canonical_sha256(ROOT / "harness/repository_isolation.py")
                and manifest["isolation"]["receipt_sha256"] == sha(ISOLATION)
                and manifest["isolation"]["isolation_version"] == "oneiros_repository_isolation_v6")
        require("every_qualified_target_v6_and_formally_requalified", all(
            p["isolation_v6_admissible"] and p["requalification"] == "requalified"
            for p in manifest["per_target"] if p["kept"]) and sorted(
            p["key"] for p in manifest["per_target"] if p["kept"]) == sorted(manifest["qualified_targets"]))
        require("rehearsal_rule", manifest["rehearsal_rule"]["passed"])
        admitted = [r for r in prompts["rows"] if r.get("admitted")]
        require("admitted_prompts_fit_clean_untruncated",
                sorted(r["target_key"] for r in admitted) == sorted(i["target_key"] for i in job["items"])
                and all(r["v22_prompt_tokens"] <= FROZEN_CONTRACT["prompt_token_limit"]
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
    inputs = (MANIFEST, JOB, V23_MANIFEST, V23_JOB, V22_MANIFEST, V22_JOB, PROMPTS, ISOLATION,
              PREP, suite_rel, LEDGER,
              *(f"{CANARY_DIR}/{n}" for n in ("pipeline_receipt.json", "canary_receipt_v2.json",
                                              "atheris_canary_receipt_v3.json")))
    receipt = {
        "schema_version": gate.PREFLIGHT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_ready": pipeline_ready, "blockers": blockers, "checks": checks,
        "authorization": "NOT EVALUATED HERE: the separate read-only launch gate reports "
                         "pipeline_ready, gpu_authorized and launch_ready and revalidates "
                         "every input below",
        "source": {"commit": head, **identity},
        "fetch": {"rc": fetch["rc"], "remote_sha": fetch["remote_sha"]},
        "job": {"path": JOB, "file_sha256": sha(JOB) if (ROOT / JOB).exists() else None,
                "job_sha256": job.get("job_sha256"), "items": items,
                "targets": sorted(i["target_key"] for i in job.get("items", []))},
        "cohort": {"qualified": cohort.get("qualified", []),
                   "generation": cohort.get("generation", []),
                   "pre_generation_exclusions": cohort.get("pre_generation_exclusions", []),
                   "expected": cohort.get("expected", {})},
        "expected_candidates": candidates,
        "telemetry_schema": TELEMETRY_SCHEMA, "generator_version": GENERATOR_VERSION,
        "atheris_design_version": ATHERIS_DESIGN,
        "study_mode": "engineering_dress_rehearsal",
        "coverage_gate": manifest.get("coverage_gate"),
        "launch_contract": manifest.get("launch_contract"),
        "preparation": {"path": PREP, "sha256": sha(PREP) if (ROOT / PREP).is_file() else None},
        "generation_outputs": {"base": f"{GENERATIONS}/base", "sft": f"{GENERATIONS}/sft"},
        "manifest": {"path": MANIFEST, "sha256": sha(MANIFEST) if manifest else None},
        "prompt_records": {"path": PROMPTS, "sha256": sha(PROMPTS) if present else None},
        "model": model, "adapter_manifest_sha256": adapter_sha256(),
        "generation_contract": FROZEN_CONTRACT,
        "inputs": {rel: sha(rel) for rel in inputs if (ROOT / rel).exists()},
        "gate_evidence": gate_evidence,
        "live_view_policy": manifest.get("live_view_policy"),
        "engineering_gate_requirements": manifest.get("engineering_gate_requirements"),
        "analysis_contract": manifest.get("analysis_contract"),
        "proposed_analysis_NOT_EXECUTED": (
            f".venv-gpu/Scripts/python.exe scripts/native_generated_tests_analyse.py analyse "
            f"--manifest {MANIFEST} --job {JOB} --prep {PREP} --preflight {out_rel} "
            f"--execution-contract results/sft_root_cause/native_v24_execution/"
            f"execute_contract_{CONDITION}.json --results results/sft_root_cause/native_v24_execution/"
            f"results_{CONDITION}.jsonl --generations {GENERATIONS} --condition {CONDITION} "
            f"--study-mode engineering_dress_rehearsal --out results/sft_root_cause/"
            f"native_v24_analysis.json"),
        "proposed_gpu_commands_NOT_EXECUTED": [
            f".venv-gpu/Scripts/python.exe scripts/gpu_run.py start --name native_v24_generate_{arm} "
            f"--exclusive-key {EXCLUSIVE_KEY} "
            f"-- .venv-gpu/Scripts/python.exe scripts/native_generated_tests_generate.py run "
            f"--job {JOB} --condition {CONDITION} --arm {arm} "
            f"--out {GENERATIONS}/{arm} --backend hf --preflight {out_rel} "
            f"--authorization {AUTHORIZATION}" for arm in ("base", "sft")],
        "proposed_launch_gate_check_NOT_EXECUTED": [
            f".venv-gpu/Scripts/python.exe scripts/native_generated_tests_launch_gate.py "
            f"--preflight {out_rel} --authorization {AUTHORIZATION} --job {JOB} --arm {arm} "
            f"--out {GENERATIONS}/{arm}" for arm in ("base", "sft")],
        "proposed_base_completion_verification_NOT_EXECUTED": (
            f".venv-gpu/Scripts/python.exe scripts/native_generated_tests_launch_gate.py "
            f"--preflight {out_rel} --authorization {AUTHORIZATION} --job {JOB} --arm sft "
            f"--out {GENERATIONS}/sft   (reports base_arm_verification; SFT launches only when "
            f"launch_ready=true)"),
        "proposed_execution_NOT_EXECUTED": (
            "wsl -u root -- bash scripts/wsl_native_python.sh scripts/native_generated_tests_execute_wsl.py "
            f"run --prep {PREP} --manifest {MANIFEST} --job {JOB} --generations "
            f"{GENERATIONS} --condition {CONDITION} --out results/sft_root_cause/native_v24_execution"),
        "estimates_not_executed": {
            "candidates": candidates,
            "gpu_hours_range": [round(candidates * 200 / (2 * 60) / 3600, 2),
                                round(candidates * 1024 / (2 * 60) / 3600, 2)],
            "basis": "strictly sequential arms (exclusive key), 552 candidates each; batch 2 at ~60 generated tokens/s "
                     "per sequence; 200 (typical) to 1024 (maximum) new tokens per candidate; "
                     "model load ~1 min per arm; telemetry adds only CUDA synchronisation",
            "gpu_memory_gib": "~4-6 PyTorch allocated (1.5B bf16 weights ~3.1 GB + LoRA + KV "
                              "cache ~0.2 GB for 2 x 3072 tokens) on the 24 GB RTX 4500; row-scoped "
                              "and process-lifetime peaks are recorded separately",
            "storage_mb": "<20 for both arms (raw output + extracted module + telemetry per "
                          "candidate, at most ~1024 tokens each); sandbox execution of 1,104 "
                          "candidates ~1-3 CPU-hours in WSL"},
        "confirmation_acquisition": "NOT AUTHORISED",
        "root_cause_established": False, "generalization_established": False,
        "sft_benefit_established": False, "atheris_superiority_established": False,
        "gpu_generation_launched": False,
        "protected_access_audit": {k: v for k, v in evidence.items() if k != "opens_checked"},
    }
    publish_file_atomically(ROOT / out_rel, (json.dumps(receipt, indent=1, sort_keys=True)
                                             + "\n").encode("utf-8"))
    print(json.dumps({"pipeline_ready": pipeline_ready, "blockers": blockers,
                      "source_commit": head, "items": items,
                      "expected_candidates": candidates}, indent=1))
    return 0 if pipeline_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
