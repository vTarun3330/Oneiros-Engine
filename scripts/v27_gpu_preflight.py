"""v2.7 Phase 8: CPU-only GPU-evaluation PREFLIGHT for base vs historical SFT A@431 vs historical
relearning on the 29 native-executable targets of the frozen confirmation panel.

Loads NO model weights (model identity hashes the local snapshot files; the tokenizer is loaded
on CPU), launches nothing, writes no authorisation. The receipt uses the launch gate's preflight
schema (harness/native_launch_gate.py: oneiros_native_generated_tests_preflight_v2_5) so the
existing gate can later verify it, plus the v2.7 study fields. Every check records evidence.

Arms are resolved ONLY from accepted historical receipts and refused on any ambiguity or hash
mismatch:
- base:    Qwen/Qwen2.5-Coder-1.5B-Instruct @ the generator's pinned immutable revision;
- sft:     A@431 = results/v4_2_development_selection_receipt.json arms["A@431"] (adapter path
           and SHA-256), which must equal the generator's frozen ``sft_adapter``;
- relearn: results/v4_2_relearning_receipt.json ``adapter`` with its SHA-256 bound by
           results/v4_2_failure_taxonomy_relearning.json. The native generator supports only
           the arms ("base", "sft"); the relearning arm is therefore reported BLOCKED until a
           reviewed generator extension exists (no source change is made here).

    python scripts/v27_gpu_preflight.py --out results/<preflight>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PANEL = "results/sft_root_cause_v27_confirmation_panel_r4.json"
SUBSET = "results/sft_root_cause_v27_common_subset_r4_v2.json"
BUNDLE = "results/sft_root_cause_v27_phase6_evidence_bundle_v2.json"
JOB = "results/sft_root_cause_v27_generation_job_r4.json"
MANIFEST = "results/sft_root_cause_v27_generation_manifest_r4.json"
STAGE2 = "results/sft_root_cause_v27_phase6b_stage2_receipt.json"
SELECTION = "results/v4_2_development_selection_receipt.json"
RELEARN = "results/v4_2_relearning_receipt.json"
RELEARN_HASH = "results/v4_2_failure_taxonomy_relearning.json"
EXEC_CANARY = "results/sft_root_cause/v27_confirmation/phase6/exec_canaries/canary_receipt_v2.json"
VISIBLE = "results/sft_root_cause/v27_confirmation/r4_panel/model_visible_bundle.json"
OUT_ROOT = "results/sft_root_cause/v27_generations_r4"
EXEC_OUT = "results/sft_root_cause/v27_execution_r4"
CONDITION = "primary_whole_module"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PROTECTED_HINTS = ("sealed", "locked_validation", "final_test")


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rel_of(path_text: str) -> str:
    p = Path(str(path_text).replace("\\", "/"))
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


class Checks:
    def __init__(self):
        self.items = {}

    def add(self, name: str, ok: bool, evidence) -> bool:
        self.items[name] = {"pass": bool(ok), "evidence": evidence}
        return bool(ok)

    @property
    def failed(self):
        return [k for k, v in self.items.items() if not v["pass"]]


def resolve_arms(checks: Checks, gen) -> dict:
    sel = json.loads((ROOT / SELECTION).read_text(encoding="utf-8"))
    a431 = sel["arms"]["A@431"]
    chosen = sel["decision"]["selected_development_sft_candidate"]
    a_path = a431["adapter_path"]
    a_dir = str(Path(a_path).parent.as_posix())
    a_actual = sha(a_path) if (ROOT / a_path).is_file() else None
    cfg = json.loads((ROOT / a_dir / "adapter_config.json").read_text(encoding="utf-8"))
    checks.add("sft_A431_resolved_unambiguously",
               chosen == "Arm A checkpoint 431" and a_actual == a431["adapter_sha256"]
               and a_dir == gen.CONTRACT["sft_adapter"]
               and cfg.get("base_model_name_or_path") == gen.CONTRACT["base_model"],
               {"receipt": SELECTION, "decision": chosen, "adapter_path": a_path,
                "receipt_sha256": a431["adapter_sha256"], "file_sha256": a_actual,
                "generator_contract_sft_adapter": gen.CONTRACT["sft_adapter"],
                "adapter_base_model": cfg.get("base_model_name_or_path"),
                "adapter_config_revision": cfg.get("revision"),
                "lora": {k: cfg.get(k) for k in ("r", "lora_alpha", "target_modules")}})
    rl = json.loads((ROOT / RELEARN).read_text(encoding="utf-8"))
    r_dir = rel_of(rl["adapter"])
    r_file = f"{r_dir}/adapter_model.safetensors"
    tax = json.loads((ROOT / RELEARN_HASH).read_text(encoding="utf-8"))
    bound = sorted({a.get("adapter_sha256") for a in tax.get("artifacts", [])
                    if a.get("adapter_sha256")})
    r_actual = sha(r_file) if (ROOT / r_file).is_file() else None
    rcfg = json.loads((ROOT / r_dir / "adapter_config.json").read_text(encoding="utf-8"))
    others = sorted(p.parent.relative_to(ROOT).as_posix() for p in
                    (ROOT / r_dir).parents[1].rglob("adapter_model.safetensors"))
    checks.add("relearn_resolved_unambiguously",
               len(bound) == 1 and r_actual == bound[0]
               and rcfg.get("base_model_name_or_path") == gen.CONTRACT["base_model"],
               {"receipt": RELEARN, "adapter_dir": r_dir, "hash_receipt": RELEARN_HASH,
                "receipt_sha256": bound, "file_sha256": r_actual,
                "selection_rule": rl["checkpoint_selection"]["rule"],
                "selected_step": rl["checkpoint_selection"]["selected_checkpoint_step"],
                "other_adapter_copies_in_run": [o for o in others if o != r_dir],
                "note": "the receipt names one explicit checkpoint; other copies are ignored",
                "adapter_base_model": rcfg.get("base_model_name_or_path")})
    return {"base": {"model": gen.CONTRACT["base_model"],
                     "revision": gen.CONTRACT["base_revision"], "adapter": None},
            "sft": {"label": "A@431", "adapter_dir": a_dir, "adapter_sha256": a_actual,
                    "receipt": SELECTION},
            "relearn": {"label": "relearning checkpoint-141", "adapter_dir": r_dir,
                        "adapter_sha256": r_actual, "receipt": RELEARN,
                        "hash_receipt": RELEARN_HASH}}


def mock_rehearsal(checks: Checks, cohort) -> dict:
    from scripts.native_generation_io import verify_single_arm
    tmp = Path(tempfile.mkdtemp(prefix="oneiros_v27_mock_"))
    try:
        done = subprocess.run([sys.executable, "scripts/native_generated_tests_generate.py",
                               "run", "--job", JOB, "--condition", CONDITION, "--arm", "base",
                               "--out", str(tmp / "base"), "--backend", "mock"],
                              cwd=ROOT, capture_output=True, text=True, timeout=1800)
        ok = done.returncode == 0
        verified = None
        if ok:
            verified = verify_single_arm(tmp / "base", "base", cohort)
            ok = (verified["rows"], verified["candidates"]) == (
                cohort["expected"]["rows_per_arm"], cohort["expected"]["candidates_per_arm"])
        checks.add("mock_generation_rehearsal_verified", ok,
                   {"returncode": done.returncode, "stderr_tail": done.stderr[-300:],
                    "rows": (verified or {}).get("rows"),
                    "candidates": (verified or {}).get("candidates"),
                    "expected": cohort["expected"], "backend": "mock (no model)"})
        return verified or {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def gpu_state() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                        "utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    name, total, used, util = [x.strip() for x in q.stdout.strip().split(",")]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
            "utilization_pct": int(util),
            "python_compute_processes": [l for l in apps.splitlines() if "python" in l.lower()]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists (preflights are written once)")
    from harness.acquisition_receipt import ProtectedAccessMonitor
    ProtectedAccessMonitor.install(ROOT)
    scope = ProtectedAccessMonitor.scope()
    from harness import native_launch_gate as gate
    from harness import native_generated_test_job_v25 as jobs
    from scripts import native_generated_tests_generate as gen
    from scripts import native_generated_tests_execute_wsl as ex
    from scripts.native_generation_io import resolve_cohort
    checks = Checks()

    # --- repository identity ---------------------------------------------------------------
    state = gate.checkout_state(ROOT)
    identity = gate.source_identity(ROOT)
    checks.add("checkout_clean_and_synced", not gate.checkout_problems(state),
               {"head": state["head"], "remote_sha": state["fetch"].get("remote_sha"),
                "problems": gate.checkout_problems(state),
                "review_materials": "locally excluded, present and untouched"})

    # --- cohort and panel binding -----------------------------------------------------------
    cohort = resolve_cohort(ROOT / JOB, ROOT / MANIFEST, CONDITION)
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    subset = json.loads((ROOT / SUBSET).read_text(encoding="utf-8"))
    panel = json.loads((ROOT / PANEL).read_text(encoding="utf-8"))
    bundle = {t["target_id"]: t for t in
              json.loads((ROOT / BUNDLE).read_text(encoding="utf-8"))["targets"]}
    ids = cohort["generation"]
    ids_sha = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    checks.add("cohort_is_the_29_native_executable_targets",
               len(ids) == 29 and ids_sha == subset["native_executable_ids_sha256"]
               and sorted(ids) == subset["sets"]["native_executable"],
               {"targets": len(ids), "ids_sha256": ids_sha})
    checks.add("cohort_bound_to_frozen_34_target_panel",
               manifest["panel"]["sha256"] == sha(PANEL) and len(panel["targets"]) == 34
               and set(ids) <= {t["target_id"] for t in panel["targets"]},
               {"panel_sha256": sha(PANEL), "panel_status": panel["status"]})
    reach = {k: bundle[k]["native_reach"]["reach"] for k in ids}
    checks.add("native_target_reach_bound_for_every_target",
               all(v == "reach_verified" for v in reach.values()),
               {"reach_verified": sum(v == "reach_verified" for v in reach.values()),
                "bundle_sha256": sha(BUNDLE)})
    excl = manifest["panel_policy_exclusions"]
    checks.add("five_policy_exclusions_reported_separately",
               len(excl) == 5 and not set(e["target_key"] for e in excl) & set(ids),
               excl)
    checks.add("atheris_not_used_for_selection",
               manifest.get("atheris_independent") is True
               and sorted(ids) != sorted(subset["sets"]["fuzzable"]),
               {"cohort_definition": "native_executable (reach AND sandbox control); the "
                                     "Atheris fuzzable subset is a separate descriptive view"})

    # --- job, prompts, leakage, budgets ----------------------------------------------------------
    job_file = json.loads((ROOT / JOB).read_text(encoding="utf-8"))
    job = jobs.validate_job_file(job_file, CONDITION, ROOT)
    visible = json.loads((ROOT / VISIBLE).read_text(encoding="utf-8"))
    same_prompts = all(i["prompt_sha256"] == visible[i["target_key"]]["prompt_sha256"]
                       for i in job["items"])
    tokens = [i["prompt_tokens"] for i in job["items"]]
    checks.add("job_valid_prompts_frozen_no_leakage_no_truncation",
               same_prompts and not job["refused"] and max(tokens) <= jobs.PROMPT_TOKEN_LIMIT
               and max(tokens) + jobs.COMPLETION_TOKENS <= jobs.SEQUENCE_LIMIT,
               {"items": len(job["items"]), "refused": job["refused"],
                "prompt_tokens_max": max(tokens), "prompt_tokens_min": min(tokens),
                "limits": job["sequence_fit"], "builder": job_file["builder"],
                "prompts_equal_frozen_visible_bundle": same_prompts,
                "leakage_scan": "harness.native_generated_test_leakage via build_item "
                                "(verifier-only material); zero refusals"})

    # --- model, tokenizer, adapters ------------------------------------------------------------
    model = gen.model_identity()
    checks.add("base_model_revision_immutable_and_local",
               bool(HEX40.match(gen.CONTRACT["base_revision"])),
               {"base_model": gen.CONTRACT["base_model"],
                "revision": gen.CONTRACT["base_revision"], **model})
    arms = resolve_arms(checks, gen)
    adapter_manifest = gen.adapter_sha256()

    # --- generation and evaluation contracts -------------------------------------------------
    c = gen.CONTRACT
    checks.add("identical_generation_contract_for_every_arm",
               c["candidates"] == 8 and c["seeds"] == [42, 43, 44] and c["reranking"] == "none"
               and c["raw_output_retained"] is True and c["duplicates"] == "kept"
               and c["extraction"] == "whole_output_single_fence_strip"
               and c["max_new_tokens"] == jobs.COMPLETION_TOKENS
               and c["prompt_token_limit"] == jobs.PROMPT_TOKEN_LIMIT,
               {"contract": c, "generator_version": gen.GENERATOR_VERSION,
                "note": "one frozen contract; the arm changes only the adapter"})
    exec_canary = json.loads((ROOT / EXEC_CANARY).read_text(encoding="utf-8"))
    checks.add("identical_evaluator_sandbox_and_timeouts",
               exec_canary["passed"] is True
               and exec_canary["executor_sha256"] == gen.sha256_file(ROOT / "scripts/"
                                                                     "native_generated_tests_"
                                                                     "execute_wsl.py"),
               {"executor_design": ex.DESIGN_VERSION, "limits": ex.LIMITS,
                "module_limits": ex.MODULE_LIMITS,
                "executor_sha256": exec_canary["executor_sha256"],
                "inner_sha256": exec_canary["inner_sha256"],
                "sandbox_canaries": EXEC_CANARY})

    # --- outputs, resources ----------------------------------------------------------------------
    outputs = {"base": f"{OUT_ROOT}/base", "sft": f"{OUT_ROOT}/sft",
               "relearn": f"{OUT_ROOT}/relearn", "execution": EXEC_OUT}
    checks.add("output_paths_new_and_unique",
               not any((ROOT / p).exists() for p in outputs.values())
               and len(set(outputs.values())) == len(outputs), outputs)
    gpu = gpu_state()
    free_disk = shutil.disk_usage(ROOT).free
    checks.add("gpu_idle_and_resources_sufficient",
               gpu["utilization_pct"] < 10 and not gpu["python_compute_processes"]
               and gpu["memory_total_mib"] - gpu["memory_used_mib"] > 10_000
               and free_disk > 20 * 2 ** 30,
               {**gpu, "disk_free_gib": round(free_disk / 2 ** 30, 1)})

    # --- end-to-end CPU rehearsal (mock backend: no model) ---------------------------------------
    mock_rehearsal(checks, cohort)

    # --- protected data ------------------------------------------------------------------------------
    inputs = {p: sha(p) for p in (PANEL, SUBSET, BUNDLE, JOB, MANIFEST, STAGE2, SELECTION,
                                  RELEARN, RELEARN_HASH, EXEC_CANARY, VISIBLE,
                                  manifest["requalification_records"]["path"],
                                  f"{arms['sft']['adapter_dir']}/adapter_model.safetensors",
                                  f"{arms['relearn']['adapter_dir']}/adapter_model.safetensors")}
    evidence = scope.close()
    checks.add("no_validation_or_sealed_final_access",
               not evidence.get("protected_accesses")
               and not any(h in p for p in inputs for h in PROTECTED_HINTS),
               {"protected_accesses": evidence.get("protected_accesses"),
                "opens_checked": evidence.get("opens_checked"),
                "policy": evidence.get("protection_policy")})

    rows, cands = cohort["expected"]["rows_per_arm"], cohort["expected"]["candidates_per_arm"]
    py = ".venv-gpu/Scripts/python.exe"
    auth = "results/sft_root_cause_v27_gpu_authorization_r4.json"
    commands = {
        arm: (f"{py} scripts/gpu_run.py start --name v27_generate_{arm} --exclusive-key "
              f"v27_generation -- {py} scripts/native_generated_tests_generate.py run --job {JOB} "
              f"--condition {CONDITION} --arm {arm} --out {outputs[arm]} --backend hf "
              f"--preflight {args.out} --authorization {auth}")
        for arm in ("base", "sft")}
    ready = not checks.failed
    receipt = {
        "schema_version": gate.PREFLIGHT_SCHEMA,
        "study": "oneiros_v27_repository_native_confirmation_evaluation",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_ready": ready,
        "three_arm_study_ready": False,
        "launched": False, "weights_loaded": False, "authorization_written": False,
        "source": {"commit": state["head"],
                   "executable_tree_sha256": identity["executable_tree_sha256"],
                   "protocol_sha256": identity["protocol_sha256"]},
        "inputs": inputs,
        "job": {"path": JOB, "file_sha256": sha(JOB), "job_sha256": job["job_sha256"]},
        "manifest": {"path": MANIFEST, "sha256": sha(MANIFEST)},
        "generation_outputs": {"base": outputs["base"], "sft": outputs["sft"]},
        "model": model,
        "adapter_manifest_sha256": adapter_manifest,
        "arms": {**arms,
                 "relearn": {**arms["relearn"], "status": "BLOCKED",
                             "reason": "the native generator (scripts/native_generated_tests_"
                                       "generate.py) and the execution/IO pipeline accept only "
                                       "arms ('base', 'sft'), with 'sft' pinned to A@431; a "
                                       "relearning run needs a reviewed, tested generator "
                                       "extension (a third arm bound to this adapter hash), "
                                       "a new preflight and its own authorisation"}},
        "cohort": {"targets": len(ids), "ids_sha256": ids_sha, "ids": sorted(ids),
                   "rows_per_arm": rows, "candidates_per_arm": cands,
                   "panel_policy_exclusions": excl},
        "checks": checks.items, "failed_checks": checks.failed,
        "estimates": {
            "basis": "same generator and contract, v2.4 rehearsal: 552 candidates per arm took "
                     "1,878 s (base) and 5,838 s (adapter); scaled linearly to 696",
            "runtime_minutes": {"base": round(1878 * cands / 552 / 60),
                                "sft": round(5838 * cands / 552 / 60),
                                "relearn_if_unblocked": round(5838 * cands / 552 / 60)},
            "vram": "1.5B parameters in bf16 ~3.1 GiB plus KV cache for batch 2 x 3,072 tokens; "
                    "well under the 24 GiB card",
            "disk": "v2.4 generations for 2 arms used 2.5 MiB; expect < 10 MiB per arm plus "
                    "execution evidence < 200 MiB"},
        "commands": {"generate": commands,
                     "sequence": "base first; the gate admits sft only after the base arm "
                                 "passes the exact single-arm verifier",
                     "authorization_required": auth,
                     "execute_after_generation": (
                         "wsl.exe -u root --cd <repo> -- bash scripts/wsl_isolated.sh bash "
                         "scripts/wsl_native_python.sh scripts/native_generated_tests_execute_wsl"
                         f".py run --prep {manifest['requalification_records']['path']} "
                         f"--manifest {MANIFEST} --job {JOB} --base-generations "
                         f"{outputs['base']} --sft-generations {outputs['sft']} --condition "
                         f"{CONDITION} --out {EXEC_OUT}")},
        "analysis_plan": {
            "primary": "paired exploratory comparison on the SAME 29 targets: per-target Kill@8 "
                       "(a candidate fails on the buggy revision and passes on the fixed "
                       "revision), base vs A@431 (and base vs relearning when unblocked)",
            "statistics": "exact paired counts, two-sided exact McNemar, Wilson 95% intervals; "
                          "repository-level descriptive summary and a one-target-per-repository "
                          "sensitivity (14 repositories)",
            "secondary": "Kill@1/2/4, parse, execution and reference validity per requested/"
                         "parsed/executed candidate, function-level reference validity, "
                         "duplicate/diversity, failure taxonomy, complexity/defect-family slices",
            "claims": "no promotion or generalisation claim from a non-significant result; the "
                      "predeclared promotion gate (p < 0.01) is not expected to be reachable "
                      "(80% power unattainable at alpha 0.01 under the assumed discordance)",
            "excluded": "environment/harness failures are reported, never counted as model "
                        "failures; the 5 policy exclusions are reported separately",
            "atheris": "the 3-target fuzzable subset is a separate descriptive adapter-coverage "
                       "analysis and never selects or filters the model-evaluation cohort"},
    }
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(json.dumps({"pipeline_ready": ready, "three_arm_study_ready": False,
                      "failed_checks": checks.failed, "sha256": sha(args.out)}, indent=1))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
