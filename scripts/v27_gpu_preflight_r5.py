"""v2.7 r5: CPU-only, END-TO-END preflight for the three-arm repository-native evaluation
(base vs A@431 vs relearning checkpoint-141) on the 29 native-executable panel targets.

Supersedes the r4 two-arm preflight (preserved; never authorised). Loads NO model weights,
launches nothing, writes no authorisation. Schema oneiros_native_generated_tests_preflight_v2_7
(harness/native_launch_gate.py), authorisable only by oneiros_native_gpu_authorization_v4.

Binds: the frozen 29-target cohort and the original 34-target panel; the five policy
exclusions; exact prompts; the frozen arm registry with all three adapter identities (verified
on disk); the base snapshot, tokenizer and chat template; Python, PyTorch, Transformers, PEFT,
Tokenizers, Accelerate, CUDA runtime, driver and device; the generation contract; evaluator,
sandbox and timeouts; the analysis implementation and frozen plan; and ``gate_evidence``
(full-suite, synthetic-pipeline, sandbox-canary and Atheris-canary receipts plus the quarantine
ledger) exactly as the analyser requires. A no-model mock rehearsal generates all three arms on
the real job and loads them authoritatively (261 rows, 2,088 candidates).

Machine-bound inputs (git-ignored adapters and preparation records) are labelled and hashed
byte-exact; tracked JSON is hashed as committed (LF) bytes.

    python scripts/v27_gpu_preflight_r5.py --out results/<preflight>.json --full-suite R
        --synthetic R --sandbox-canaries R --atheris-canaries R --ledger R
        [--tag r5] [--supersedes results/<earlier preflight>.json]

``--tag`` names every output path and the authorisation file (default r5, the first use);
a successor uses a new tag so its outputs and authorisation never collide with an earlier
preflight's. ``--supersedes`` names the preflight this one replaces (default r4).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
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
SUBSET = "results/sft_root_cause_v27_common_subset_r4_v3.json"
BUNDLE = "results/sft_root_cause_v27_phase6_evidence_bundle_v3.json"
JOB = "results/sft_root_cause_v27_generation_job_r5.json"
MANIFEST = "results/sft_root_cause_v27_generation_manifest_r5.json"
R4_PREFLIGHT = "results/sft_root_cause_v27_gpu_preflight_r4.json"
VISIBLE = "results/sft_root_cause/v27_confirmation/r4_panel/model_visible_bundle.json"
ANALYSER = "scripts/native_generated_tests_analyse.py"
def tagged_paths(tag: str) -> dict:
    if not tag.isalnum():
        raise SystemExit("REFUSED: --tag must be alphanumeric")
    return {"OUT_ROOT": f"results/sft_root_cause/v27_generations_{tag}",
            "EXEC_OUT": f"results/sft_root_cause/v27_execution_{tag}",
            "ANALYSIS_OUT": f"results/sft_root_cause_v27_analysis_{tag}.json",
            "AUTH": f"results/sft_root_cause_v27_gpu_authorization_{tag}.json",
            "RUN_NAME_PREFIX": f"v27{tag}_generate_",
            "EXCLUSIVE_KEY": f"v27{tag}_generation"}
CONDITION = "primary_whole_module"
REPO_WSL = "/mnt/c/Users/Student2/Desktop/Capstone/oneiros"


def stored_commands(tag: str, preflight: str, arms, prep: str) -> dict:
    """The authoritative commands a preflight stores. Every tagged identifier (output,
    execution, analysis and authorisation paths, gpu_run run names and exclusive key) is
    derived from ``tag`` alone, so a successor preflight never carries a predecessor's."""
    from scripts import native_generated_tests_analyse as analysis

    t = tagged_paths(tag)
    py = ".venv-gpu/Scripts/python.exe"
    out = {arm: f"{t['OUT_ROOT']}/{arm}" for arm in arms}
    generate = {arm: (f"{py} scripts/gpu_run.py start --name {t['RUN_NAME_PREFIX']}{arm} "
                      f"--exclusive-key {t['EXCLUSIVE_KEY']} -- {py} "
                      f"scripts/native_generated_tests_generate.py run --job {JOB} --condition "
                      f"{CONDITION} --arm {arm} --out {out[arm]} --backend hf --preflight "
                      f"{preflight} --authorization {t['AUTH']}") for arm in arms}
    gates = {arm: (f"{py} scripts/native_generated_tests_launch_gate.py --preflight {preflight} "
                   f"--authorization {t['AUTH']} --job {JOB} --arm {arm} --out {out[arm]}")
             for arm in arms}
    execute = (f"wsl.exe -u root --cd {REPO_WSL} -- bash scripts/wsl_isolated.sh bash "
               f"scripts/wsl_native_python.sh scripts/native_generated_tests_execute_wsl.py run "
               f"--prep {prep} --manifest {MANIFEST} --job {JOB} --generations {t['OUT_ROOT']} "
               f"--condition {CONDITION} --out {t['EXEC_OUT']}")
    analyse = (f"{py} {ANALYSER} analyse --manifest {MANIFEST} --job {JOB} --prep "
               f"{prep} --preflight {preflight} "
               f"--execution-contract {t['EXEC_OUT']}/execute_contract_{CONDITION}.json --results "
               f"{t['EXEC_OUT']}/results_{CONDITION}.jsonl --generations {t['OUT_ROOT']} "
               f"--condition {CONDITION} --study-mode {analysis.EXPLORATORY} "
               f"--out {t['ANALYSIS_OUT']}")
    return {"generate": generate, "gates": gates, "execute": execute, "analyse": analyse}


TAG_TOKEN = re.compile(r"(?:v27|_)(r\d+)(?![0-9A-Za-z])")
FROZEN_INPUT_PREFIX = "results/sft_root_cause/v27_confirmation/"


def command_tag_problems(tag: str, cmds: dict, preflight: str) -> list:
    """Empty when every tagged identifier in the stored commands (generation, execution and
    analysis paths, authorisation, gpu_run run names and exclusive key) names ``tag``. The
    frozen job, manifest and preparation inputs, and the preflight's own path, are exempt."""
    t = tagged_paths(tag)
    frozen = {preflight, JOB, MANIFEST}
    texts = {**{f"generate.{a}": c for a, c in cmds["generate"].items()},
             **{f"gate.{a}": c for a, c in cmds["gates"].items()},
             "execute": cmds["execute"], "analyse": cmds["analyse"]}
    problems = [f"{where}: {token} is tagged {found}, not {tag}"
                for where, text in texts.items() for token in text.split()
                if token not in frozen and not token.startswith(FROZEN_INPUT_PREFIX)
                for found in TAG_TOKEN.findall(token) if found != tag]
    for arm, c in cmds["generate"].items():
        for need in (f"--name {t['RUN_NAME_PREFIX']}{arm} ",
                     f"--exclusive-key {t['EXCLUSIVE_KEY']} ",
                     f"--out {t['OUT_ROOT']}/{arm} ", f"--authorization {t['AUTH']}"):
            if need not in c:
                problems.append(f"generate.{arm}: missing {need.strip()}")
    return problems


HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.items = {}

    def add(self, name, ok, evidence):
        self.items[name] = {"pass": bool(ok), "evidence": evidence}

    @property
    def failed(self):
        return [k for k, v in self.items.items() if not v["pass"]]


def environment(samples: int = 5) -> dict:
    """Frozen software/hardware identity plus GPU load as the MEDIAN of several samples one
    second apart (a single instantaneous sample is spiked by desktop compositing: r5 attempt 1
    sampled 43% with no compute process on the GPU)."""
    import statistics
    import time
    from scripts.native_generated_tests_generate import library_versions
    libs = library_versions()
    readings = []
    for i in range(samples):
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
                                 "memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True).stdout.strip()
        readings.append([x.strip() for x in driver.split(",")])
        if i + 1 < samples:
            time.sleep(1)
    name, drv, total = readings[-1][:3]
    used = statistics.median(int(r[3]) for r in readings)
    util = statistics.median(int(r[4]) for r in readings)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    return {"python": platform.python_version(), "platform": platform.platform(),
            "libraries": libs, "cuda_runtime": libs.get("cuda"), "driver": drv,
            "device": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
            "utilization_pct": int(util), "utilization_samples_pct": [int(r[4]) for r in readings],
            "utilization_rule": f"median of {samples} samples, 1 s apart",
            "python_compute_processes": [l for l in apps.splitlines() if "python" in l.lower()]}


def mock_rehearsal(checks: Checks, cohort) -> None:
    """All three arms on the real job with the mock backend (no model), then the authoritative
    three-arm loader: exact rows/candidates, registry-bound identities."""
    from scripts import native_generation_io as gio
    tmp = Path(tempfile.mkdtemp(prefix="oneiros_v27r5_mock_"))
    try:
        codes = {}
        for arm in cohort["arms"]:
            done = subprocess.run([sys.executable, "scripts/native_generated_tests_generate.py",
                                   "run", "--job", JOB, "--condition", CONDITION, "--arm", arm,
                                   "--out", str(tmp / arm), "--backend", "mock"],
                                  cwd=ROOT, capture_output=True, text=True, timeout=1800)
            codes[arm] = {"returncode": done.returncode, "stderr_tail": done.stderr[-200:]}
        per_arm = {arm: gio.verify_single_arm(tmp / arm, arm, cohort) for arm in cohort["arms"]
                   if codes[arm]["returncode"] == 0}
        loaded = gio.load_arm_generations(gio.arm_dirs(tmp, {}, cohort["arms"], CONDITION),
                                          cohort) if len(per_arm) == 3 else {"rows": {}}
        rows = len(loaded["rows"])
        cands = sum(len(r["candidates"]) for r in loaded["rows"].values())
        exp = cohort["expected"]
        checks.add("mock_three_arm_generation_and_authoritative_loading",
                   all(c["returncode"] == 0 for c in codes.values())
                   and all((v["rows"], v["candidates"]) == (exp["rows_per_arm"],
                                                            exp["candidates_per_arm"])
                           for v in per_arm.values())
                   and (rows, cands) == (exp["rows_total"], exp["candidates_total"])
                   and all(v["identity"].get("arm_registry_sha256") == cohort["arm_registry"]["sha256"]
                           for v in per_arm.values()),
                   {"returncodes": codes, "per_arm": {a: [v["rows"], v["candidates"]]
                                                      for a, v in per_arm.items()},
                    "loaded_rows": rows, "loaded_candidates": cands, "expected": exp,
                    "adapter_manifest_sha256": {a: v["identity"].get("adapter_manifest_sha256")
                                                for a, v in per_arm.items()}})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r5")
    ap.add_argument("--supersedes", default=R4_PREFLIGHT)
    ap.add_argument("--supersedes-reason", default=None)
    for name in ("full-suite", "synthetic", "sandbox-canaries", "atheris-canaries", "ledger"):
        ap.add_argument(f"--{name}", required=True)
    args = ap.parse_args(argv)
    paths = tagged_paths(args.tag)
    OUT_ROOT, EXEC_OUT = paths["OUT_ROOT"], paths["EXEC_OUT"]
    ANALYSIS_OUT, AUTH = paths["ANALYSIS_OUT"], paths["AUTH"]
    supersedes = args.supersedes
    if not (ROOT / supersedes).is_file():
        raise SystemExit(f"REFUSED: superseded preflight {supersedes} missing")
    if supersedes != R4_PREFLIGHT and not args.supersedes_reason:
        raise SystemExit("REFUSED: --supersedes-reason is required for a non-r4 predecessor")
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists (preflights are written once)")
    from harness.acquisition_receipt import ProtectedAccessMonitor
    ProtectedAccessMonitor.install(ROOT)
    scope = ProtectedAccessMonitor.scope()
    from harness import native_arm_registry as reg
    from harness import native_launch_gate as gate
    from harness import native_generated_test_job_v25 as jobs
    from scripts import native_generated_tests_analyse as analysis
    from scripts import native_generated_tests_execute_wsl as ex
    from scripts import native_generated_tests_generate as gen
    from scripts.native_generation_io import resolve_cohort
    checks = Checks()

    state = gate.checkout_state(ROOT)
    identity = gate.source_identity(ROOT)
    checks.add("checkout_clean_and_synced", not gate.checkout_problems(state),
               {"head": state["head"], "remote_sha": state["fetch"].get("remote_sha"),
                "problems": gate.checkout_problems(state),
                "review_materials": "locally excluded; present and untouched"})

    cohort = resolve_cohort(ROOT / JOB, ROOT / MANIFEST, CONDITION)
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    subset = json.loads((ROOT / SUBSET).read_text(encoding="utf-8"))
    panel = json.loads((ROOT / PANEL).read_text(encoding="utf-8"))
    bundle = {t["target_id"]: t for t in json.loads((ROOT / BUNDLE).read_text("utf-8"))["targets"]}
    ids = cohort["generation"]
    ids_sha = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    checks.add("cohort_is_the_29_native_executable_targets",
               len(ids) == 29 and ids_sha == subset["native_executable_ids_sha256"],
               {"targets": len(ids), "ids_sha256": ids_sha, "subset_receipt": SUBSET})
    checks.add("cohort_bound_to_frozen_34_target_panel",
               manifest["panel"]["sha256"] == sha(PANEL) and len(panel["targets"]) == 34,
               {"panel_sha256": sha(PANEL), "status": panel["status"]})
    checks.add("native_target_reach_bound_for_every_target",
               all(bundle[k]["native_reach"]["reach"] == "reach_verified" for k in ids),
               {"bundle": BUNDLE, "bundle_sha256": sha(BUNDLE)})
    excl = manifest["panel_policy_exclusions"]
    checks.add("five_policy_exclusions_reported_separately",
               len(excl) == 5 and not {e["target_key"] for e in excl} & set(ids), excl)

    job_file = json.loads((ROOT / JOB).read_text(encoding="utf-8"))
    job = jobs.validate_job_file(job_file, CONDITION, ROOT)
    visible = json.loads((ROOT / VISIBLE).read_text(encoding="utf-8"))
    tokens = [i["prompt_tokens"] for i in job["items"]]
    checks.add("job_valid_prompts_frozen_no_leakage_no_truncation",
               not job["refused"] and all(i["prompt_sha256"] == visible[i["target_key"]]
                                          ["prompt_sha256"] for i in job["items"])
               and max(tokens) + jobs.COMPLETION_TOKENS <= jobs.SEQUENCE_LIMIT,
               {"items": len(job["items"]), "prompt_tokens_max": max(tokens),
                "limits": job["sequence_fit"], "builder": job_file["builder"]})

    registry = reg.summary()
    problems = reg.verify(ROOT)
    checks.add("arm_registry_frozen_bound_and_verified_on_disk",
               not problems and cohort["arm_registry"] == registry == job_file["arm_registry"]
               and list(cohort["arms"]) == registry["arms"],
               {"registry": registry, "problems": problems,
                "arms": {a: {k: reg.REGISTRY["arms"][a].get(k) for k in
                             ("label", "adapter_dir", "adapter_model_sha256",
                              "adapter_config_sha256", "adapter_manifest_sha256")}
                         for a in registry["arms"]}})
    adapters = {a: gen.adapter_sha256(a, registry) for a in registry["arms"]}
    checks.add("adapters_resolve_only_through_the_registry",
               adapters == registry["adapter_manifest_sha256"]
               and gen.arm_adapter_dir("sft", registry) == ROOT / gen.CONTRACT["sft_adapter"],
               adapters)

    model = gen.model_identity()
    checks.add("base_model_revision_immutable_and_local",
               bool(HEX40.match(gen.CONTRACT["base_revision"]))
               and gen.CONTRACT["base_revision"] == reg.BASE_REVISION,
               {"base_model": gen.CONTRACT["base_model"],
                "revision": gen.CONTRACT["base_revision"], **model})
    env = environment()
    free_disk = shutil.disk_usage(ROOT).free
    # Pipeline readiness gates the frozen identity and capacity only; GPU LOAD is a launch-time
    # condition (recorded here, re-checked immediately before each launch): desktop graphics
    # can load the GPU without any compute process (r5 attempt 1).
    checks.add("environment_frozen_and_capacity_sufficient",
               not env["python_compute_processes"]
               and env["memory_total_mib"] - env["memory_used_mib"] > 10_000
               and free_disk > 20 * 2 ** 30 and all(env["libraries"].get(k) for k in
                                                      ("torch", "transformers", "peft",
                                                       "tokenizers", "accelerate", "cuda")),
               {**env, "disk_free_gib": round(free_disk / 2 ** 30, 1),
                "gpu_load_policy": "recorded, not gated; re-check that no other compute or "
                                   "graphics workload is active immediately before launch"})

    c = gen.CONTRACT
    checks.add("identical_generation_contract_for_every_arm",
               c["candidates"] == 8 and c["seeds"] == [42, 43, 44] and c["reranking"] == "none"
               and c["raw_output_retained"] is True and c["duplicates"] == "kept"
               and c["extraction"] == "whole_output_single_fence_strip",
               {"contract": c, "generator_version": gen.GENERATOR_VERSION,
                "only_difference_between_arms": "the registered adapter"})

    receipts = {"full_suite": args.full_suite, "synthetic_pipeline": args.synthetic,
                "sandbox_canaries": args.sandbox_canaries,
                "atheris_canaries": args.atheris_canaries}
    gate_evidence = {"receipts": {k: {"path": v, "sha256": sha(v)} for k, v in receipts.items()},
                     "quarantine_ledger": {"path": args.ledger, "sha256": sha(args.ledger)}}
    evidence_problems = {k: analysis.receipt_problems(k, gate_evidence["receipts"][k], ROOT)
                         for k in receipts}
    evidence_problems["quarantine_ledger"] = analysis.ledger_problems(
        gate_evidence["quarantine_ledger"], ROOT, {})
    checks.add("gate_evidence_satisfies_the_analyser",
               not any(evidence_problems.values()),
               {"gate_evidence": gate_evidence, "problems": evidence_problems})
    exec_canary = json.loads((ROOT / args.sandbox_canaries).read_text(encoding="utf-8"))
    checks.add("identical_evaluator_sandbox_and_timeouts",
               exec_canary.get("passed") is True,
               {"executor_design": ex.DESIGN_VERSION, "limits": ex.LIMITS,
                "module_limits": ex.MODULE_LIMITS, "executor_sha256": sha(
                    "scripts/native_generated_tests_execute_wsl.py"),
                "inner_sha256": sha("scripts/native_sandbox_inner.sh")})
    checks.add("analysis_frozen_before_output",
               manifest["analysis_plan"]["sha256"] == analysis.analysis_plan_sha256()
               and manifest["study_mode"] == analysis.EXPLORATORY
               and cohort["study_mode"] == analysis.EXPLORATORY,
               {"plan": analysis.ANALYSIS_PLAN_V27,
                "plan_sha256": analysis.analysis_plan_sha256(),
                "implementation": ANALYSER, "implementation_sha256": sha(ANALYSER)})

    cmds = stored_commands(args.tag, args.out, registry["arms"],
                           manifest["requalification_records"]["path"])
    outputs = {**{a: f"{OUT_ROOT}/{a}" for a in registry["arms"]}, "execution": EXEC_OUT,
               "analysis": ANALYSIS_OUT}
    checks.add("output_paths_new_unique_and_not_r4",
               not any((ROOT / p).exists() for p in outputs.values())
               and len(set(outputs.values())) == len(outputs)
               and all(f"_{args.tag}" in p for p in outputs.values())
               and not (ROOT / AUTH).exists()
               and not command_tag_problems(args.tag, cmds, args.out),
               {**outputs, "authorization": AUTH, "run_name_prefix": paths["RUN_NAME_PREFIX"],
                "exclusive_key": paths["EXCLUSIVE_KEY"],
                "command_tag_problems": command_tag_problems(args.tag, cmds, args.out)})
    mock_rehearsal(checks, cohort)

    machine_bound = {manifest["requalification_records"]["path"]: "derived preparation records "
                     "(git-ignored)", VISIBLE: "frozen model-visible bundle (git-ignored)",
                     **{f"{reg.REGISTRY['arms'][a]['adapter_dir']}/adapter_model.safetensors":
                        f"{a} adapter weights (git-ignored)" for a in ("sft", "relearn")}}
    inputs = {p: sha(p) for p in (PANEL, SUBSET, BUNDLE, JOB, MANIFEST, supersedes, ANALYSER,
                                  *receipts.values(), args.ledger, *machine_bound)}
    for rec in (reg.REGISTRY["arms"]["sft"]["receipts"], reg.REGISTRY["arms"]["relearn"]["receipts"]):
        for r in rec.values():
            inputs[r["path"]] = sha(r["path"])
    evidence = scope.close()
    checks.add("no_validation_or_sealed_final_access", not evidence.get("protected_accesses"),
               {"protected_accesses": evidence.get("protected_accesses"),
                "opens_checked": evidence.get("opens_checked"),
                "policy": evidence.get("protection_policy")})

    generate, gates = cmds["generate"], cmds["gates"]
    execute, analyse = cmds["execute"], cmds["analyse"]
    ready = not checks.failed
    exp = cohort["expected"]
    receipt = {
        "schema_version": gate.PREFLIGHT_SCHEMA_V27,
        "study": "oneiros_v27_repository_native_three_arm_exploratory_confirmation_r5",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_ready": ready, "launched": False, "weights_loaded": False,
        "authorization_written": False,
        "tag": args.tag,
        "supersedes": {"path": supersedes, "sha256": sha(supersedes),
                       "status": ("valid two-arm generation preflight, never authorised; "
                                  "scientifically incomplete for the three-arm study"
                                  if supersedes == R4_PREFLIGHT else args.supersedes_reason)},
        "source": {"commit": state["head"],
                   "executable_tree_sha256": identity["executable_tree_sha256"],
                   "protocol_sha256": identity["protocol_sha256"]},
        "inputs": inputs, "machine_bound_inputs": machine_bound,
        "job": {"path": JOB, "file_sha256": sha(JOB), "job_sha256": job["job_sha256"]},
        "manifest": {"path": MANIFEST, "sha256": sha(MANIFEST)},
        "arm_registry": registry, "adapters": registry["adapter_manifest_sha256"],
        "adapter_manifest_sha256": None,
        "generation_outputs": {a: outputs[a] for a in registry["arms"]},
        "model": model, "environment": env,
        "gate_evidence": gate_evidence,
        "cohort": {"targets": len(ids), "ids_sha256": ids_sha, "ids": sorted(ids),
                   "arms": list(cohort["arms"]), "expected": exp,
                   "panel_policy_exclusions": excl},
        "checks": checks.items, "failed_checks": checks.failed,
        "estimates": {"basis": "same generator and contract, v2.4: 552 candidates per arm took "
                               "1,878 s (base) and 5,838 s (adapter); scaled to 696",
                      "runtime_minutes": {"base": 39, "sft": 123, "relearn": 123,
                                          "total": 285},
                      "vram": "1.5B bf16 ~3.1 GiB + KV cache for batch 2 x 3,072 tokens",
                      "disk": "< 10 MiB per arm of generations; execution evidence < 300 MiB"},
        "commands": {
            "authorization_required": AUTH,
            "order": ["base", "sft", "relearn"],
            "generate": generate, "read_only_gate_before_each_arm": gates,
            "sequential_verification": "each trained arm's gate re-verifies the completed base "
                                       "arm with the exact single-arm verifier before launch",
            "execute_after_all_three_arms": execute,
            "analyse_after_execution": analyse,
            "shell_note": "run from PowerShell, or prefix MSYS_NO_PATHCONV=1 in Git Bash so "
                          "/mnt/c paths are not rewritten",
            "launch_time_check": "nvidia-smi --query-gpu=utilization.gpu,memory.used "
                                 "--format=csv -l 1 (several seconds): no other compute or "
                                 "heavy graphics workload; nvidia-smi --query-compute-apps="
                                 "process_name --format=csv: no python process"},
        "analysis_plan": analysis.ANALYSIS_PLAN_V27,
        "analysis_plan_sha256": analysis.analysis_plan_sha256(),
    }
    out.write_bytes((json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
                    .encode("utf-8"))
    print(json.dumps({"pipeline_ready": ready, "failed_checks": checks.failed,
                      "sha256": sha(args.out)}, indent=1))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
