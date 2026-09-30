"""End-to-end synthetic pipeline for native generated tests (no model, no GPU).

sealed prompt -> leakage scan -> sequence-fit job -> generation (the real runner, with a
deterministic synthetic backend) -> extraction -> sandboxed buggy/fixed execution with reruns
(WSL executor ``run``) -> durable result rows -> frozen analysis (engineering mode).

The toy cohort is prepared by the formal preparation path (``synthetic-prepare``). The
synthetic backend emits designed candidates per slot, so the expected class of every row is
known; the receipt records each check and binds the source hashes of every component.

    python scripts/native_pipeline_synthetic.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "sft_root_cause" / "native_v24_synthetic"
RECEIPT = ROOT / "results" / "sft_root_cause" / "native_v24_canaries" / "pipeline_receipt.json"
COMPONENTS = ("harness/native_generated_test_prompt.py", "harness/native_generated_test_leakage.py",
              "scripts/native_generated_tests_generate.py",
              "scripts/native_generated_tests_execute_wsl.py", "scripts/native_sandbox_inner.sh",
              "scripts/native_rehearsal_prepare_wsl.py", "scripts/native_generated_tests_analyse.py",
              "scripts/native_pipeline_synthetic.py", "harness/native_launch_gate.py",
              "scripts/native_generation_io.py", "scripts/receipt_sanitize.py",
              "scripts/native_execution_results.py", "scripts/gpu_run.py")
CANARY_DIR = ROOT / "results" / "sft_root_cause" / "native_v24_canaries"
LOCK_RACER = r'''
import json, sys, time, uuid
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import gpu_run
runs, go = Path(sys.argv[2]), Path(sys.argv[3])
while not go.exists():
    pass
held = gpu_run.acquire_exclusive("native_v24_generation", runs, sys.argv[4], uuid.uuid4().hex)
print(json.dumps({"won": held is not None}))
sys.stdout.flush()
time.sleep(3)
'''


def exclusive_lock_canary(rounds: int = 4, racers: int = 5) -> dict:
    """v2.4 I.1: genuinely concurrent starters race for one exclusive key; exactly one wins."""
    import shutil
    import sys
    import tempfile
    import time
    work = Path(tempfile.mkdtemp(prefix="oneiros_lock_canary_"))
    winners = []
    try:
        script = work / "racer.py"
        script.write_text(LOCK_RACER, encoding="utf-8")
        for r in range(rounds):
            runs, go = work / f"runs{r}", work / f"go{r}"
            runs.mkdir()
            procs = [subprocess.Popen([sys.executable, str(script), str(ROOT), str(runs), str(go),
                                       f"racer{i}"], stdout=subprocess.PIPE, text=True)
                     for i in range(racers)]
            time.sleep(1.5)
            go.write_text("go")
            winners.append(sum(json.loads(p.communicate(timeout=120)[0].strip().splitlines()[-1])
                               ["won"] for p in procs))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"rounds": rounds, "racers": racers, "winners": winners,
            "ok": winners == [1] * rounds}

# slot -> (designed candidate for "add", expected class; for "crash" the kill slot differs)
DESIGN = {
    0: ("from toy.core import {name}\n\n\ndef test_a():\n    assert {kill}\n", "KILL"),
    1: ("```python\nfrom toy.core import {name}\n\n\ndef test_b():\n    assert {ok}\n```", "pass_both"),
    2: ("import inspect\nfrom toy.core import {name}\n\n\ndef test_c():\n    inspect.getsource({name})\n",
        "policy_refused"),
    3: ("def test_d():\n    assert 1 == 1\n", "target_not_reached"),
    4: ("def test_(:\n", "syntax_failure"),
    5: ("from toy.core import {name}\n\n\ndef test_f():\n    {name}({args})\n    assert False\n",
        "fixed_side_failure"),
    6: ("from toy.core import no_such_name\n\n\ndef test_g():\n    assert no_such_name()\n",
        "fabricated_import"),
    7: ("import pytest\nfrom toy.core import {name}\n\n\n@pytest.mark.skip\ndef test_h():\n"
        "    assert {ok}\n", "skipped_or_xfail"),
}
TARGET_FORMS = {"add": {"kill": "add(2, 3) == 5", "ok": "add(2, 0) == 2", "args": "1, 1",
                        "kill_class": "semantic_kill"},
                "crash": {"kill": "crash(1) == 10", "ok": "crash(2) == 10",
                          "args": "2", "kill_class": "crash_kill"}}


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    return "/mnt/" + resolved.drive[0].lower() + resolved.as_posix()[2:]


def wsl(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    script = ROOT / "scripts" / "wsl_native_python.sh"
    return subprocess.run(["wsl.exe", "-u", "root", "--", "bash", wsl_path(script), *args],
                          capture_output=True, text=True, timeout=timeout)


def synthetic_backend_for(qualname: str, arm: str):
    form = TARGET_FORMS[qualname]

    def backend(prompt: str, n: int, seed: int):
        out = []
        for slot in range(n):
            template, _ = DESIGN[slot]
            if slot == 0 and arm == "base":          # base never kills; sft kills at slot 0
                template = DESIGN[1][0]
            out.append(template.format(name=qualname, **form))
        return out
    return backend


def expected_class(qualname: str, arm: str, slot: int) -> str:
    if slot == 0:
        return TARGET_FORMS[qualname]["kill_class"] if arm == "sft" else "pass_both"
    return DESIGN[slot][1]


def main() -> int:
    from harness.native_generated_test_leakage import scan
    from harness.native_generated_test_prompt import build_prompt
    from scripts import native_generated_tests_analyse as analysis
    from scripts import native_generated_tests_generate as gen
    checks = {}
    prep = wsl("scripts/native_generated_tests_execute_wsl.py", "synthetic-prepare", wsl_path(OUT))
    checks["formal_preparation"] = prep.returncode == 0
    if not checks["formal_preparation"]:
        print(prep.stdout[-1500:], prep.stderr[-1500:])
        return finish(checks, {})
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    from scripts.native_rehearsal_prepare_wsl import tag_of
    tag = {t["key"]: tag_of(t["key"]) for t in manifest["targets"]}
    sealed, scans = [], {}
    for key in manifest["kept_targets"]:
        view = json.loads((OUT / "exports" / "buggy_view" / f"{tag[key]}.json").read_text())
        s = build_prompt(view["dto"], view["buggy_source"])
        verifier = json.loads((OUT / "exports" / "verifier" / f"{tag[key]}.json").read_text())
        scans[key] = scan(s, verifier)
        sealed.append(s)
    checks["prompts_leakage_clean"] = all(v["ok"] for v in scans.values())
    job = gen.build_job(sealed, scans, gen.sequence_fit(sealed, lambda text: len(text.split())))
    job_file = OUT / "job.json"
    job_bytes = (json.dumps({"primary_whole_module": job}, indent=1) + "\n").encode("utf-8")
    job_file.write_bytes(job_bytes)
    checks["job_complete"] = len(job["items"]) == len(manifest["kept_targets"])
    # amendment v2.3: an explicit cohort manifest (qualified / generation / exclusions)
    from scripts.native_generation_io import cohort_fields
    cohort_manifest = OUT / "manifest_cohort.json"
    records = OUT / "records.jsonl"                # v2.4 C: the exact declared preparation
    cohort_manifest.write_text(json.dumps({
        **manifest, **cohort_fields(manifest["targets"], "job.json", job_bytes,
                                    "primary_whole_module", {}),
        "nature": "ENGINEERING SYNTHETIC PIPELINE TEST",
        "study_mode": "engineering_dress_rehearsal",
        "requalification_records": {"path": records.relative_to(ROOT).as_posix(),
                                    "sha256": hashlib.sha256(records.read_bytes()).hexdigest()}},
        indent=1), encoding="utf-8")
    gen_dir = OUT / "generations"
    if gen_dir.exists():
        import shutil
        shutil.rmtree(gen_dir)
    qual = {t["key"]: t["target"] for t in manifest["targets"]}
    for arm in gen.ARMS:                           # separate arm directories (v2.3 C)
        per_target = {item["target_key"]: synthetic_backend_for(qual[item["target_key"]], arm)
                      for item in job["items"]}
        by_prompt = {item["prompt"]: per_target[item["target_key"]] for item in job["items"]}
        gen.run(job, arm, "primary_whole_module", gen_dir / arm,
                {"backend": "synthetic", "arm": arm, "condition": "primary_whole_module",
                 "job_sha256": job["job_sha256"],
                 "job_file_sha256": hashlib.sha256(job_bytes).hexdigest(),
                 "adapter_manifest_sha256": None if arm == "base" else "synthetic-adapter"},
                gen.texts_backend(lambda prompt, n, seed: by_prompt[prompt](prompt, n, seed)))
    results_dir = OUT / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    run = wsl("scripts/native_generated_tests_execute_wsl.py", "run",
              "--prep", wsl_path(OUT / "records.jsonl"), "--manifest", wsl_path(cohort_manifest),
              "--job", wsl_path(job_file), "--generations", wsl_path(gen_dir),
              "--condition", "primary_whole_module",
              "--out", wsl_path(results_dir), timeout=3600)
    checks["execution_complete"] = run.returncode == 0
    rows = [json.loads(l) for l in (results_dir / "results_primary_whole_module.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()] if checks["execution_complete"] else []
    mismatches = [(r["key"], r["class"], expected_class(qual[r["target_key"]], r["arm"], r["slot"]))
                  for r in rows if r["class"] != expected_class(qual[r["target_key"]], r["arm"],
                                                                r["slot"])]
    checks["every_row_classified_as_designed"] = bool(rows) and not mismatches
    checks["grid_complete"] = len(rows) == 2 * 3 * 8 * len(job["items"]) and all(
        r.get("generation", {}).get("finish_reason") == "eos" for r in rows)
    checks["kills_rerun_and_fixed_valid"] = all(
        r["classification"].get("rerun_agrees") is True and r["fixed_valid"]
        for r in rows if r["class"] in ("semantic_kill", "crash_kill"))
    # synthetic gate evidence: the real, current sandbox and Atheris canary receipts and an
    # empty synthetic ledger; no full-suite or pipeline receipt can exist yet, and synthetic
    # evidence is never a v2.4 preflight - so the stage-receipt subgate must fail
    ledger = OUT / "quarantine_ledger.json"
    ledger.write_bytes((json.dumps({"schema_version": analysis.LEDGER_SCHEMA, "entries": []},
                                   indent=1) + "\n").encode("utf-8"))
    receipts = {}
    for kind, name in (("sandbox_canaries", "canary_receipt_v2.json"),
                       ("atheris_canaries", "atheris_canary_receipt_v3.json")):
        path = CANARY_DIR / name
        if path.is_file():
            receipts[kind] = {"path": path.relative_to(ROOT).as_posix(),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    evidence = OUT / "synthetic_gate_evidence.json"
    evidence.write_bytes((json.dumps({
        "schema_version": analysis.SYNTHETIC_EVIDENCE_SCHEMA,
        "gate_evidence": {"receipts": receipts, "quarantine_ledger": {
            "path": ledger.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()}}}, indent=1)
        + "\n").encode("utf-8"))
    artifact = OUT / "analysis.json"
    try:
        analysis.main(["analyse", "--manifest", str(cohort_manifest), "--job", str(job_file),
                       "--prep", str(records), "--preflight", str(evidence),
                       "--execution-contract",
                       str(results_dir / "execute_contract_primary_whole_module.json"),
                       "--results", str(results_dir / "results_primary_whole_module.jsonl"),
                       "--generations", str(gen_dir), "--condition", "primary_whole_module",
                       "--study-mode", "engineering_dress_rehearsal", "--out", str(artifact)])
        result = json.loads(artifact.read_text(encoding="utf-8"))
        gate = result["engineering_gate"]
        # the 2-target, 1-repository toy must FAIL coverage/repositories (v2.4 G), and the
        # synthetic evidence must fail the stage-receipt subgate, while every other subgate is
        # really evaluated and passes; every arm comparison is suppressed
        checks["analysis_engineering_mode"] = (
            "SUPPRESSED" in result["decisions"]
            and result["engineering_gate_passed"] is False
            and gate["subgates"] == {"coverage_gate_passed": False,
                                     "repository_gate_passed": False,
                                     "stage_receipts_gate_passed": False,
                                     "canaries_gate_passed": True,
                                     "artifact_integrity_gate_passed": True,
                                     "unexplained_failures_gate_passed": True}
            and sorted(gate["evidence_problems"]["stage_receipts"]) == [
                "full_suite: missing", "synthetic_pipeline: missing"]
            and gate["evidence_problems"]["preflight"] == [
                "synthetic evidence is not a v2.4 preflight"]
            and result["arm_comparison"].startswith("SUPPRESSED")
            and "unique_bugs_killed" not in result and "kill_at_8" not in result
            and result["cohort"]["generation_targets"] == 2
            and "generation_telemetry" in result)
        if not checks["analysis_engineering_mode"]:
            print("gate:", json.dumps(gate, default=str)[:1500])
        killed = sorted({r["target_key"] for r in rows if r["arm"] == "sft"
                         and r["class"] in ("semantic_kill", "crash_kill")})
        checks["sft_kills_both_toy_targets"] = killed == sorted(
            i["target_key"] for i in job["items"]) and not any(
            r["class"] in ("semantic_kill", "crash_kill") for r in rows if r["arm"] == "base")
    except Exception as exc:
        checks["analysis_engineering_mode"] = False
        print("analysis failed:", exc)
    lock = exclusive_lock_canary()
    checks["exclusive_lock_single_winner"] = lock["ok"]
    return finish(checks, {"mismatches": mismatches[:10], "rows": len(rows),
                           "execution_tail": run.stdout[-500:], "exclusive_lock": lock})


def finish(checks: dict, detail: dict) -> int:
    receipt = {"schema_version": "oneiros_native_pipeline_synthetic_v1",
               "components_sha256": {c: hashlib.sha256((ROOT / c).read_bytes()).hexdigest()
                                     for c in COMPONENTS},
               "checks": checks, "passed": bool(checks) and all(checks.values()), **detail}
    from scripts.receipt_sanitize import scrub_json     # tracked receipt: no user paths
    receipt = scrub_json(receipt)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "checks": checks,
                      "mismatches": detail.get("mismatches")}, indent=1))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
