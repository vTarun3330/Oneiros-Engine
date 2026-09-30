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

OUT = ROOT / "results" / "sft_root_cause" / "native_v21_synthetic"
RECEIPT = ROOT / "results" / "sft_root_cause" / "native_v21_canaries" / "pipeline_receipt.json"
COMPONENTS = ("harness/native_generated_test_prompt.py", "harness/native_generated_test_leakage.py",
              "scripts/native_generated_tests_generate.py",
              "scripts/native_generated_tests_execute_wsl.py", "scripts/native_sandbox_inner.sh",
              "scripts/native_rehearsal_prepare_wsl.py", "scripts/native_generated_tests_analyse.py",
              "scripts/native_pipeline_synthetic.py")

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
    job_file.write_text(json.dumps({"primary_whole_module": job}, indent=1), encoding="utf-8")
    checks["job_complete"] = len(job["items"]) == len(manifest["kept_targets"])
    gen_dir = OUT / "generations"
    if gen_dir.exists():
        import shutil
        shutil.rmtree(gen_dir)
    qual = {t["key"]: t["target"] for t in manifest["targets"]}
    for arm in gen.ARMS:
        per_target = {item["target_key"]: synthetic_backend_for(qual[item["target_key"]], arm)
                      for item in job["items"]}
        by_prompt = {item["prompt"]: per_target[item["target_key"]] for item in job["items"]}
        gen.run(job, arm, "primary_whole_module", gen_dir,
                {"backend": "synthetic", "arm": arm, "condition": "primary_whole_module",
                 "job_sha256": job["job_sha256"]},
                lambda prompt, n, seed: by_prompt[prompt](prompt, n, seed))
    results_dir = OUT / "results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    run = wsl("scripts/native_generated_tests_execute_wsl.py", "run",
              "--prep", wsl_path(OUT / "records.jsonl"), "--manifest", wsl_path(OUT / "manifest.json"),
              "--generations", wsl_path(gen_dir), "--condition", "primary_whole_module",
              "--out", wsl_path(results_dir), timeout=3600)
    checks["execution_complete"] = run.returncode == 0
    rows = [json.loads(l) for l in (results_dir / "results_primary_whole_module.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()] if checks["execution_complete"] else []
    mismatches = [(r["key"], r["class"], expected_class(qual[r["target_key"]], r["arm"], r["slot"]))
                  for r in rows if r["class"] != expected_class(qual[r["target_key"]], r["arm"],
                                                                r["slot"])]
    checks["every_row_classified_as_designed"] = bool(rows) and not mismatches
    checks["grid_complete"] = len(rows) == 2 * 3 * 8 * len(manifest["kept_targets"])
    checks["kills_rerun_and_fixed_valid"] = all(
        r["classification"].get("rerun_agrees") is True and r["fixed_valid"]
        for r in rows if r["class"] in ("semantic_kill", "crash_kill"))
    artifact = OUT / "analysis.json"
    try:
        analysis.main(["analyse", "--manifest", str(OUT / "manifest.json"), "--results",
                       str(results_dir / "results_primary_whole_module.jsonl"),
                       "--condition", "primary_whole_module",
                       "--study-mode", "engineering_dress_rehearsal", "--out", str(artifact)])
        result = json.loads(artifact.read_text(encoding="utf-8"))
        checks["analysis_engineering_mode"] = ("SUPPRESSED" in result["decisions"]
                                               and result["unique_bugs_killed"] == {"base": 0, "sft": 2})
    except Exception as exc:
        checks["analysis_engineering_mode"] = False
        print("analysis failed:", exc)
    return finish(checks, {"mismatches": mismatches[:10], "rows": len(rows),
                           "execution_tail": run.stdout[-500:]})


def finish(checks: dict, detail: dict) -> int:
    receipt = {"schema_version": "oneiros_native_pipeline_synthetic_v1",
               "components_sha256": {c: hashlib.sha256((ROOT / c).read_bytes()).hexdigest()
                                     for c in COMPONENTS},
               "checks": checks, "passed": bool(checks) and all(checks.values()), **detail}
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "checks": checks,
                      "mismatches": detail.get("mismatches")}, indent=1))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
