"""Shared helpers for the analysis CLI tests: v2.4 gate evidence built against the CURRENT
source, the full analysis invocation, and v2.4 Atheris contracts/grids built by the real
``prepare_contract`` (fake probe only). ``world`` is the cohort-pipeline fixture dict."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harness import native_launch_gate as gate
from scripts import native_atheris_results as ar
from scripts import native_generated_tests_analyse as an
from scripts import native_generated_tests_atheris_wsl as ath

REPO = Path(__file__).resolve().parent.parent
COND = "primary_whole_module"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=1, sort_keys=True) + "\n").encode())
    return path


def valid_receipts(world, drop=(), mutate=None) -> dict:
    """Receipts that validate against the CURRENT repository source."""
    from scripts import native_pipeline_synthetic as ps
    base = world["tmp"] / "receipts"
    receipts = {
        "full_suite": {"schema_version": "oneiros_native_full_suite_receipt_v2", "exit": 0,
                       "failed": 0, "passed": 10, "skipped": 0, "tree_clean_at_start": True,
                       "source_commit": "abc",
                       "executable_tree_sha256": gate.source_identity(REPO)["executable_tree_sha256"]},
        "synthetic_pipeline": {"schema_version": "oneiros_native_pipeline_synthetic_v1",
                               "passed": True,
                               "components_sha256": {c: sha(REPO / c) for c in ps.COMPONENTS}},
        "sandbox_canaries": {"schema_version": "oneiros_native_sandbox_canaries_v2", "passed": True,
                             "executor_sha256": sha(REPO / "scripts/native_generated_tests_execute_wsl.py"),
                             "inner_sha256": sha(REPO / "scripts/native_sandbox_inner.sh"),
                             "prepare_sha256": sha(REPO / "scripts/native_rehearsal_prepare_wsl.py")},
        "atheris_canaries": {"schema_version": "oneiros_native_atheris_canaries_v3", "passed": True,
                             "design_version": ar.DESIGN_VERSION,
                             "script_sha256": sha(REPO / "scripts/native_generated_tests_atheris_wsl.py"),
                             "inner_sha256": sha(REPO / "scripts/native_sandbox_inner.sh"),
                             "verdicts_sha256": sha(REPO / "scripts/native_atheris_results.py"),
                             "checks": {k: True for k in an.ATHERIS_REQUIRED_CHECKS}},
    }
    if mutate:
        mutate(receipts)
    entries = {}
    for kind, value in receipts.items():
        if kind in drop:
            continue
        path = write(base / f"{kind}.json", value)
        entries[kind] = {"path": path.relative_to(world["tmp"]).as_posix(), "sha256": sha(path)}
    return entries


def evidence(world, drop=(), mutate=None, ledger=None, preflight_mutate=None) -> Path:
    receipts = valid_receipts(world, drop, mutate)
    ledger_path = write(world["tmp"] / "ledger.json",
                        ledger if ledger is not None else
                        {"schema_version": an.LEDGER_SCHEMA, "entries": []})
    rel = lambda p: Path(p).relative_to(world["tmp"]).as_posix()  # noqa: E731
    inputs = {rel(p): sha(p) for p in (world["manifest"], world["job"], world["prep"], ledger_path)}
    inputs.update({e["path"]: e["sha256"] for e in receipts.values()})
    pre = {"schema_version": gate.PREFLIGHT_SCHEMA, "pipeline_ready": True,
           "job": {"path": rel(world["job"])}, "manifest": {"path": rel(world["manifest"])},
           "inputs": inputs,
           "gate_evidence": {"receipts": receipts,
                             "quarantine_ledger": {"path": rel(ledger_path),
                                                   "sha256": sha(ledger_path)}}}
    if preflight_mutate:
        preflight_mutate(pre)
    return write(world["tmp"] / "preflight_v2_4.json", pre)


def analyse(world, results, contract, root, preflight=None, extra=(), out="analysis.json",
            generations=None, study_mode="engineering_dress_rehearsal"):
    args = ["analyse", "--manifest", str(world["manifest"]), "--job", str(world["job"]),
            "--prep", str(world["prep"]), "--preflight", str(preflight or evidence(world)),
            "--execution-contract", str(contract), "--results", str(results),
            *(generations or ["--generations", str(root)]), "--condition", COND,
            "--study-mode", study_mode, "--out", str(world["tmp"] / out), *extra]
    an.main(args, root=world["tmp"])
    return json.loads((world["tmp"] / out).read_text())


def probe_eligible(view, module, qualname):
    return {"eligible": True, "plan": [{"name": "x", "type": "int", "keyword": False}],
            "returns": "int", "python": "3.11.15"}


def atheris_contract(world, ineligible=()):
    """A real v2.4 contract (live views + frozen eligibility) with a fake probe."""
    contract, prepared, eligibility = ath.prepare_contract(
        world["prep"], world["manifest"], 600, (42, 43, 44), root=world["tmp"],
        probe_fn=probe_eligible)
    for key in ineligible:                       # applicability decided per target
        eligibility[key] = {**eligibility[key], "status": "atheris_ineligible",
                            "reason": "variadic_signature", "plan": None, "returns": None}
    contract = {**contract, "eligibility": eligibility,
                "eligibility_sha256": ar.canonical_sha(eligibility)}
    return contract, prepared, eligibility


def load_atheris(world, rpath, cpath, prepared):
    return ar.load_results(
        rpath, cpath, qualified=sorted(prepared["rows"]), manifest_sha256=sha(world["manifest"]),
        prep={"path": prepared["path"], "sha256": prepared["sha256"]}, prep_rows=prepared["rows"],
        script_sha256=sha(REPO / "scripts/native_generated_tests_atheris_wsl.py"),
        inner_sha256=sha(REPO / "scripts/native_sandbox_inner.sh"),
        verdicts_sha256=sha(REPO / "scripts/native_atheris_results.py"))


def fuzz_evidence(mode: str, seed: int, budget: int = 600, **over) -> dict:
    """Exactly the field set the real ``fuzz`` returns for one completed search."""
    row = {"mode": mode, "seed": seed, "budget_cpu_seconds": budget,
           "tolerance_cpu_seconds": ar.tolerance(budget), "supervisor_reason": "cpu_budget_exhausted",
           "wall_seconds": budget + 1.5, "aggregate_cpu_seconds": budget + 0.4,
           "main_cpu_seconds": budget + 0.4, "worker_cpu_seconds": 0.0,
           "end_reason": "cpu_budget_exhausted", "exit": -9, "reached": True,
           "within_budget": True, "cleanup_ok": True, "views_unchanged": True,
           "witnesses": 0, "confirmations": [], "confirmed": 0, "replay_errors": 0,
           "kill": False, "replay_cpu_seconds": 0.5, "replay_cleanup_ok": True}
    if mode == "posthoc":
        row.update(corpus=12, corpus_truncated=False, dropped_partial_inputs=0, opaque_results=0)
    else:
        row["dropped_partial_witnesses"] = 0
    if mode == "differential":
        row["label"] = "oracle-assisted upper bound"
    row.update(over)
    return row
