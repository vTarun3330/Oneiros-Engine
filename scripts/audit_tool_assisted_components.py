"""Machine-readable audit of the tool-assisted components (Phases 1 and 8).

Static facts are recomputed from the source on every run (canonical hashes,
whether a file reads the canonical records.json, touches the reference or
gold tests, calls the reference/mutant classifier, imports real Atheris,
deep-copies arguments, and whether it has tests).  The classification and
its rationale are the audit's judgement and are recorded next to the
evidence that supports them.  Environment readiness is probed live when
``--probe-wsl`` is given; otherwise the last recorded probe is kept.

Reads source code only.  Opens no dataset split and no prior result artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.source_identity import canonical_sha256

SCHEMA = "oneiros_tool_assisted_component_audit_v1"
CLASSES = ("reusable_unchanged", "reusable_after_fix", "diagnostic_only",
           "unsafe_for_this_experiment", "missing")

COMPONENTS = {
    "harness/verifier_guided_repair.py": {
        "classification": "unsafe_for_this_experiment",
        "tests": ["tests/test_verifier_guided_repair.py"],
        "findings": [
            "reference is structurally out of scope (good)",
            "feedback reports pass/fail on the code under test; passing implies the "
            "mutant is not killed, so this reveals kill-relevant information",
            "asks the model to 'correct the asserted value' after an assertion "
            "failure, i.e. can repair a legitimate kill away",
            "treats every exception as repairable, including exceptions raised inside "
            "the target, which may be the defect itself",
            "no frame attribution, no duplicate handling, no budget accounting, no "
            "lineage, no matched control",
        ],
        "disposition": "left unchanged (bound by earlier receipts); superseded for this "
                       "experiment by harness/execution_feedback.py and "
                       "harness/tool_assisted_generation.py; its leakage-scan idea is kept",
    },
    "harness/safe_execution.py": {
        "classification": "reusable_unchanged",
        "tests": ["tests/test_safe_execution.py"],
        "findings": [
            "fresh isolated interpreter (-I), temp working directory, restricted "
            "builtins and imports, tracing deadline, hard parent timeout, POSIX "
            "limits where available (not enforced on Windows)",
            "execute_assertions runs against the shown code only; classify_assertions "
            "compares reference and mutant and is evaluator-only",
            "each implementation executes the candidate text afresh in its own "
            "namespace, so mutable arguments are never shared",
            "does not report which frame raised",
        ],
        "disposition": "primitives reused by harness/buggy_side_execution.py, which "
                       "adds frame attribution without modifying this file",
    },
    "harness/execution_harness.py": {
        "classification": "diagnostic_only",
        "tests": [],
        "findings": ["legacy differential golden-vs-mutant harness; uses the golden "
                     "function, so it must never feed a prompt"],
        "disposition": "not used",
    },
    "harness/metamorphic_relations.py": {
        "classification": "diagnostic_only",
        "tests": ["tests/test_metamorphic_relations.py"],
        "findings": ["relations proposed blind from argument shape; verification in "
                     "its caller uses the reference"],
        "disposition": "not used; a metamorphic arm would be a different intervention",
    },
    "harness/oracle_diagnosis.py": {
        "classification": "diagnostic_only",
        "tests": ["tests/test_oracle_diagnosis.py"],
        "findings": ["executes the candidate's call on reference and mutant; its own "
                     "docstring forbids importing it from prompt-building code"],
        "disposition": "not used",
    },
    "harness/rehearsal_evaluator.py": {
        "classification": "reusable_unchanged",
        "tests": ["tests/test_rehearsal_receipt.py"],
        "findings": ["imports the sealed evaluator's pure scoring helpers; retains raw "
                     "outputs and hashes; refuses the consumed test split; accepts "
                     "exactly candidates_per_target slots per record"],
        "disposition": "scores all three arms unchanged, after generation is frozen",
    },
    "harness/native_repository_eval.py": {
        "classification": "reusable_after_fix",
        "tests": ["tests/test_native_repository_eval.py"],
        "findings": ["fail-closed statuses separate environment failures from "
                     "behavioural outcomes", "requires WSL-built per-bug environments; "
                     "no generated test has been executed natively yet"],
        "disposition": "readiness blocked on environment reliability (see readiness)",
    },
    "baseline/atheris_harness.py": {
        "classification": "diagnostic_only",
        "tests": ["tests/test_atheris_harness.py", "tests/test_atheris_panel_and_comparison.py",
                  "tests/test_atheris_seed_aggregate.py"],
        "findings": ["real atheris package driving libFuzzer, labelled distinct from the "
                     "simulated baseline.coverage_fuzzer", "differential oracle uses the "
                     "reference (baseline only, never a model input)",
                     "deep-copies arguments per implementation",
                     "integer inputs are bounded to [-1000, 1000]; branches outside that "
                     "range are unreachable by construction",
                     "its own comments record that an earlier validation sweep ran "
                     "without coverage instrumentation (since fixed)"],
        "disposition": "baseline for a separate comparison; not part of the repair arms",
    },
    "scripts/measure_repair_loop_headroom.py": {
        "classification": "unsafe_for_this_experiment",
        "tests": [],
        "findings": ["reads the canonical records.json", "uses the superseded "
                     "observation taxonomy that exposes pass/fail on the code under test"],
        "disposition": "not used",
    },
    "scripts/measure_metamorphic_ceiling.py": {
        "classification": "diagnostic_only",
        "tests": [],
        "findings": ["reads the canonical records.json and executes the reference; "
                     "a ceiling measurement, never a model input"],
        "disposition": "not used",
    },
    "scripts/diagnose_atheris_survivors.py": {
        "classification": "unsafe_for_this_experiment",
        "tests": [],
        "findings": ["default input is the validation-split Atheris task file",
                     "uses gold assertions as a diagnostic oracle"],
        "disposition": "not run",
    },
    "scripts/run_atheris_wsl.sh": {
        "classification": "reusable_unchanged",
        "tests": [],
        "findings": ["one target per process, per-target working directory, wall-limit "
                     "and per-input timeout, finalises every result file"],
        "disposition": "used only for the readiness smoke",
    },
}

MISSING = {
    "buggy-side executor with frame attribution": "harness/buggy_side_execution.py",
    "closed, versioned, hashable feedback taxonomy": "harness/execution_feedback.py",
    "bounded repair loop with a call-, sequence-, cap- and rendered-input-token-"
    "matched sham-feedback control, lineage, "
    "outcome-blind final-slot policy and resume journal": "harness/tool_assisted_generation.py",
}

STATIC_PATTERNS = {
    "reads_canonical_records_json": r"records\.json",
    "mentions_reference_or_golden": r"golden|reference_code|\breference\b",
    "calls_reference_mutant_classifier": r"classify_assertions|evaluate_candidate_slots",
    "imports_real_atheris": r"^\s*import atheris",
    "deep_copies_arguments": r"deepcopy",
}


def static_facts(relative: str) -> dict:
    path = ROOT / relative
    if not path.exists():
        return {"exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    facts = {"exists": True, "canonical_sha256": canonical_sha256(path)}
    for name, pattern in STATIC_PATTERNS.items():
        facts[name] = bool(re.search(pattern, text, flags=re.MULTILINE))
    return facts


def probe_wsl() -> dict:
    def run(command: str) -> str:
        completed = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command],
                                   capture_output=True, timeout=120)
        return completed.stdout.decode("utf-8", "replace").replace("\x00", "").strip()
    return {"atheris_python_and_version": run(
        '/opt/atheris311/bin/python -c "import sys, importlib.metadata as m; '
        'print(sys.version.split()[0], m.version(\'atheris\'))"'),
        "probed_utc": datetime.now(timezone.utc).isoformat()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_tool_assisted_audit.json")
    parser.add_argument("--probe-wsl", action="store_true")
    args = parser.parse_args(argv)
    previous = (json.loads(args.output.read_text(encoding="utf-8"))
                if args.output.exists() else {})
    components = {}
    for relative, entry in COMPONENTS.items():
        if entry["classification"] not in CLASSES:
            raise SystemExit(f"unknown classification for {relative}")
        components[relative] = {**entry, "static": static_facts(relative),
                                "tests_present": {test: (ROOT / test).exists()
                                                  for test in entry["tests"]}}
    readiness = {
        "atheris": {
            "probe": probe_wsl() if args.probe_wsl else (
                previous.get("readiness", {}).get("atheris", {}).get("probe")),
            "is_actual_atheris": True,
            "wsl_distribution": "Ubuntu-24.04 (WSL2)",
            "instrumentation_smoke": {
                "target": "synthetic int branch x > 500 (reachable within the [-1000, "
                          "1000] adapter range); reference returns x",
                "coverage_instrumented": True, "libfuzzer_coverage": "cov 2 -> 3 (NEW)",
                "outcome": "killed", "kill_kind": "semantic_kill", "runs_to_kill": 11,
                "negative_check": "a branch at x > 1000 was unreachable: the adapter "
                                  "bounds integers to [-1000, 1000]",
            },
            "crash_vs_semantic_separated": True,
            "blocking_defects": [
                "input adapter ranges are fixed ([-1000,1000] ints, <=8-element lists): "
                "reachability must be reported per target, not assumed",
                "earlier Atheris artifacts on validation predate the instrumentation "
                "fix and are not reusable for a reportable comparison",
                "no permitted (non-validation) Atheris task panel exists for the "
                "tool-assisted pilot panel yet",
            ],
            "status": "ready_for_smoke_not_for_comparison",
        },
        "native_repository": {
            "host": "WSL2 Ubuntu 24.04; per-bug interpreters 3.7-3.11",
            "official_test_pilot": {"tasks": 5, "reproduced": 3, "infrastructure_failures": 2,
                                    "reproduced_tasks": ["bugsinpy::thefuck::1",
                                                         "bugsinpy::sanic::1",
                                                         "bugsinpy::tornado::1"],
                                    "failures": {"bugsinpy::httpie::1": "pytest exit 4 "
                                                 "(collection)", "bugsinpy::fastapi::1":
                                                 "flit_core<4 unresolvable"}},
            "generated_tests_executed_natively": 0,
            "environment_failures_counted_as_model_outcomes": False,
            "blocking_defects": [
                "40% infrastructure failure on the official-test pilot",
                "no generated repository test has been executed natively",
                "all 13 train repository projects are in the frozen control's training "
                "arm, so any native run on them is an engineering smoke, not evidence",
                "the local ingestion cache is a blob-less partial clone; environments "
                "must be rebuilt from upstream URLs",
            ],
            "status": "not_ready_for_reportable_comparison",
            "smoke_plan": [
                {"case": "bugsinpy::thefuck::1", "purpose": "reproduced official test; "
                 "inject 8 generated tests, expect buggy/fixed statuses recorded per test"},
                {"case": "bugsinpy::sanic::1", "purpose": "second reproduced project"},
                {"case": "bugsinpy::tornado::1", "purpose": "third reproduced project"},
                {"case": "negative control: a generated test that passes on both "
                 "revisions", "purpose": "must be passes_both, never a kill"},
                {"case": "negative control: environment build forced to fail",
                 "purpose": "must be environment_unavailable, never a model failure or "
                            "positive example"},
            ],
            "smoke_runtime_estimate": {"cpu_wsl_minutes": "15-45 (environment builds, "
                                                          "900 s timeout each)",
                                       "gpu_minutes": "about 1 (8 samples x 3 tasks)"},
            "artifacts_needed_before_reportable_comparison": [
                "a frozen repository panel outside every training arm (none exists in "
                "the train shard today)",
                "per-task environment receipts (interpreter, dependency lock, commit "
                "hashes) with a reliability target met on official tests",
                "a smoke receipt showing generated tests reach both revisions and every "
                "environment failure is excluded from rates",
                "a predeclared analysis with separate environment-failure accounting",
            ],
        },
    }
    audit = {
        "schema_version": SCHEMA,
        "label": "component audit and readiness; reads source only, opens no split",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classes": list(CLASSES),
        "components": components,
        "missing_components_now_implemented": MISSING,
        "checks_performed": [
            "fixed-code or gold-test leakage", "validation/sealed-data access",
            "fabricated expected values", "parser differences from the successor protocol",
            "raw-output loss", "candidate reranking", "candidate budgets",
            "deep-copy of mutable arguments", "filesystem/network/process safety",
            "simulated vs actual Atheris labelling"],
        "readiness": readiness,
    }
    args.output.write_bytes((json.dumps(audit, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({relative: entry["classification"]
                      for relative, entry in components.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
