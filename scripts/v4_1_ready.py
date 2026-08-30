"""Audit and materialize the GPU-ready Oneiros V4.1 execution queue.

This command is intentionally launch-safe: it can run CPU verification and
print Modal commands, but it never submits a GPU job and never exposes the
sealed final-test command.  Use it before every paid run so stale tests,
corpus drift, branch drift, or an unreviewed protocol change fails closed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import verify_corpus, write_json
from scripts.research_ablations import build_ablation_plan
from utils.reproducibility import source_tree_sha256


BRANCH = "experiment/research-eval-ablations"
BASELINE_COMMIT = "1d2cca8"
CORPUS_VERSION = "v4_1_research_hardened_candidate"
INTEGRATION_RUN = "v4_1_integration_32_seed42"
SELECTED_RUN = "v4_1_selected_candidate"


def _command(*parts: object) -> str:
    return " ".join(str(part) for part in parts)


def _run(arguments: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_commands() -> list[str]:
    return [
        "py -3.12 scripts/build_corpus_v4_1.py --workers 16 --offline",
        "py -3.12 scripts/audit_prompt_lineage.py --corpus-dir data/corpus/v4_1_research_hardened_candidate",
        "py -3.12 scripts/verify_v4_1_local.py",
        "py -3.12 scripts/audit_sft_readiness.py --corpus-version v4_1_research_hardened_candidate --split train --output results/v4_1_research_hardened_candidate_train_readiness.json",
        "py -3.12 scripts/research_ablations.py smoke --output results/v4_1_research_metrics_local_smoke.json",
        "py -3.12 scripts/preflight_sft_run.py --corpus-version v4_1_research_hardened_candidate --max-pairs 32 --epochs 1 --batch-size 1 --learning-rate 0.00005 --lr-scheduler-type constant_with_warmup --real-target-fraction 0.20 --repository-prompt-token-limit 1024 --repository-completion-token-limit 1024 --minimum-monitor-checkpoints 1 --min-function-kill-rate 0.58 --output results/v4_1_integration_32_preflight.json",
    ]


def integration_commands() -> dict[str, str]:
    common = (
        "--phase sft --corpus-version v4_1_research_hardened_candidate "
        "--run-name v4_1_integration_32_seed42 --seed 42 --max-pairs 32 "
        "--evaluation-split ablation_dev --sft-epochs 1 "
        "--sft-learning-rate 0.00005 --sft-lr-scheduler-type constant_with_warmup "
        "--sft-batch-size 1 --sft-repository-completion-token-limit 1024 "
        "--sft-min-monitor-checkpoints 1 --sft-monitor-validation-functions 32 "
        "--sft-monitor-patience 1 --sft-monitor-min-function-kill-rate 0.58"
    )
    return {
        "fresh": f"py -3.12 scripts/modal_train.py --fresh {common}",
        "resume": f"py -3.12 scripts/modal_train.py {common}",
    }


def selected_model_commands() -> dict[str, Any]:
    validation = [
        _command(
            "py -3.12 scripts/modal_train.py --phase sft_eval",
            f"--corpus-version {CORPUS_VERSION}",
            f"--run-name {SELECTED_RUN}",
            f"--seed {seed} --evaluation-split val",
        )
        for seed in (42, 43, 44)
    ]
    return {
        "locked_validation": validation,
        "dpo_if_all_completed_sft_seeds_pass_0_58": _command(
            "py -3.12 scripts/modal_train.py --phase dpo",
            f"--corpus-version {CORPUS_VERSION}",
            f"--run-name {SELECTED_RUN} --seed 42 --evaluation-split val",
        ),
        "final_test": {
            "status": "SEALED_NO_COMMAND_EMITTED",
            "reason": (
                "The one-time final command is created only after the selected adapter, "
                "generation settings, evaluator, candidate count, and signed decision record are frozen."
            ),
        },
    }


def build_execution_queue() -> dict[str, Any]:
    ablations = build_ablation_plan("v4_1_ablation", CORPUS_VERSION)
    return {
        "schema_version": "oneiros_v4_1_gpu_ready_queue_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "branch": BRANCH,
        "baseline_commit": BASELINE_COMMIT,
        "corpus_version": CORPUS_VERSION,
        "safety": {
            "gpu_auto_launch": False,
            "final_test_command_emitted": False,
            "locked_validation_used_for_ablation_selection": False,
            "dpo_gate": "all completed locked SFT seeds >= 0.58",
        },
        "stages": [
            {
                "id": "local_verification",
                "kind": "cpu",
                "commands": local_commands(),
                "must_pass_before_gpu": True,
            },
            {
                "id": "integration_32",
                "kind": "gpu_integration_not_research_result",
                "commands": integration_commands(),
                "promotion_gate": "terminal monitor and persisted artifacts complete",
            },
            {
                "id": "ablation_screening",
                "kind": "gpu_training_and_ablation_dev_evaluation",
                "plan": ablations,
                "selection_split": "ablation_dev",
            },
            {
                "id": "selected_model",
                "kind": "gpu_training_then_locked_validation",
                "commands": selected_model_commands(),
            },
            {
                "id": "native_repository_evaluation",
                "kind": "cpu_evaluation",
                "status": "HARNESS_IMPLEMENTED_NOT_VALIDATED_ON_REAL_PROJECTS",
                "harness": "harness.native_repository_eval",
                "commands": [
                    "py -3.12 scripts/evaluate_native_repository.py"
                    " --generated results/<run>/repository_generations.json"
                    " --bugsinpy-root data/bugsinpy_v2_ingestion/BugsInPy"
                    " --repository-cache data/bugsinpy_v2_ingestion/repositories"
                    " --output results/<run>/native_repository_eval.json"
                ],
                "prerequisite": (
                    "Provisioned BugsInPy checkouts and historical interpreters "
                    "(scripts/provision_bugsinpy_runtimes.py). No GPU required."
                ),
                "reporting_constraint": (
                    "Report executed_candidates alongside inconclusive_candidates. "
                    "Do not quote a real-repository kill rate until a real project "
                    "run exists."
                ),
            },
            {
                "id": "realistic_mutation_transfer",
                "kind": "benchmark_deferred",
                "status": "WAITING_FOR_STABLE_FROZEN_SFT_AND_LICENSED_HELD_OUT_DATA",
            },
        ],
    }


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def doctor(check_modal: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    branch = _run(["git", "branch", "--show-current"])
    branch_name = branch.stdout.strip()
    checks.append(_check("active_experiment_branch", branch.returncode == 0 and branch_name == BRANCH, branch_name))

    ancestor = _run(["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"])
    checks.append(_check("v4_baseline_is_ancestor", ancestor.returncode == 0, BASELINE_COMMIT))

    tracked = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    checks.append(_check("tracked_worktree_clean", tracked.returncode == 0 and not tracked.stdout.strip(), tracked.stdout.strip() or "clean"))

    corpus_dir = ROOT / "data" / "corpus" / CORPUS_VERSION
    try:
        manifest = verify_corpus(corpus_dir)
        checks.append(_check("canonical_corpus_hashes", True, manifest.get("corpus_sha256")))
    except Exception as error:  # fail-closed diagnostic
        manifest = {}
        checks.append(_check("canonical_corpus_hashes", False, str(error)))

    current_hash = source_tree_sha256(ROOT)
    test_path = ROOT / "results" / "v4_1_local_test_status.json"
    test_status = _json(test_path) if test_path.exists() else {}
    tests_current = (
        test_status.get("returncode") == 0
        and test_status.get("failed") == 0
        and test_status.get("source_tree_sha256") == current_hash
        and test_status.get("sealed_final_test_accessed") is False
    )
    checks.append(_check("source_bound_tests_current", tests_current, {
        "passed": test_status.get("passed"),
        "recorded_source_sha256": test_status.get("source_tree_sha256"),
        "current_source_sha256": current_hash,
    }))

    readiness_path = ROOT / "results" / f"{CORPUS_VERSION}_train_readiness.json"
    readiness = _json(readiness_path) if readiness_path.exists() else {}
    checks.append(_check("locked_train_readiness", readiness.get("ready") is True, readiness_path.as_posix()))

    preflight_path = ROOT / "results" / "v4_1_integration_32_preflight.json"
    preflight = _json(preflight_path) if preflight_path.exists() else {}
    preflight_current = (
        preflight.get("ready") is True
        and preflight.get("local_test_status", {}).get("source_tree_sha256") == current_hash
        and preflight.get("gates", {}).get("zero_sequence_overflows") is True
    )
    checks.append(_check("integration_preflight_current", preflight_current, preflight_path.as_posix()))

    frozen = _json(ROOT / "research" / "v4_1" / "FROZEN_EVALUATION_CONFIG.json")
    frozen_ok = (
        frozen.get("candidate_count") == 8
        and frozen.get("seeds") == [42, 43, 44]
        and frozen.get("design_selection_split") == "ablation_dev"
        and frozen.get("sealed_final_split") == "test"
        and frozen.get("locked_validation_gate_per_completed_seed") == 0.58
    )
    checks.append(_check("frozen_evaluation_protocol", frozen_ok, frozen.get("schema_version")))

    results = _json(ROOT / "ABLATION_RESULTS.json")
    checks.append(_check("sealed_final_test_unaccessed", results.get("final_test_accessed") is False, results.get("selection_status")))

    if check_modal:
        modal = _run([sys.executable, "-m", "modal", "profile", "current"])
        checks.append(_check("modal_authentication", modal.returncode == 0, modal.stdout.strip() or modal.stderr.strip()))

    hard_gates = {item["name"] for item in checks if item["name"] != "tracked_worktree_clean"}
    ready = all(item["passed"] for item in checks if item["name"] in hard_gates)
    return {
        "schema_version": "oneiros_v4_1_doctor_v1",
        "ready_for_32_pair_integration": ready,
        "tracked_changes_warning": bool(tracked.stdout.strip()),
        "modal_credit_balance_verified": False,
        "modal_credit_note": (
            "Modal has no universal free-credit gate in this script. Use "
            "scripts/modal_train_failover.py --dry-run with the account's actual credit limit."
        ),
        "sealed_final_test_accessed": False,
        "checks": checks,
    }


def run_cpu_commands(commands: Iterable[str]) -> None:
    for command in commands:
        print(f"\n>>> {command}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, shell=True, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Write the complete launch-safe execution queue")
    plan.add_argument("--output", type=Path, default=ROOT / "research" / "v4_1" / "GPU_READY_QUEUE.json")
    audit = sub.add_parser("doctor", help="Run read-only readiness checks")
    audit.add_argument("--check-modal", action="store_true")
    audit.add_argument("--output", type=Path)
    sub.add_parser("run-local", help="Run every CPU-safe verification and preflight command")
    sub.add_parser("show-integration", help="Print fresh and resume commands; never launch them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan":
        payload = build_execution_queue()
        write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
    elif args.command == "doctor":
        payload = doctor(args.check_modal)
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0 if payload["ready_for_32_pair_integration"] else 1
    elif args.command == "run-local":
        run_cpu_commands(local_commands())
    else:
        print(json.dumps(integration_commands(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
