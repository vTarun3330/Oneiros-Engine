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
from scripts.research_ablations import (
    ADMISSIBLE_PROMPT_BUDGETS, INTEGRATION_PROMPT_BUDGET, build_ablation_plan,
)
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


def _validate_budget(prompt_token_limit: int) -> None:
    if prompt_token_limit not in ADMISSIBLE_PROMPT_BUDGETS:
        raise ValueError("Prompt budget must be a predeclared admissible budget: 1024 or 1280")


def integration_preflight_path(prompt_token_limit: int) -> Path:
    _validate_budget(prompt_token_limit)
    return ROOT / "results" / f"v4_1_integration_32_p{prompt_token_limit}_preflight.json"


def integration_preflight_command(prompt_token_limit: int = INTEGRATION_PROMPT_BUDGET) -> str:
    output = integration_preflight_path(prompt_token_limit).relative_to(ROOT).as_posix()
    return (
        "py -3.12 scripts/preflight_sft_run.py"
        f" --corpus-version {CORPUS_VERSION} --max-pairs 32 --epochs 1 --batch-size 1"
        " --learning-rate 0.00005 --lr-scheduler-type constant_with_warmup"
        " --real-target-fraction 0.20 --repository-prompt-token-limit 1024"
        " --repository-completion-token-limit 1024 --minimum-monitor-checkpoints 1"
        " --min-function-kill-rate 0.58 --evaluation-split ablation_dev"
        f" --prompt-token-limit {prompt_token_limit} --output {output}"
    )


def local_commands(prompt_token_limit: int = INTEGRATION_PROMPT_BUDGET) -> list[str]:
    return [
        "py -3.12 scripts/build_corpus_v4_1.py --workers 16 --offline",
        "py -3.12 scripts/audit_prompt_lineage.py --corpus-dir data/corpus/v4_1_research_hardened_candidate",
        "py -3.12 scripts/verify_v4_1_local.py",
        "py -3.12 scripts/audit_sft_readiness.py --corpus-version v4_1_research_hardened_candidate --split train --output results/v4_1_research_hardened_candidate_train_readiness.json",
        "py -3.12 scripts/research_ablations.py smoke --output results/v4_1_research_metrics_local_smoke.json",
        integration_preflight_command(prompt_token_limit),
    ]


def integration_commands(prompt_token_limit: int = INTEGRATION_PROMPT_BUDGET) -> dict[str, str]:
    _validate_budget(prompt_token_limit)
    common = (
        "--phase sft --corpus-version v4_1_research_hardened_candidate "
        f"--run-name {INTEGRATION_RUN}_p{prompt_token_limit} --seed 42 --max-pairs 32 "
        f"--sft-prompt-token-limit {prompt_token_limit} --sft-real-target-fraction 0.20 "
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


def selected_model_commands(prompt_token_limit: int = INTEGRATION_PROMPT_BUDGET) -> dict[str, Any]:
    _validate_budget(prompt_token_limit)
    validation = [
        _command(
            "py -3.12 scripts/modal_train.py --phase sft_eval",
            f"--corpus-version {CORPUS_VERSION}",
            f"--run-name {SELECTED_RUN}_p{prompt_token_limit}",
            f"--sft-prompt-token-limit {prompt_token_limit}",
            f"--seed {seed} --evaluation-split val",
        )
        for seed in (42, 43, 44)
    ]
    return {
        "status": "TEMPLATES_ONLY_NO_CONFIGURATION_SELECTED",
        "requires_before_execution": (
            "Frozen group-J budget decision and all other ablation decisions; "
            "train and confirm the selected configuration on ablation_dev first. "
            "Regenerate these templates if the accepted budget differs."
        ),
        "locked_validation": validation,
        "dpo_if_all_completed_sft_seeds_pass_0_58": _command(
            "py -3.12 scripts/modal_train.py --phase dpo",
            f"--corpus-version {CORPUS_VERSION}",
            f"--run-name {SELECTED_RUN}_p{prompt_token_limit} --seed 42 --evaluation-split val",
            f"--sft-prompt-token-limit {prompt_token_limit}",
        ),
        "final_test": {
            "status": "SEALED_NO_COMMAND_EMITTED",
            "reason": (
                "The one-time final command is created only after the selected adapter, "
                "generation settings, evaluator, candidate count, and signed decision record are frozen."
            ),
        },
    }


def build_execution_queue(prompt_token_limit: int = INTEGRATION_PROMPT_BUDGET) -> dict[str, Any]:
    _validate_budget(prompt_token_limit)
    ablations = build_ablation_plan(
        "v4_1_ablation", CORPUS_VERSION, screening_prompt_token_limit=prompt_token_limit,
    )
    return {
        "schema_version": "oneiros_v4_1_gpu_ready_queue_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "branch": BRANCH,
        "baseline_commit": BASELINE_COMMIT,
        "corpus_version": CORPUS_VERSION,
        "prompt_budget": {
            "integration_and_candidate_templates": prompt_token_limit,
            "status": "NOT_A_SELECTED_RESEARCH_WINNER",
            "frozen_runtime_defaults_modified": False,
            "group_J_runs_before_other_ablation_groups": True,
        },
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
                "commands": local_commands(prompt_token_limit),
                "must_pass_before_gpu": True,
            },
            {
                "id": "integration_32",
                "kind": "gpu_integration_not_research_result",
                "commands": integration_commands(prompt_token_limit),
                "preflight_command": integration_preflight_command(prompt_token_limit),
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
                "commands": selected_model_commands(prompt_token_limit),
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


def integration_preflight_mismatches(
    preflight: dict[str, Any], current_hash: str, manifest: dict[str, Any],
    prompt_token_limit: int,
) -> list[str]:
    """Reject green but wrong-scope/stale evidence, not just ready=false."""
    _validate_budget(prompt_token_limit)
    expected = {
        "ready": True,
        "local_test_status.source_tree_sha256": current_hash,
        "local_test_status.returncode": 0,
        "local_test_status.failed": 0,
        "local_test_status.sealed_final_test_accessed": False,
        "corpus.version": CORPUS_VERSION,
        "corpus.corpus_id": manifest.get("corpus_id"),
        "selection.requested_pairs": 32,
        "selection.retained_pairs": 32,
        "selection.execution_mode_filter": None,
        "evaluation_panel.evaluation_split": "ablation_dev",
        "evaluation_panel.prompt_token_limit": prompt_token_limit,
        "evaluation_panel.prompt_budget_failures": 0,
        "sampling.target_real_fraction": 0.20,
        "sampling.balanced_sampling_enabled": True,
        "sampling.synthetic_balance_fraction": 0.0,
        "sampling.synthetic_balance_mode": "none",
        "tokenization.prompt_token_limit": prompt_token_limit,
        "tokenization.repository_prompt_token_limit": 1024,
        "tokenization.completion_token_limit": 128,
        "tokenization.repository_completion_token_limit": 1024,
        "tokenization.sequence_token_limit": 2048,
        "tokenization.prompt_information_variant": "full",
        "tokenization.output_instruction_variant": "self_contained",
        "training.epochs": 1,
        "training.batch_size": 1,
        "training.learning_rate": 0.00005,
        "training.lr_scheduler_type": "constant_with_warmup",
        "training.min_function_kill_rate": 0.58,
        "training.optimizer_schedule.minimum_monitor_checkpoints": 1,
        "gates.zero_sequence_overflows": True,
        "gates.evaluation_panel_fully_promptable": True,
        "gates.terminal_checkpoint_monitor_enabled": True,
        "gates.minimum_monitor_schedule_reached": True,
    }
    for filename, field in (
        ("records.json", "records_sha256"), ("splits.json", "splits_sha256"),
        ("ablation_dev_manifest.json", "ablation_dev_sha256"),
        ("leakage_audit.json", "leakage_audit_sha256"),
    ):
        expected[f"corpus.{field}"] = manifest.get("files", {}).get(filename, {}).get("sha256")
    mismatches = []
    missing = object()
    for field, required in expected.items():
        actual: Any = preflight
        for key in field.split("."):
            actual = actual.get(key, missing) if isinstance(actual, dict) else missing
        if actual is missing or actual != required:
            mismatches.append(field)
    if any(not value for value in preflight.get("gates", {}).values()):
        mismatches.append("gates_not_all_passed")
    if not preflight.get("evaluation_panel", {}).get("function_records", 0):
        mismatches.append("empty_evaluation_panel")
    panel = preflight.get("evaluation_panel", {})
    if panel.get("function_records") != panel.get("promptable_function_records"):
        mismatches.append("evaluation_panel_count_mismatch")
    if not preflight.get("training", {}).get("planned_validation_checkpoints"):
        mismatches.append("missing_terminal_monitor_schedule")
    return mismatches


def doctor(
    check_modal: bool = False, prompt_token_limit: int = INTEGRATION_PROMPT_BUDGET,
) -> dict[str, Any]:
    _validate_budget(prompt_token_limit)
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

    preflight_path = integration_preflight_path(prompt_token_limit)
    preflight = _json(preflight_path) if preflight_path.exists() else {}
    mismatches = integration_preflight_mismatches(
        preflight, current_hash, manifest, prompt_token_limit,
    )
    checks.append(_check("integration_preflight_current", not mismatches, {
        "path": preflight_path.as_posix(), "scope_mismatches": mismatches,
    }))

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
        "integration_prompt_token_limit": prompt_token_limit,
        "integration_run_name": f"{INTEGRATION_RUN}_p{prompt_token_limit}",
        "research_configuration_selected": False,
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
    local = sub.add_parser("run-local", help="Run every CPU-safe verification and preflight command")
    show = sub.add_parser("show-integration", help="Print fresh and resume commands; never launch them")
    for command_parser in (plan, audit, local, show):
        command_parser.add_argument(
            "--prompt-token-limit", type=int, choices=ADMISSIBLE_PROMPT_BUDGETS,
            default=INTEGRATION_PROMPT_BUDGET,
            help="Explicit integration/candidate budget, not a selected research winner",
        )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan":
        payload = build_execution_queue(args.prompt_token_limit)
        write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
    elif args.command == "doctor":
        payload = doctor(args.check_modal, args.prompt_token_limit)
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0 if payload["ready_for_32_pair_integration"] else 1
    elif args.command == "run-local":
        run_cpu_commands(local_commands(args.prompt_token_limit))
    else:
        print(json.dumps(integration_commands(args.prompt_token_limit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
