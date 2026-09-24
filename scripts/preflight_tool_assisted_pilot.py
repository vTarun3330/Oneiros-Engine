"""Frozen CPU design receipt for the execution-feedback pilot (Phase 9).

Run on a clean committed tree; the receipt is committed on its own and the
runner accepts only a HEAD that differs from the receipt's commit by the
receipt alone.

Checks, all before any GPU work:

* the frozen control adapter matches its training result;
* the panel receipt verifies and resolves completely through admission;
* for every panel record the canonical prompt built from the PERMITTED view
  is byte-identical to the one built from the full record, so no hidden
  field can influence what the model sees;
* every canonical prompt fits its budget;
* generation settings are the unchanged successor settings;
* budgets for B and C are identical by construction;
* every protected-data flag is false.

No model weights are loaded; the tokenizer is.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.execution_supervision_sidecar import sha256_file
from harness.source_identity import canonical_sha256

SCHEMA = "oneiros_tool_assisted_design_receipt_v1"
RECEIPT = ROOT / "results" / "v4_3_tool_assisted_design_receipt.json"
BOUND_SOURCES = (
    "harness/buggy_side_execution.py", "harness/execution_feedback.py",
    "harness/tool_assisted_generation.py", "harness/safe_execution.py",
    "harness/candidate_policy.py", "harness/generation_adapter.py",
    "harness/generation_rng.py", "harness/prompt_factory.py",
    "harness/rehearsal_evaluator.py", "harness/sealed_final_evaluator.py",
    "harness/evaluation_admission.py", "harness/parallel_execution.py",
    "metrics/research_evaluation.py", "engine/generator.py",
    "engine/test_generation_prompt.py", "engine/prompt_budget.py",
    "scripts/freeze_tool_assisted_panel.py", "scripts/run_tool_assisted_pilot.py",
    "scripts/analyse_tool_assisted_pilot.py", "scripts/audit_tool_assisted_components.py",
    "scripts/preflight_tool_assisted_pilot.py",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def preflight() -> dict:
    from harness.generation_adapter import build_prompts, successor_settings
    from harness.prompt_factory import prompt_factory
    from harness.safe_execution import DEFAULT_TIMEOUT_SECONDS
    from harness.tool_assisted_generation import (
        FINAL_SLOTS, FINAL_SLOT_POLICY, INITIAL_SAMPLES, LOOP_VERSION, MAX_ECHO_CHARS,
        MAX_REPAIRS_PER_TARGET, SECOND_ROUND_SLOTS, SEQUENCES_PER_TARGET, permitted_view,
    )
    from harness.execution_feedback import FEEDBACK_SCHEMA_VERSION, REPAIRABLE, RETAINED
    from scripts import analyse_tool_assisted_pilot as analysis
    from scripts.run_tool_assisted_pilot import CONTROL_ADAPTER, CONTROL_RESULT, PANEL, \
        load_panel_scope

    problems: list[str] = []
    status = git("status", "--porcelain")
    if status:
        problems.append("working tree is not clean")
    control = json.loads(CONTROL_RESULT.read_text(encoding="utf-8"))
    if sha256_file(CONTROL_ADAPTER / "adapter_model.safetensors") != control["adapter_sha256"]:
        problems.append("frozen control adapter differs from its training result")
    panel, scope, _ = load_panel_scope()
    if not all(panel["checks"].values()):
        problems.append("panel disjointness checks do not all pass")

    settings = successor_settings()
    if settings.problems():
        problems.append(f"invalid generation settings: {settings.problems()}")
    build_prompt = prompt_factory(settings.prompt_settings())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name,
                                              revision=settings.base_model_revision)
    views, mismatches = [], []
    for record in scope.eligible:
        view = permitted_view(record)
        views.append(view)
        if build_prompt(view) != build_prompt(record):
            mismatches.append(str(record["id"]))
    if mismatches:
        problems.append(f"prompt from permitted view differs for {len(mismatches)} records")
    token_ids, generable, failures = build_prompts(tokenizer, views, settings, build_prompt)
    if failures or len(generable) != len(views):
        problems.append(f"canonical prompt budget failures: {len(failures)}")
    lengths = sorted(len(ids) for ids in token_ids)

    records = panel["records"]
    per_record = 577.753 / 542  # measured canonical Kill@8 seconds per record
    estimate = {
        "basis": {"canonical_kill_at_8_seconds_per_record": per_record,
                  "assumed_single_sequence_generation_seconds": "0.8-2.5",
                  "note": "B/C round 2 is 8 single-sequence calls per record plus repairs"},
        "arm_a_minutes": round(records * per_record * 1.2 / 60, 1),
        "arms_b_c_generation_minutes": {"low": round(records * 10 / 60),
                                        "high": round(records * 30 / 60)},
        "score_b_and_c_cpu_minutes": round(2 * records * 0.35 / 60, 1),
        "gpu": "one local RTX 4500 Ada; stages strictly sequential",
        "storage_megabytes": {"loop_lineage_files": round(records * 0.06, 1),
                              "journal": round(records * 0.04, 1),
                              "three_evaluation_artifacts": 12},
    }
    return {
        "schema_version": SCHEMA,
        "ready": not problems,
        "problems": problems,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
                "clean": not status},
        "label": "CPU design receipt; no generation has been run",
        "question": "does bounded execution feedback improve test generation beyond the "
                    "same inference compute spent on resampling?",
        "arms": {
            "A": "frozen control adapter, unchanged successor protocol, 8 samples, no "
                 "feedback (canonical historical control)",
            "B": "same adapter; shared round 1 of 8 + 8 single-sequence canonical "
                 "resamples; same outcome-blind final-slot policy as C; no diagnostics",
            "C": "same adapter; shared round 1 of 8 + 8 single-sequence slots, each a "
                 "repair of a demonstrably invalid round-1 candidate or else the identical "
                 "canonical sample B receives",
            "primary_comparison": "C versus B",
        },
        "adapter": {"path": CONTROL_ADAPTER.relative_to(ROOT).as_posix(),
                    "sha256": control["adapter_sha256"]},
        "generation_settings": settings.to_dict(),
        "budgets": {"initial_samples": INITIAL_SAMPLES, "second_round_slots": SECOND_ROUND_SLOTS,
                    "sequences_per_target_b_and_c": SEQUENCES_PER_TARGET,
                    "max_repairs_per_target": MAX_REPAIRS_PER_TARGET,
                    "max_new_tokens_per_sequence": settings.generation_completion_token_limit,
                    "final_slots": FINAL_SLOTS, "max_echo_chars": MAX_ECHO_CHARS,
                    "arm_a_sequences_per_target": settings.candidates_per_function,
                    "compute_matched": "B and C: identical sequences, calls and token cap "
                                       "per target by construction; A is not compute-"
                                       "matched and is secondary"},
        "executor_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "final_slot_policy": FINAL_SLOT_POLICY,
        "feedback": {"schema_version": FEEDBACK_SCHEMA_VERSION, "repairable": list(REPAIRABLE),
                     "retained_silently": list(RETAINED), "loop_version": LOOP_VERSION},
        "panel": {"path": PANEL.relative_to(ROOT).as_posix(), "sha256": sha256_file(PANEL),
                  "records": panel["records"], "lineages": panel["lineages"],
                  "record_ids_sha256": panel["record_ids_sha256"],
                  "qualification": panel["qualification"],
                  "prompt_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2],
                                    "max": lengths[-1]},
                  "prompt_from_permitted_view_identical": not mismatches},
        "gates": {
            "min_gain_pp": analysis.MIN_GAIN_PP,
            "validity_margin_pp": analysis.VALIDITY_MARGIN_PP,
            "max_diversity_loss": analysis.MAX_DIVERSITY_LOSS,
            "max_wall_ratio": analysis.MAX_WALL_RATIO,
            "interval": "lineage-cluster percentile bootstrap, 90%, "
                        f"{analysis.BOOTSTRAP_REPLICATES} replicates, seed "
                        f"{analysis.BOOTSTRAP_SEED}",
            "pass": "Kill@8 C-B >= +5 pp with lower bound > 0; reference-valid per requested "
                    "lower bound >= -3 pp; exact-unique ratio loss <= 0.05; C wall <= 1.5x B; "
                    "compute matched",
            "fail": "Kill@8 C-B upper bound < +5 pp, or reference-valid upper bound < -3 pp",
            "inconclusive": "anything else; does not pass",
            "syntax_only_cannot_pass": "the gate is on Kill@8; parse and execution rates "
                                       "never gate",
        },
        "stopping_rules": [
            "stages run strictly sequentially: generate-a, generate-bc, score-b, score-c, "
            "then the frozen analysis on the CPU",
            "the analysis refuses partial results and uncontrolled compute",
            "an infrastructure crash resumes from the journal; no completed model call is "
            "repeated and no setting changes",
            "no rerun with changed prompts, budgets, taxonomy or thresholds",
        ],
        "rollback_rule": "no weights are written and nothing is promoted; if any integrity "
                         "check fails after the run, all artifacts are preserved, the result "
                         "is declared invalid and the frozen control remains the reference",
        "confirmation_condition": "only a PASS on this pilot permits a REQUEST for explicit "
                                  "authorization to design a confirmation on untouched data; "
                                  "the 100 unopened confirmation lineages were reserved for "
                                  "the execution-supervision line and are not opened by this "
                                  "experiment",
        "runtime_and_storage_estimate": estimate,
        "source_files_sha256": {path: canonical_sha256(ROOT / path) for path in BOUND_SOURCES},
        "tracked_artifacts_sha256": {
            path: sha256_file(ROOT / path) for path in (
                "results/v4_3_tool_assisted_panel.json",
                "results/v4_3_tool_assisted_audit.json")},
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False,
                    "canonical_records_json_opened": False,
                    "fixed_code_in_loop": False, "gold_tests_in_loop": False},
        "generation_launched": False,
        "next_permitted_step": ("stop for explicit approval before any GPU generation"
                                if not problems else "fix preflight problems; no GPU work"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECEIPT)
    args = parser.parse_args(argv)
    receipt = preflight()
    args.output.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: receipt[key] for key in (
        "ready", "problems", "panel", "runtime_and_storage_estimate")}, indent=2)[:3000])
    return 0 if receipt["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
