"""Freeze the locked-validation measurement before it is allowed to run.

Locked validation is the last development-side measurement in this project.
After it, the only thing left is the sealed final test, which stays closed.
That makes this preflight the place where every degree of freedom is spent in
advance: the two arms, the protocol, the budgets, the evaluator, the split,
the output directories, and above all the decision rule.

The rule is written here, before any number exists, because a threshold chosen
after seeing the result is not a threshold. The four-arm development experiment
that precedes this one was decided by a rule frozen the same way, and it
returned a negative verdict; that only meant anything because the rule could
not move.

This preflight reads no validation record. Split identity is bound through the
development view's manifest - per-split content and record-id hashes - which
is metadata about the shard, not its payload. The sealed test is excluded by
construction: it is absent from the development view entirely.

CPU only. Launches nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import immutable_revision_for  # noqa: E402
from harness import model_identity as mid  # noqa: E402
from harness.successor_protocol import (  # noqa: E402
    SUCCESSOR_PROTOCOL, contract_source_hashes, protocol_sha256,
)
from harness.source_identity import (  # noqa: E402
    EVALUATION_DEFINING_SOURCES, HASH_SCHEME_VERSION, canonical_sha256,
    scheme_block as source_identity_scheme_block,
)
from utils.reproducibility import source_tree_sha256  # noqa: E402

SCHEMA_VERSION = "oneiros_locked_validation_preflight_v1"

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CORPUS_VERSION = "v4_1_research_hardened_candidate"
EVALUATION_SPLIT = "val"
SEALED_SPLIT = "test"

#: The files that decide what a generation/evaluation run actually does. A
#: whole-tree hash is the wrong test here: it moves when preflight or reporting
#: code is edited, which either blocks honest tooling work or teaches people to
#: wave real drift through. The comparison is made file by file.
RUNTIME_COMPONENTS = (
    "scripts/train_on_dataset.py",
    "engine/generator.py",
    "engine/model_runtime.py",
    "engine/prompt_budget.py",
    "engine/test_generation_prompt.py",
    "harness/candidate_policy.py",
    "harness/safe_execution.py",
    "harness/corpus_view.py",
    "harness/successor_protocol.py",
    "metrics/research_evaluation.py",
    "config/settings.py",
)

#: Arm B of this measurement. Selected by the frozen four-arm development
#: experiment; its adapter hash is pinned so a different checkpoint cannot be
#: substituted after the rule was written.
#:
#: It is evaluated WHERE IT LIVES, through --adapter-dir. Nothing is copied,
#: moved, or written into this directory, and no marker or metadata file is
#: created anywhere to make an evaluation look like a training run.
ARM_A_431_RUN = "local_sft_armA_baseline_successor_s42"
ARM_A_431_ADAPTER_DIR = f"checkpoints/{ARM_A_431_RUN}/sft_adapter"
ARM_A_431_ADAPTER = f"{ARM_A_431_ADAPTER_DIR}/adapter_model.safetensors"
ARM_A_431_ADAPTER_SHA256 = (
    "e67dd599a37cbf2a738791c5cdf889cb4938a2ad53c991dca2fefae503b6f9e7"
)

DEVELOPMENT_SELECTION_RECEIPT = "results/v4_2_development_selection_receipt.json"
FROZEN_DEVELOPMENT_RECEIPT = "results/v4_2_frozen_development_evaluation_receipt.json"

#: Output run names. Deliberately distinct from every development run name, so
#: a locked-validation run cannot land on top of a development artifact.
BASE_RUN_NAME = "locked_val_base_qwen_s42"
ARM_A_RUN_NAME = "locked_val_armA_ckpt431_s42"

DEVELOPMENT_RUN_NAMES = (
    "local_base_qwen_ablationdev_successor_s42_v3",
    "local_sft_armA_baseline_successor_s42",
    "local_sft_armB_o1_nocollide_s42",
    "local_eval_armA_ckpt150_matched",
)

#: Predeclared decision rule. Every threshold below is chosen before any
#: locked-validation number exists, and may not be changed afterwards.
DECISION_RULE = {
    "frozen_before_results": True,
    "comparison": "Arm A checkpoint 431 vs the immutable base model, paired on the val panel",
    "all_criteria_must_hold": True,
    "criteria": {
        "1_practical_kill_gain": {
            "kill_at_8_absolute_points": ">= +3.0",
            "net_functions_gained": ">= +15",
            "paired_mcnemar_two_sided_p": "< 0.01",
            "why": (
                "On development, A@431 beat base by +9.04 Kill@8 points with p=8.2e-05. "
                "A genuine effect of that size must survive an unseen split comfortably. "
                "The bar is set above the development-stage +2.0/+10/p<0.05 because this "
                "is the promotion decision, not a screening decision, and because "
                "ablation_dev selected this checkpoint and so flattered it."
            ),
        },
        "2_reference_validity": {
            "max_absolute_drop_points": 12.0,
            "why": (
                "This is the criterion that matters most and the one most likely to fail. "
                "Every SFT arm lost about 11 points of reference validity against base on "
                "development: A@431 -11.05, A@150 -10.70, B@150 -11.67. The threshold is "
                "set at 12.0 so that the known regression does not by itself block "
                "promotion, while a materially worse one does. This is a deliberate, "
                "declared tolerance of a real defect, not a claim that the defect is "
                "acceptable in the finished system. If A@431 is promoted, it is promoted "
                "WITH this regression and the regression must be reported every time the "
                "model is described."
            ),
        },
        "3_execution_and_parse": {
            "max_absolute_drop_points_each": 1.0,
            "why": "A model that emits less runnable code than base is not an improvement.",
        },
        "4_diversity": {
            "max_relative_drop_exact_unique": 0.10,
            "max_relative_drop_input_shape_unique": 0.10,
            "max_relative_increase_redundancy": 0.10,
            "why": "Kills bought by emitting the same assertion repeatedly are not kills.",
        },
        "5_integrity": {
            "raw_output_present_and_hash_matching": "100%",
            "prompt_budget_failures": 0,
            "max_completion_limit_rate": 0.01,
            "both_arms_share_run_contract_sha256": True,
            "both_arms_share_evaluation_scope_sha256": True,
        },
    },
    "if_all_criteria_hold": "Promote Arm A checkpoint 431 as the final candidate.",
    "if_any_criterion_fails": (
        "Retain the immutable base model as the final candidate. Do not retrain, do not "
        "tune prompts, do not change thresholds, do not add data, and do not select a "
        "different checkpoint."
    ),
    "forbidden_after_seeing_results": [
        "retraining of any kind",
        "prompt changes",
        "threshold changes",
        "adding or reweighting data",
        "selecting a different checkpoint",
        "re-running an arm to obtain a different sample",
        "opening the sealed final test",
    ],
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


def view_problems(view: dict) -> list[str]:
    """Everything that can be wrong with a corpus view, independent of disk.

    Kept separate from :func:`collect` so the sealed-split refusal can be
    tested for what it is - a rule about a manifest - rather than by standing
    up a whole fake repository and hoping the crash arrives after the check.
    """
    found: list[str] = []
    if SEALED_SPLIT in view.get("included_splits", []):
        found.append("development view includes the sealed split")
    if sorted(view.get("sealed_splits_excluded", [])) != [SEALED_SPLIT]:
        found.append("development view does not declare the sealed split excluded")
    if EVALUATION_SPLIT not in view.get("splits", {}):
        found.append(f"development view has no {EVALUATION_SPLIT!r} shard")
    return found


#: Placeholder written into the commands on the first pass, then replaced once
#: the receipt's own hash is known. A receipt whose commands must quote that
#: receipt's SHA-256 cannot state it in one pass: writing the hash changes the
#: bytes being hashed. So the file is written once with this marker, hashed,
#: and rewritten with the marker replaced - which leaves the final file's hash
#: equal to the value its own commands carry.
RECEIPT_SHA_PLACEHOLDER = "<<FROZEN_PREFLIGHT_RECEIPT_SHA256>>"


def resolved_commands(receipt: dict, receipt_sha256: str) -> dict[str, list[str]]:
    """The stored commands with the receipt's own digest substituted in.

    A receipt cannot state its own SHA-256 - writing the digest changes the
    bytes being hashed - so the stored command carries a placeholder and this
    puts the real value in. Every consumer goes through here, so the launch
    command and the tested command are the same string.
    """
    return {
        name: [receipt_sha256 if part == RECEIPT_SHA_PLACEHOLDER else part
               for part in arm["command"]]
        for name, arm in (receipt.get("arms") or {}).items()
    }


def evaluation_command(run_name: str, *, adapter_run: str | None,
                       revision: str, receipt_path: str,
                       receipt_sha256: str) -> list[str]:
    """The exact command for one arm. Built here so it cannot drift.

    Every option the protocol does not own is written out, including the model
    and tokenizer revision, which the development commands left to the pinned
    default. The options the protocol DOES own - parse mode, raw-output
    retention, seed, completion limit, sequence limit - are deliberately absent:
    SUCCESSOR_PROTOCOL.OWNED_CLI_OPTIONS makes passing them beside
    --successor-protocol a conflict rather than an override, precisely so a
    value typed twice cannot disagree with the receipt that names the protocol.
    Those values are resolved and recorded in `resolved_generation_settings`
    instead, and checked against the runtime by assert_runtime_matches().
    """
    phase = "sft_eval" if adapter_run else "base_eval"
    command = [
        ".venv-gpu/Scripts/python.exe", "scripts/train_on_dataset.py",
        "--phase", phase,
        "--run-name", run_name,
        "--successor-protocol",
        "--evaluation-split", EVALUATION_SPLIT,
        "--prompt-information-variant", "full",
        "--output-instruction-variant", "self_contained",
        "--base-model-name", BASE_MODEL,
        "--base-model-revision", revision,
        "--attention-implementation", "sdpa",
        "--sft-prompt-token-limit", "1024",
        "--frozen-preflight-receipt", receipt_path,
        "--expected-preflight-receipt-sha256", receipt_sha256,
    ]
    if adapter_run:
        command += [
            "--adapter-dir", ARM_A_431_ADAPTER_DIR,
            "--expected-adapter-sha256", ARM_A_431_ADAPTER_SHA256,
            "--adapter-source-run", adapter_run,
        ]
    return command


def collect(problems: list[str], receipt_path: str = "results/v4_2_locked_validation_preflight.json") -> dict:
    """Build the receipt, appending to ``problems`` for anything that refuses."""
    revision = immutable_revision_for(BASE_MODEL)
    if not mid.is_immutable_revision(revision):
        problems.append(
            f"base model revision is not immutable: {mid.describe_revision(revision)}"
        )

    identity = mid.build(
        model_name=BASE_MODEL, model_revision=revision or "",
        tokenizer_name=BASE_MODEL, tokenizer_revision=revision or "",
        source_tree_sha256=source_tree_sha256(ROOT),
        protocol_name=SUCCESSOR_PROTOCOL["protocol_name"],
        protocol_sha256=protocol_sha256(),
    )
    problems.extend(mid.problems(identity, label="locked validation"))

    # ---- adapter identity ------------------------------------------------
    adapter_path = ROOT / ARM_A_431_ADAPTER
    if not adapter_path.is_file():
        problems.append(f"Arm A adapter is missing: {ARM_A_431_ADAPTER}")
        adapter_sha = None
    else:
        adapter_sha = sha256_file(adapter_path)
        if adapter_sha != ARM_A_431_ADAPTER_SHA256:
            problems.append(
                "Arm A adapter hash does not match the selected development checkpoint: "
                f"expected {ARM_A_431_ADAPTER_SHA256}, found {adapter_sha}"
            )

    # ---- the development decision this measurement inherits --------------
    sel_path = ROOT / DEVELOPMENT_SELECTION_RECEIPT
    selection = None
    if not sel_path.is_file():
        problems.append(f"development selection receipt is missing: {DEVELOPMENT_SELECTION_RECEIPT}")
    else:
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
        if selection["decision"]["selected_development_sft_candidate"] != "Arm A checkpoint 431":
            problems.append("development selection receipt does not select Arm A checkpoint 431")
        if selection["decision"]["promote_b_at_150"] is not False:
            problems.append("development selection receipt does not reject Arm B")

    # ---- corpus and split identity, without reading a val record ---------
    corpus_dir = ROOT / "data" / "corpus" / CORPUS_VERSION
    view_manifest_path = corpus_dir / "development_view" / "manifest.json"
    if not view_manifest_path.is_file():
        problems.append("development corpus view manifest is missing")
        view = {}
    else:
        view = json.loads(view_manifest_path.read_text(encoding="utf-8"))
        problems.extend(view_problems(view))

    val_descriptor = view.get("splits", {}).get(EVALUATION_SPLIT, {})

    # ---- output directories cannot collide with development artifacts ----
    for run_name in (BASE_RUN_NAME, ARM_A_RUN_NAME):
        if run_name in DEVELOPMENT_RUN_NAMES:
            problems.append(f"output run name collides with a development run: {run_name}")
        for parent in ("results", "checkpoints"):
            target = ROOT / parent / run_name
            if target.exists():
                problems.append(f"output directory already exists and would be reused: {parent}/{run_name}")

    # ---- protocol and evaluator identity ---------------------------------
    contracts = contract_source_hashes(ROOT)
    runtime = {name: sha256_file(ROOT / name) for name in RUNTIME_COMPONENTS}

    # ---- the runner decides which weights are loaded ---------------------
    from scripts import train_on_dataset as runner
    from harness.successor_protocol import assert_runtime_matches
    from config import model_config

    runtime_drift = assert_runtime_matches(ROOT)
    problems.extend(runtime_drift)

    # Values the protocol owns are not passed as flags (that would be a
    # conflict); they are resolved from the code that will execute and
    # recorded here, so nothing is left to a silent default.
    resolved_generation_settings = {
        "seed": SUCCESSOR_PROTOCOL["generation_seed"],
        "candidates_per_function": runner.TESTS_PER_PAIR,
        "candidate_parse_mode": SUCCESSOR_PROTOCOL["candidate_parse_mode"],
        "retain_raw_output": SUCCESSOR_PROTOCOL["retain_raw_output"],
        "allow_test_function_candidates": SUCCESSOR_PROTOCOL["allow_test_function_candidates"],
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "do_sample": True,
        "generation_completion_token_limit": SUCCESSOR_PROTOCOL["function_generation_completion_limit"],
        "prompt_token_limit": 1024,
        "max_sequence_tokens": SUCCESSOR_PROTOCOL["max_sequence_tokens"],
        "max_assertions": SUCCESSOR_PROTOCOL["max_assertions"],
        "base_model_name": BASE_MODEL,
        "base_model_revision": revision,
        "tokenizer_name": BASE_MODEL,
        "tokenizer_revision": revision,
        "attention_implementation": "sdpa",
        "resolution": (
            "Options owned by the named protocol are set by --successor-protocol "
            "and verified against the runtime by assert_runtime_matches(); every "
            "other option is written out explicitly in the commands above."
        ),
    }
    expected = {
        "seed": 42, "candidates_per_function": 8, "candidate_parse_mode": "whole_output",
        "retain_raw_output": True, "temperature": 0.7, "top_p": 0.9,
        "generation_completion_token_limit": 1024, "prompt_token_limit": 1024,
        "max_sequence_tokens": 3072,
    }
    for key, want in expected.items():
        if resolved_generation_settings[key] != want:
            problems.append(
                f"resolved generation setting {key} is "
                f"{resolved_generation_settings[key]!r}, expected {want!r}")

    runner_binding = {
        "entrypoint": "scripts/train_on_dataset.py",
        "runner_source_sha256": sha256_file(ROOT / "scripts" / "train_on_dataset.py"),
        "runner_source_canonical_sha256": canonical_sha256(
            ROOT / "scripts" / "train_on_dataset.py"),
        "adapter_resolution_source_sha256": runner._adapter_resolution_source_sha256(),
        "adapter_resolution_source_canonical_sha256":
            runner._adapter_resolution_source_canonical_sha256(),
        "adapter_required_files": list(runner.EXTERNAL_ADAPTER_REQUIRED_FILES),
        "why": (
            "The evaluator hash says how a candidate was scored. It says nothing "
            "about whether the intended adapter was the one on the GPU. The runner "
            "and the adapter-resolution logic decide that, so both are frozen here "
            "and both are written into every run contract."
        ),
    }

    # ---- portable source binding -----------------------------------------
    # Raw hashes prove which bytes this GPU machine loaded; canonical hashes
    # can be re-derived from any checkout. Three evaluation-defining files are
    # CRLF here and LF in Git, so without the second family a fresh clone
    # would appear to disagree with this receipt about code that never changed.
    source_binding = source_identity_scheme_block(ROOT)
    git_blob_mismatches = []
    for role, relative in sorted(EVALUATION_DEFINING_SOURCES.items()):
        stored = git("rev-parse", f"HEAD:{relative}")
        computed = source_binding["sources"][role]["git_blob_sha1"]
        source_binding["sources"][role]["git_blob_sha1_committed"] = stored or None
        matches = bool(stored) and stored == computed
        source_binding["sources"][role]["matches_committed_content"] = matches
        if not matches:
            git_blob_mismatches.append(relative)
    source_binding["working_tree_matches_committed_content"] = not git_blob_mismatches
    source_binding["uncommitted_or_divergent_sources"] = git_blob_mismatches
    # Everything needed to re-identify this measurement's code and weights, in
    # one place, so a reader never has to assemble it from three sections.
    source_binding["git_commit"] = git("rev-parse", "HEAD")
    source_binding["adapter_sha256"] = adapter_sha
    source_binding["adapter_source_path"] = ARM_A_431_ADAPTER_DIR
    source_binding["base_model_revision"] = revision
    source_binding["historical_artifacts_use_the_earlier_raw_only_scheme"] = {
        "applies_to": [
            FROZEN_DEVELOPMENT_RECEIPT,
            DEVELOPMENT_SELECTION_RECEIPT,
            "results/local_base_qwen_ablationdev_successor_s42_v3/",
            "results/local_sft_armA_baseline_successor_s42/",
            "results/local_sft_armB_o1_nocollide_s42/",
            "results/local_eval_armA_ckpt150_matched/",
        ],
        "note": (
            "Development artifacts predate this scheme and record raw on-machine "
            "SHA-256 only. They are not rewritten: their hashes are correct for the "
            "machine that produced them, and altering a frozen receipt to adopt a "
            "newer convention would destroy the thing that made it evidence. Read "
            "them as raw-only, and compare them against a working tree with the "
            "same line endings."
        ),
    }
    if git_blob_mismatches:
        problems.append(
            "evaluation-defining sources differ from committed content (or are "
            f"uncommitted): {git_blob_mismatches}. Locked validation must run from "
            "code that can be recovered from the repository.")

    frozen_dev_path = ROOT / FROZEN_DEVELOPMENT_RECEIPT
    dev_contract_drift = []
    if frozen_dev_path.is_file():
        dev = json.loads(frozen_dev_path.read_text(encoding="utf-8"))
        dev_files = {
            "evaluator": "metrics/research_evaluation.py",
            "candidate_policy": "harness/candidate_policy.py",
            "timeout_policy_safe_execution": "harness/safe_execution.py",
            "prompt_builder": "engine/test_generation_prompt.py",
            "prompt_budget": "engine/prompt_budget.py",
            "generator": "engine/generator.py",
            "model_runtime": "engine/model_runtime.py",
        }
        for key, rel in dev_files.items():
            if sha256_file(ROOT / rel) != dev["contract_source_hashes"][key]:
                dev_contract_drift.append(rel)
        if dev_contract_drift:
            problems.append(
                "evaluator/prompt/policy sources changed since the development "
                f"measurement: {dev_contract_drift}. Locked validation would not be "
                "comparable with the development results it inherits."
            )

    porcelain = git("status", "--porcelain")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": (
            "Freeze the locked-validation measurement of exactly two arms before it runs. "
            "The sealed final test remains closed."
        ),
        "launched": False,
        "cpu_only_preflight": True,
        "model_identity": identity,
        "arms": {
            "A_base_control": {
                "kind": "immutable base model, no adapter",
                "run_name": BASE_RUN_NAME,
                "model_name": BASE_MODEL,
                "model_revision": revision,
                "adapter": None,
                "command": evaluation_command(
                    BASE_RUN_NAME, adapter_run=None, revision=revision or "",
                    receipt_path=receipt_path, receipt_sha256=RECEIPT_SHA_PLACEHOLDER),
            },
            "B_arm_a_checkpoint_431": {
                "kind": "selected development SFT candidate",
                "run_name": ARM_A_RUN_NAME,
                "model_name": BASE_MODEL,
                "model_revision": revision,
                "adapter_provenance": "external_evaluation_adapter",
                "adapter_source_run": ARM_A_431_RUN,
                "adapter_source_path": ARM_A_431_ADAPTER_DIR,
                "adapter_checkpoint_step": 431,
                "adapter_sha256_expected": ARM_A_431_ADAPTER_SHA256,
                "adapter_sha256_found": adapter_sha,
                "command": evaluation_command(
                    ARM_A_RUN_NAME, adapter_run=ARM_A_431_RUN, revision=revision or "",
                    receipt_path=receipt_path, receipt_sha256=RECEIPT_SHA_PLACEHOLDER),
                "note": (
                    "The adapter is read where it already lives, through --adapter-dir. "
                    "Nothing is copied or moved, the source directory is not written to, "
                    "and no marker or metadata file is created in it or in the output "
                    "directory. --expected-adapter-sha256 is checked before the model is "
                    "loaded, so the wrong weights are refused rather than measured."
                ),
                "staging_required": False,
                "source_directory_modified": False,
            },
        },
        "receipt_self_reference": {
            "placeholder": RECEIPT_SHA_PLACEHOLDER,
            "note": (
                "Each arm's command carries --expected-preflight-receipt-sha256 with "
                "this placeholder in place of a digest. A file cannot contain its own "
                "SHA-256, because writing the digest changes the bytes being hashed, so "
                "no fixed point exists. Substitute this receipt's SHA-256 - the value "
                "the preflight prints, and the hash of this file as written - for the "
                "placeholder before running. scripts/preflight_locked_validation.py's "
                "resolved_commands() does exactly that, and is what the tests execute, "
                "so the tested command and the launch command are the same string."
            ),
        },
        "resolved_generation_settings": resolved_generation_settings,
        "runner_binding": runner_binding,
        "source_identity": source_binding,
        "protocol_runtime_drift": runtime_drift,
        "protocol": dict(
            SUCCESSOR_PROTOCOL,
            protocol_sha256=protocol_sha256(),
            evaluation_split=EVALUATION_SPLIT,
            final_test_measurement=False,
            reranking="none - candidates are judged as generated",
            feedback_rounds=0,
            diversity_mode="none",
        ),
        "contract_source_hashes": contracts,
        "runtime_component_hashes": runtime,
        "contract_sources_match_development_measurement": not dev_contract_drift,
        "corpus": {
            "corpus_version": CORPUS_VERSION,
            "source_corpus_id": view.get("source_corpus_id"),
            "source_records_sha256": view.get("source_records_sha256"),
            "source_splits_sha256": view.get("source_splits_sha256"),
            "included_splits": view.get("included_splits"),
            "sealed_splits_excluded": view.get("sealed_splits_excluded"),
        },
        "evaluation_split_identity": {
            "split": EVALUATION_SPLIT,
            "shard_filename": val_descriptor.get("filename"),
            "shard_sha256": val_descriptor.get("sha256"),
            "record_ids_sha256": val_descriptor.get("record_ids_sha256"),
            "record_count": val_descriptor.get("record_count"),
            "records_read_by_this_preflight": 0,
            "binding_method": (
                "split identity is bound through the development view manifest's content "
                "and record-id hashes, which are metadata about the shard. No validation "
                "record was opened."
            ),
        },
        "sealed_final_test": {
            "accessed": False,
            "split": SEALED_SPLIT,
            "status": "sealed - not opened, not inspected, absent from the development view",
        },
        "output_isolation": {
            "base_run_name": BASE_RUN_NAME,
            "arm_a_run_name": ARM_A_RUN_NAME,
            "development_run_names_that_must_not_be_touched": list(DEVELOPMENT_RUN_NAMES),
            "collides_with_development_artifacts": False,
        },
        "inherited_development_decision": {
            "receipt": DEVELOPMENT_SELECTION_RECEIPT,
            "receipt_sha256": sha256_file(sel_path) if sel_path.is_file() else None,
            "selected_candidate": (selection or {}).get("decision", {}).get("selected_development_sft_candidate"),
            "arm_b_rejected": (selection or {}).get("decision", {}).get("arm_b_o1_sidecar"),
        },
        "decision_rule": DECISION_RULE,
        "reproducibility": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(porcelain),
            "git_status_porcelain_sha256": hashlib.sha256(porcelain.encode()).hexdigest(),
            "source_tree_sha256": source_tree_sha256(ROOT),
            "python_version": sys.version.split()[0],
        },
        "expected_gpu_time": {
            "development_panel_reference": "542 functions took 643-884 s per arm on an RTX 4500 Ada",
            "val_shard_record_count": val_descriptor.get("record_count"),
            "note": (
                "The evaluated function count is decided at run time after repository "
                "records and ineligible records are held out; on ablation_dev, 594 shard "
                "records yielded 542 evaluated functions. It is NOT predicted here."
            ),
            "estimate_per_arm_seconds": [900, 1300],
            "estimate_total_minutes": [30, 45],
        },
        "preflight_problems": problems,
        "ready_to_launch": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/v4_2_locked_validation_preflight.json")
    args = parser.parse_args()

    problems: list[str] = []
    receipt = collect(problems, receipt_path=args.output)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    # A refused preflight must not overwrite the receipt already on disk.
    # Running this after locked validation completed - the runs' output
    # directories now exist, so it refuses - previously still wrote the refusal
    # over the committed receipt that the locked-validation result receipt
    # points at, destroying immutable evidence as a side effect of a read-only
    # question. A refusal now reports and writes nothing.
    if problems and out.exists():
        print("=" * 96)
        print("LOCKED-VALIDATION PREFLIGHT - REFUSED, existing receipt left untouched")
        print("=" * 96)
        print(f"receipt : {args.output} (unchanged, sha256 {sha256_file(out)})")
        for p in problems:
            print(f"  - {p}")
        return 1

    # The commands must quote this receipt's SHA-256, and a file cannot contain
    # its own hash: writing the digest changes the bytes being hashed, so no
    # fixed point exists. Pretending otherwise - iterating until the numbers
    # "settle" - would just be a loop that never converges.
    #
    # So the stored commands carry a placeholder, and the digest is substituted
    # when the command is used. resolved_commands() below does that
    # substitution, main() prints the runnable form, and the tests run it. The
    # receipt says plainly that this is what the placeholder means.
    out.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    receipt_sha = sha256_file(out)

    print("=" * 96)
    print("LOCKED-VALIDATION PREFLIGHT (CPU only; nothing was launched)")
    print("=" * 96)
    print(f"receipt : {args.output}")
    print(f"sha256  : {receipt_sha}")
    print()
    print(f"base model      : {BASE_MODEL} @ {receipt['model_identity']['base_model_revision']}")
    print(f"arm A adapter   : {receipt['arms']['B_arm_a_checkpoint_431']['adapter_sha256_found']}")
    print(f"protocol        : {receipt['protocol']['protocol_name']} {protocol_sha256()}")
    print(f"split           : {EVALUATION_SPLIT}  shard {receipt['evaluation_split_identity']['shard_sha256']}")
    print(f"val records     : {receipt['evaluation_split_identity']['record_count']} "
          f"(0 read by this preflight)")
    print(f"sealed test     : {receipt['sealed_final_test']['status']}")
    print(f"output runs     : {BASE_RUN_NAME}, {ARM_A_RUN_NAME}")
    print()
    if problems:
        print(f"REFUSED - {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("ALL GATES PASSED - ready to launch on explicit approval.")
    print()
    print("Exact commands (receipt digest substituted for the stored placeholder):")
    for name, command in resolved_commands(receipt, receipt_sha).items():
        print(f"\n# {name}")
        print("  " + " \\\n    ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
