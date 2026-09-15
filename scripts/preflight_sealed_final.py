"""Freeze the one-time sealed final measurement. Opens nothing.

The sealed split is the only measurement here that cannot be repeated. Once it
has been seen, every later choice is contaminated by having seen it - so
everything a result depends on has to be frozen first, and the freezing itself
must not require looking.

Two consequences shape this file.

**The candidate is the base model.** Locked validation retained it: Arm A@431
gained +3.43 Kill@8 points on val but at McNemar p=0.0783 against a required
p<0.01, and it cost 11.87 points of reference validity. So the sealed run
measures the immutable base model as the Oneiros final candidate, with no
adapter. ``harness.sealed_final`` still requires an ``adapter_sha256``, and the
honest value is the evaluator's own convention for a base arm,
sha256("<model>@<revision>") - which keeps "which weights answered" answerable
rather than null.

**Split identity is bound without enumerating the sealed split.** The corpus
manifest already records a digest over ``splits.json`` and one over
``records.json``. Those bind membership for every split at once: any change to
which ids are sealed changes them. That is a stronger binding than a test-only
digest and needs no sealed access, so this preflight reads no sealed record,
enumerates no sealed id, and hashes no sealed payload.

No token is issued here. Authorization is a separate, explicit act; a preflight
that minted its own token would defeat the one-time protection it exists to
set up. The guard machinery is instead rehearsed against mock bundles in a
temporary directory, proving the refusals work before anything real depends on
them.

CPU only. Launches nothing. Opens nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import immutable_revision_for  # noqa: E402
from harness.comparison_contract import base_model_adapter_identity  # noqa: E402
from harness.sealed_final import (  # noqa: E402
    REQUIRED_BUNDLE_FIELDS, SEALED_SPLIT, Authorization, FinalBundle,
    SealedAccessError, SealedFinalGuard, issue_authorization,
    refuse_sealed_split_for_development,
)
from harness.source_identity import (  # noqa: E402
    EVALUATION_DEFINING_SOURCES, HASH_SCHEME_VERSION, canonical_sha256,
    git_blob_sha1, raw_sha256,
)
from harness.successor_protocol import SUCCESSOR_PROTOCOL, protocol_sha256  # noqa: E402

SCHEMA_VERSION = "oneiros_sealed_final_readiness_v1"

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CORPUS_VERSION = "v4_1_research_hardened_candidate"
FINAL_RUN_NAME = "sealed_final_base_qwen_s42"
FINAL_ENTRYPOINT = "scripts/run_sealed_final_test.py"

#: Committed evidence this decision rests on. Hashed, not summarised.
DECISION_ARTIFACTS = (
    "results/v4_2_frozen_development_evaluation_receipt.json",
    "results/v4_2_development_selection_receipt.json",
    "results/v4_2_locked_validation_preflight.json",
    "results/v4_2_locked_validation_result_receipt.json",
)

#: Baselines the final measurement is reported against.
BASELINE_ARTIFACTS = (
    "results/v4_2_baseline_bundle_val.json",
    "results/v4_2_atheris_tasks_val.manifest.json",
)

#: Run names that must never be written by the sealed run.
PROTECTED_RUN_NAMES = (
    "local_base_qwen_ablationdev_successor_s42_v3",
    "local_sft_armA_baseline_successor_s42",
    "local_sft_armB_o1_nocollide_s42",
    "local_eval_armA_ckpt150_matched",
    "locked_val_base_qwen_s42",
    "locked_val_armA_ckpt431_s42",
)

PROHIBITIONS_AFTER_EXECUTION = [
    "retraining of any kind",
    "prompt changes",
    "threshold changes",
    "adding, reweighting or re-selecting data",
    "selecting a different checkpoint or model",
    "re-running the sealed final test",
    "re-scoring the sealed final result under a different parser or policy",
]


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def build_bundle(problems: list[str]) -> tuple[FinalBundle, dict]:
    """The frozen configuration, assembled without touching the sealed split."""
    revision = immutable_revision_for(BASE_MODEL)
    if not revision or len(revision) != 40:
        problems.append(f"base model revision is not an immutable SHA: {revision!r}")

    view = json.loads((ROOT / "data" / "corpus" / CORPUS_VERSION
                       / "development_view" / "manifest.json").read_text(encoding="utf-8"))
    if SEALED_SPLIT in (view.get("included_splits") or []):
        problems.append("development view includes the sealed split")

    protocol = SUCCESSOR_PROTOCOL
    fields = {
        # The candidate. No adapter: the base model is the final candidate.
        "adapter_path": "none - the immutable base model is the final candidate",
        "adapter_sha256": base_model_adapter_identity(BASE_MODEL, revision or ""),
        "adapter_source_tree_sha256": git("rev-parse", "HEAD"),
        "base_model_name": BASE_MODEL,
        "base_model_revision": revision,
        # Corpus and split membership, bound by manifest digests rather than by
        # reading the sealed shard.
        "corpus_version": CORPUS_VERSION,
        "corpus_records_sha256": view.get("source_records_sha256"),
        "split_ids_sha256": view.get("source_splits_sha256"),
        # Generation and scoring.
        "prompt_schema_version": "oneiros_unified_test_generation_v2",
        "prompt_budgets": {
            "prompt_token_limit": 1024,
            "repository_prompt_token_limit": 1024,
            "generation_completion_token_limit": protocol["function_generation_completion_limit"],
            "max_sequence_tokens": protocol["max_sequence_tokens"],
        },
        "candidates_per_target": protocol["candidates_per_function"],
        "seeds": {"generation_seed": protocol["generation_seed"]},
        "sampling": {
            "temperature": protocol["temperature"],
            "top_p": protocol["top_p"],
            "do_sample": True,
            "candidate_parse_mode": protocol["candidate_parse_mode"],
            "retain_raw_output": protocol["retain_raw_output"],
            "reranking": "none",
            "feedback_rounds": 0,
            "diversity_mode": "none",
        },
        "timeout_seconds": {
            "policy_source_canonical_sha256": canonical_sha256(
                ROOT / "harness/safe_execution.py"),
        },
        "evaluator_version": {
            "protocol_name": protocol["protocol_name"],
            "protocol_sha256": protocol_sha256(),
            "evaluator_canonical_sha256": canonical_sha256(
                ROOT / "metrics/research_evaluation.py"),
            "source_identity_scheme": HASH_SCHEME_VERSION,
        },
        "baseline_versions": {
            rel: sha256_file(ROOT / rel) for rel in BASELINE_ARTIFACTS
            if (ROOT / rel).is_file()
        },
        "checkpoint_selection_rule": (
            "No checkpoint is selected. Locked validation retained the immutable "
            "base model: Arm A@431 gained +3.4346 Kill@8 points on val with "
            "McNemar exact p=0.0783 against a required p<0.01, and lost 11.8726 "
            "points of reference validity. The final candidate is therefore the "
            "base model with no adapter, per "
            "results/v4_2_locked_validation_result_receipt.json."
        ),
    }
    bundle = FinalBundle(fields=fields)
    missing = bundle.missing_fields()
    if missing:
        problems.append(f"bundle is not fully frozen; missing: {missing}")
    if not fields["baseline_versions"]:
        problems.append("no baseline artifacts were found to pin")
    return bundle, view


def rehearse_with_mocks() -> dict:
    """Prove the guard refuses what it must, using mock bundles only.

    Nothing here touches the real corpus, the real state file or the real audit
    log. If any refusal below stopped working, the sealed split could be opened
    twice, or opened against a bundle edited after authorization - which is the
    entire failure mode the guard exists to prevent.
    """
    results: dict = {}
    mock_fields = {name: f"mock-{name}" for name in REQUIRED_BUNDLE_FIELDS}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        guard = SealedFinalGuard(tmp_path / "state.json", tmp_path / "audit.log")
        frozen = FinalBundle(fields=mock_fields)

        unfrozen = FinalBundle(fields={**mock_fields, "adapter_sha256": ""})
        try:
            issue_authorization(unfrozen, "mock", "mock")
            results["unfrozen_bundle_refused"] = False
        except SealedAccessError:
            results["unfrozen_bundle_refused"] = True

        try:
            guard.open_sealed_split(SEALED_SPLIT, frozen, None, "mock-no-token")
            results["missing_token_refused"] = False
        except SealedAccessError:
            results["missing_token_refused"] = True

        try:
            guard.open_sealed_split(SEALED_SPLIT, frozen, "not-a-real-token", "mock")
            results["unknown_token_refused"] = False
        except SealedAccessError:
            results["unknown_token_refused"] = True

        auth = issue_authorization(frozen, "mock-issuer", "mock rehearsal")
        guard.register(auth)
        try:
            guard.open_sealed_split(SEALED_SPLIT, frozen, auth.token, "mock-run")
            results["valid_token_granted_once"] = True
        except SealedAccessError:
            results["valid_token_granted_once"] = False

        try:
            guard.open_sealed_split(SEALED_SPLIT, frozen, auth.token, "mock-rerun")
            results["spent_token_refused"] = False
        except SealedAccessError:
            results["spent_token_refused"] = True

        auth2 = issue_authorization(frozen, "mock-issuer", "second mock")
        guard.register(auth2)
        edited = FinalBundle(fields={**mock_fields, "seeds": "tampered-after-issue"})
        try:
            guard.open_sealed_split(SEALED_SPLIT, edited, auth2.token, "mock-tampered")
            results["bundle_edited_after_authorization_refused"] = False
        except SealedAccessError:
            results["bundle_edited_after_authorization_refused"] = True

        try:
            refuse_sealed_split_for_development(SEALED_SPLIT, "scripts/train_on_dataset.py")
            results["development_command_refused"] = False
        except SealedAccessError:
            results["development_command_refused"] = True

        guard.open_sealed_split("val", frozen, None, "mock-dev-split")
        results["development_split_needs_no_token"] = True
        results["audit_entries_written"] = len(guard.audit_entries())
    results["all_refusals_hold"] = all(
        v for k, v in results.items() if isinstance(v, bool))
    return results


def collect(problems: list[str]) -> dict:
    bundle, view = build_bundle(problems)

    sources = {
        role: {
            "path": rel,
            "raw_sha256": raw_sha256(ROOT / rel),
            "canonical_sha256": canonical_sha256(ROOT / rel),
            "git_blob_sha1": git_blob_sha1(ROOT / rel),
        }
        for role, rel in sorted(EVALUATION_DEFINING_SOURCES.items())
    }
    uncommitted = [
        item["path"] for role, item in sources.items()
        if git("rev-parse", f"HEAD:{item['path']}") != item["git_blob_sha1"]
    ]
    if uncommitted:
        problems.append(
            f"evaluation-defining sources differ from committed content: {uncommitted}")

    entry = ROOT / FINAL_ENTRYPOINT
    if not entry.is_file():
        problems.append(f"authorized final entrypoint is missing: {FINAL_ENTRYPOINT}")

    for run_name in PROTECTED_RUN_NAMES:
        if run_name == FINAL_RUN_NAME:
            problems.append(f"final run name collides with an existing run: {run_name}")
    for parent in ("results", "checkpoints"):
        if (ROOT / parent / FINAL_RUN_NAME).exists():
            problems.append(f"final output directory already exists: {parent}/{FINAL_RUN_NAME}")

    decision = {}
    for rel in DECISION_ARTIFACTS:
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"decision artifact missing: {rel}")
            continue
        decision[rel] = sha256_file(path)

    locked = json.loads(
        (ROOT / "results/v4_2_locked_validation_result_receipt.json").read_text(encoding="utf-8"))
    if locked.get("promote_arm_a_431") is not False:
        problems.append("locked validation does not record Arm A@431 as rejected")

    rehearsal = rehearse_with_mocks()
    if not rehearsal.get("all_refusals_hold"):
        problems.append(f"mock guard rehearsal did not hold: {rehearsal}")

    porcelain = git("status", "--porcelain")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": (
            "Freeze the one-time sealed final measurement of the selected final "
            "candidate. This preflight opens nothing and authorizes nothing."),
        "sealed_split_accessed": False,
        "sealed_records_read": 0,
        "sealed_ids_enumerated": 0,
        "sealed_payload_hashed": False,
        "split_identity_binding": (
            "Split membership is bound by the corpus manifest digests over "
            "records.json and splits.json, which cover every split at once. No "
            "sealed record was read, no sealed id enumerated, and no sealed "
            "payload hashed."),
        "authorization_token_issued": False,
        "token_policy": (
            "A preflight that minted its own token would defeat the one-time "
            "protection it exists to set up. Authorization is a separate explicit "
            "act, performed once, against this bundle hash."),
        "final_candidate": {
            "model": BASE_MODEL,
            "model_revision": bundle.fields["base_model_revision"],
            "tokenizer": BASE_MODEL,
            "tokenizer_revision": bundle.fields["base_model_revision"],
            "adapter": None,
            "adapter_identity_digest": bundle.fields["adapter_sha256"],
            "why": (
                "Locked validation retained the base model. Arm A@431 is rejected "
                "for promotion; the O1 sidecar / Arm B was rejected on development."),
        },
        "frozen_bundle": bundle.to_dict(),
        "bundle_sha256": bundle.sha256(),
        "evaluation_defining_sources": sources,
        "source_identity_scheme": HASH_SCHEME_VERSION,
        "decision_evidence": decision,
        "protected_run_names": list(PROTECTED_RUN_NAMES),
        "output_isolation": {
            "final_run_name": FINAL_RUN_NAME,
            "results_dir": f"results/{FINAL_RUN_NAME}",
            "checkpoints_dir": f"checkpoints/{FINAL_RUN_NAME}",
            "collides_with_existing_runs": False,
            "may_not_write_into": list(PROTECTED_RUN_NAMES),
        },
        "authorized_entrypoint": {
            "path": FINAL_ENTRYPOINT,
            "canonical_sha256": canonical_sha256(entry) if entry.is_file() else None,
            "raw_sha256": raw_sha256(entry) if entry.is_file() else None,
            "separate_from_development_entrypoint": True,
            "development_entrypoint": "scripts/train_on_dataset.py",
        },
        "mock_guard_rehearsal": rehearsal,
        "prohibited_after_execution": PROHIBITIONS_AFTER_EXECUTION,
        "interpretation_limits": [
            "The sealed final test measures the immutable base model. No SFT model "
            "was promoted, so no Oneiros SFT generalization claim follows from it.",
            "Repository records are excluded from every kill rate; no real native "
            "repository result is claimed.",
            "SFT improved development Kill@8 and produced a positive but "
            "statistically insufficient locked-validation Kill@8 result, alongside a "
            "large reference-validity regression.",
        ],
        "reproducibility": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(porcelain),
            "python_version": sys.version.split()[0],
        },
        "preflight_problems": problems,
        "ready_for_authorization": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/v4_2_sealed_final_readiness_receipt.json")
    args = parser.parse_args()

    problems: list[str] = []
    receipt = collect(problems)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))

    print("=" * 96)
    print("SEALED-FINAL READINESS PREFLIGHT (CPU only; nothing opened, nothing authorized)")
    print("=" * 96)
    print(f"receipt        : {args.output}")
    print(f"sha256         : {sha256_file(out)}")
    print(f"bundle_sha256  : {receipt['bundle_sha256']}")
    print()
    print(f"final candidate: {BASE_MODEL} @ {receipt['final_candidate']['model_revision']}")
    print(f"adapter        : none (base model is the final candidate)")
    print(f"identity digest: {receipt['final_candidate']['adapter_identity_digest']}")
    print(f"entrypoint     : {FINAL_ENTRYPOINT}")
    print(f"output run     : {FINAL_RUN_NAME}")
    print(f"sealed access  : accessed={receipt['sealed_split_accessed']} "
          f"records_read={receipt['sealed_records_read']} "
          f"payload_hashed={receipt['sealed_payload_hashed']}")
    print(f"token issued   : {receipt['authorization_token_issued']}")
    print(f"mock rehearsal : all refusals hold = "
          f"{receipt['mock_guard_rehearsal']['all_refusals_hold']}")
    print()
    if problems:
        print(f"NOT READY - {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("READY FOR AUTHORIZATION - no sealed data was touched. Awaiting explicit approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
