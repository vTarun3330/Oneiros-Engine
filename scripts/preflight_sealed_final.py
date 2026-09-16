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
from harness.sealed_final_evaluator import (  # noqa: E402
    EVALUATOR_VERSION, environment_problems, evaluator_source_hashes,
)
from harness.generation_rng import (  # noqa: E402
    SEED_APPLICATION_VERSION, SEEDED_GENERATORS,
)
from harness.successor_protocol import SUCCESSOR_PROTOCOL, protocol_sha256  # noqa: E402

SCHEMA_VERSION = "oneiros_sealed_final_readiness_v6"

#: Where this preflight writes by default, and the only receipt the entrypoint
#: accepts. It is derived from, never independent of, the version above: the v5
#: receipt shipped an ``exact_command`` naming the v4 file because that path was
#: hardcoded in one place and the schema bumped in another.
DEFAULT_RECEIPT_PATH = "results/v4_2_sealed_final_executable_receipt_v6.json"

#: v1 receipts were emitted before the final evaluator existed. The first
#: sealed entrypoint called the guard, spent the token, and only then reached
#: a comment saying the measurement was unimplemented - so a v1 receipt could
#: have been read as executable authorization for a run that could not produce
#: a result. v2 states executability explicitly and the entrypoint refuses any
#: receipt that does not.
SUPERSEDED_SCHEMA_VERSIONS = (
    "oneiros_sealed_final_readiness_v1",
    "oneiros_sealed_final_readiness_v2",
    "oneiros_sealed_final_readiness_v3",
    "oneiros_sealed_final_readiness_v4",
    #: v5 bound the prompt correctly, but its own ``exact_command`` named the v4
    #: receipt - a path the entrypoint refuses - and its frozen bundle carried a
    #: Git commit id in a field called ``adapter_source_tree_sha256``.
    "oneiros_sealed_final_readiness_v5",
)

BASE_MODEL ="Qwen/Qwen2.5-Coder-1.5B-Instruct"
CORPUS_VERSION = "v4_1_research_hardened_candidate"
FINAL_RUN_NAME = "sealed_final_base_qwen_s42"
FINAL_ENTRYPOINT = "scripts/run_sealed_final_test.py"
FINAL_EVALUATOR = "harness/sealed_final_evaluator.py"
SEALED_LOADER = "harness/sealed_final_loader.py"
GENERATION_ADAPTER = "harness/generation_adapter.py"
SMOKE_MODULE = "harness/sealed_final_smoke.py"
RNG_MODULE = "harness/generation_rng.py"
PROMPT_FACTORY_MODULE = "harness/prompt_factory.py"

#: Committed evidence this decision rests on. Hashed, not summarised.
DECISION_ARTIFACTS = (
    "results/v4_2_frozen_development_evaluation_receipt.json",
    "results/v4_2_development_selection_receipt.json",
    "results/v4_2_locked_validation_preflight.json",
    "results/v4_2_locked_validation_result_receipt.json",
)

#: Baseline scope, chosen explicitly. Option B of the two offered.
#:
#: The v3 bundle pinned val baseline artifacts - Atheris task manifests and a
#: baseline bundle - inside a SEALED-FINAL freeze. Those baselines were never
#: run on sealed targets under the sealed budget, so retaining them there would
#: have implied a comparison the measurement cannot support. That is exactly the
#: silent retention that turns a single-arm result into an apparent benchmark.
#:
#: So the sealed final test is scoped to the immutable-base Oneiros candidate
#: alone. No baseline is bundled, none is claimed, and the receipt says so.
#: Running frozen baseline runners on the sealed targets would be option A and
#: would require its own freeze, its own rehearsals, and its own authorization.
BASELINE_SCOPE = "oneiros_immutable_base_only_no_comparative_baseline"
BASELINE_ARTIFACTS: tuple = ()
BASELINE_SCOPE_STATEMENT = (
    "This sealed final test measures the immutable-base Oneiros candidate only. "
    "No baseline runner is included in the bundle and none is executed on the "
    "sealed targets. The result therefore CANNOT support any comparative claim "
    "against Atheris or any other baseline. Development and locked-validation "
    "baseline artifacts exist for other splits and must not be presented "
    "alongside this result as if they were sealed-final comparisons."
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
    from harness.generation_adapter import successor_settings
    _settings = successor_settings()
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
        # A Git commit id, named as one. Filling a ``*_sha256`` field from
        # ``git rev-parse HEAD`` labelled a 40-hex SHA-1 as a SHA-256 digest of
        # a source tree, which it never was.
        "candidate_source_tree_git_commit": git("rev-parse", "HEAD"),
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
        "generation_batch_size": _settings.generation_batch_size,
        "allow_test_function_candidates": _settings.allow_test_function_candidates,
        "prompt_information_variant": _settings.prompt_information_variant,
        "output_instruction_variant": _settings.output_instruction_variant,
        "attention_implementation": _settings.attention_implementation,
        "tokenizer_name": BASE_MODEL,
        "tokenizer_revision": revision,
        "seeds": {
            "generation_seed": protocol["generation_seed"],
            "seed_application_version": SEED_APPLICATION_VERSION,
            "seeded_generators": list(SEEDED_GENERATORS),
            "applied_immediately_before_generation": True,
        },
        "baseline_scope": BASELINE_SCOPE,
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
            "baseline_scope": BASELINE_SCOPE,
            "baselines_bundled": [],
            "comparative_claims_supported": False,
            "statement": BASELINE_SCOPE_STATEMENT,
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
    if fields["baseline_versions"].get("comparative_claims_supported") is not False:
        problems.append("the bundle must not claim comparative baseline support")
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


def collect(problems: list[str], output_path: str = DEFAULT_RECEIPT_PATH) -> dict:
    """Build the receipt that will be written to ``output_path``.

    The path is threaded in rather than assumed, so the receipt's own
    ``exact_command`` can name the file it is about to become.
    """
    output_path = str(output_path).replace("\\", "/")
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

    # ---- is the measurement actually runnable? ---------------------------
    evaluator_problems: list[str] = []
    try:
        evaluator_source = evaluator_source_hashes()
        evaluator_source["loader_module"] = SEALED_LOADER
        evaluator_source["loader_canonical_sha256"] = canonical_sha256(ROOT / SEALED_LOADER)
        evaluator_source["loader_raw_sha256"] = raw_sha256(ROOT / SEALED_LOADER)
        evaluator_source["entrypoint_canonical_sha256"] = canonical_sha256(ROOT / FINAL_ENTRYPOINT)
        evaluator_source["adapter_module"] = GENERATION_ADAPTER
        evaluator_source["adapter_canonical_sha256"] = canonical_sha256(ROOT / GENERATION_ADAPTER)
        evaluator_source["smoke_module"] = SMOKE_MODULE
        evaluator_source["smoke_canonical_sha256"] = canonical_sha256(ROOT / SMOKE_MODULE)
        evaluator_source["rng_module"] = RNG_MODULE
        evaluator_source["rng_canonical_sha256"] = canonical_sha256(ROOT / RNG_MODULE)
        evaluator_source["seed_application_version"] = SEED_APPLICATION_VERSION
        from harness.sealed_final_smoke import SMOKE_VERSION
        evaluator_source["smoke_version"] = SMOKE_VERSION
        from harness.prompt_factory import (
            PROMPT_FACTORY_VERSION, prompt_factory_source_hashes,
        )
        evaluator_source["prompt_factory_module"] = PROMPT_FACTORY_MODULE
        evaluator_source["prompt_factory_version"] = PROMPT_FACTORY_VERSION
        evaluator_source["prompt_binding"] = prompt_factory_source_hashes(ROOT)
        from harness.generation_adapter import successor_settings as _succ
        _prompt_settings = _succ().prompt_settings()
        evaluator_source["frozen_prompt_settings"] = _prompt_settings.to_dict()
        if _prompt_settings.problems():
            evaluator_problems.append(
                f"frozen prompt settings invalid: {_prompt_settings.problems()}")
        from harness.generation_adapter import ADAPTER_VERSION, successor_settings
        evaluator_source["adapter_version"] = ADAPTER_VERSION
        _settings = successor_settings()
        evaluator_source["frozen_generation_settings"] = _settings.to_dict()
        if _settings.candidate_parse_mode != "whole_output":
            evaluator_problems.append(
                "frozen settings do not use whole_output parsing")
        if _settings.generation_batch_size != 2:
            evaluator_problems.append(
                f"frozen generation_batch_size is {_settings.generation_batch_size}; "
                "locked validation used 2 and batch shape changes sampling")
        if _settings.seed != 42:
            evaluator_problems.append(f"frozen seed is {_settings.seed}, expected 42")
        if _settings.problems():
            evaluator_problems.append(f"frozen settings invalid: {_settings.problems()}")
    except Exception as exc:  # noqa: BLE001
        evaluator_source = {}
        evaluator_problems.append(f"final evaluator source is unavailable: {exc!r}")

    try:
        from harness import sealed_final_loader as _loader
        for symbol in ("sealed_records", "sealed_batch_generator", "select_split_records",
                       "adapt_records", "build_sealed_generator"):
            if not callable(getattr(_loader, symbol, None)):
                evaluator_problems.append(f"sealed loader is missing {symbol}")
    except Exception as exc:  # noqa: BLE001
        evaluator_problems.append(f"sealed loader is not importable: {exc!r}")

    # Resolve every symbol the generation path will call. Importability of the
    # loader proved nothing before, because the broken imports sat inside a
    # function body and would only have raised after the token was spent.
    try:
        from engine.generator import Phi3Generator
        from engine.prompt_budget import compact_unified_user_prompt
        from engine.test_generation_prompt import build_unified_user_prompt, format_chat_prompt
        from scripts.train_on_dataset import build_pair_prompt, _record_to_pair
        from harness.generation_adapter import generate_candidate_slots
        from harness.sealed_final_smoke import (
            prepare_generator, run_model_smoke, smoke_problems, synthetic_batch)
        from harness.generation_rng import seed_generation_rngs
        for name, obj in (("Phi3Generator._parse_output", getattr(Phi3Generator, "_parse_output", None)),
                          ("compact_unified_user_prompt", compact_unified_user_prompt),
                          ("build_unified_user_prompt", build_unified_user_prompt),
                          ("format_chat_prompt", format_chat_prompt),
                          ("build_pair_prompt", build_pair_prompt),
                          ("_record_to_pair", _record_to_pair),
                          ("generate_candidate_slots", generate_candidate_slots),
                          ("run_model_smoke", run_model_smoke),
                          ("smoke_problems", smoke_problems),
                          ("prepare_generator", prepare_generator),
                          ("synthetic_batch", synthetic_batch),
                          ("seed_generation_rngs", seed_generation_rngs)):
            if not callable(obj):
                evaluator_problems.append(f"real generation symbol is missing: {name}")
        if hasattr(Phi3Generator, "generate_candidates"):
            evaluator_problems.append(
                "Phi3Generator.generate_candidates reappeared; the sealed path must "
                "not depend on an API the locked-validation path does not use")
    except Exception as exc:  # noqa: BLE001
        evaluator_problems.append(f"real generation path does not resolve: {exc!r}")

    # A dry environment check with injected stubs: proves the pre-authorization
    # gate runs and reports, without a GPU, a model, or any sealed access.
    probe = environment_problems(
        output_dir=ROOT / "results" / FINAL_RUN_NAME,
        model_name=BASE_MODEL,
        model_revision=bundle.fields["base_model_revision"] or "",
        expected_candidates=int(bundle.fields["candidates_per_target"]),
        model_files_present=None, cuda_available=None, free_disk_bytes=None,
        require_cuda=False,
    )
    if probe:
        evaluator_problems.extend(f"environment gate (dry): {item}" for item in probe)

    evaluator_executable = not evaluator_problems
    evaluator_status = (
        "executable - the final evaluator is implemented, importable and tested"
        if evaluator_executable else
        "NOT EXECUTABLE - authorization must be refused while this state holds")
    if not evaluator_executable:
        problems.extend(evaluator_problems)

    # The command names THIS receipt, at the path it is actually being written
    # to. A hardcoded version string here is how v5 came to instruct an operator
    # to present the v4 receipt: the schema moved, the literal did not, and no
    # test looked past argv[1]. The path is a parameter for exactly that reason.
    exact_command = [
        ".venv-gpu/Scripts/python.exe", FINAL_ENTRYPOINT,
        "--executable-receipt", output_path,
        "--expected-receipt-sha256", "<this receipt's sha256, printed on generation>",
        "--authorization-token", "<issued once, separately, against the bundle hash>",
        "--i-understand-this-is-one-time-and-irreversible",
    ]

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
        "final_evaluator_executable": evaluator_executable,
        "final_evaluator_status": evaluator_status,
        "final_evaluator_source": evaluator_source,
        "supersedes": {
            "schema_versions": list(SUPERSEDED_SCHEMA_VERSIONS),
            "artifacts": [
                "results/v4_2_sealed_final_readiness_receipt.json",
                "results/v4_2_sealed_final_executable_receipt.json",
                "results/v4_2_sealed_final_executable_receipt_v3.json",
                "results/v4_2_sealed_final_executable_receipt_v4.json",
                "results/v4_2_sealed_final_executable_receipt_v5.json",
            ],
            "why": (
                "v1 was emitted before the final evaluator existed, and its "
                "entrypoint called the guard before reaching any measurement, so a "
                "valid token could have been spent to discover that evaluation was "
                "unavailable. v2 implemented the evaluator but its loader imported "
                "build_test_generation_prompt and called "
                "Phi3Generator.generate_candidates - neither of which exists - and "
                "never set parse_mode, which defaults to first_assertion; the sealed "
                "run would therefore have been scored by the legacy parser while "
                "claiming the successor protocol, and no test caught it because every "
                "test injected a mock generator. v3 fixed the nonexistent APIs but "
                "never applied the frozen seed, generated one target at a time, and "
                "loaded a second model after the token was spent. v4 fixed those but "
                "its sealed loader still imported a trainer helper that read mutable "
                "prompt globals. v5 bound the prompt correctly, but a read-only audit "
                "found two receipt defects: its own exact_command named the v4 "
                "receipt, which the entrypoint refuses, so the one-time operator "
                "instruction selected a file that could not authorize anything; and "
                "its frozen bundle carried a 40-hex Git commit id in a field named "
                "adapter_source_tree_sha256, labelling a SHA-1 commit as a SHA-256 "
                "source-tree digest. v6 derives exact_command from the receipt's own "
                "output path and renames the field to "
                "candidate_source_tree_git_commit. All five are retained as evidence, "
                "none is executable authorization, and the entrypoint refuses every "
                "one of their schema versions outright."),
        },
        "exact_command": exact_command,
        "baseline_scope": {
            "scope": BASELINE_SCOPE,
            "baselines_bundled": [],
            "comparative_claims_supported": False,
            "statement": BASELINE_SCOPE_STATEMENT,
            "option_chosen": "B - scope the sealed final test to immutable-base Oneiros only",
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
    parser.add_argument("--output", default=DEFAULT_RECEIPT_PATH)
    args = parser.parse_args()

    problems: list[str] = []
    receipt = collect(problems, args.output)
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
