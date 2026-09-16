"""The authorized entrypoint for the one-time sealed final measurement.

Separate from ``scripts/train_on_dataset.py`` on purpose: the development
entrypoint refuses the sealed split outright, and this is the only path that
can open it.

**The ordering is the whole design, and it has been wrong twice.** The first
version called the guard, spent the token, and only then reached a comment
saying the measurement was unimplemented. The second implemented it but through
a loader calling two functions that do not exist, with the parse mode left at
its legacy default. Both would have cost the single authorization, and the
second would not even have raised - it would have produced a plausible number
under the wrong parser.

So the token is the last thing presented, and everything it depends on is
proven first, on the real path:

1. receipt hash, schema, bundle freeze, evaluator/adapter/loader/smoke/RNG
   identity, prior-run state, model files, revision, output path, disk, CUDA;
2. the model is loaded **once** and a two-record synthetic batch is generated,
   scored and hash-checked through the same adapter the sealed run uses;
3. only then is the token presented;
4. the RNG is reset to the frozen seed and the **same prepared generator** runs
   the sealed batch immediately - no second model load, because a load that
   failed after the token would waste an authorization on code never proven;
5. an irreversible run-state receipt is persisted either way.

This file refuses by default and is committed so the procedure is reviewable in
advance, not so it is convenient to execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.sealed_final import (  # noqa: E402
    SEALED_SPLIT, FinalBundle, SealedAccessError, SealedFinalGuard,
)
from harness.sealed_final_evaluator import (  # noqa: E402
    EVALUATOR_VERSION, FinalEvaluationError, default_free_disk_bytes,
    environment_problems, evaluator_source_hashes, run_final_evaluation,
)
from harness.source_identity import canonical_sha256  # noqa: E402

#: The only receipt that may authorize a run. It is a fallback for --check-only
#: convenience ONLY: an actual irreversible run must name its receipt
#: explicitly, so a stale file can never be selected by omission.
EXECUTABLE_RECEIPT = "results/v4_2_sealed_final_executable_receipt_v6.json"

#: Receipt schema versions this entrypoint refuses outright.
#:   v1 predates the final evaluator entirely.
#:   v2 described a loader calling two functions that do not exist, and never
#:      set parse_mode.
#:   v3 fixed those but never applied the frozen seed, generated one target at
#:      a time where locked validation used two, left several generation
#:      semantics unbound, loaded the model a second time after the token was
#:      spent, and implied a baseline comparison it could not support.
REFUSED_SCHEMA_VERSIONS = (
    "oneiros_sealed_final_readiness_v1",
    "oneiros_sealed_final_readiness_v2",
    "oneiros_sealed_final_readiness_v3",
    #   v4 bound seed, batch and generator reuse, but the sealed loader still
    #      imported a trainer helper that read mutable prompt globals, and the
    #      prompt-defining sources were not bound by hash.
    "oneiros_sealed_final_readiness_v4",
    #   v5 bound the prompt, but its own exact_command named the v4 receipt -
    #      which this entrypoint refuses - so the one-time operator instruction
    #      pointed at a file that could not authorize anything; and its frozen
    #      bundle recorded a Git commit id under adapter_source_tree_sha256.
    "oneiros_sealed_final_readiness_v5",
)
REQUIRED_SCHEMA_VERSION = "oneiros_sealed_final_readiness_v6"

STATE_PATH = "results/sealed_final_state.json"
AUDIT_LOG_PATH = "results/sealed_final_audit.log"
RUN_STATE_PATH = "results/sealed_final_run_state.json"
FINAL_RUN_NAME = "sealed_final_base_qwen_s42"


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt_problems(receipt_path: Path, expected_sha256: str) -> tuple[dict, list[str]]:
    """Load the executable receipt, refusing anything that is not exactly it."""
    if not receipt_path.is_file():
        return {}, [f"executable receipt not found: {receipt_path}"]
    if not expected_sha256:
        return {}, ["--expected-receipt-sha256 is required"]
    actual = sha256_file(receipt_path)
    if actual != str(expected_sha256).strip().lower():
        return {}, [
            "executable receipt hash mismatch\n"
            f"  expected: {expected_sha256}\n  found   : {actual}"]
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"executable receipt is not valid JSON: {exc}"]

    problems: list[str] = []
    schema = receipt.get("schema_version")
    if schema in REFUSED_SCHEMA_VERSIONS:
        problems.append(
            f"receipt schema {schema} is refused: it predates the verified "
            f"seed/batch/generator-reuse corrections. Only "
            f"{REQUIRED_SCHEMA_VERSION} may authorize a run.")
    elif schema != REQUIRED_SCHEMA_VERSION:
        problems.append(f"receipt schema {schema!r} is not {REQUIRED_SCHEMA_VERSION!r}")
    if receipt.get("final_evaluator_executable") is not True:
        problems.append(
            "this receipt does not declare the final evaluator executable; it "
            "cannot authorize a run")
    if receipt.get("ready_for_authorization") is not True:
        problems.append("receipt is not marked ready_for_authorization")
    if receipt.get("sealed_split_accessed") is not False:
        problems.append("receipt claims the sealed split was already accessed")
    return receipt, problems


def evaluator_binding_problems(receipt: dict) -> list[str]:
    """Every module that decides the number must be the approved one."""
    problems: list[str] = []
    recorded = receipt.get("final_evaluator_source") or {}
    if not recorded:
        return ["receipt records no final evaluator source identity"]

    from harness.generation_adapter import adapter_source_hashes

    current = evaluator_source_hashes()
    if recorded.get("evaluator_version") != EVALUATOR_VERSION:
        problems.append(
            f"evaluator version differs: receipt {recorded.get('evaluator_version')!r}, "
            f"runtime {EVALUATOR_VERSION!r}")
    for field in ("canonical_sha256", "measurement_logic_canonical_sha256"):
        if recorded.get(field) != current.get(field):
            problems.append(
                f"final evaluator {field} differs from the approved receipt")

    expected = {
        "adapter_canonical_sha256": adapter_source_hashes()["canonical_sha256"],
        "smoke_canonical_sha256": canonical_sha256(ROOT / "harness/sealed_final_smoke.py"),
        "loader_canonical_sha256": canonical_sha256(ROOT / "harness/sealed_final_loader.py"),
        "rng_canonical_sha256": canonical_sha256(ROOT / "harness/generation_rng.py"),
        "entrypoint_canonical_sha256": canonical_sha256(ROOT / "scripts/run_sealed_final_test.py"),
    }
    for field, value in expected.items():
        if recorded.get(field) != value:
            problems.append(
                f"{field} differs from the approved receipt: "
                f"{recorded.get(field)} vs {value}")

    # Every source that can change a rendered prompt. The prompt factory's own
    # hash is checked here too, via prompt_binding, rather than being repeated
    # as a top-level field - two copies of one fact are two facts that can
    # disagree. A prompt that changed
    # between freeze and run would change the measurement without changing any
    # recorded setting.
    from harness.prompt_factory import prompt_binding_problems
    problems.extend(prompt_binding_problems(recorded.get("prompt_binding"), ROOT))
    return problems


def settings_binding_problems(receipt: dict) -> list[str]:
    """The frozen generation semantics must match the runtime's, field by field."""
    from harness.generation_adapter import GenerationSettings

    frozen = (receipt.get("final_evaluator_source") or {}).get("frozen_generation_settings")
    if not frozen:
        return ["receipt records no frozen generation settings"]
    try:
        settings = GenerationSettings(**frozen)
    except TypeError as exc:
        return [f"frozen generation settings do not match the settings contract: {exc}"]

    problems = list(settings.problems())
    if settings.candidate_parse_mode != "whole_output":
        problems.append("frozen settings do not use whole_output parsing")
    if settings.generation_batch_size != 2:
        problems.append(
            f"frozen generation_batch_size is {settings.generation_batch_size}; "
            "locked validation used 2 and batch shape changes sampling")
    if settings.seed != 42:
        problems.append(f"frozen seed is {settings.seed}, expected 42")

    prompt_settings = settings.prompt_settings()
    problems.extend(prompt_settings.problems())
    recorded_prompt = (receipt.get("final_evaluator_source") or {}).get(
        "frozen_prompt_settings")
    if recorded_prompt != prompt_settings.to_dict():
        problems.append(
            "frozen prompt settings differ from the approved receipt: "
            f"{recorded_prompt} vs {prompt_settings.to_dict()}")
    return problems


def frozen_settings_from(receipt: dict):
    from harness.generation_adapter import GenerationSettings
    return GenerationSettings(
        **receipt["final_evaluator_source"]["frozen_generation_settings"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # No default for a real run. A receipt selected by omission is a receipt
    # nobody looked at, and the file that would have been selected is exactly
    # the one a version bump leaves stale.
    parser.add_argument("--executable-receipt", default=None)
    parser.add_argument("--expected-receipt-sha256", default=None)
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument(
        "--i-understand-this-is-one-time-and-irreversible", action="store_true",
        dest="acknowledged")
    parser.add_argument("--run-name", default=FINAL_RUN_NAME)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run every pre-authorization check, including the real two-record "
             "model smoke, and stop. Presents no token, opens nothing, writes "
             "no state.")
    args = parser.parse_args(argv)

    required_missing = [name for name, value in (
        ("--executable-receipt", args.executable_receipt),
        ("--expected-receipt-sha256", args.expected_receipt_sha256),
        ("--authorization-token", args.authorization_token),
    ) if not value]
    if not args.check_only and (required_missing or not args.acknowledged):
        print("REFUSED: the sealed final test was not run. No token was presented.\n")
        print("This opens the one measurement that cannot be repeated. It requires:\n")
        print("  --executable-receipt <path>   (no default; name it explicitly)")
        print("  --expected-receipt-sha256 <sha256>")
        print("  --authorization-token <token>")
        print("  --i-understand-this-is-one-time-and-irreversible")
        for name in required_missing:
            print(f"\nmissing: {name}")
        if not args.acknowledged:
            print("missing: --i-understand-this-is-one-time-and-irreversible")
        print("\nAfter execution these are prohibited, without exception:")
        for item in ("retraining of any kind", "prompt changes", "threshold changes",
                     "adding, reweighting or re-selecting data",
                     "selecting a different checkpoint or model",
                     "re-running the sealed final test",
                     "re-scoring the result under a different parser or policy"):
            print(f"  - {item}")
        return 2

    # ---- PHASE 1: every non-sealed prerequisite, before any token ---------
    problems: list[str] = []
    # --check-only is allowed to fall back, because it authorizes nothing; a
    # real run reached this line only by naming its receipt above.
    receipt_path = args.executable_receipt or EXECUTABLE_RECEIPT
    receipt, receipt_issues = receipt_problems(
        ROOT / receipt_path, args.expected_receipt_sha256 or "")
    problems.extend(receipt_issues)

    settings = None
    if receipt and not receipt_issues:
        problems.extend(evaluator_binding_problems(receipt))
        problems.extend(settings_binding_problems(receipt))
        if not problems:
            settings = frozen_settings_from(receipt)

        bundle = FinalBundle(fields=receipt.get("frozen_bundle", {}).get("fields", {}))
        if bundle.missing_fields():
            problems.append(f"bundle is not frozen; missing: {bundle.missing_fields()}")
        elif bundle.sha256() != receipt.get("bundle_sha256"):
            problems.append(
                "the frozen bundle does not hash to the receipt's recorded value; "
                "something was edited after the freeze")

        if (ROOT / RUN_STATE_PATH).exists():
            problems.append(
                f"a final run state already exists at {RUN_STATE_PATH}; the sealed "
                "final test has already been executed and must never run twice")

        try:
            from harness import sealed_final_loader
            for symbol in ("sealed_records", "sealed_batch_generator",
                           "build_sealed_generator", "select_split_records",
                           "adapt_records"):
                if not callable(getattr(sealed_final_loader, symbol, None)):
                    problems.append(f"sealed loader is missing {symbol}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"sealed loader is not importable: {exc!r}")

        candidate = receipt.get("final_candidate") or {}
        try:
            import torch
            cuda = lambda: bool(torch.cuda.is_available())  # noqa: E731
        except Exception:
            cuda = lambda: False  # noqa: E731

        def model_files_present(name: str, revision: str) -> bool:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(name, revision=revision, local_files_only=True)
                return True
            except Exception:
                return False

        problems.extend(environment_problems(
            output_dir=ROOT / "results" / args.run_name,
            model_name=candidate.get("model") or "",
            model_revision=candidate.get("model_revision") or "",
            expected_candidates=int(
                (receipt.get("frozen_bundle", {}).get("fields", {})
                 .get("candidates_per_target")) or 0),
            model_files_present=model_files_present,
            cuda_available=cuda,
            free_disk_bytes=default_free_disk_bytes,
        ))

    # ---- PHASE 1b: load the model ONCE and prove the real path -----------
    # The generator built here is the generator the sealed run uses. Building a
    # second one after the token was spent meant a failure in that load would
    # waste the single authorization on code that had never been proven.
    smoke_result = None
    prepared = None
    build_prompt = None
    if not problems and settings is not None:
        try:
            from harness.sealed_final_loader import build_sealed_generator
            from harness.sealed_final_smoke import run_model_smoke, smoke_problems

            prepared, settings, build_prompt = build_sealed_generator(receipt)
            prepared.load_model()
            print("Running required pre-authorization two-record model smoke on "
                  "synthetic records (no sealed data, no token)...", flush=True)
            smoke_result = run_model_smoke(
                settings=settings, build_pair_prompt=build_prompt,
                generator=prepared)
            problems.extend(smoke_problems(smoke_result, settings))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"pre-authorization model smoke failed: {exc!r}")

    if problems:
        print("REFUSED before authorization. No token was presented and none was spent.\n")
        for item in problems:
            print(f"  - {item}")
        return 1

    if args.check_only:
        print("PRE-AUTHORIZATION CHECKS PASSED, including the real two-record smoke.")
        print(f"smoke    : {smoke_result['records_in_batch']} records padded together, "
              f"{smoke_result['candidate_slots_total']} slots, "
              f"parse_mode={smoke_result['observed_parse_mode']}, "
              f"seed_applied={smoke_result['seed_applied']}, "
              f"sealed_data_touched={smoke_result['sealed_data_touched']}")
        print("No token was presented. Nothing was opened. No state was written.")
        print(f"evaluator: {EVALUATOR_VERSION}")
        print(f"bundle   : {receipt.get('bundle_sha256')}")
        return 0

    # ---- PHASE 2: only now is the single authorization presented ---------
    guard = SealedFinalGuard(ROOT / STATE_PATH, ROOT / AUDIT_LOG_PATH)
    bundle = FinalBundle(fields=receipt["frozen_bundle"]["fields"])
    try:
        guard.open_sealed_split(
            SEALED_SPLIT, bundle, args.authorization_token,
            caller=f"{Path(__file__).name}:{args.run_name}")
    except SealedAccessError as exc:
        print(f"REFUSED: {exc}")
        return 1

    # ---- PHASE 3: measure immediately, on the already-proven generator ---
    output_dir = ROOT / "results" / args.run_name
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from harness.generation_rng import rng_state_fingerprint, seed_generation_rngs
        from harness.sealed_final_loader import sealed_batch_generator, sealed_records
        from harness.sealed_final_smoke import generator_identity

        # The smoke consumed randomness. Reset to the frozen seed immediately
        # before real generation, and record that the reset actually moved the
        # state rather than asserting a reset that no-oped.
        before = rng_state_fingerprint()
        seed_record = seed_generation_rngs(settings.seed)
        seed_record["rng_state_before_reset"] = before
        seed_record["rng_state_after_reset"] = rng_state_fingerprint()
        seed_record["reset_immediately_before_sealed_generation"] = True

        identity = generator_identity(prepared, settings)
        identity["reused_from_pre_authorization_smoke"] = True
        identity["smoke_generator_object_id"] = (
            smoke_result["generator_identity"]["python_object_id"])
        identity["same_object_as_smoke"] = (
            identity["python_object_id"] == identity["smoke_generator_object_id"])

        outcome = run_final_evaluation(
            load_records=sealed_records,
            generate_batch=sealed_batch_generator(prepared, settings, build_prompt),
            output_dir=output_dir,
            bundle_sha256=receipt["bundle_sha256"],
            frozen_settings=settings.to_dict(),
            candidates_per_target=settings.candidates_per_function,
            generation_batch_size=settings.generation_batch_size,
            allow_test_function=settings.allow_test_function_candidates,
            log=lambda message: print(message, flush=True),
            seed_record=seed_record,
            generator_identity=identity,
        )
        status = "completed"
        error = None
    except Exception as exc:  # noqa: BLE001
        outcome, status, error = {}, "failed_after_authorization", repr(exc)

    # ---- PHASE 4: the run is irreversible either way ----------------------
    (ROOT / RUN_STATE_PATH).write_text(json.dumps({
        "schema_version": "oneiros_sealed_final_run_state_v2",
        "status": status,
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle_sha256": receipt["bundle_sha256"],
        "executable_receipt_sha256": args.expected_receipt_sha256,
        "evaluator_source": evaluator_source_hashes(),
        "pre_authorization_smoke": smoke_result,
        "authorization_spent": True,
        "may_never_run_again": True,
        "result": outcome,
        "error": error,
    }, indent=2, default=str) + "\n", encoding="utf-8")

    if status != "completed":
        print(f"The authorized run FAILED after the token was spent: {error}")
        print(f"Run state recorded at {RUN_STATE_PATH}. The sealed split may not be "
              "reopened; do not retry.")
        return 1

    print(f"Final measurement complete. Artifact: {outcome['artifact_path']}")
    print(f"sha256: {outcome['artifact_sha256']}")
    print("This authorization is spent. Nothing may be retrained, tuned, re-scored "
          "or re-run on the basis of this result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
