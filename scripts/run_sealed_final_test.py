"""The authorized entrypoint for the one-time sealed final measurement.

Separate from ``scripts/train_on_dataset.py`` on purpose: the development
entrypoint refuses the sealed split outright, and this is the only path that
can open it.

**The ordering here is the whole design.** The first version of this file
called the guard, the guard spent the token, and control then reached a comment
saying the measurement was not implemented. A real authorization would have
been consumed irreversibly with nothing to show for it, and the one split that
cannot be measured twice would have been marked opened.

So the token is now the *last* thing presented, not the first. In order:

1. every non-sealed prerequisite is checked - receipt hash, bundle freeze,
   evaluator presence and source identity, model files, revision, output path,
   disk, CUDA;
2. the evaluator is proven importable and callable;
3. only then is the token presented to the guard;
4. the measurement runs immediately, on the already-checked code path;
5. an irreversible run-state receipt is persisted.

A valid token must never be spent to discover that evaluation is unavailable.
Every refusal below happens before ``open_sealed_split`` is reached.

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

EXECUTABLE_RECEIPT = "results/v4_2_sealed_final_executable_receipt_v3.json"

#: Receipt schema versions this entrypoint refuses outright.
#:   v1 predates the final evaluator entirely.
#:   v2 described a loader that called two functions which do not exist and
#:      never set parse_mode, so it would have measured under the legacy
#:      parser while claiming the successor protocol.
REFUSED_SCHEMA_VERSIONS = (
    "oneiros_sealed_final_readiness_v1",
    "oneiros_sealed_final_readiness_v2",
)
REQUIRED_SCHEMA_VERSION = "oneiros_sealed_final_readiness_v3"
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
            f"receipt schema {schema} is refused: it predates the verified real "
            "generation path. Only " + REQUIRED_SCHEMA_VERSION + " may authorize a run.")
    elif schema != REQUIRED_SCHEMA_VERSION:
        problems.append(
            f"receipt schema {schema!r} is not {REQUIRED_SCHEMA_VERSION!r}")
    if receipt.get("final_evaluator_executable") is not True:
        problems.append(
            "this receipt does not declare the final evaluator executable; it is a "
            "pre-implementation readiness artifact and cannot authorize a run")
    if receipt.get("ready_for_authorization") is not True:
        problems.append("receipt is not marked ready_for_authorization")
    if receipt.get("sealed_split_accessed") is not False:
        problems.append("receipt claims the sealed split was already accessed")
    return receipt, problems


def evaluator_binding_problems(receipt: dict) -> list[str]:
    """The evaluator that will run must be the evaluator that was approved."""
    problems: list[str] = []
    recorded = receipt.get("final_evaluator_source") or {}
    if not recorded:
        return ["receipt records no final evaluator source identity"]
    current = evaluator_source_hashes()
    if recorded.get("evaluator_version") != EVALUATOR_VERSION:
        problems.append(
            f"evaluator version differs: receipt {recorded.get('evaluator_version')!r}, "
            f"runtime {EVALUATOR_VERSION!r}")
    from harness.generation_adapter import adapter_source_hashes
    from harness.source_identity import canonical_sha256
    adapter = recorded.get("adapter_canonical_sha256")
    if adapter != adapter_source_hashes()["canonical_sha256"]:
        problems.append(
            "shared generation adapter differs from the approved receipt: "
            f"{adapter} vs {adapter_source_hashes()['canonical_sha256']}")
    smoke = recorded.get("smoke_canonical_sha256")
    if smoke != canonical_sha256(ROOT / "harness" / "sealed_final_smoke.py"):
        problems.append("smoke module differs from the approved receipt")
    for field in ("canonical_sha256", "measurement_logic_canonical_sha256"):
        if recorded.get(field) != current.get(field):
            problems.append(
                f"final evaluator {field} differs from the approved receipt: "
                f"{recorded.get(field)} vs {current.get(field)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable-receipt", default=EXECUTABLE_RECEIPT)
    parser.add_argument("--expected-receipt-sha256", default=None)
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument(
        "--i-understand-this-is-one-time-and-irreversible", action="store_true",
        dest="acknowledged")
    parser.add_argument("--run-name", default=FINAL_RUN_NAME)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run every pre-authorization check and stop. Presents no token, "
             "opens nothing, writes no state.")
    args = parser.parse_args(argv)

    required_missing = [name for name, value in (
        ("--expected-receipt-sha256", args.expected_receipt_sha256),
        ("--authorization-token", args.authorization_token),
    ) if not value]
    if not args.check_only and (required_missing or not args.acknowledged):
        print("REFUSED: the sealed final test was not run. No token was presented.\n")
        print("This opens the one measurement that cannot be repeated. It requires:\n")
        print(f"  --executable-receipt                             (default: {EXECUTABLE_RECEIPT})")
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
    receipt, receipt_issues = receipt_problems(
        ROOT / args.executable_receipt, args.expected_receipt_sha256 or "")
    problems.extend(receipt_issues)

    if receipt and not receipt_issues:
        problems.extend(evaluator_binding_problems(receipt))

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

        # The loader must exist and be importable BEFORE the token is
        # presented. Its absence was the original defect in a second guise: a
        # spent authorization followed by an ImportError is exactly the outcome
        # this whole redesign exists to make impossible.
        try:
            from harness import sealed_final_loader
            for symbol in ("sealed_records", "sealed_generator"):
                if not callable(getattr(sealed_final_loader, symbol, None)):
                    problems.append(f"sealed loader is missing {symbol}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"sealed loader is not importable: {exc!r}")

        recorded_loader = (receipt.get("final_evaluator_source") or {}).get(
            "loader_canonical_sha256")
        if recorded_loader:
            from harness.source_identity import canonical_sha256 as _canonical
            current_loader = _canonical(ROOT / "harness" / "sealed_final_loader.py")
            if recorded_loader != current_loader:
                problems.append(
                    "sealed loader source differs from the approved receipt: "
                    f"{recorded_loader} vs {current_loader}")
        else:
            problems.append("receipt records no sealed loader source identity")

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

    # ---- PHASE 1b: the real model/prompt/parser path, on synthetic data ---
    # Nothing here may be skipped. Every sealed defect so far survived because
    # the real path was never executed: a nonexistent prompt function, a
    # nonexistent generator method, and a parse mode left at its legacy
    # default. Mocks cannot catch any of those; only running the model can.
    smoke_result = None
    if not problems:
        try:
            from harness.generation_adapter import successor_settings
            from harness.sealed_final_smoke import run_model_smoke, smoke_problems
            settings = successor_settings()
            print("Running required pre-authorization model smoke on a synthetic "
                  "record (no sealed data, no token)...", flush=True)
            smoke_result = run_model_smoke(
                model_name=(receipt.get("final_candidate") or {}).get("model") or "",
                model_revision=(receipt.get("final_candidate") or {}).get(
                    "model_revision") or "",
                settings=settings,
            )
            problems.extend(smoke_problems(smoke_result, settings))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"pre-authorization model smoke failed: {exc!r}")

    if problems:
        print("REFUSED before authorization. No token was presented and none was spent.\n")
        for item in problems:
            print(f"  - {item}")
        return 1

    if args.check_only:
        print("PRE-AUTHORIZATION CHECKS PASSED, including the real model smoke.")
        if smoke_result:
            print(f"smoke    : {smoke_result['candidate_slots']} candidates, "
                  f"parse_mode={smoke_result['observed_parse_mode']}, "
                  f"killed={smoke_result['killing_candidates']}, "
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

    # ---- PHASE 3: measure immediately, on the already-checked path -------
    output_dir = ROOT / "results" / args.run_name
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from harness.sealed_final_loader import (  # type: ignore
            sealed_generator, sealed_records,
        )
        outcome = run_final_evaluation(
            load_records=sealed_records,
            generate=sealed_generator(receipt),
            output_dir=output_dir,
            bundle_sha256=receipt["bundle_sha256"],
            frozen_settings=receipt["frozen_bundle"]["fields"],
            candidates_per_target=int(
                receipt["frozen_bundle"]["fields"]["candidates_per_target"]),
            log=lambda message: print(message, flush=True),
        )
        status = "completed"
        error = None
    except (FinalEvaluationError, ImportError, Exception) as exc:  # noqa: BLE001
        outcome, status, error = {}, "failed_after_authorization", repr(exc)

    # ---- PHASE 4: the run is irreversible either way ----------------------
    (ROOT / RUN_STATE_PATH).write_text(json.dumps({
        "schema_version": "oneiros_sealed_final_run_state_v1",
        "status": status,
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle_sha256": receipt["bundle_sha256"],
        "executable_receipt_sha256": args.expected_receipt_sha256,
        "evaluator_source": evaluator_source_hashes(),
        "authorization_spent": True,
        "may_never_run_again": True,
        "result": outcome,
        "error": error,
    }, indent=2) + "\n", encoding="utf-8")

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
