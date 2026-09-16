"""Operational rehearsal of the final-measurement flow, on a permitted split.

**Not a measurement of the model.** Every artifact is labelled
"operational rehearsal - not eligible for model selection or final-performance
claims", and this script refuses the consumed ``test`` split outright.

It exists because the sealed-final attempt failed on a record-admission defect
that a two-record synthetic smoke could never have caught. A rehearsal runs
every stage a future independent final protocol would use - admission and
scoping, the generation adapter, the parser, the safe executor, raw-output
retention, the progress writer, the scoring flow - against a real split at full
scale, so the next thing that cannot be repeated is not the first thing to run
this code.

Base model only. No adapter, no training, no weight update.

    # CPU gate, no model loaded:
    python scripts/run_rehearsal_evaluation.py --dry-run

    # Full GPU rehearsal:
    .venv-gpu/Scripts/python.exe scripts/run_rehearsal_evaluation.py \
        --receipt results/v4_2_rehearsal_receipt.json \
        --expected-receipt-sha256 <sha>
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

from harness.evaluation_admission import (  # noqa: E402
    REFUSED_SPLITS, AdmissionError, RefusedSplitError,
    admission_binding_problems, refuse_refused_split, scope_split,
)
from harness.rehearsal_evaluator import (  # noqa: E402
    REHEARSAL_LABEL, REHEARSAL_VERSION, RehearsalError,
    run_rehearsal_evaluation,
)

DEFAULT_RECEIPT = "results/v4_2_rehearsal_receipt.json"
DEFAULT_SPLIT = "ablation_dev"
DEFAULT_RUN_NAME = "rehearsal_ablationdev_base_qwen_s42"
CORPUS_VERSION = "v4_1_research_hardened_candidate"


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scope(split: str, corpus_version: str = CORPUS_VERSION):
    """Resolve the split into its evaluable targets, in adapted form.

    Refuses a refused split before opening anything, then delegates scoping to
    the shared admission function so this path cannot drift from the one a
    future final protocol would use.
    """
    refuse_refused_split(split)
    from scripts.train_on_dataset import _record_to_pair

    corpus_dir = ROOT / "data" / "corpus" / corpus_version
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    return scope_split(splits, records, split, adapt=_record_to_pair)


def receipt_problems(receipt_path: Path, expected_sha256: str):
    """Load the rehearsal receipt, refusing anything that is not exactly it."""
    if not receipt_path.is_file():
        return {}, [f"rehearsal receipt not found: {receipt_path}"]
    actual = sha256_file(receipt_path)
    if expected_sha256 and actual != str(expected_sha256).strip().lower():
        return {}, [
            "rehearsal receipt hash mismatch\n"
            f"  expected: {expected_sha256}\n  found   : {actual}"]
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"rehearsal receipt is not valid JSON: {exc}"]

    problems = []
    if receipt.get("schema_version") != REHEARSAL_VERSION:
        problems.append(
            f"receipt schema {receipt.get('schema_version')!r} is not {REHEARSAL_VERSION!r}")
    if receipt.get("operational_rehearsal") is not True:
        problems.append("receipt does not declare itself an operational rehearsal")
    if receipt.get("eligible_for_model_selection") is not False:
        problems.append("receipt does not disclaim model selection")
    split = receipt.get("input_split")
    if split in REFUSED_SPLITS:
        problems.append(f"receipt names a refused split: {split!r}")
    problems.extend(admission_binding_problems(
        (receipt.get("source_hashes") or {}).get("admission"), ROOT))
    return receipt, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT)
    parser.add_argument("--expected-receipt-sha256", default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve the scope and report it. Loads no model, generates "
             "nothing, writes no result.")
    args = parser.parse_args(argv)

    print("=" * 84)
    print("OPERATIONAL REHEARSAL")
    print(REHEARSAL_LABEL.upper())
    print("=" * 84)

    # ---- the split gate, before anything is opened ------------------------
    try:
        refuse_refused_split(args.split)
    except RefusedSplitError as exc:
        print(f"REFUSED: {exc}")
        return 2

    problems = []
    receipt = {}
    receipt_sha = ""
    receipt_path = ROOT / args.receipt
    if receipt_path.is_file():
        receipt, problems = receipt_problems(
            receipt_path, args.expected_receipt_sha256 or "")
        receipt_sha = sha256_file(receipt_path)
    elif not args.dry_run:
        problems.append(f"rehearsal receipt not found: {args.receipt}")

    if receipt and receipt.get("input_split") and receipt["input_split"] != args.split:
        problems.append(
            f"receipt freezes split {receipt['input_split']!r}, "
            f"--split says {args.split!r}")

    # ---- scope resolution -------------------------------------------------
    try:
        scope = load_scope(args.split)
    except (AdmissionError, RefusedSplitError) as exc:
        print(f"REFUSED: {exc}")
        return 1

    expected = (receipt.get("expected_target_count") if receipt else None)
    if expected is not None and scope.target_count != expected:
        problems.append(
            f"scope resolved {scope.target_count} targets, receipt froze {expected}")

    budget_failures = sum(
        1 for r in scope.eligible if r.get("prompt_budget_failure"))

    print(f"split                 : {args.split}")
    print(f"records requested     : {scope.requested}")
    print(f"function-mode targets : {scope.target_count}")
    print(f"repository excluded   : {len(scope.excluded_repository)}")
    print(f"unknown-mode excluded : {len(scope.excluded_unknown_mode)}")
    print(f"admission failures    : 0 (scoping raises rather than counting)")
    print(f"prompt-budget failures: {budget_failures}")
    print(f"scope sha256          : {scope.scope_sha256()}")
    print(f"refused splits        : {sorted(REFUSED_SPLITS)} - not read")
    print(f"receipt               : {args.receipt} ({receipt_sha[:16] or 'absent'})")

    if problems:
        print("\nGATE FAILED:")
        for item in problems:
            print(f"  - {item}")
        return 1

    if args.dry_run:
        print("\nCPU GATE PASSED. No model was loaded, nothing was generated, "
              "nothing was written.")
        return 0

    # ---- GPU rehearsal ----------------------------------------------------
    from harness.generation_adapter import (
        GenerationSettings, generate_candidate_slots,
    )
    from harness.generation_rng import rng_state_fingerprint, seed_generation_rngs
    from harness.prompt_factory import prompt_factory
    from engine.generator import Phi3Generator

    settings = GenerationSettings(**receipt["frozen_generation_settings"])
    bad = settings.problems()
    if bad:
        print(f"REFUSED: frozen settings are invalid: {bad}")
        return 1

    output_dir = ROOT / "results" / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"REFUSED: rehearsal output directory is not empty: {output_dir}")
        return 1

    print(f"\nloading {settings.base_model_name} @ {settings.base_model_revision} "
          f"(base model only, no adapter)...", flush=True)
    generator = Phi3Generator(
        model_name=settings.base_model_name,
        model_revision=settings.base_model_revision,
        attention_implementation=settings.attention_implementation,
    )
    generator.temperature = settings.temperature
    generator.top_p = settings.top_p
    generator.parse_mode = settings.candidate_parse_mode
    generator.load_model()
    build_prompt = prompt_factory(settings.prompt_settings())

    before = rng_state_fingerprint()
    seed_record = seed_generation_rngs(settings.seed)
    seed_record["rng_state_before_reset"] = before
    seed_record["rng_state_after_reset"] = rng_state_fingerprint()
    seed_record["applied_immediately_before_generation"] = True

    def generate_batch(batch):
        rows = [dict(record) for record in batch]
        accounting = generate_candidate_slots(
            generator, rows, settings, build_prompt, seed_before_generation=False)
        outputs = []
        for position in range(len(rows)):
            slots = accounting[position]["candidate_slots"]
            outputs.append([
                {"raw_output": slot.get("raw_output", ""), "code": slot.get("code")}
                for slot in slots
            ])
        return outputs

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        artifact = run_rehearsal_evaluation(
            load_records=lambda: scope.eligible,
            generate_batch=generate_batch,
            output_dir=output_dir,
            split_name=args.split,
            scope_summary=scope.to_dict(),
            frozen_settings=settings.to_dict(),
            receipt_sha256=receipt_sha,
            candidates_per_target=settings.candidates_per_function,
            generation_batch_size=settings.generation_batch_size,
            allow_test_function=settings.allow_test_function_candidates,
            log=lambda message: print(message, flush=True),
            seed_record=seed_record,
            generator_identity={
                "python_object_id": id(generator),
                "model_name": settings.base_model_name,
                "model_revision": settings.base_model_revision,
                "parse_mode": generator.parse_mode,
                "adapter": None,
            },
        )
    except (RehearsalError, Exception) as exc:  # noqa: BLE001
        print(f"\nREHEARSAL FAILED at {started}: {exc!r}")
        print("Stopping. Do not switch splits and do not retry with changed "
              "settings; investigate the failure first.")
        return 1

    print("\n" + "=" * 84)
    print(f"REHEARSAL COMPLETE - {REHEARSAL_LABEL}")
    print("=" * 84)
    for k in (1, 2, 4, 8):
        key = f"kill_at_{k}"
        if key in artifact:
            print(f"  Kill@{k:<2}: {artifact[key]:.6f}")
    print(f"  targets : {artifact['function_validation_records']}")
    print(f"  runtime : {artifact['wall_time_seconds']}s")
    print(f"  raw ok  : {artifact['raw_output_integrity']['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
