"""Freeze the operational rehearsal. CPU only; opens no model, launches nothing.

**This receipt describes a rehearsal, not a measurement of the model.** It
carries three explicit denials - ``final_test_measurement``,
``eligible_for_model_selection`` and ``supports_performance_claim`` are all
false - because ``ablation_dev`` selected the checkpoints it scored, so a number
produced under this receipt is an operational fact about the pipeline and
nothing more.

Why a receipt at all, for a run that decides nothing: the sealed final test
failed because the code between "authorized" and "measuring" had never been
executed against a real split. The rehearsal exists to execute it. That is only
worth doing if what gets rehearsed is pinned - the same admission rule, the same
generation settings, the same sources - and pinned in something a machine can
verify rather than a paragraph someone remembers to read.

The consumed split is refused before any corpus file is opened.

    python scripts/build_rehearsal_receipt.py
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

from harness.evaluation_admission import (  # noqa: E402
    REFUSED_SPLITS, admission_source_hashes, refuse_refused_split, scope_split,
)
from harness.rehearsal_evaluator import (  # noqa: E402
    REHEARSAL_LABEL, REHEARSAL_VERSION,
)
from harness.source_identity import canonical_sha256, raw_sha256  # noqa: E402

DEFAULT_OUTPUT = "results/v4_2_rehearsal_receipt.json"
REHEARSAL_SPLIT = "ablation_dev"
CORPUS_VERSION = "v4_1_research_hardened_candidate"
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

#: The count this rehearsal must resolve. It is the number the four-arm
#: development evaluation reported, so a receipt that froze anything else would
#: be rehearsing a different scope than the pipeline has ever used.
EXPECTED_TARGETS = 542

#: Every source that can change a rehearsal number. Bound by hash, file by
#: file - never a whole-tree digest, which changes for reasons that have
#: nothing to do with the measurement.
REHEARSAL_DEFINING_SOURCES = {
    "prompt_factory": "harness/prompt_factory.py",
    "prompt_engine": "engine/test_generation_prompt.py",
    "prompt_compaction": "engine/prompt_budget.py",
    "generation_adapter": "harness/generation_adapter.py",
    "generation_rng": "harness/generation_rng.py",
    "admission": "harness/evaluation_admission.py",
    "rehearsal_evaluator": "harness/rehearsal_evaluator.py",
    "scoring": "harness/sealed_final_evaluator.py",
    "safe_execution": "harness/safe_execution.py",
    "candidate_policy": "harness/candidate_policy.py",
    "record_adaptation": "scripts/train_on_dataset.py",
    "runner": "scripts/run_rehearsal_evaluation.py",
    "receipt_builder": "scripts/build_rehearsal_receipt.py",
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def source_hashes() -> dict:
    out = {}
    for role, relative in sorted(REHEARSAL_DEFINING_SOURCES.items()):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"rehearsal-defining source missing: {relative}")
        out[role] = {
            "path": relative,
            "raw_sha256": raw_sha256(path),
            "canonical_sha256": canonical_sha256(path),
        }
    return out


def build(split: str = REHEARSAL_SPLIT, problems=None) -> dict:
    """Resolve the scope and freeze everything a rehearsal depends on."""
    problems = problems if problems is not None else []

    # Before the corpus is opened, not after.
    refuse_refused_split(split)
    if split in REFUSED_SPLITS:
        raise RuntimeError(f"refusing to freeze a refused split: {split!r}")

    from config.settings import immutable_revision_for
    from harness.generation_adapter import successor_settings
    from scripts.train_on_dataset import _record_to_pair

    revision = immutable_revision_for(BASE_MODEL)
    if not revision or len(revision) != 40:
        problems.append(f"base model revision is not an immutable SHA: {revision!r}")

    settings = successor_settings()
    bad = settings.problems()
    if bad:
        problems.extend(f"generation settings: {item}" for item in bad)

    corpus_dir = ROOT / "data" / "corpus" / CORPUS_VERSION
    splits = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    records = json.loads((corpus_dir / "records.json").read_text(encoding="utf-8"))
    scope = scope_split(splits, records, split, adapt=_record_to_pair)

    if scope.target_count != EXPECTED_TARGETS:
        problems.append(
            f"{split} resolved {scope.target_count} function-mode targets, "
            f"expected {EXPECTED_TARGETS}")

    budget_failures = sum(1 for r in scope.eligible if r.get("prompt_budget_failure"))
    if budget_failures:
        problems.append(f"{budget_failures} target(s) fail the prompt budget")

    frozen = settings.to_dict()
    for key, expected in (
        ("seed", 42), ("generation_batch_size", 2), ("candidates_per_function", 8),
        ("candidate_parse_mode", "whole_output"), ("retain_raw_output", True),
        ("prompt_token_limit", 1024), ("generation_completion_token_limit", 1024),
    ):
        if frozen.get(key) != expected:
            problems.append(f"frozen {key} is {frozen.get(key)!r}, expected {expected!r}")

    return {
        "schema_version": REHEARSAL_VERSION,
        "label": REHEARSAL_LABEL,
        # The three denials, stated as data so a consumer can check them.
        "operational_rehearsal": True,
        "final_test_measurement": False,
        "eligible_for_model_selection": False,
        "supports_performance_claim": False,
        "purpose": (
            "Execute the stages a future independent final protocol would use - "
            "admission and scoping, generation, parsing, safe execution, "
            "raw-output retention, progress writing and scoring - against a "
            "real permitted split at full scale. It measures the pipeline, not "
            "the model."),
        "interpretation_limits": [
            "ablation_dev selected the checkpoints it scored, so no number from "
            "this run is a generalization estimate.",
            "No model is selected, promoted, tuned or compared on the basis of "
            "this run.",
            "No baseline is bundled or executed; no Oneiros-versus-Atheris or "
            "any other comparison is supported.",
        ],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_split": split,
        "refused_splits": sorted(REFUSED_SPLITS),
        "consumed_split_policy": (
            "The consumed split is refused before any corpus file is opened. "
            "This rehearsal never reads it and cannot be pointed at it."),
        "corpus_version": CORPUS_VERSION,
        "expected_target_count": scope.target_count,
        "admission_scope": scope.to_dict(),
        "evaluation_scope_sha256": scope.scope_sha256(),
        "repository_exclusions": {
            "count": len(scope.excluded_repository),
            "unknown_mode_count": len(scope.excluded_unknown_mode),
            "reason": scope.to_dict()["repository_exclusion_reason"],
        },
        "prompt_budget_failures": budget_failures,
        "candidate": {
            "model": BASE_MODEL,
            "model_revision": revision,
            "tokenizer": BASE_MODEL,
            "tokenizer_revision": revision,
            "adapter": None,
            "adapter_note": "base model only; no adapter is loaded and none exists to load",
            "training": "none - this run updates no weights",
        },
        "frozen_generation_settings": frozen,
        # Two different things, under two different keys. Merging the nested
        # admission binding in under "admission" silently replaced that role's
        # own file entry, so harness/evaluation_admission.py was skipped by the
        # file-level verifier - the one source whose change most directly
        # changes which records get measured.
        "source_hashes": source_hashes(),
        "admission_binding": admission_source_hashes(ROOT),
        "reproducibility": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "python_version": sys.version.split()[0],
        },
        "receipt_problems": problems,
        "ready_for_rehearsal": not problems,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default=REHEARSAL_SPLIT)
    args = parser.parse_args(argv)

    problems: list[str] = []
    try:
        receipt = build(args.split, problems)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: {exc}")
        return 1

    print("=" * 84)
    print("REHEARSAL RECEIPT (CPU only; no model, no generation, nothing launched)")
    print(REHEARSAL_LABEL.upper())
    print("=" * 84)
    print(f"split           : {receipt['input_split']}")
    print(f"targets         : {receipt['expected_target_count']}")
    print(f"repo excluded   : {receipt['repository_exclusions']['count']}")
    print(f"scope sha256    : {receipt['evaluation_scope_sha256']}")
    print(f"model           : {receipt['candidate']['model']} @ "
          f"{receipt['candidate']['model_revision']}")
    print(f"adapter         : {receipt['candidate']['adapter']}")
    print(f"refused splits  : {receipt['refused_splits']} - not read")
    print(f"sources bound   : {len(REHEARSAL_DEFINING_SOURCES)}")

    if problems:
        print(f"\nNOT READY - {len(problems)} problem(s); nothing written:")
        for item in problems:
            print(f"  - {item}")
        return 1

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    out.write_bytes(payload)
    print(f"\nreceipt         : {args.output}")
    print(f"sha256          : {hashlib.sha256(payload).hexdigest()}")
    print("\nREADY. This authorizes nothing: it freezes what a rehearsal would run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
