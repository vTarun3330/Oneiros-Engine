"""Record where the re-scored artifacts live, and that Git will not move them.

The three files the oracle dataset depends on total tens of megabytes and are
matched by ``.gitignore:64`` (``results/**``). They are therefore NOT in the
repository, and a clone or ``git pull`` on another machine produces a checkout
that looks complete and is missing every input the build needs.

This receipt IS committed. It carries the hashes, so a second machine can prove
it received the same bytes by some other route, and it says plainly that no
such route is automatic.

Nothing here copies anything. Naming the gap is the point; silently assuming
Git closed it is what this prevents.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import sha256_file, write_json

EVALUATOR_SOURCES = {
    "evaluator_source_sha256": "metrics/research_evaluation.py",
    "candidate_policy_source_sha256": "harness/candidate_policy.py",
    "safe_execution_source_sha256": "harness/safe_execution.py",
    "prompt_builder_source_sha256": "engine/test_generation_prompt.py",
    "rescore_tool_sha256": "scripts/rescore_harness_failures.py",
    "verifier_sha256": "scripts/verify_successor_generation.py",
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _ignored(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    rule = _git("check-ignore", "-v", relative)
    tracked = _git("ls-files", "--error-unmatch", relative)
    return {
        "git_ignored": bool(rule),
        "git_ignore_rule": rule.split("\t")[0] if rule else None,
        "tracked_in_git": bool(tracked) and tracked != "unknown",
    }


def _describe(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{role} artifact missing: {path}")
    return {
        "role": role,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "megabytes": round(path.stat().st_size / (1024 * 1024), 2),
        **_ignored(path),
    }


def build(original: Path, derived: Path, overlay: Path) -> dict[str, Any]:
    files = [
        _describe(original, "original_generation_artifact_immutable"),
        _describe(derived, "derived_rescored_artifact"),
        _describe(overlay, "rescore_overlay_all_attempts"),
    ]

    derived_payload = json.loads(derived.read_text(encoding="utf-8"))
    parent_claim = derived_payload.get("derived_from_sha256")
    original_actual = files[0]["sha256"]
    if parent_claim != original_actual:
        raise SystemExit(
            "the derived artifact claims parent " + str(parent_claim)
            + " but the original hashes to " + original_actual)

    unsynced = [f["path"] for f in files if f["git_ignored"]]
    return {
        "schema_version": "oneiros_rescore_artifact_receipt_v1",
        "source_commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "sealed_final_test_accessed": False,
        "evaluation_split": derived_payload.get("evaluation_split"),
        "files": files,
        "parent_chain_verified": True,
        "parent_chain_note": (
            "the derived artifact's derived_from_sha256 matches the original "
            "file's actual SHA-256, checked here rather than trusted"),
        "execution_and_evaluator_hashes": {
            field: sha256_file(ROOT / relative)
            for field, relative in EVALUATOR_SOURCES.items()},
        "run_contract_sha256": derived_payload.get("run_contract_sha256"),
        "retention": {
            "machine": "DESKTOP-A1EGVDN",
            "directory": (ROOT / "results").as_posix(),
            "artifacts_are_git_ignored": bool(unsynced),
            "git_ignored_paths": unsynced,
            "git_alone_does_not_synchronise_these": True,
            "consequence": (
                "a clone or git pull on the laptop yields a checkout that looks "
                "complete and contains none of these files. Every script that "
                "consumes them will fail with a missing path, or - worse - a "
                "caller may silently fall back to a smaller artifact that does "
                "exist. They must be copied by some other means, and their "
                "SHA-256 above checked after the copy."),
        },
        "protocol_note": (
            "the successor kill@8 carried by the derived artifact is a "
            "TRAIN-ONLY whole_output diagnostic. It is not a validation or "
            "generalisation result, and it is not comparable with any legacy "
            "first_assertion number."),
        "train_only_successor_kill_at_8": derived_payload.get("function_kill_rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "results"
    parser.add_argument("--original", type=Path, default=base
                        / "local_base_qwen_train_successor_s42"
                        / "base_validation_train_parse-whole-output_completion1024_seed_42.json")
    parser.add_argument("--derived", type=Path, default=base
                        / "local_base_qwen_train_successor_s42_rescored"
                        / "base_validation_train_parse-whole-output_completion1024_seed_42.rescored.json")
    parser.add_argument("--overlay", type=Path, default=base
                        / "local_base_qwen_train_successor_s42" / "RESCORE_OVERLAY.json")
    parser.add_argument("--output", type=Path,
                        default=base / "v4_2_rescore_artifact_receipt.json")
    arguments = parser.parse_args()

    receipt = build(arguments.original, arguments.derived, arguments.overlay)
    write_json(arguments.output, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
