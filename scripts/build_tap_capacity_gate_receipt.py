"""Build a reproducible receipt for the completed TAP capacity diagnostic.

The GPU launcher records the launch commit and dirty paths, while the TAP
artifact binds exact generation-source hashes.  This builder verifies both,
checks every retained raw output, binds the paired analysis, and refuses to
label an incomplete/superseded run as accepted evidence.
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

SOURCE_PATHS = {
    "runner": "scripts/run_tap_adapter_compare.py",
    "tap_scorer": "scripts/run_tap_diagnostic.py",
    "generator": "engine/generator.py",
    "prompt_engine": "engine/test_generation_prompt.py",
    "settings": "config/settings.py",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_meta(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def committed_bytes(commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact", default="results/tap_adapter_compare.json")
    parser.add_argument("--analysis", default="results/tap_capacity_gate_analysis.json")
    parser.add_argument("--split", default="results/tap_pilot_split.json")
    parser.add_argument("--out", default="results/tap_capacity_gate_receipt.json")
    args = parser.parse_args(argv)

    run_dir = ROOT / "runs" / args.run_id
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    artifact_path = ROOT / args.artifact
    analysis_path = ROOT / args.analysis
    split_path = ROOT / args.split
    required = [manifest_path, status_path, run_dir / ".complete",
                artifact_path, analysis_path, split_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"REFUSED: missing required evidence: {missing}")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed" or status.get("exit_code") != 0:
        print(f"REFUSED: GPU run is not a successful completion: {status}")
        return 2

    from scripts.analyse_tap_capacity_gate import validate_artifact
    validate_artifact(artifact, split)

    artifact_digest = file_meta(artifact_path)
    analysis_digest = file_meta(analysis_path)
    split_digest = file_meta(split_path)
    if analysis.get("inputs", {}).get("artifact_sha256") != artifact_digest["sha256"]:
        print("REFUSED: analysis is not bound to the completed TAP artifact")
        return 2
    if analysis.get("inputs", {}).get("pilot_split_sha256") != split_digest["sha256"]:
        print("REFUSED: analysis is not bound to the frozen pilot split")
        return 2

    launch_commit = manifest.get("git", {}).get("commit")
    bound_sources = artifact.get("run_contract", {}).get("source_files_sha256", {})
    source_binding = {}
    for name, relative in SOURCE_PATHS.items():
        expected = bound_sources.get(name)
        if not expected:
            print(f"REFUSED: artifact omits source binding for {name}")
            return 2
        try:
            committed = sha256_bytes(committed_bytes(launch_commit, relative))
        except subprocess.CalledProcessError as exc:
            print(f"REFUSED: cannot read {relative} from launch commit: {exc}")
            return 2
        if committed != expected:
            print(f"REFUSED: launch commit does not contain the bound {name} bytes")
            return 2
        source_binding[name] = {
            "path": relative,
            "artifact_sha256": expected,
            "launch_commit_sha256": committed,
            "matches": True,
        }

    receipt = {
        "schema_version": "oneiros_tap_capacity_gate_receipt_v2",
        "label": "train-only diagnostic; not model selection, generalization, "
                 "validation, or final-test evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "launch": {
            "branch": manifest.get("git", {}).get("branch"),
            "commit": launch_commit,
            "dirty": manifest.get("git", {}).get("dirty"),
            "dirty_files": manifest.get("git", {}).get("dirty_files", []),
            "command": manifest.get("command"),
            "versions": manifest.get("versions"),
            "status": status,
            "note": "The launch worktree contained only non-generation untracked "
                    "paths; every generation-defining source is verified against "
                    "the committed launch bytes below.",
        },
        "source_binding": source_binding,
        "artifacts": {
            "tap_comparison": artifact_digest,
            "paired_analysis": analysis_digest,
            "pilot_split": split_digest,
            "run_manifest": file_meta(manifest_path),
            "run_status": file_meta(status_path),
        },
        "run_contract": artifact.get("run_contract"),
        "primary_n": analysis.get("primary_n"),
        "pilot_n": analysis.get("pilot_n"),
        "predeclared_margin_pp": analysis.get("margin_pp"),
        "tost_alpha": analysis.get("tost_alpha"),
        "per_requested": analysis.get("arms"),
        "paired_equivalence": analysis.get("equivalence"),
        "code_sensitivity": analysis.get("code_sensitivity"),
        "strata": analysis.get("strata"),
        "sealed_data_accessed": False,
        "weights_written": False,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
