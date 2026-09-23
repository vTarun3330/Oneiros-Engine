"""Build the final provenance receipt for a completed 7B TAP capacity gate."""
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

from scripts.run_tap_7b_capacity_gate import MODEL_NAME, MODEL_REVISION  # noqa: E402

SOURCE_PATHS = {
    "runner": "scripts/run_tap_7b_capacity_gate.py",
    "tap_helpers": "scripts/run_tap_adapter_compare.py",
    "tap_scorer": "scripts/run_tap_diagnostic.py",
    "generator": "engine/generator.py",
    "settings": "config/settings.py",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalised_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def file_meta(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": sha256_bytes(data)}


def git_blob(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"REFUSED: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate", default="results/tap_7b_capacity.json")
    parser.add_argument("--analysis", default="results/tap_7b_capacity_analysis.json")
    parser.add_argument("--protocol", default="results/tap_7b_capacity_protocol.json")
    parser.add_argument("--baseline", default="results/tap_adapter_compare.json")
    parser.add_argument("--gate1-receipt", default="results/tap_capacity_gate_receipt.json")
    parser.add_argument("--items", default="results/tap_train_items.jsonl")
    parser.add_argument("--split", default="results/tap_pilot_split.json")
    parser.add_argument("--out", default="results/tap_7b_capacity_receipt.json")
    args = parser.parse_args(argv)

    paths = {
        "candidate": ROOT / args.candidate,
        "analysis": ROOT / args.analysis,
        "protocol": ROOT / args.protocol,
        "baseline_1_5b": ROOT / args.baseline,
        "gate1_receipt": ROOT / args.gate1_receipt,
        "items": ROOT / args.items,
        "pilot_split": ROOT / args.split,
        "run_manifest": ROOT / "runs" / args.run_id / "manifest.json",
        "run_status": ROOT / "runs" / args.run_id / "status.json",
        "run_complete_marker": ROOT / "runs" / args.run_id / ".complete",
    }
    try:
        for name, path in paths.items():
            require(path.exists(), f"missing {name}: {path}")

        candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
        analysis = json.loads(paths["analysis"].read_text(encoding="utf-8"))
        protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
        status = json.loads(paths["run_status"].read_text(encoding="utf-8"))

        require(protocol.get("status") == "ready", "protocol was not ready before launch")
        require(candidate.get("status") == "complete", "candidate artifact is incomplete")
        require(candidate.get("model") == MODEL_NAME, "candidate model mismatch")
        require(candidate.get("model_revision") == MODEL_REVISION, "candidate revision mismatch")
        require(candidate.get("training_performed") is False, "capacity run trained weights")
        require(candidate.get("weights_written") is False, "capacity run wrote weights")
        require(status.get("state") == "completed" and status.get("exit_code") == 0,
                "supervised run did not complete with exit 0")

        candidate_meta = file_meta(paths["candidate"])
        require(
            analysis.get("inputs", {}).get("candidate", {}).get("sha256")
            == candidate_meta["sha256"],
            "analysis is not bound to the candidate artifact",
        )
        for analysis_key, path_key in (
            ("baseline", "baseline_1_5b"),
            ("split", "pilot_split"),
            ("protocol", "protocol"),
        ):
            require(
                analysis.get("inputs", {}).get(analysis_key, {}).get("sha256")
                == file_meta(paths[path_key])["sha256"],
                f"analysis is not bound to {analysis_key}",
            )

        contract = candidate.get("run_contract", {})
        require(contract.get("items_file_sha256") == file_meta(paths["items"])["sha256"],
                "candidate does not bind the frozen item file")
        require(contract.get("permitted_split") == "train", "candidate is not train-only")
        require(protocol.get("panel", {}).get("validation_read") is False,
                "protocol claims validation access")
        require(protocol.get("panel", {}).get("sealed_final_read") is False,
                "protocol claims sealed-final access")

        launch_commit = manifest.get("git", {}).get("commit")
        require(isinstance(launch_commit, str) and len(launch_commit) == 40,
                "run manifest omits immutable launch commit")
        bound_sources = contract.get("source_files_sha256", {})
        source_binding = {}
        for name, relative in SOURCE_PATHS.items():
            expected = bound_sources.get(name)
            require(bool(expected), f"candidate omits source hash for {name}")
            runtime = (ROOT / relative).read_bytes()
            require(sha256_bytes(runtime) == expected,
                    f"current {name} bytes differ from the run binding")
            committed = git_blob(launch_commit, relative)
            require(normalised_sha256(committed) == normalised_sha256(runtime),
                    f"launch commit does not contain the bound {name} source")
            source_binding[name] = {
                "path": relative,
                "artifact_sha256": expected,
                "launch_blob_sha256": sha256_bytes(committed),
                "normalised_sha256": normalised_sha256(runtime),
                "matches": True,
            }

        snapshot = (
            Path.home() / ".cache" / "huggingface" / "hub"
            / "models--Qwen--Qwen2.5-Coder-7B-Instruct" / "snapshots" / MODEL_REVISION
        )
        require(snapshot.is_dir(), "exact 7B snapshot is not present in the local cache")
        expected_shards = sorted(snapshot.glob("model-*-of-*.safetensors"))
        require(len(expected_shards) == 4, "local snapshot does not expose four weight shards")

        receipt = {
            "schema_version": "oneiros_tap_7b_capacity_receipt_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "label": analysis.get("label"),
            "run_id": args.run_id,
            "launch": {"manifest": manifest, "status": status},
            "model": {
                "name": MODEL_NAME,
                "revision": MODEL_REVISION,
                "local_snapshot": str(snapshot),
                "weight_shards": [path.name for path in expected_shards],
                "runtime_profile": candidate.get("runtime_profile"),
            },
            "source_binding": source_binding,
            "artifacts": {name: file_meta(path) for name, path in paths.items()
                          if name != "run_complete_marker"},
            "run_contract": contract,
            "acceptance_policy": analysis.get("acceptance_policy"),
            "decision": analysis.get("decision"),
            "integrity": {
                "full_raw_outputs_verified_by_analysis": True,
                "train_only": True,
                "validation_read": False,
                "sealed_final_read": False,
                "weights_written": False,
                "training_performed": False,
            },
        }
        output = ROOT / args.out
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"written: {output}")
        return 0
    except (RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
