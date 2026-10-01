"""Immutable manifest of the completed v2.4 engineering-rehearsal artifacts.

The generation, execution and analysis artifacts are Git-ignored and local to the GPU machine;
this tracked receipt records each one's path, size, SHA-256 and row/candidate counts, the
bound preflight/authorisation/source/model/adapter identities, and a byte-verified second copy
in an archival directory. Every value is recomputed from the files; written once.

    python scripts/native_v24_artifact_manifest.py --archive <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_rehearsal_rebuild_v22 import publish_once

OUT = "results/sft_root_cause_native_v24_completed_artifacts_manifest.json"
GEN = "results/sft_root_cause/native_v24_generations"
EXE = "results/sft_root_cause/native_v24_execution"
COND = "primary_whole_module"
ARTIFACTS = (f"{GEN}/base/contract_{COND}_base.json", f"{GEN}/base/generations_{COND}_base.jsonl",
             f"{GEN}/sft/contract_{COND}_sft.json", f"{GEN}/sft/generations_{COND}_sft.jsonl",
             f"{EXE}/execute_contract_{COND}.json", f"{EXE}/results_{COND}.jsonl",
             "results/sft_root_cause/native_v24_analysis_pre_atheris.json")
RUNS = ("runs/20261001-013100-native_v24_generate_base",
        "runs/20261001-020339-native_v24_generate_sft")
BOUND = ("results/sft_root_cause_native_generated_tests_preflight_v2_4_r2.json",
         "results/sft_root_cause_native_gpu_authorization_v2.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(rel: str) -> dict:
    path = ROOT / rel
    out = {"path": rel, "bytes": path.stat().st_size, "sha256": sha(path)}
    if rel.endswith(".jsonl"):
        rows = [json.loads(l) for l in path.read_bytes().decode("utf-8").splitlines()]
        out["rows"] = len(rows)
        if "/native_v24_generations/" in rel:
            out["candidates"] = sum(len(r["candidates"]) for r in rows)
        else:
            out["rows_by_arm"] = {a: sum(r["arm"] == a for r in rows) for a in ("base", "sft")}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args(argv)
    archive = Path(args.archive)
    files = [*ARTIFACTS, *(p.relative_to(ROOT).as_posix() for run in RUNS
                           for p in sorted((ROOT / run).iterdir()) if p.is_file())]
    mismatched = [rel for rel in files
                  if not (archive / rel).is_file() or sha(archive / rel) != sha(ROOT / rel)]
    contracts = {arm: json.loads((ROOT / f"{GEN}/{arm}/contract_{COND}_{arm}.json")
                                 .read_text(encoding="utf-8"))["identity"]
                 for arm in ("base", "sft")}
    pre = json.loads((ROOT / BOUND[0]).read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / ARTIFACTS[-1]).read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "oneiros_native_v24_completed_artifacts_v1",
        "study_mode": "engineering_dress_rehearsal",
        "note": "generation, execution and analysis artifacts are Git-ignored and local-only "
                "on the GPU machine; this receipt binds their exact bytes",
        "artifacts": [entry(rel) for rel in ARTIFACTS],
        "gpu_runs": [{"run_dir": run, "status": json.loads((ROOT / run / "status.json")
                                                           .read_text(encoding="utf-8"))}
                     for run in RUNS],
        "bound": {rel: sha(ROOT / rel) for rel in BOUND},
        "source_commit": pre["source"]["commit"],
        "executable_tree_sha256": pre["source"]["executable_tree_sha256"],
        "model": pre["model"], "adapter_manifest_sha256": pre["adapter_manifest_sha256"],
        "generation_identities": contracts,
        "engineering_gate_passed": analysis["engineering_gate_passed"],
        "atheris": "NOT RUN: contract preview found 5 cattrs targets failing as infrastructure "
                   "(system python3.11 attrs 23.2.0 instead of the prepared environment); no "
                   "fuzzing budget spent",
        "archive": {"directory_name": archive.name, "files": len(files),
                    "byte_identical": not mismatched, "mismatched": mismatched},
        "claims": {"root_cause": False, "generalization": False, "sft_benefit": False,
                   "atheris_superiority": False},
    }
    if mismatched:
        raise SystemExit(f"REFUSED: archive differs for {mismatched[:3]}")
    status = publish_once(OUT, manifest)
    print(json.dumps({"status": status, "files": len(files),
                      "artifacts": {a["path"]: a["sha256"][:12] for a in manifest["artifacts"]}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
