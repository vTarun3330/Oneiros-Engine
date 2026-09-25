"""Build the machine-readable next-direction design package (CPU only).

Computes, from local upstream sources and the permitted train shard only:

* the reference universe every future candidate must be disjoint from, with
  source hashes and a digest of the sorted repository names;
* executable self-checks proving the isolation machinery rejects a known
  benchmark patch and a renamed benchmark function and admits a novel one;
* the environment available for native execution (probed from WSL).

It then records the repository-native protocol, the optional sampling-budget
study, estimates, blockers, required decisions and the proposed order.

It creates no evaluation split, mines nothing from the network, loads no
model, and opens no protected split or canonical records.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.repository_isolation import (
    CandidateBug, ISOLATION_VERSION, build_reference_universe, check_candidate,
    corpus_sources_are_covered,
)
from harness.source_identity import canonical_sha256
from scripts.audit_cross_split_near_duplicates import NEAR_DUPLICATE_JACCARD

SCHEMA = "oneiros_next_direction_design_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_closed_pilots() -> dict:
    receipt = json.loads((ROOT / "results" / "v4_3_tool_assisted_decision_receipt.json")
                         .read_text(encoding="utf-8"))
    checks = {key: sha(ROOT / receipt[key]["path"]) == receipt[key]["sha256"]
              for key in ("analysis", "lineage_manifest", "design_receipt", "panel",
                          "journal")}
    for arm, entry in receipt["evaluations"].items():
        checks[f"eval_{arm}"] = (sha(ROOT / entry["envelope"]["path"])
                                 == entry["envelope"]["sha256"])
    dose = json.loads((ROOT / "results" / "v4_3_execution_dose_decision_receipt.json")
                      .read_text(encoding="utf-8"))
    return {"tool_assisted": {"verdict": receipt["decision"]["outcome"],
                              "artifact_hashes_verify": all(checks.values()),
                              "protected_flags_false": not any(receipt["leakage"].values())},
            "execution_dose": {"outcome": dose["decision"]["outcome"],
                               "protected_flags_false": not any(
                                   value for value in dose["leakage"].values()
                                   if isinstance(value, bool))}}


def probe_wsl() -> dict:
    command = ('for p in python3.7 python3.8 python3.9 python3.10 python3.11 python3.12 '
               'python3.13; do command -v $p >/dev/null && echo "$p $($p -c \'import sys;'
               'print(sys.version.split()[0])\')"; done; for t in uv pyenv docker podman git '
               'conda; do printf "%s=" $t; command -v $t >/dev/null && echo present || echo '
               'absent; done; nproc; free -g | awk \'/Mem/{print $2}\'; df -BG / | '
               'awk \'NR==2{print $4}\'')
    try:
        out = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], capture_output=True,
                             timeout=120).stdout.decode("utf-8", "replace").replace("\x00", "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": str(exc)}
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    interpreters = [line for line in lines if line.startswith("python")]
    tools = dict(line.split("=", 1) for line in lines if "=" in line)
    tail = [line for line in lines if "=" not in line and not line.startswith("python")]
    return {"interpreters": interpreters, "tools": tools,
            "cpus": tail[0] if tail else None, "memory_gb": tail[1] if len(tail) > 1 else None,
            "free_disk": tail[2] if len(tail) > 2 else None,
            "probed_utc": datetime.now(timezone.utc).isoformat()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results"
                        / "v4_3_next_direction_design.json")
    args = parser.parse_args(argv)

    corpus_manifest = json.loads((ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
                                  / "manifest.json").read_text(encoding="utf-8"))
    universe = build_reference_universe(ROOT)
    summary = universe.summary()
    bugsinpy_patch = (ROOT / "data" / "BugsInPy_repo" / "projects" / "tqdm" / "bugs" / "1"
                      / "bug_patch.txt").read_text(encoding="utf-8")
    mbpp_first = json.loads((ROOT / "data" / "mbpp" / "mbpp_full.jsonl")
                            .read_text(encoding="utf-8").splitlines()[0])["code"]
    self_checks = {
        "known_benchmark_patch_rejected": check_candidate(
            CandidateBug(repository="https://github.com/tqdm/tqdm", patch=bugsinpy_patch),
            universe),
        "renamed_mbpp_function_rejected": check_candidate(
            CandidateBug(repository="example/unseen-repository",
                         target_function=mbpp_first.replace("min_cost", "cheapest_path")),
            universe),
        "fork_of_excluded_repository_rejected": check_candidate(
            CandidateBug(repository="someone/renamed-fork", fork_parent="pandas-dev/pandas"),
            universe),
        "novel_function_admitted": check_candidate(
            CandidateBug(repository="example/unseen-repository",
                         target_function="def zz(q):\n    return {k: q[k] * 3 for k in "
                                         "sorted(q) if k.startswith('x')}\n"),
            universe),
    }
    self_checks_pass = (not self_checks["known_benchmark_patch_rejected"]["admissible"]
                        and not self_checks["renamed_mbpp_function_rejected"]["admissible"]
                        and not self_checks["fork_of_excluded_repository_rejected"]["admissible"]
                        and self_checks["novel_function_admitted"]["admissible"])
    names = summary["repository_names"]
    design = json.loads((ROOT / "docs" / "next_direction_design_content.json")
                        .read_text(encoding="utf-8"))
    report = {
        "schema_version": SCHEMA,
        "label": "design only; no split created, no mining, no model, no protected access",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "starting_state": {"closed_pilots": verify_closed_pilots()},
        "isolation": {
            "version": ISOLATION_VERSION,
            "near_duplicate_threshold": NEAR_DUPLICATE_JACCARD,
            "method": "AST-normalised, docstring-stripped code; exact Jaccard over 5-token "
                      "shingles; inverted-index candidates (reused from "
                      "scripts/audit_cross_split_near_duplicates.py)",
            "corpus_sources": sorted(corpus_manifest["records_by_source"]),
            "corpus_sources_fully_covered_by_upstream_universe":
                corpus_sources_are_covered(corpus_manifest),
            "why_protected_splits_need_not_be_opened": (
                "every split, the consumed test split included, is built only from MBPP, "
                "HumanEval, BugsInPy, SWE-bench Verified and curated examples; the universe "
                "below is the full upstream copy of each, a superset of every split"),
            "excluded_repository_names": names,
            "excluded_repository_names_sha256": hashlib.sha256(
                json.dumps(names).encode("utf-8")).hexdigest(),
            "reference_universe": {key: value for key, value in summary.items()
                                   if key != "repository_names"},
            "self_checks": self_checks,
            "self_checks_pass": self_checks_pass,
        },
        "environment_probe": probe_wsl(),
        **design,
        "source_files_sha256": {relative: canonical_sha256(ROOT / relative) for relative in (
            "harness/repository_isolation.py", "scripts/build_next_direction_design.py",
            "scripts/audit_cross_split_near_duplicates.py",
            "docs/next_direction_design_content.json")},
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False,
                    "canonical_records_json_opened": False, "network_accessed": False,
                    "new_split_created": False},
    }
    if not self_checks_pass:
        print("REFUSED: isolation self-checks failed")
        return 2
    args.output.write_bytes((json.dumps(report, indent=2, default=sorted) + "\n")
                            .encode("utf-8"))
    print(json.dumps({"self_checks_pass": self_checks_pass,
                      "excluded_repositories": len(names),
                      "reference_functions": summary["reference_functions"],
                      "known_patches": summary["known_patches"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
