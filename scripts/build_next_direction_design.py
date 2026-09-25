"""Build the next-direction design package and the reference-universe receipt (CPU only).

From local upstream sources, the curated seed definition in code, the legacy
real-bug files and the permitted train shard only, it:

* indexes the complete reference universe and runs the evidence-based
  coverage audit (concrete counts and bound inputs per corpus source);
* writes ``results/v4_3_reference_universe_receipt.json``, a deterministic
  hash-bound receipt whose SHA-256 every future isolation record must carry;
* runs executable self-checks with fully evidenced candidates (a known
  benchmark patch, a renamed benchmark function and a fork of an excluded
  repository are refused; an evidence-free candidate is refused; a novel,
  fully evidenced candidate is admitted);
* binds the prospective power analysis and records the protocol content.

It mines nothing, creates no split, loads no model, and opens no protected
split or canonical records.json.  The claim it supports is narrow: disjointness
is enforced against the complete indexed source universe under the recorded
checks.
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
    CLAIM, ISOLATION_VERSION, CandidateBug, audit_source_coverage, build_reference_universe,
    check_candidate, reference_universe_receipt, verify_receipt,
)
from harness.source_identity import canonical_sha256
from scripts.audit_cross_split_near_duplicates import NEAR_DUPLICATE_JACCARD

SCHEMA = "oneiros_next_direction_design_v2"
RECEIPT_PATH = ROOT / "results" / "v4_3_reference_universe_receipt.json"
POWER_PATH = ROOT / "results" / "v4_3_repository_native_power_analysis.json"
UNIVERSE_SOURCES = ("harness/repository_isolation.py", "scripts/audit_cross_split_near_duplicates.py",
                    "scripts/build_next_direction_design.py", "scripts/build_corpus_v1.py",
                    "harness/corpus_view.py")
NOVEL = ("def merge_ranges(pairs):\n    pairs = sorted(pairs)\n    merged = [pairs[0]]\n"
         "    for lo, hi in pairs[1:]:\n        if lo <= merged[-1][1]:\n"
         "            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))\n"
         "        else:\n            merged.append((lo, hi))\n    return merged\n")
NOVEL_PATCH = ("--- a/src/ranges.py\n+++ b/src/ranges.py\n"
               "-        if lo < merged[-1][1]:\n+        if lo <= merged[-1][1]:\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidenced(**overrides) -> CandidateBug:
    """A fully evidenced synthetic candidate; overrides make it collide or fail."""
    values = dict(repository="example-org/ranges",
                  repository_url="https://github.com/example-org/ranges", repository_id="1",
                  fork_status="not_fork", buggy_commit="1" * 40, fixed_commit="2" * 40,
                  patch=NOVEL_PATCH, target_function=NOVEL, target_file="src/ranges.py",
                  target_module="ranges", issue_id="example-org/ranges#1", licence_spdx="MIT",
                  licence_sha256="b" * 64)
    values.update(overrides)
    return CandidateBug(**values)


def verify_closed_pilots() -> dict:
    receipt = json.loads((ROOT / "results" / "v4_3_tool_assisted_decision_receipt.json")
                         .read_text(encoding="utf-8"))
    checks = {key: sha(ROOT / receipt[key]["path"]) == receipt[key]["sha256"]
              for key in ("analysis", "lineage_manifest", "design_receipt", "panel", "journal")}
    for arm, entry in receipt["evaluations"].items():
        checks[f"eval_{arm}"] = sha(ROOT / entry["envelope"]["path"]) == entry["envelope"]["sha256"]
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
               'print(sys.version.split()[0])\')"; done; for t in uv pip-compile pyenv docker '
               'podman git conda; do printf "%s=" $t; command -v $t >/dev/null && echo present '
               '|| echo absent; done; nproc; free -g | awk \'/Mem/{print $2}\'; df -BG / | '
               'awk \'NR==2{print $4}\'')
    try:
        out = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], capture_output=True,
                             timeout=120).stdout.decode("utf-8", "replace").replace("\x00", "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": str(exc)}
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    tools = dict(line.split("=", 1) for line in lines if "=" in line)
    tail = [line for line in lines if "=" not in line and not line.startswith("python")]
    return {"interpreters": [line for line in lines if line.startswith("python")],
            "tools": tools, "cpus": tail[0] if tail else None,
            "memory_gb": tail[1] if len(tail) > 1 else None,
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
    coverage = audit_source_coverage(universe, corpus_manifest)
    receipt = reference_universe_receipt(
        universe, {relative: canonical_sha256(ROOT / relative) for relative in UNIVERSE_SOURCES})
    if not verify_receipt(receipt):
        print("REFUSED: reference-universe receipt does not verify")
        return 2
    RECEIPT_PATH.write_bytes((json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False)
                              + "\n").encode("utf-8"))
    receipt_hash = receipt["receipt_sha256"]

    bugsinpy_patch = (ROOT / "data" / "BugsInPy_repo" / "projects" / "tqdm" / "bugs" / "1"
                      / "bug_patch.txt").read_text(encoding="utf-8")
    mbpp_first = json.loads((ROOT / "data" / "mbpp" / "mbpp_full.jsonl")
                            .read_text(encoding="utf-8").splitlines()[0])["code"]
    self_checks = {
        "known_benchmark_patch_in_excluded_repository_refused": check_candidate(
            evidenced(repository="tqdm/tqdm", repository_url="https://github.com/tqdm/tqdm",
                      issue_id="tqdm/tqdm#1", patch=bugsinpy_patch), universe, receipt_hash),
        "renamed_mbpp_function_refused": check_candidate(
            evidenced(target_function=mbpp_first.replace("min_cost", "cheapest_path")),
            universe, receipt_hash),
        "fork_of_excluded_repository_refused": check_candidate(
            evidenced(fork_status="fork", fork_parent="pandas-dev/pandas", fork_parent_id="7"),
            universe, receipt_hash),
        "evidence_free_candidate_refused": check_candidate(
            CandidateBug(repository="example-org/ranges"), universe, receipt_hash),
        "novel_fully_evidenced_candidate_admitted": check_candidate(
            evidenced(), universe, receipt_hash),
    }
    expected = {"known_benchmark_patch_in_excluded_repository_refused": False,
                "renamed_mbpp_function_refused": False,
                "fork_of_excluded_repository_refused": False,
                "evidence_free_candidate_refused": False,
                "novel_fully_evidenced_candidate_admitted": True}
    self_checks_pass = all(self_checks[name]["admissible"] is admissible
                           for name, admissible in expected.items()) and \
        self_checks["evidence_free_candidate_refused"]["insufficient_evidence"]

    power = json.loads(POWER_PATH.read_text(encoding="utf-8"))
    design = json.loads((ROOT / "docs" / "next_direction_design_content.json")
                        .read_text(encoding="utf-8"))
    names = receipt["collections"]["repository_names"]
    report = {
        "schema_version": SCHEMA,
        "label": "design only; no split created, no mining, no model, no protected access",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "starting_state": {"closed_pilots": verify_closed_pilots()},
        "isolation": {
            "version": ISOLATION_VERSION,
            "claim": CLAIM,
            "near_duplicate_threshold": NEAR_DUPLICATE_JACCARD,
            "method": "AST-normalised, docstring-stripped code; exact Jaccard over 5-token "
                      "shingles; inverted-index candidates (reused from "
                      "scripts/audit_cross_split_near_duplicates.py)",
            "why_protected_splits_need_not_be_opened": (
                "every split, the consumed test split included, is built only from MBPP, "
                "HumanEval, BugsInPy, SWE-bench Verified and the curated seeds; the universe "
                "indexes the complete upstream copy of each, verified by the coverage audit"),
            "coverage": coverage,
            "reference_universe_receipt_path": RECEIPT_PATH.relative_to(ROOT).as_posix(),
            "reference_universe_receipt_sha256": receipt_hash,
            "reference_universe_receipt_file_sha256": sha(RECEIPT_PATH),
            "collection_sha256": receipt["collection_sha256"],
            "excluded_repository_names": names,
            "counts": receipt["collections"]["counts"],
            "bound_input_files": {source: len(files) for source, files in
                                  receipt["collections"]["input_files"].items()},
            "self_checks": self_checks,
            "self_checks_pass": self_checks_pass,
        },
        "power_analysis": {"path": POWER_PATH.relative_to(ROOT).as_posix(),
                           "sha256": sha(POWER_PATH),
                           "recommendation": power["recommendation"],
                           "evidence": {name: {key: value for key, value in entry.items()
                                               if key != "inputs_sha256"}
                                        for name, entry in power["evidence"].items()}},
        "environment_probe": probe_wsl(),
        **design,
        "source_files_sha256": {relative: canonical_sha256(ROOT / relative) for relative in (
            *UNIVERSE_SOURCES, "scripts/build_repository_native_power_analysis.py",
            "docs/next_direction_design_content.json")},
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False,
                    "canonical_records_json_opened": False, "non_train_records_opened": False,
                    "network_accessed": False, "new_split_created": False},
    }
    if not self_checks_pass or not coverage["covered"]:
        print(f"REFUSED: self_checks_pass={self_checks_pass} coverage={coverage['problems']}")
        return 2
    args.output.write_bytes((json.dumps(report, indent=2, default=sorted) + "\n")
                            .encode("utf-8"))
    print(json.dumps({"coverage": coverage["covered"], "self_checks_pass": self_checks_pass,
                      "receipt_sha256": receipt_hash,
                      "bound_input_files": report["isolation"]["bound_input_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
