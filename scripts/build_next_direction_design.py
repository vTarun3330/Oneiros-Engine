"""Build the next-direction design package and the reference-universe receipt (CPU only).

From the corpus manifest, local upstream sources, the curated seed definition
in code, the legacy real-bug files and the permitted train shard only, it:

* indexes the complete reference universe, computes its deterministic receipt
  and FREEZES the universe against it (collections, collection hashes, internal
  receipt hash, every input file and every canonical source re-verified);
* runs executable self-checks against the frozen universe with synthetic,
  fully authenticated evidence (a known benchmark patch, a renamed benchmark
  function and a fork of an excluded repository are refused by the overlap
  stage; a partial patch and a pre-cutoff fix are refused by authentication;
  an evidence-free candidate is refused; a novel candidate is admitted; an
  arbitrary receipt hash cannot be attached);
* verifies the power artifact by recomputing it from its verified evidence;
* verifies both closed pilots, each evaluation by envelope AND raw result hash.

Nothing is published unless every gate passes.  The receipt, the design and a
copy of the verified power artifact are published together as ONE immutable
generation (``harness.atomic_publish.publish_bundle``): staged, re-verified from
the staged bytes, renamed into place, and selected by replacing a single
pointer.  A failure or crash at any point leaves readers with the complete
previous generation.

It mines nothing, makes no network call, creates no split, loads no model, and
opens no protected split or canonical records.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atomic_publish import PublicationRefused, publish_bundle
from harness.closed_pilot_evidence import sha256_file, verify_closed_pilots
from harness.isolation_evidence_fixtures import synthetic_candidate
from harness.repository_isolation import (
    ACQUISITION_REQUIREMENTS, CANONICAL_SOURCES, CLAIM, DIFF_POLICY, INSUFFICIENT,
    ISOLATION_VERSION, TEMPORAL_CUTOFF_EPOCH, TEMPORAL_RULE, CandidateBug,
    FrozenReferenceUniverse, ReferenceUniverseMismatch, build_reference_universe,
    canonical_diff, check_candidate, extract_functions, freeze_reference_universe,
    isolation_record_is_current, reference_universe_receipt,
)
from harness.source_identity import canonical_sha256
from scripts.audit_cross_split_near_duplicates import NEAR_DUPLICATE_JACCARD
from scripts.build_repository_native_power_analysis import (
    EvidenceRefused, verify_power_artifact,
)

SCHEMA = "oneiros_next_direction_design_v4"
POWER = "results/v4_3_repository_native_power_analysis.json"
#: Crash-safe bundle store: one immutable generation per build, one pointer.
BUNDLE_STORE = "results/next_direction_bundle"
RECEIPT_FILE = "reference_universe_receipt.json"
DESIGN_FILE = "next_direction_design.json"
POWER_FILE = "repository_native_power_analysis.json"
DESIGN_SOURCES = (*CANONICAL_SOURCES, "harness/isolation_evidence_fixtures.py",
                  "harness/atomic_publish.py", "harness/closed_pilot_evidence.py",
                  "scripts/build_repository_native_power_analysis.py",
                  "docs/next_direction_design_content.json")
NOVEL = ("def merge_ranges(pairs):\n    pairs = sorted(pairs)\n    merged = [pairs[0]]\n"
         "    for lo, hi in pairs[1:]:\n        if lo <= merged[-1][1]:\n"
         "            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))\n"
         "        else:\n            merged.append((lo, hi))\n    return merged\n")
BUGGY = NOVEL.replace("lo <= merged", "lo < merged")


def novel_candidate(**overrides: Any) -> CandidateBug:
    return synthetic_candidate(buggy_text=BUGGY, fixed_text=NOVEL, target_function=BUGGY,
                               target_file="src/ranges.py", target_module="ranges", **overrides)


TQDM_BUGGY = ("def tenumerate(iterable, start=0, total=None, tqdm_class=None, **tqdm_kwargs):\n"
              "    total = total or 0\n    tqdm_kwargs = dict(tqdm_kwargs, total=total)\n"
              "    if tqdm_class is None:\n        raise ValueError('tqdm_class is required')\n"
              "    return enumerate(tqdm_class(iterable, start, **tqdm_kwargs))\n")
TQDM_FIXED = TQDM_BUGGY.replace("return enumerate(tqdm_class(iterable, start, **tqdm_kwargs))",
                                "return enumerate(tqdm_class(iterable, **tqdm_kwargs), start)")
SETTING = ("TIMEOUT = 10\n\n\n", "TIMEOUT = 30\n\n\n")


def _with_body_line(function: str, line: str) -> str:
    """``function`` with ``line`` inserted as its first body statement."""
    lines = function.splitlines()
    body = next(text for text in lines[1:] if text.strip())
    indent = body[:len(body) - len(body.lstrip())]
    return "\n".join([lines[0], indent + line, *lines[1:]]) + "\n"


def run_self_checks(frozen: FrozenReferenceUniverse, root: Path = ROOT
                    ) -> tuple[dict[str, Any], bool]:
    """Synthetic, fully authenticated candidates against the frozen universe."""
    mbpp = json.loads((root / "data/mbpp/mbpp_full.jsonl").read_text(
        encoding="utf-8").splitlines()[0])["code"].replace("\r\n", "\n")
    renamed = extract_functions(mbpp.replace("min_cost", "cheapest_path"))[0] + "\n"
    partial = canonical_diff("src/ranges.py", SETTING[0] + BUGGY, SETTING[1] + BUGGY)
    auth = f"{INSUFFICIENT}:authentication:"
    # name: (candidate, reasons that must appear, whether schema and authentication pass)
    candidates = {
        "known_benchmark_patch_in_excluded_repository_refused": (synthetic_candidate(
            repository="tqdm/tqdm", buggy_text=TQDM_BUGGY, fixed_text=TQDM_FIXED,
            target_function=TQDM_BUGGY, target_file="tqdm/contrib/__init__.py",
            target_module="contrib"),
            {"patch_identical_to_known_bug", "repository_in_reference_universe:tqdm/tqdm"}, True),
        "renamed_mbpp_function_refused": (synthetic_candidate(
            buggy_text=renamed, fixed_text=_with_body_line(renamed, "assert True"),
            target_function=renamed, target_file="src/paths.py", target_module="paths"),
            {"function_near_duplicate_of_reference"}, True),
        "fork_of_excluded_repository_refused": (novel_candidate(
            fork_parent="pandas-dev/pandas"),
            {"fork_parent_in_reference_universe:pandas-dev/pandas"}, True),
        "partial_patch_with_only_an_unrelated_change_refused": (synthetic_candidate(
            buggy_text=SETTING[0] + BUGGY, fixed_text=SETTING[1] + NOVEL, target_function=BUGGY,
            target_file="src/ranges.py", target_module="ranges", patch=partial),
            {auth + "submitted_patch_does_not_match_derived_diff"}, False),
        "fix_before_temporal_cutoff_refused": (novel_candidate(
            committer_epoch=TEMPORAL_CUTOFF_EPOCH - 1, buggy_epoch=TEMPORAL_CUTOFF_EPOCH - 7200),
            {auth + "fixed_commit_before_temporal_cutoff"}, False),
        "novel_fully_evidenced_candidate_admitted": (novel_candidate(), set(), True),
    }
    checks: dict[str, Any] = {}
    passed = True
    for name, (candidate, expected, authenticates) in candidates.items():
        record = check_candidate(candidate, frozen)
        ok = (expected <= set(record["reasons"])
              and record["admissible"] is (not expected)
              and (record["stages"]["schema"] == [] and record["stages"]["authentication"] == [])
              is authenticates
              and isolation_record_is_current(record, frozen))
        checks[name] = {**record, "self_check_passed": ok}
        passed &= ok
    evidence_free = check_candidate(CandidateBug(repository="example-org/ranges"), frozen)
    ok = evidence_free["admissible"] is False and evidence_free["insufficient_evidence"] is True
    checks["evidence_free_candidate_refused"] = {**evidence_free, "self_check_passed": ok}
    passed &= ok
    try:
        check_candidate(novel_candidate(), "a" * 64)  # type: ignore[arg-type]
        refused = False
    except TypeError:
        refused = True
    record = check_candidate(novel_candidate(), frozen)
    forged = {**record, "reference_universe_sha256": "a" * 64}
    ok = refused and not isolation_record_is_current(forged, frozen)
    checks["arbitrary_receipt_hash_refused"] = {
        "check_candidate_with_a_hash_raises_type_error": refused,
        "record_with_a_substituted_hash_is_not_current": not isolation_record_is_current(
            forged, frozen),
        "self_check_passed": ok}
    passed &= ok
    return checks, passed


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_receipt(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8")


def build_artifacts(root: Path, probe: bool = True) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Every gate runs here, in memory; raises on any failure."""
    universe = build_reference_universe(root)
    receipt = reference_universe_receipt(universe)
    frozen = freeze_reference_universe(universe, receipt, root=root)
    coverage = receipt["coverage"]
    if not coverage["covered"]:
        raise PublicationRefused(f"coverage fails: {coverage['problems']}")
    self_checks, self_checks_pass = run_self_checks(frozen, root)
    if not self_checks_pass:
        raise PublicationRefused("isolation self-checks fail")
    power_bytes = (root / POWER).read_bytes()
    power = verify_power_artifact(root, power_bytes)
    pilots = verify_closed_pilots(root)
    if not (pilots["all_evaluations_verify"] and pilots["tool_assisted"]["artifact_hashes_verify"]
            and pilots["tool_assisted"]["protected_flags_false"]
            and pilots["execution_dose"]["protected_flags_false"]):
        raise PublicationRefused(f"closed-pilot evidence does not verify: {pilots}")
    receipt_bytes = encode_receipt(receipt)
    design = json.loads((root / "docs/next_direction_design_content.json").read_text(
        encoding="utf-8"))
    names = receipt["collections"]["repository_names"]
    report = {
        "schema_version": SCHEMA,
        "label": "design only; no split created, no mining, no model, no protected access",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "starting_state": {"closed_pilots": pilots},
        "isolation": {
            "version": ISOLATION_VERSION,
            "claim": CLAIM,
            "near_duplicate_threshold": NEAR_DUPLICATE_JACCARD,
            "method": "AST-normalised, docstring-stripped code; exact Jaccard over 5-token "
                      "shingles; inverted-index candidates (reused from "
                      "scripts/audit_cross_split_near_duplicates.py)",
            "stages": ["evidence schema", "evidence authentication (offline)",
                       "source-universe overlap"],
            "acquisition_requirements": ACQUISITION_REQUIREMENTS,
            "why_protected_splits_need_not_be_opened": (
                "every split, the consumed test split included, is built only from MBPP, "
                "HumanEval, BugsInPy, SWE-bench Verified and the curated seeds; the universe "
                "indexes the complete upstream copy of each, verified by the coverage audit"),
            "diff_policy": DIFF_POLICY,
            "temporal_rule": TEMPORAL_RULE,
            "integrity": ("record_sha256 detects accidental modification only; it is a "
                          "self-hash, not a signature. Downstream use must keep the evidence "
                          "sidecar, reload the same frozen universe, rerun check_candidate and "
                          "compare the record byte for byte (revalidate_isolation_record)"),
            "coverage": coverage,
            "reference_universe_receipt_file": f"{BUNDLE_STORE}/generations/<current>/"
                                               f"{RECEIPT_FILE}",
            "reference_universe_receipt_sha256": frozen.receipt_sha256,
            "reference_universe_receipt_file_sha256": sha256_bytes(receipt_bytes),
            "frozen_universe_verification": frozen.verification,
            "collection_sha256": receipt["collection_sha256"],
            "excluded_repository_names": names,
            "counts": receipt["collections"]["counts"],
            "bound_input_files": {source: len(files) for source, files in
                                  receipt["collections"]["input_files"].items()},
            "bound_code_inputs": {source: sorted(files) for source, files in
                                  receipt["collections"]["code_input_files"].items()},
            "self_checks": self_checks,
            "self_checks_pass": self_checks_pass,
        },
        "publication": ("crash-safe bundle: receipt, design and the verified power artifact "
                        f"are one immutable generation under {BUNDLE_STORE}, selected by its "
                        "CURRENT pointer; readers accept only a complete manifest-verified "
                        "generation (harness/atomic_publish.py)"),
        "power_analysis": {"path": POWER, "bundle_copy": POWER_FILE,
                           "sha256": sha256_bytes(power_bytes),
                           "verified_by_recomputation": True,
                           "recommendation": power["recommendation"],
                           "evidence": {name: {key: value for key, value in entry.items()
                                               if key != "inputs_sha256"}
                                        for name, entry in power["evidence"].items()}},
        "environment_probe": probe_wsl() if probe else {"probe_skipped": True},
        **design,
        "source_files_sha256": {relative: canonical_sha256(root / relative)
                                for relative in DESIGN_SOURCES},
        "leakage": {"splits_opened": ["train"], "validation_accessed": False,
                    "ablation_dev_accessed": False, "test_accessed": False,
                    "sealed_final_test_accessed": False, "confirmation_opened": False,
                    "canonical_records_json_opened": False, "non_train_records_opened": False,
                    "network_accessed": False, "new_split_created": False},
    }
    design_bytes = (json.dumps(report, indent=2, default=sorted) + "\n").encode("utf-8")
    files = {RECEIPT_FILE: receipt_bytes, DESIGN_FILE: design_bytes, POWER_FILE: power_bytes}
    return files, {"universe": universe, "power_bytes": power_bytes}


def verify_staged(root: Path, universe: Any, power_bytes: bytes,
                  staged: Mapping[str, Path]) -> None:
    """Re-verify the proposed bundle from its staged bytes."""
    receipt_bytes = staged[RECEIPT_FILE].read_bytes()
    receipt = json.loads(receipt_bytes)
    try:
        frozen = freeze_reference_universe(universe, receipt, root=root)
    except ReferenceUniverseMismatch as exc:
        raise PublicationRefused(f"staged receipt does not freeze: {exc}") from exc
    design = json.loads(staged[DESIGN_FILE].read_bytes())
    isolation = design["isolation"]
    problems = []
    if isolation["reference_universe_receipt_sha256"] != frozen.receipt_sha256:
        problems.append("design names a different receipt hash")
    if isolation["reference_universe_receipt_file_sha256"] != sha256_bytes(receipt_bytes):
        problems.append("design does not bind the staged receipt file")
    staged_power = staged[POWER_FILE].read_bytes()
    if not (design["power_analysis"]["sha256"] == sha256_bytes(power_bytes)
            == sha256_bytes(staged_power) == sha256_file(root / POWER)):
        problems.append("power artifact changed during the build")
    if not (isolation["coverage"]["covered"] and isolation["self_checks_pass"]):
        problems.append("staged design records a failed gate")
    if any(value for key, value in design["leakage"].items() if key != "splits_opened"):
        problems.append("staged design reports protected access")
    if problems:
        raise PublicationRefused("; ".join(problems))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / BUNDLE_STORE)
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args(argv)
    try:
        files, context = build_artifacts(ROOT, probe=not args.no_probe)
        generation = publish_bundle(
            args.store, files,
            verify=lambda staged: verify_staged(ROOT, context["universe"],
                                                context["power_bytes"], staged))
    except (PublicationRefused, ReferenceUniverseMismatch, EvidenceRefused) as exc:
        print(f"REFUSED (the current generation is unchanged): {exc}")
        return 2
    receipt_bytes = files[RECEIPT_FILE]
    receipt = json.loads(receipt_bytes)
    print(json.dumps({"generation": generation, "coverage": receipt["coverage"]["covered"],
                      "receipt_sha256": receipt["receipt_sha256"],
                      "receipt_file_sha256": sha256_bytes(receipt_bytes),
                      "bound_input_files": sum(len(files) for files in
                                               receipt["collections"]["input_files"].values())},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
