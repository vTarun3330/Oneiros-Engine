"""Tests for the fail-closed isolation checks, source coverage and universe receipt."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from harness.repository_isolation import (
    CLAIM, INSUFFICIENT, MIN_FUNCTION_SHINGLES, CandidateBug, ReferenceUniverse,
    audit_source_coverage, build_reference_universe, canonical_repository, check_candidate,
    code_shingles, evidence_problems, extract_functions, isolation_record_is_current,
    normalised_patch, patch_hash, reference_universe_receipt, verify_receipt,
)
from scripts.audit_cross_split_near_duplicates import NEAR_DUPLICATE_JACCARD, normalise, shingles

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = "a" * 64
PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
index 111..222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@ def total(xs):
 def total(xs):
-    return sum(xs[1:])
+    return sum(xs)
"""
NOVEL = ("def merge_ranges(pairs):\n    pairs = sorted(pairs)\n    merged = [pairs[0]]\n"
         "    for lo, hi in pairs[1:]:\n        if lo <= merged[-1][1]:\n"
         "            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))\n"
         "        else:\n            merged.append((lo, hi))\n    return merged\n")
NEW_PATCH = ("--- a/src/ranges.py\n+++ b/src/ranges.py\n"
             "-        if lo < merged[-1][1]:\n+        if lo <= merged[-1][1]:\n")
REFERENCE_FUNCTION = ("def rolling_mean(values, window):\n    out = []\n"
                      "    for i in range(len(values) - window + 1):\n"
                      "        out.append(sum(values[i:i + window]) / window)\n    return out\n")


def complete_candidate(**overrides) -> CandidateBug:
    values = dict(
        repository="example-org/ranges", repository_url="https://github.com/example-org/ranges",
        repository_id="123456", fork_status="not_fork", buggy_commit="1" * 40,
        fixed_commit="2" * 40, patch=NEW_PATCH, target_function=NOVEL,
        target_file="src/ranges.py", target_module="ranges", issue_id="example-org/ranges#42",
        licence_spdx="MIT", licence_sha256="b" * 64)
    values.update(overrides)
    return CandidateBug(**values)


def _universe() -> ReferenceUniverse:
    universe = ReferenceUniverse()
    universe.add_repository("https://github.com/pandas-dev/pandas.git", "BugsInPy")
    universe.add_repository("youtube_dl", "legacy_real_bugs")
    universe.commits.add("c" * 40)
    universe.instance_ids.add("django__django-1234")
    universe.patch_hashes.add(patch_hash(PATCH))
    universe.patch_shingles["known::1"] = shingles(normalised_patch(PATCH).replace("\n", " "))
    universe.add_function_source("mbpp::1", REFERENCE_FUNCTION, "mbpp")
    return universe


# --- Phase 1: fail-closed evidence ---------------------------------------------

def test_complete_novel_candidate_is_admitted_and_carries_the_receipt_hash():
    record = check_candidate(complete_candidate(), _universe(), RECEIPT)
    assert record["admissible"] is True and record["reasons"] == []
    assert record["reference_universe_sha256"] == RECEIPT and record["claim"] == CLAIM
    assert isolation_record_is_current(record, RECEIPT)
    assert not isolation_record_is_current(record, "d" * 64)


def _refused(candidate: CandidateBug, expected: str) -> None:
    record = check_candidate(candidate, _universe(), RECEIPT)
    assert record["admissible"] is False and record["insufficient_evidence"] is True
    assert f"{INSUFFICIENT}:{expected}" in record["reasons"], record["reasons"]


@pytest.mark.parametrize("candidate, expected", [
    (CandidateBug(), "repository_not_canonical_owner_name"),
    (CandidateBug(repository="example-org/ranges"), "patch_empty"),
    (complete_candidate(patch=""), "patch_empty"),
    (complete_candidate(patch="--- a/x\n+++ b/x\n context only\n"), "patch_empty"),
    (complete_candidate(target_function=""), "target_function_missing_or_unparseable"),
    (complete_candidate(target_function="   \n\t"), "target_function_missing_or_unparseable"),
    (complete_candidate(target_function="x = (1,"), "target_function_missing_or_unparseable"),
    (complete_candidate(target_function="value = 3\n"), "target_function_missing_or_unparseable"),
    (complete_candidate(target_function="def f():\n    return 1\n"),
     "target_function_too_short_for_comparison"),
    (complete_candidate(buggy_commit=""), "buggy_commit_not_full_sha"),
    (complete_candidate(fixed_commit=""), "fixed_commit_not_full_sha"),
    (complete_candidate(fixed_commit="2" * 7), "fixed_commit_not_full_sha"),
    (complete_candidate(buggy_commit="G" * 40), "buggy_commit_not_full_sha"),
    (complete_candidate(fork_status="unknown"), "fork_status_unknown"),
    (complete_candidate(fork_status=""), "fork_status_unknown"),
    (complete_candidate(fork_status="fork"), "fork_parent_identity_missing"),
    (complete_candidate(repository="not a repo"), "repository_not_canonical_owner_name"),
    (complete_candidate(repository="Example/Ranges"), "repository_not_canonical_owner_name"),
    (complete_candidate(repository_url="https://gitlab.com/example-org/ranges"),
     "repository_url_unverified_or_inconsistent"),
    (complete_candidate(repository_id=""), "repository_identity_missing"),
    (complete_candidate(licence_spdx=""), "licence_evidence_missing_or_not_admitted"),
    (complete_candidate(licence_spdx="GPL-3.0"), "licence_evidence_missing_or_not_admitted"),
    (complete_candidate(licence_sha256=""), "licence_file_hash_missing"),
    (complete_candidate(issue_id=""), "issue_identity_missing"),
    (complete_candidate(issue_id="other/repo#1"), "issue_repository_mismatch"),
    (complete_candidate(target_file=""), "target_file_missing"),
    (complete_candidate(target_module=""), "target_module_missing"),
])
def test_missing_or_malformed_evidence_refuses_admission(candidate, expected):
    _refused(candidate, expected)


def test_empty_candidate_is_refused_on_every_mandatory_field():
    problems = evidence_problems(CandidateBug())
    assert len(problems) >= 10 and all(p.startswith(INSUFFICIENT) for p in problems)


def test_fork_with_verified_parent_is_checked_against_the_universe():
    record = check_candidate(complete_candidate(
        fork_status="fork", fork_parent="pandas-dev/pandas", fork_parent_id="99"),
        _universe(), RECEIPT)
    assert "fork_parent_in_reference_universe:pandas-dev/pandas" in record["reasons"]
    assert record["insufficient_evidence"] is False


def test_repository_commit_issue_patch_and_function_lineage_are_rejected():
    universe = _universe()
    assert any(r.startswith("repository_in_reference_universe") for r in check_candidate(
        complete_candidate(repository="someone/youtube-dl",
                           repository_url="https://github.com/someone/youtube-dl",
                           issue_id="someone/youtube-dl#1"), universe, RECEIPT)["reasons"])
    assert "fixed_commit_is_known_benchmark_commit" in check_candidate(
        complete_candidate(fixed_commit="c" * 40), universe, RECEIPT)["reasons"]
    reformatted = PATCH.replace("return sum(xs)", "return   sum( xs )")
    assert "patch_near_duplicate_of_known_bug" in check_candidate(
        complete_candidate(patch=reformatted), universe, RECEIPT)["reasons"]
    renamed = REFERENCE_FUNCTION.replace("rolling_mean", "moving_average")
    record = check_candidate(complete_candidate(target_function=renamed), universe, RECEIPT)
    assert "function_near_duplicate_of_reference" in record["reasons"]
    assert record["nearest_function"]["jaccard"] >= NEAR_DUPLICATE_JACCARD


def test_a_decision_without_a_receipt_hash_is_impossible():
    with pytest.raises(ValueError):
        check_candidate(complete_candidate(), _universe(), "")


def test_helpers_reuse_the_frozen_audit():
    assert normalise('def f():\n    """doc"""\n    return 1\n') == "def f():\n    return 1"
    assert canonical_repository("https://github.com/Pandas-Dev/pandas.git/") == \
        ("pandas-dev/pandas", "pandas")
    assert len(extract_functions("def a():\n    return 1\n\nclass K:\n    def b(self):\n"
                                 "        return 2\n")) == 2
    assert len(code_shingles(NOVEL)) >= MIN_FUNCTION_SHINGLES


# --- Phase 2: evidence-based coverage on a synthetic root -------------------------

def _write_fake_root(root: Path) -> None:
    import pandas as pd
    bug = root / "data/BugsInPy_repo/projects/demo/bugs/1"
    bug.mkdir(parents=True)
    (root / "data/BugsInPy_repo/projects/demo/project.info").write_text(
        'github_url="https://github.com/demo-org/demo"\n', encoding="utf-8")
    (bug / "bug.info").write_text(f'buggy_commit_id="{"3" * 40}"\nfixed_commit_id="{"4" * 40}"\n',
                                  encoding="utf-8")
    (bug / "bug_patch.txt").write_text(PATCH, encoding="utf-8")
    (root / "data/swebench_verified_source").mkdir(parents=True)
    pd.DataFrame([{"repo": "acme/widgets", "instance_id": "acme__widgets-1",
                   "base_commit": "5" * 40, "patch": PATCH}]).to_parquet(
        root / "data/swebench_verified_source/SWE-bench_Verified.test.parquet")
    (root / "data/bugsinpy").mkdir(parents=True)
    (root / "data/bugsinpy/bugsinpy_metadata.json").write_text(
        json.dumps([{"project": "demo", "commit_buggy": "6" * 40}]), encoding="utf-8")
    (root / "data/bugsinpy/bugsinpy_demo_1_buggy.py").write_text(REFERENCE_FUNCTION,
                                                                 encoding="utf-8")
    (root / "data/mbpp").mkdir(parents=True)
    (root / "data/mbpp/mbpp_full.jsonl").write_text(
        json.dumps({"task_id": 1, "code": REFERENCE_FUNCTION}) + "\n", encoding="utf-8")
    (root / "data/humaneval").mkdir(parents=True)
    (root / "data/humaneval/humaneval_cache.json").write_text(json.dumps([{
        "task_id": "HumanEval/0", "prompt": "def f(x):\n", "canonical_solution": "    return x\n"}]),
        encoding="utf-8")


FAKE_MANIFEST = {"records_by_source": {"mbpp": 1, "humaneval": 1, "BugsInPy": 1,
                                       "SWE-bench Verified": 1}}


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "repo"
    _write_fake_root(root)
    return root


def _fake_universe(root: Path) -> ReferenceUniverse:
    universe = build_reference_universe(root, include_train_shard=False)
    # The synthetic root has no train view; mark it so only the tested source varies.
    universe.counts["train_shard"] = {"records": 1, "declared_records": 1}
    universe.inputs["train_shard"] = {"synthetic": "0" * 64}
    return universe


def test_synthetic_universe_with_every_input_is_covered(fake_root):
    audit = audit_source_coverage(_fake_universe(fake_root), FAKE_MANIFEST)
    assert audit["covered"] is True, audit["problems"]


def test_declared_but_unindexed_and_unknown_sources_fail(fake_root):
    universe = _fake_universe(fake_root)
    universe.counts.pop("humaneval")
    audit = audit_source_coverage(universe, FAKE_MANIFEST)
    assert "declared source 'humaneval' is not indexed" in audit["problems"]
    unknown = audit_source_coverage(_fake_universe(fake_root),
                                    {"records_by_source": {**FAKE_MANIFEST["records_by_source"],
                                                           "new_source": 3}})
    assert "unknown corpus source 'new_source'" in unknown["problems"]
    universe = _fake_universe(fake_root)
    universe.counts["mbpp"] = {"tasks": 0, "functions": 0}
    assert "source 'mbpp' indexed with zero items" in audit_source_coverage(
        universe, FAKE_MANIFEST)["problems"]


@pytest.mark.parametrize("relative", [
    "data/BugsInPy_repo/projects/demo/project.info",
    "data/BugsInPy_repo/projects/demo/bugs/1/bug.info",
    "data/BugsInPy_repo/projects/demo/bugs/1/bug_patch.txt",
    "data/swebench_verified_source/SWE-bench_Verified.test.parquet",
    "data/bugsinpy/bugsinpy_metadata.json",
    "data/mbpp/mbpp_full.jsonl",
    "data/humaneval/humaneval_cache.json",
])
def test_removing_any_required_upstream_input_fails(fake_root, relative):
    (fake_root / relative).unlink()
    try:
        universe = _fake_universe(fake_root)
    except (FileNotFoundError, OSError):
        return
    assert audit_source_coverage(universe, FAKE_MANIFEST)["covered"] is False


def test_curated_seeds_are_indexed_from_their_original_definition(fake_root):
    from harness.bugsinpy_loader import CURATED_BUGSINPY_BUGS
    universe = _fake_universe(fake_root)
    curated = universe.counts["manual_curated_examples"]
    assert curated["definition_ids"] == sorted(f"curated::{b['id']}"
                                               for b in CURATED_BUGSINPY_BUGS)
    assert "harness/bugsinpy_loader.py" in universe.inputs["manual_curated_examples"]


# --- Phase 3: receipt ------------------------------------------------------------

def test_receipt_is_deterministic_self_verifying_and_invalidates_old_decisions(fake_root):
    first = reference_universe_receipt(_fake_universe(fake_root), {"x.py": "1" * 64})
    second = reference_universe_receipt(_fake_universe(fake_root), {"x.py": "1" * 64})
    assert first == second and verify_receipt(first)
    tampered = json.loads(json.dumps(first))
    tampered["collections"]["commits"].append("f" * 40)
    assert not verify_receipt(tampered)
    changed = _fake_universe(fake_root)
    changed.commits.add("e" * 40)
    changed_receipt = reference_universe_receipt(changed, {"x.py": "1" * 64})
    assert changed_receipt["receipt_sha256"] != first["receipt_sha256"]
    record = check_candidate(complete_candidate(), _fake_universe(fake_root),
                             first["receipt_sha256"])
    assert not isolation_record_is_current(record, changed_receipt["receipt_sha256"])


# --- the real universe -------------------------------------------------------------

@pytest.fixture(scope="module")
def real_universe():
    if not (ROOT / "data" / "BugsInPy_repo" / "projects").exists():
        pytest.skip("upstream sources not present")
    return build_reference_universe(ROOT)


def test_complete_real_universe_passes_coverage(real_universe):
    manifest = json.loads((ROOT / "data/corpus/v4_1_research_hardened_candidate/manifest.json")
                          .read_text(encoding="utf-8"))
    audit = audit_source_coverage(real_universe, manifest)
    assert audit["covered"] is True, audit["problems"]
    assert real_universe.counts["BugsInPy"]["projects"] == 17
    assert real_universe.counts["BugsInPy"]["bugs"] == 501
    assert real_universe.counts["SWE-bench Verified"]["instances"] == 500
    assert real_universe.counts["mbpp"]["tasks"] == 974
    assert real_universe.counts["humaneval"]["tasks"] == 164
    assert len(real_universe.inputs["BugsInPy"]) == 17 + 2 * 501


def test_all_eight_curated_corpus_seeds_are_indexed(real_universe):
    manifest = json.loads((ROOT / "data/corpus/v4_1_research_hardened_candidate/manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["records_by_source"]["manual_curated_examples"] == 8
    curated = real_universe.counts["manual_curated_examples"]
    indexed = set(curated["definition_ids"])
    train = set(real_universe.counts["train_shard"]["curated_ids"])
    assert len(train) == 7 and train <= indexed
    # The only generator of curated records is this definition, so the non-train
    # seed is one of the indexed definitions that are not in train.
    eighth = indexed - train
    assert len(eighth) >= 8 - len(train)
    assert "harness/bugsinpy_loader.py" in real_universe.inputs["manual_curated_examples"]
    assert "scripts/build_corpus_v1.py" in real_universe.inputs["manual_curated_examples"]


def test_tracked_receipt_and_design_verify():
    receipt_path = ROOT / "results" / "v4_3_reference_universe_receipt.json"
    design_path = ROOT / "results" / "v4_3_next_direction_design.json"
    if not receipt_path.exists() or not design_path.exists():
        pytest.skip("receipt or design not present")
    for path in (receipt_path, design_path):
        assert b"\r" not in path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert verify_receipt(receipt)
    from harness.source_identity import canonical_sha256
    for relative, expected in receipt["source_files_sha256"].items():
        assert canonical_sha256(ROOT / relative) == expected, relative
    for files in receipt["collections"]["input_files"].values():
        for relative, expected in files.items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert design["isolation"]["reference_universe_receipt_sha256"] == receipt["receipt_sha256"]
    assert design["isolation"]["coverage"]["covered"] is True
    assert design["isolation"]["self_checks_pass"] is True
    assert "proven" not in json.dumps(design["isolation"]).lower()
    assert not any(value for key, value in design["leakage"].items() if key != "splits_opened")
