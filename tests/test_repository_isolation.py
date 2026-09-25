"""Tests for the fail-closed isolation checks, the frozen universe, coverage and evidence."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from harness.isolation_evidence_fixtures import (
    files_from_patch, reissue_response, synthetic_candidate,
)
from harness.repository_isolation import (
    CANONICAL_SOURCES, CLAIM, CODE_INPUTS, CODE_ROOT, CORPUS_MANIFEST, INSUFFICIENT,
    MIN_FUNCTION_SHINGLES, ApiResponse, CandidateBug, FrozenReferenceUniverse,
    ReferenceUniverse, ReferenceUniverseMismatch, audit_source_coverage,
    authentication_problems, build_reference_universe, canonical_diff, canonical_repository,
    check_candidate,
    code_shingles, evidence_problems, extract_functions, freeze_reference_universe,
    isolation_record_is_current, load_frozen_reference_universe, normalised_patch,
    overlap_problems, patch_hash, reference_universe_receipt, verify_receipt,
)
from harness.source_identity import canonical_sha256
from scripts.audit_cross_split_near_duplicates import NEAR_DUPLICATE_JACCARD, normalise, shingles

ROOT = Path(__file__).resolve().parent.parent
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
BUGGY = NOVEL.replace("lo <= merged", "lo < merged")
REFERENCE_FUNCTION = ("def rolling_mean(values, window):\n    out = []\n"
                      "    for i in range(len(values) - window + 1):\n"
                      "        out.append(sum(values[i:i + window]) / window)\n    return out\n")


BUILDER_ARGUMENTS = {"buggy_text", "fixed_text", "fork_parent", "issue_number", "repository_id",
                     "extra_files", "committer_epoch", "author_epoch", "buggy_epoch"}


def complete_candidate(**overrides) -> CandidateBug:
    """Consistent synthetic evidence; builder arguments rebuild it, others override fields."""
    arguments = dict(buggy_text=BUGGY, fixed_text=NOVEL, target_function=BUGGY,
                     target_file="src/ranges.py", target_module="ranges")
    arguments.update({key: overrides.pop(key) for key in list(overrides)
                      if key in BUILDER_ARGUMENTS})
    return dataclasses.replace(synthetic_candidate(**arguments), **overrides)


# --- a synthetic data root ----------------------------------------------------------

FAKE_SOURCES = {"mbpp": 1, "humaneval": 1, "BugsInPy": 1, "SWE-bench Verified": 1}


def _write_manifest(root: Path, records_by_source: dict) -> None:
    path = root / CORPUS_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records_by_source": records_by_source}), encoding="utf-8")


def _write_fake_root(root: Path, mbpp_code: str = REFERENCE_FUNCTION) -> None:
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
        json.dumps({"task_id": 1, "code": mbpp_code}) + "\n", encoding="utf-8")
    (root / "data/humaneval").mkdir(parents=True)
    (root / "data/humaneval/humaneval_cache.json").write_text(json.dumps([{
        "task_id": "HumanEval/0", "prompt": "def f(x):\n", "canonical_solution": "    return x\n"}]),
        encoding="utf-8")
    (root / "data/train_stub.json").write_text('{"records": 1}\n', encoding="utf-8")
    _write_manifest(root, FAKE_SOURCES)


def _fake_universe(root: Path, code_root: Path = CODE_ROOT) -> ReferenceUniverse:
    universe = build_reference_universe(root, include_train_shard=False, code_root=code_root)
    # The synthetic root has no train view: bind a stub so only the tested input varies.
    universe.counts["train_shard"] = {"records": 1, "declared_records": 1}
    universe.bind("train_shard", root, root / "data/train_stub.json")
    return universe


def _freeze(root: Path, code_root: Path = CODE_ROOT):
    universe = _fake_universe(root, code_root)
    receipt = reference_universe_receipt(universe, code_root)
    return freeze_reference_universe(universe, receipt, root=root, code_root=code_root), receipt


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "repo"
    _write_fake_root(root)
    return root


@pytest.fixture
def frozen(fake_root):
    return _freeze(fake_root)[0]


# --- Phase 1: the universe is cryptographically bound to every decision ------------

def test_complete_candidate_is_admitted_with_a_current_record(frozen):
    record = check_candidate(complete_candidate(), frozen)
    assert record["admissible"] is True and record["reasons"] == [], record["reasons"]
    assert record["reference_universe_sha256"] == frozen.receipt_sha256
    assert record["claim"] == CLAIM
    assert record["stages"] == {"schema": [], "authentication": [], "overlap": []}
    assert isolation_record_is_current(record, frozen)


def test_an_arbitrary_receipt_hash_cannot_be_attached_to_an_isolation_record(frozen):
    arbitrary = "a" * 64
    candidate = complete_candidate()
    # No call shape accepts a caller-supplied hash.
    with pytest.raises(TypeError):
        check_candidate(candidate, arbitrary)
    with pytest.raises(TypeError):
        check_candidate(candidate, frozen, arbitrary)
    with pytest.raises(TypeError):
        check_candidate(candidate, frozen.universe)
    with pytest.raises(TypeError):
        FrozenReferenceUniverse(universe=frozen.universe, receipt_sha256=arbitrary)
    with pytest.raises(TypeError):
        FrozenReferenceUniverse._create(object(), frozen.universe, {}, {})
    with pytest.raises(AttributeError):
        frozen._receipt_sha256 = arbitrary
    # Editing a record to carry the arbitrary hash never makes it current, even
    # when its record digest is recomputed to match.
    record = check_candidate(candidate, frozen)
    forged = {**record, "reference_universe_sha256": arbitrary}
    assert not isolation_record_is_current(forged, frozen)
    from harness.repository_isolation import _record_digest
    forged["record_sha256"] = _record_digest(forged)
    assert not isolation_record_is_current(forged, frozen)
    with pytest.raises(TypeError):
        isolation_record_is_current(record, arbitrary)


def test_a_record_edited_after_the_decision_is_not_current(frozen):
    record = check_candidate(complete_candidate(), frozen)
    assert not isolation_record_is_current({**record, "admissible": False}, frozen)


def test_a_sealed_universe_cannot_be_mutated(frozen):
    with pytest.raises(AttributeError):
        frozen.universe.commits.add("f" * 40)
    with pytest.raises(AttributeError):
        frozen.universe.commits = set()
    with pytest.raises(TypeError):
        frozen.universe.functions["x#0"] = frozenset({"a"})
    receipt = frozen.receipt
    receipt["receipt_sha256"] = "a" * 64
    assert frozen.receipt["receipt_sha256"] != "a" * 64


def test_universe_a_cannot_be_paired_with_receipt_b(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _write_fake_root(root_a)
    _write_fake_root(root_b, mbpp_code=NOVEL)
    universe_a = _fake_universe(root_a)
    receipt_b = reference_universe_receipt(_fake_universe(root_b))
    assert verify_receipt(receipt_b)
    with pytest.raises(ReferenceUniverseMismatch) as refused:
        freeze_reference_universe(universe_a, receipt_b, root=root_a)
    assert any("function_fingerprints" in problem for problem in refused.value.problems)


def test_isolation_record_is_current_refuses_any_other_universe(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _write_fake_root(root_a)
    _write_fake_root(root_b, mbpp_code=NOVEL)
    frozen_a, frozen_b = _freeze(root_a)[0], _freeze(root_b)[0]
    assert frozen_a.receipt_sha256 != frozen_b.receipt_sha256
    record = check_candidate(complete_candidate(), frozen_a)
    assert isolation_record_is_current(record, frozen_a)
    assert not isolation_record_is_current(record, frozen_b)


def _mutate_commit(universe, root):
    universe.commits.add("e" * 40)


def _mutate_patch(universe, root):
    universe.patch_hashes.add(patch_hash("-a\n+b\n"))


def _mutate_function(universe, root):
    key = next(iter(universe.functions))
    universe.functions[key] = frozenset({"changed shingle"})


def _mutate_collection_count(universe, root):
    universe.counts["mbpp"]["tasks"] += 1


@pytest.mark.parametrize("mutate", [_mutate_commit, _mutate_patch, _mutate_function,
                                    _mutate_collection_count])
def test_a_changed_universe_invalidates_the_frozen_universe(fake_root, mutate):
    universe = _fake_universe(fake_root)
    receipt = reference_universe_receipt(universe)
    mutate(universe, fake_root)
    with pytest.raises(ReferenceUniverseMismatch):
        freeze_reference_universe(universe, receipt, root=fake_root)


def test_a_changed_input_file_invalidates_the_frozen_universe(fake_root):
    receipt = _freeze(fake_root)[1]
    universe = _fake_universe(fake_root)
    (fake_root / "data/humaneval/humaneval_cache.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ReferenceUniverseMismatch) as refused:
        freeze_reference_universe(universe, receipt, root=fake_root)
    assert "input file changed or missing: data/humaneval/humaneval_cache.json" in \
        refused.value.problems


def _code_root_copy(tmp_path: Path) -> Path:
    code_root = tmp_path / "code"
    for relative in {*CANONICAL_SOURCES, *(p for paths in CODE_INPUTS.values() for p in paths)}:
        (code_root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CODE_ROOT / relative, code_root / relative)
    return code_root


@pytest.mark.parametrize("relative", ["utils/dataset_identity.py", "harness/bugsinpy_loader.py",
                                      "scripts/audit_cross_split_near_duplicates.py",
                                      "harness/corpus_view.py"])
def test_a_changed_source_file_invalidates_the_frozen_universe(fake_root, tmp_path, relative):
    code_root = _code_root_copy(tmp_path)
    frozen_, receipt = _freeze(fake_root, code_root)
    assert frozen_.verification["canonical_sources_verified"] == len(CANONICAL_SOURCES)
    universe = _fake_universe(fake_root, code_root) if relative != \
        "harness/bugsinpy_loader.py" else None
    with (code_root / relative).open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    if universe is None:
        # The curated definition no longer matches the imported module: refused at build.
        with pytest.raises(ValueError):
            _fake_universe(fake_root, code_root)
        universe = _fake_universe(fake_root)
    with pytest.raises(ReferenceUniverseMismatch) as refused:
        freeze_reference_universe(universe, receipt, root=fake_root, code_root=code_root)
    assert any(relative in problem for problem in refused.value.problems)


def test_a_changed_collection_in_a_resealed_receipt_is_refused(fake_root):
    universe = _fake_universe(fake_root)
    receipt = reference_universe_receipt(universe)
    tampered = json.loads(json.dumps(receipt))
    tampered["collections"]["commits"].append("f" * 40)
    assert not verify_receipt(tampered)
    with pytest.raises(ReferenceUniverseMismatch):
        freeze_reference_universe(universe, tampered, root=fake_root)
    # Re-seal every hash so the receipt is internally consistent: still refused,
    # because the recomputed universe differs.
    from harness.repository_isolation import _sha_json
    tampered["collection_sha256"]["commits"] = _sha_json(tampered["collections"]["commits"])
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = _sha_json(tampered)
    assert verify_receipt(tampered)
    with pytest.raises(ReferenceUniverseMismatch) as refused:
        freeze_reference_universe(universe, tampered, root=fake_root)
    assert "collection 'commits' differs from the receipt" in refused.value.problems


def test_a_receipt_missing_a_canonical_source_is_refused(fake_root):
    universe = _fake_universe(fake_root)
    receipt = reference_universe_receipt(universe)
    receipt["source_files_sha256"].pop("utils/dataset_identity.py")
    from harness.repository_isolation import _sha_json
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha_json(receipt)
    with pytest.raises(ReferenceUniverseMismatch) as refused:
        freeze_reference_universe(universe, receipt, root=fake_root)
    assert "receipt does not bind exactly the canonical source set" in refused.value.problems


def test_receipt_is_deterministic(fake_root):
    first = reference_universe_receipt(_fake_universe(fake_root))
    second = reference_universe_receipt(_fake_universe(fake_root))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert verify_receipt(first)


# --- Phase 2: every coverage input is bound -----------------------------------------------

def test_receipt_binds_the_manifest_and_classification_code(fake_root):
    frozen_, receipt = _freeze(fake_root)
    inputs = receipt["collections"]["input_files"]["corpus_manifest"]
    assert list(inputs) == [CORPUS_MANIFEST]
    code = receipt["collections"]["code_input_files"]
    assert "utils/dataset_identity.py" in code["corpus_manifest"]
    assert set(code["manual_curated_examples"]) == {"harness/bugsinpy_loader.py",
                                                    "scripts/build_corpus_v1.py"}
    for relative in ("harness/bugsinpy_loader.py", "utils/dataset_identity.py",
                     "harness/corpus_view.py", "harness/corpus.py",
                     "harness/function_complexity.py", "harness/source_identity.py"):
        assert relative in receipt["source_files_sha256"]
    assert receipt["coverage"]["corpus_manifest_sha256"] == \
        hashlib.sha256((fake_root / CORPUS_MANIFEST).read_bytes()).hexdigest()
    assert frozen_.verification["corpus_manifest_sha256"] == \
        receipt["coverage"]["corpus_manifest_sha256"]


@pytest.mark.parametrize("records_by_source", [
    {**FAKE_SOURCES, "mbpp": 2},                                  # changed count
    {key: value for key, value in FAKE_SOURCES.items() if key != "humaneval"},  # dropped
    {**FAKE_SOURCES, "manual_curated_examples": 8},               # added source
])
def test_a_changed_manifest_refuses_the_old_receipt(fake_root, records_by_source):
    receipt = _freeze(fake_root)[1]
    _write_manifest(fake_root, records_by_source)
    with pytest.raises(ReferenceUniverseMismatch) as refused:
        freeze_reference_universe(_fake_universe(fake_root), receipt, root=fake_root)
    assert f"input file changed or missing: {CORPUS_MANIFEST}" in refused.value.problems
    assert "collection 'counts' differs from the receipt" in refused.value.problems


def test_coverage_detects_a_manifest_changed_after_indexing(fake_root):
    universe = _fake_universe(fake_root)
    assert audit_source_coverage(universe, fake_root)["covered"] is True
    _write_manifest(fake_root, {**FAKE_SOURCES, "mbpp": 9})
    audit = audit_source_coverage(universe, fake_root)
    assert "corpus manifest changed since it was indexed" in audit["problems"]


def test_declared_but_unindexed_unknown_and_zero_sources_fail(fake_root):
    universe = _fake_universe(fake_root)
    universe.counts.pop("humaneval")
    assert "declared source 'humaneval' is not indexed" in \
        audit_source_coverage(universe)["problems"]
    _write_manifest(fake_root, {**FAKE_SOURCES, "new_source": 3})
    assert "unknown corpus source 'new_source'" in \
        audit_source_coverage(_fake_universe(fake_root))["problems"]
    _write_manifest(fake_root, FAKE_SOURCES)
    universe = _fake_universe(fake_root)
    universe.counts["mbpp"] = {"tasks": 0, "functions": 0}
    assert "source 'mbpp' indexed with zero items" in audit_source_coverage(universe)["problems"]


@pytest.mark.parametrize("relative", [
    CORPUS_MANIFEST,
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
    assert audit_source_coverage(universe)["covered"] is False


def test_curated_seeds_are_indexed_from_their_original_definition(fake_root):
    from harness.bugsinpy_loader import CURATED_BUGSINPY_BUGS
    universe = _fake_universe(fake_root)
    curated = universe.counts["manual_curated_examples"]
    assert curated["definition_ids"] == sorted(f"curated::{b['id']}"
                                               for b in CURATED_BUGSINPY_BUGS)
    assert "harness/bugsinpy_loader.py" in universe.code_inputs["manual_curated_examples"]


# --- Phase 3: evidence schema, authentication and overlap are separate stages ---------

def _schema_refused(candidate: CandidateBug, expected: str, frozen_) -> None:
    record = check_candidate(candidate, frozen_)
    assert record["admissible"] is False and record["insufficient_evidence"] is True
    assert f"{INSUFFICIENT}:{expected}" in record["reasons"], record["reasons"]
    assert record["stages"]["authentication"] == ["not_run:schema_invalid"]


@pytest.mark.parametrize("overrides, expected", [
    (dict(patch=""), "patch_empty"),
    (dict(patch="--- a/x\n+++ b/x\n context only\n"), "patch_empty"),
    (dict(target_function=""), "target_function_missing_or_unparseable"),
    (dict(target_function="   \n\t"), "target_function_missing_or_unparseable"),
    (dict(target_function="x = (1,"), "target_function_missing_or_unparseable"),
    (dict(target_function="value = 3\n"), "target_function_missing_or_unparseable"),
    (dict(target_function="def f():\n    return 1\n"), "target_function_too_short_for_comparison"),
    (dict(buggy_commit=""), "buggy_commit_not_full_sha"),
    (dict(fixed_commit="2" * 7), "fixed_commit_not_full_sha"),
    (dict(buggy_commit="G" * 40), "buggy_commit_not_full_sha"),
    (dict(target_blob_buggy=""), "target_blob_buggy_not_full_sha"),
    (dict(licence_blob=""), "licence_blob_not_full_sha"),
    (dict(fork_status="unknown"), "fork_status_unknown"),
    (dict(fork_status=""), "fork_status_unknown"),
    (dict(fork_status="fork"), "fork_parent_identity_missing"),
    (dict(fork_parent="pandas-dev/pandas", fork_parent_node_id=""),
     "fork_parent_node_id_missing"),
    (dict(fork_parent="pandas-dev/pandas", fork_parent_id=""), "fork_parent_id_missing"),
    (dict(repository="Example/Ranges"), "repository_not_canonical_owner_name"),
    (dict(repository_url="https://gitlab.com/example-org/ranges"),
     "repository_url_unverified_or_inconsistent"),
    (dict(repository_id=""), "repository_identity_missing"),
    (dict(repository_node_id=""), "repository_node_id_missing"),
    (dict(licence_spdx="GPL-3.0"), "licence_evidence_missing_or_not_admitted"),
    (dict(licence_sha256=""), "licence_file_hash_missing"),
    (dict(licence_path=""), "licence_path_missing"),
    (dict(issue_id=""), "issue_identity_missing"),
    (dict(issue_id="other/repo#1"), "issue_repository_mismatch"),
    (dict(target_file=""), "target_file_missing"),
    (dict(target_file="../x.py"), "target_file_missing"),
    (dict(target_module=""), "target_module_missing"),
    (dict(repository_metadata=ApiResponse()), "repository_metadata_response_missing"),
    (dict(issue_metadata=ApiResponse()), "issue_metadata_response_missing"),
    (dict(git_objects=()), "git_object_evidence_missing"),
])
def test_missing_or_malformed_evidence_refuses_admission(frozen, overrides, expected):
    _schema_refused(complete_candidate(**overrides), expected, frozen)


def test_evidence_free_candidates_are_refused(frozen):
    _schema_refused(CandidateBug(), "repository_not_canonical_owner_name", frozen)
    _schema_refused(CandidateBug(repository="example-org/ranges"), "patch_empty", frozen)
    assert len(evidence_problems(CandidateBug())) >= 15


def test_retrieval_timestamp_and_api_urls_are_required():
    base = complete_candidate()
    stale = dataclasses.replace(base.repository_metadata, retrieved_utc="yesterday")
    assert f"{INSUFFICIENT}:repository_metadata_retrieval_timestamp_missing" in \
        evidence_problems(dataclasses.replace(base, repository_metadata=stale))
    wrong = dataclasses.replace(base.issue_metadata,
                                url="https://api.github.com/repos/other/repo/issues/1")
    assert f"{INSUFFICIENT}:issue_metadata_url_missing_or_inconsistent" in \
        evidence_problems(dataclasses.replace(base, issue_metadata=wrong))


def _authentication(candidate: CandidateBug) -> list[str]:
    assert evidence_problems(candidate) == [], evidence_problems(candidate)
    problems, _ = authentication_problems(candidate)
    return [problem.rsplit(":", 1)[1] for problem in problems]


def test_consistent_evidence_authenticates():
    problems, derived = authentication_problems(complete_candidate())
    assert problems == []
    assert derived["changed_files"] == ["src/ranges.py"]
    assert derived["submitted_patch_matches_derived_diff"] is True
    assert derived["target_qualname"] == "merge_ranges"
    assert derived["temporal_rule_satisfied"] is True


def _other(**overrides):
    return complete_candidate(**overrides)


AUTHENTICATION_CASES = {
    "repository_metadata_hash_mismatch": lambda c: dataclasses.replace(
        c, repository_metadata=dataclasses.replace(c.repository_metadata,
                                                   body=c.repository_metadata.body + b" ")),
    "repository_id_mismatch": lambda c: dataclasses.replace(
        c, repository_metadata=reissue_response(c.repository_metadata, id=999)),
    "repository_node_id_mismatch": lambda c: dataclasses.replace(
        c, repository_metadata=reissue_response(c.repository_metadata, node_id="R_otherxxxx")),
    "repository_renamed_or_redirected": lambda c: dataclasses.replace(
        c, repository_metadata=reissue_response(c.repository_metadata,
                                                full_name="new-org/ranges")),
    "repository_html_url_mismatch": lambda c: dataclasses.replace(
        c, repository_metadata=reissue_response(c.repository_metadata,
                                                html_url="https://github.com/x/y")),
    "fork_status_mismatch": lambda c: dataclasses.replace(
        c, repository_metadata=reissue_response(c.repository_metadata, fork=True)),
    "undeclared_fork_parent": lambda c: dataclasses.replace(
        c, repository_metadata=reissue_response(c.repository_metadata, parent={
            "full_name": "pandas-dev/pandas", "id": 7, "node_id": "R_pandasnode"})),
    "licence_spdx_mismatch": lambda c: dataclasses.replace(
        c, repository_metadata=reissue_response(c.repository_metadata,
                                                license={"spdx_id": "Apache-2.0"})),
    "issue_metadata_hash_mismatch": lambda c: dataclasses.replace(
        c, issue_metadata=dataclasses.replace(c.issue_metadata, sha256="0" * 64)),
    "issue_number_mismatch": lambda c: dataclasses.replace(
        c, issue_metadata=reissue_response(c.issue_metadata, number=2)),
    "issue_repository_identity_mismatch": lambda c: dataclasses.replace(
        c, issue_metadata=reissue_response(
            c.issue_metadata, repository_url="https://api.github.com/repos/other/repo")),
    "licence_file_hash_mismatch": lambda c: dataclasses.replace(c, licence_sha256="0" * 64),
    "licence_blob_not_at_licence_path_in_buggy_tree": lambda c: dataclasses.replace(
        c, licence_path="COPYING"),
    "fixed_commit_object_missing": lambda c: dataclasses.replace(
        c, git_objects=tuple(o for o in c.git_objects if not (
            o.kind == "commit" and b"parent " in o.body))),
    "fixed_commit_parent_is_not_buggy_commit": lambda c: dataclasses.replace(
        c, buggy_commit=_other(buggy_text=BUGGY + "\n# other\n").buggy_commit,
        git_objects=c.git_objects + _other(buggy_text=BUGGY + "\n# other\n").git_objects),
    "target_blob_buggy_not_at_target_file": lambda c: dataclasses.replace(
        c, target_blob_buggy=c.licence_blob),
    "target_function_not_in_buggy_file": lambda c: dataclasses.replace(c, target_function=NOVEL),
    "submitted_patch_does_not_match_derived_diff": lambda c: dataclasses.replace(
        c, patch=c.patch.replace("+        if lo <= merged", "+        if lo >= merged")),
    "target_module_inconsistent_with_target_file": lambda c: dataclasses.replace(
        c, target_module="elsewhere"),
}


@pytest.mark.parametrize("expected", sorted(AUTHENTICATION_CASES))
def test_authentication_cross_checks_content_not_format(expected):
    tampered = AUTHENTICATION_CASES[expected](complete_candidate())
    assert expected in _authentication(tampered)


def test_fork_evidence_is_authenticated_then_checked_for_overlap(frozen):
    fork = complete_candidate(fork_parent="demo-org/demo")
    record = check_candidate(fork, frozen)
    assert record["stages"]["schema"] == [] and record["stages"]["authentication"] == []
    assert "fork_parent_in_reference_universe:demo-org/demo" in record["stages"]["overlap"]
    assert record["insufficient_evidence"] is False and record["admissible"] is False


def test_overlap_stage_rejects_repository_commit_patch_and_function_lineage(frozen):
    universe = frozen.universe
    threshold = NEAR_DUPLICATE_JACCARD
    assert any(r.startswith("repository_in_reference_universe") for r in overlap_problems(
        CandidateBug(repository="someone/widgets"), universe, threshold)[0])
    assert "fixed_commit_is_known_benchmark_commit" in overlap_problems(
        CandidateBug(fixed_commit="4" * 40), universe, threshold)[0]
    reformatted = PATCH.replace("return sum(xs)", "return   sum( xs )")
    assert "patch_near_duplicate_of_known_bug" in overlap_problems(
        CandidateBug(), universe, threshold, derived_diff=reformatted)[0]
    # A submitted patch is never used for patch checks: only the derived diff is.
    reasons, nearest = overlap_problems(CandidateBug(patch=PATCH), universe, threshold)
    assert "patch_identical_to_known_bug" not in reasons
    assert nearest["patch_checks"] == "not_run:no_authenticated_diff"
    renamed = REFERENCE_FUNCTION.replace("rolling_mean", "moving_average")
    reasons, nearest = overlap_problems(CandidateBug(target_function=renamed), universe,
                                        threshold)
    assert "function_near_duplicate_of_reference" in reasons
    assert nearest["nearest_function"]["jaccard"] >= NEAR_DUPLICATE_JACCARD


KNOWN_BUGGY = ("def total(xs):\n    values = [x for x in xs if x is not None]\n"
               "    count = len(values)\n    if count == 0:\n        return 0\n"
               "    return sum(xs[1:])\n")
KNOWN_FIXED = KNOWN_BUGGY.replace("return sum(xs[1:])", "return sum(xs)")


def test_a_fully_authenticated_known_patch_is_refused_by_the_overlap_stage(frozen):
    candidate = synthetic_candidate(buggy_text=KNOWN_BUGGY, fixed_text=KNOWN_FIXED,
                                    target_function=KNOWN_BUGGY, target_file="pkg/mod.py",
                                    target_module="mod")
    record = check_candidate(candidate, frozen)
    assert record["stages"]["schema"] == [] and record["stages"]["authentication"] == []
    assert "patch_identical_to_known_bug" in record["stages"]["overlap"]


def test_the_isolation_module_makes_no_network_calls():
    source = (CODE_ROOT / "harness/repository_isolation.py").read_text(encoding="utf-8")
    fixtures = (CODE_ROOT / "harness/isolation_evidence_fixtures.py").read_text(encoding="utf-8")
    for text in (source, fixtures):
        for module in ("requests", "urllib", "http.client", "socket", "subprocess"):
            assert f"import {module}" not in text and f"from {module}" not in text


def test_helpers_reuse_the_frozen_audit():
    assert normalise('def f():\n    """doc"""\n    return 1\n') == "def f():\n    return 1"
    assert canonical_repository("https://github.com/Pandas-Dev/pandas.git/") == \
        ("pandas-dev/pandas", "pandas")
    assert len(extract_functions("def a():\n    return 1\n\nclass K:\n    def b(self):\n"
                                 "        return 2\n")) == 2
    assert len(code_shingles(NOVEL)) >= MIN_FUNCTION_SHINGLES
    assert shingles(normalised_patch(PATCH).replace("\n", " "))


# --- exact diff: the patch is derived from authenticated blobs (policy A) -------------

SETTING_BUGGY, SETTING_FIXED = "TIMEOUT = 10\n\n\n", "TIMEOUT = 30\n\n\n"
OTHER_FUNCTION = ("def rolling_total(values, window):\n    out = []\n"
                  "    for i in range(len(values) - window + 1):\n"
                  "        out.append(sum(values[i:i + window]))\n    return out\n")


def _two_change_candidate(submitted: str | None, **overrides) -> CandidateBug:
    """Buggy/fixed blobs with two real changes: the target function and a setting."""
    buggy, fixed = SETTING_BUGGY + BUGGY, SETTING_FIXED + NOVEL
    return synthetic_candidate(buggy_text=buggy, fixed_text=fixed, target_function=BUGGY,
                               target_file="src/ranges.py", target_module="ranges",
                               patch=submitted, **overrides)


def _auth_reasons(candidate: CandidateBug) -> list[str]:
    assert evidence_problems(candidate) == []
    return [problem.rsplit(":", 1)[1] for problem in authentication_problems(candidate)[0]]


def test_partial_patch_with_only_an_unrelated_valid_change_is_refused(frozen):
    # Regression for the reported case: before the fix both stages returned [].
    partial = canonical_diff("src/ranges.py", SETTING_BUGGY + BUGGY, SETTING_FIXED + BUGGY)
    candidate = _two_change_candidate(partial)
    assert evidence_problems(candidate) == []
    assert "submitted_patch_does_not_match_derived_diff" in _auth_reasons(candidate)
    record = check_candidate(candidate, frozen)
    assert record["admissible"] is False and record["insufficient_evidence"] is True
    derived = record["derived_from_verified_evidence"]
    assert derived["submitted_patch_matches_derived_diff"] is False
    # Patch lineage is taken from the derived diff, never the submitted one.
    full = canonical_diff("src/ranges.py", SETTING_BUGGY + BUGGY, SETTING_FIXED + NOVEL)
    assert record["patch_sha256"] == patch_hash(full) != patch_hash(partial)


def test_patch_omitting_one_of_two_real_changes_is_refused():
    function_only = canonical_diff("src/ranges.py", SETTING_BUGGY + BUGGY, SETTING_BUGGY + NOVEL)
    assert "submitted_patch_does_not_match_derived_diff" in _auth_reasons(
        _two_change_candidate(function_only))


def test_the_complete_two_change_patch_is_admitted_with_derived_facts(frozen):
    record = check_candidate(_two_change_candidate(None), frozen)
    assert record["admissible"] is True, record["reasons"]
    derived = record["derived_from_verified_evidence"]
    assert derived["changed_files"] == ["src/ranges.py"]
    assert derived["target_span_buggy"] == [4, 12] and derived["target_span_fixed"] == [4, 12]
    assert len(derived["changed_hunks"]) == 2
    assert derived["canonical_diff_sha256"] and record["authentication_passed"] is True


def test_patch_with_invented_lines_is_refused():
    candidate = complete_candidate()
    invented = candidate.patch.replace("+        if lo <= merged[-1][1]:",
                                       "+        if lo <= merged[-1][1] or True:")
    assert invented != candidate.patch
    assert "submitted_patch_does_not_match_derived_diff" in _auth_reasons(
        dataclasses.replace(candidate, patch=invented))
    extra_hunk = candidate.patch + "@@ -20,0 +21,1 @@\n+invented = 1\n"
    assert "submitted_patch_does_not_match_derived_diff" in _auth_reasons(
        dataclasses.replace(candidate, patch=extra_hunk))


def test_patch_from_another_file_is_refused():
    candidate = complete_candidate()
    elsewhere = canonical_diff("src/other.py", BUGGY, NOVEL)
    assert "submitted_patch_does_not_match_derived_diff" in _auth_reasons(
        dataclasses.replace(candidate, patch=elsewhere))


def test_target_file_changed_but_declared_function_unchanged_is_refused():
    buggy = BUGGY + "\n\n" + OTHER_FUNCTION
    fixed = BUGGY + "\n\n" + OTHER_FUNCTION.replace("window + 1", "window")
    reasons = _auth_reasons(synthetic_candidate(
        buggy_text=buggy, fixed_text=fixed, target_function=BUGGY, target_file="src/ranges.py",
        target_module="ranges"))
    assert "declared_target_function_unchanged" in reasons
    assert "no_changed_hunk_overlaps_target_function" in reasons


def test_a_fix_in_an_unrelated_file_is_refused():
    reasons = _auth_reasons(synthetic_candidate(
        buggy_text=BUGGY, fixed_text=BUGGY, target_function=BUGGY, target_file="src/ranges.py",
        target_module="ranges", patch=canonical_diff("src/util.py", BUGGY, NOVEL),
        extra_files={"src/util.py": (BUGGY, NOVEL)}))
    assert "target_file_unchanged_by_fix" in reasons


def test_known_benchmark_patch_disguised_as_a_novel_subset_is_refused(frozen):
    buggy, fixed = KNOWN_BUGGY + "\n\n" + BUGGY, KNOWN_FIXED + "\n\n" + NOVEL
    novel_subset = canonical_diff("src/ranges.py", buggy, KNOWN_BUGGY + "\n\n" + NOVEL)
    candidate = synthetic_candidate(buggy_text=buggy, fixed_text=fixed, target_function=BUGGY,
                                    target_file="src/ranges.py", target_module="ranges",
                                    patch=novel_subset)
    record = check_candidate(candidate, frozen)
    assert record["admissible"] is False
    assert f"{INSUFFICIENT}:authentication:submitted_patch_does_not_match_derived_diff" in \
        record["reasons"]
    # The derived diff, not the submission, carries the known benchmark change.
    assert record["patch_sha256"] == patch_hash(canonical_diff("src/ranges.py", buggy, fixed))


def test_a_complete_multi_file_fix_is_refused_under_policy_a():
    candidate = synthetic_candidate(
        buggy_text=BUGGY, fixed_text=NOVEL, target_function=BUGGY, target_file="src/ranges.py",
        target_module="ranges", extra_files={"src/util.py": (OTHER_FUNCTION, OTHER_FUNCTION
                                                             + "\n# changed\n")})
    problems, derived = authentication_problems(candidate)
    assert derived["changed_files"] == ["src/ranges.py", "src/util.py"]
    assert f"{INSUFFICIENT}:authentication:multi_file_fix_not_admitted_under_single_file_policy" \
        in problems
    from harness.repository_isolation import DIFF_POLICY
    assert DIFF_POLICY["id"].startswith("A_") and "refused" in DIFF_POLICY["multi_file_fixes"]


def test_missing_tree_evidence_for_a_changed_subtree_is_refused():
    candidate = synthetic_candidate(
        buggy_text=BUGGY, fixed_text=NOVEL, target_function=BUGGY, target_file="src/ranges.py",
        target_module="ranges", extra_files={"docs/a/notes.py": ("x = 1\n", "x = 2\n")})
    from harness.repository_isolation import git_object_id, parse_commit, parse_tree
    objects = {git_object_id(o.kind, o.body): o for o in candidate.git_objects}
    fixed_root = parse_commit(objects[candidate.fixed_commit].body)["tree"]
    docs_tree = parse_tree(objects[fixed_root].body)["docs"][1]
    stripped = tuple(o for o in candidate.git_objects
                     if git_object_id(o.kind, o.body) != docs_tree)
    reasons = _auth_reasons(dataclasses.replace(candidate, git_objects=stripped))
    assert any(reason.startswith("changed_files_not_fully_evidenced") or reason == "docs/"
               for reason in reasons), reasons


def test_a_merge_commit_is_not_a_direct_parent_fix():
    candidate = complete_candidate()
    from harness.repository_isolation import GitObject, git_object_id
    objects = {git_object_id(o.kind, o.body): o for o in candidate.git_objects}
    body = objects[candidate.fixed_commit].body
    merged = body.replace(b"\nauthor ", b"\nparent " + b"9" * 40 + b"\nauthor ", 1)
    merge_id = git_object_id("commit", merged)
    reasons = _auth_reasons(dataclasses.replace(
        candidate, fixed_commit=merge_id,
        git_objects=candidate.git_objects + (GitObject("commit", merged),)))
    assert "fixed_commit_is_not_a_direct_single_parent_commit" in reasons


# --- temporal rule (frozen; contamination-risk mitigation only) ----------------------

CUTOFF = 1735689600


@pytest.mark.parametrize("epoch, admitted", [(CUTOFF - 1, False), (CUTOFF, True),
                                             (CUTOFF + 86400, True)])
def test_temporal_boundary(frozen, epoch, admitted):
    candidate = complete_candidate(committer_epoch=epoch, buggy_epoch=CUTOFF - 7200)
    record = check_candidate(candidate, frozen)
    assert record["admissible"] is admitted, record["reasons"]
    assert (f"{INSUFFICIENT}:authentication:fixed_commit_before_temporal_cutoff" in
            record["reasons"]) is (not admitted)
    assert record["derived_from_verified_evidence"]["temporal_rule_satisfied"] is admitted


@pytest.mark.parametrize("arguments, expected", [
    (dict(committer_epoch=None), "fixed_commit_timestamp_missing"),
    (dict(author_epoch=1740787200 + 3600), "fixed_commit_author_after_committer"),
    (dict(buggy_epoch=1740787200 + 60), "buggy_commit_committed_after_fix"),
    (dict(committer_epoch=1830000000, buggy_epoch=1829990000),
     "fixed_commit_later_than_evidence_retrieval"),
    (dict(buggy_epoch=None), "buggy_commit_timestamp_missing"),
])
def test_missing_or_contradictory_timestamps_are_refused(arguments, expected):
    assert expected in _auth_reasons(complete_candidate(**arguments))


def test_the_rules_are_frozen_in_the_receipt(fake_root):
    from harness.repository_isolation import DIFF_POLICY, TEMPORAL_RULE, _sha_json
    universe = _fake_universe(fake_root)
    receipt = reference_universe_receipt(universe)
    assert receipt["temporal_rule"] == TEMPORAL_RULE and receipt["diff_policy"] == DIFF_POLICY
    assert "mitigation" in TEMPORAL_RULE["status"]
    for key, value in (("temporal_rule", {**TEMPORAL_RULE, "cutoff_epoch": 0}),
                       ("diff_policy", {**DIFF_POLICY, "multi_file_fixes": "admitted"})):
        altered = json.loads(json.dumps(receipt))
        altered[key] = value
        altered.pop("receipt_sha256")
        altered["receipt_sha256"] = _sha_json(altered)
        with pytest.raises(ReferenceUniverseMismatch):
            freeze_reference_universe(universe, altered, root=fake_root)


# --- integrity: record_sha256 is a self-hash, trust comes from revalidation ------------

def test_downstream_revalidation_rejects_an_edited_record_with_a_recomputed_hash(
        fake_root, tmp_path):
    from harness.repository_isolation import (
        _record_digest, candidate_from_sidecar, candidate_to_sidecar,
        revalidate_isolation_record,
    )
    frozen_, receipt = _freeze(fake_root)
    candidate = complete_candidate()
    record = check_candidate(candidate, frozen_)
    # Retain the authenticated evidence sidecar and the record, as a dataset would.
    (tmp_path / "evidence.json").write_text(json.dumps(candidate_to_sidecar(candidate)),
                                            encoding="utf-8")
    (tmp_path / "record.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    sidecar = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    stored = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert candidate_from_sidecar(sidecar) == candidate
    # Reload the same frozen universe from its receipt.
    reloaded = freeze_reference_universe(
        _fake_universe(fake_root), json.loads((tmp_path / "receipt.json").read_text()),
        root=fake_root)
    assert reloaded.receipt_sha256 == frozen_.receipt_sha256
    assert revalidate_isolation_record(stored, sidecar, reloaded)["valid"] is True
    for edit in ({"admissible": False}, {"reasons": ["manually_excluded"]},
                 {"admissible": False, "reasons": ["x"], "insufficient_evidence": True}):
        forged = {**stored, **edit}
        forged["record_sha256"] = _record_digest(forged)
        # The self-hash cannot tell: the forged record still looks current...
        assert isolation_record_is_current(forged, reloaded)
        # ...but revalidation from the evidence rejects it.
        result = revalidate_isolation_record(forged, sidecar, reloaded)
        assert result["valid"] is False and set(edit) <= set(result["differing_fields"])
    refused = check_candidate(dataclasses.replace(candidate, licence_sha256="0" * 64), frozen_)
    promoted = {**refused, "admissible": True, "reasons": []}
    promoted["record_sha256"] = _record_digest(promoted)
    tampered_sidecar = candidate_to_sidecar(dataclasses.replace(candidate,
                                                                licence_sha256="0" * 64))
    assert revalidate_isolation_record(promoted, tampered_sidecar, reloaded)["valid"] is False


# --- the real universe -------------------------------------------------------------

BUNDLE_STORE = ROOT / "results" / "next_direction_bundle"


@pytest.fixture(scope="module")
def current_bundle():
    from harness.atomic_publish import read_current_bundle
    if not (BUNDLE_STORE / "CURRENT").exists():
        pytest.skip("no published bundle")
    return read_current_bundle(BUNDLE_STORE)


@pytest.fixture(scope="module")
def real_frozen(current_bundle):
    if not (ROOT / "data" / "BugsInPy_repo" / "projects").exists():
        pytest.skip("upstream sources not present")
    receipt = json.loads(current_bundle["files"]["reference_universe_receipt.json"])
    return freeze_reference_universe(build_reference_universe(ROOT), receipt, root=ROOT)


def test_tracked_receipt_loads_as_a_verified_frozen_universe(real_frozen, current_bundle):
    receipt_bytes = current_bundle["files"]["reference_universe_receipt.json"]
    receipt = json.loads(receipt_bytes)
    assert real_frozen.receipt_sha256 == receipt["receipt_sha256"]
    verification = real_frozen.verification
    assert verification["input_files_verified"] == sum(
        len(files) for files in receipt["collections"]["input_files"].values())
    assert verification["canonical_sources_verified"] == len(CANONICAL_SOURCES)
    assert b"\r" not in receipt_bytes
    for relative, expected in receipt["source_files_sha256"].items():
        assert canonical_sha256(ROOT / relative) == expected, relative


def test_tracked_design_binds_the_bundle(current_bundle):
    files = current_bundle["files"]
    design = json.loads(files["next_direction_design.json"])
    isolation = design["isolation"]
    assert isolation["reference_universe_receipt_file_sha256"] == hashlib.sha256(
        files["reference_universe_receipt.json"]).hexdigest()
    assert design["power_analysis"]["sha256"] == hashlib.sha256(
        files["repository_native_power_analysis.json"]).hexdigest()
    assert isolation["coverage"]["covered"] and isolation["self_checks_pass"]
    assert all(check["self_check_passed"] for check in isolation["self_checks"].values())
    assert "proven" not in json.dumps(isolation).lower()
    assert not any(value for key, value in design["leakage"].items() if key != "splits_opened")


def test_complete_real_universe_passes_coverage(real_frozen):
    universe = real_frozen.universe
    audit = audit_source_coverage(universe, ROOT)
    assert audit["covered"] is True, audit["problems"]
    assert universe.counts["BugsInPy"]["projects"] == 17
    assert universe.counts["BugsInPy"]["bugs"] == 501
    assert universe.counts["SWE-bench Verified"]["instances"] == 500
    assert universe.counts["mbpp"]["tasks"] == 974
    assert universe.counts["humaneval"]["tasks"] == 164
    assert len(universe.inputs["BugsInPy"]) == 17 + 2 * 501
    assert CORPUS_MANIFEST in universe.inputs["corpus_manifest"]


def test_all_eight_curated_corpus_seeds_are_indexed(real_frozen):
    universe = real_frozen.universe
    manifest = json.loads((ROOT / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["records_by_source"]["manual_curated_examples"] == 8
    curated = universe.counts["manual_curated_examples"]
    indexed = set(curated["definition_ids"])
    train = set(universe.counts["train_shard"]["curated_ids"])
    assert len(indexed) == 10 and len(train) == 7 and train <= indexed
    # The only generator of curated records is this definition; all ten are
    # indexed, a conservative superset of the eight corpus seeds.
    assert len(indexed - train) >= 8 - len(train)


def test_real_self_check_candidates(real_frozen):
    from scripts.build_next_direction_design import run_self_checks
    checks, passed = run_self_checks(real_frozen)
    assert passed, {name: check["reasons"] for name, check in checks.items()}
