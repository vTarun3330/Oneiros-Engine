"""Tests for the executable isolation checks of a future repository-native set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.repository_isolation import (
    CandidateBug, ReferenceUniverse, canonical_repository, check_candidate,
    corpus_sources_are_covered, extract_functions, normalised_patch, patch_hash,
)
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
REFERENCE_FUNCTION = ("def rolling_mean(values, window):\n    out = []\n"
                      "    for i in range(len(values) - window + 1):\n"
                      "        out.append(sum(values[i:i + window]) / window)\n    return out\n")


def _universe() -> ReferenceUniverse:
    universe = ReferenceUniverse()
    universe.add_repository("https://github.com/pandas-dev/pandas.git", "BugsInPy")
    universe.add_repository("youtube_dl", "legacy_real_bugs")
    universe.commits.add("abc123")
    universe.instance_ids.add("django__django-1234")
    universe.patch_hashes.add(patch_hash(PATCH))
    universe.patch_shingles["known::1"] = shingles(normalised_patch(PATCH).replace("\n", " "))
    universe.add_function_source("mbpp::1", REFERENCE_FUNCTION)
    return universe


def test_canonical_repository_normalises_urls_and_names():
    assert canonical_repository("https://github.com/Pandas-Dev/pandas.git/") == \
        ("pandas-dev/pandas", "pandas")
    assert canonical_repository("youtube_dl") == ("", "youtube-dl")
    assert canonical_repository("github.com/ytdl-org/youtube-dl")[1] == "youtube-dl"


def test_repository_fork_commit_issue_and_patch_lineage_are_rejected():
    universe = _universe()
    assert "repository_in_reference_universe:pandas-dev/pandas" in check_candidate(
        CandidateBug(repository="pandas-dev/pandas"), universe)["reasons"]
    assert check_candidate(CandidateBug(repository="someone/youtube-dl"),
                           universe)["admissible"] is False
    assert "fork_parent_in_reference_universe:pandas-dev/pandas" in check_candidate(
        CandidateBug(repository="me/new", fork_parent="pandas-dev/pandas"),
        universe)["reasons"]
    assert "fixed_commit_is_known_benchmark_commit" in check_candidate(
        CandidateBug(repository="me/new", fixed_commit="abc123"), universe)["reasons"]
    assert "issue_is_known_benchmark_instance" in check_candidate(
        CandidateBug(repository="me/new", issue_id="django__django-1234"), universe)["reasons"]
    reformatted = PATCH.replace("return sum(xs)", "return   sum( xs )")
    result = check_candidate(CandidateBug(repository="me/new", patch=reformatted), universe)
    assert "patch_near_duplicate_of_known_bug" in result["reasons"]


def test_function_near_duplicate_survives_renaming_and_docstrings():
    universe = _universe()
    renamed = REFERENCE_FUNCTION.replace("rolling_mean", "moving_average").replace(
        "    out = []", '    """Average over a sliding window."""\n    out = []')
    result = check_candidate(CandidateBug(repository="me/new", target_function=renamed),
                             universe)
    assert "function_near_duplicate_of_reference" in result["reasons"]
    assert result["nearest_function"]["jaccard"] >= NEAR_DUPLICATE_JACCARD


def test_novel_candidate_is_admitted_with_evidence():
    novel = ("def merge_ranges(pairs):\n    pairs = sorted(pairs)\n    merged = [pairs[0]]\n"
             "    for lo, hi in pairs[1:]:\n        if lo <= merged[-1][1]:\n"
             "            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))\n"
             "        else:\n            merged.append((lo, hi))\n    return merged\n")
    result = check_candidate(CandidateBug(repository="me/new", target_function=novel,
                                          fixed_commit="fff"), _universe())
    assert result["admissible"] is True and result["reasons"] == []
    assert result["nearest_function"]["jaccard"] < NEAR_DUPLICATE_JACCARD


def test_extract_functions_keeps_unparseable_fragments():
    source = "def a():\n    return 1\n\nclass K:\n    def b(self):\n        return 2\n"
    assert len(extract_functions(source)) == 2
    assert extract_functions("+ x = (") == ["+ x = ("]


def test_reuses_the_frozen_audit_normalisation():
    assert normalise('def f():\n    """doc"""\n    return 1\n') == "def f():\n    return 1"


def test_corpus_sources_are_covered_by_the_upstream_universe():
    manifest = json.loads((ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
                           / "manifest.json").read_text(encoding="utf-8")) if (
        ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "manifest.json"
    ).exists() else None
    if manifest is None:
        pytest.skip("corpus manifest not present")
    assert corpus_sources_are_covered(manifest)
    assert not corpus_sources_are_covered({"records_by_source": {"new_source": 1}})


def test_tracked_design_is_consistent_and_protected_data_untouched():
    path = ROOT / "results" / "v4_3_next_direction_design.json"
    if not path.exists():
        pytest.skip("next-direction design not present")
    assert b"\r" not in path.read_bytes()
    design = json.loads(path.read_text(encoding="utf-8"))
    isolation = design["isolation"]
    assert isolation["self_checks_pass"] is True
    assert isolation["corpus_sources_fully_covered_by_upstream_universe"] is True
    names = isolation["excluded_repository_names"]
    assert hashlib.sha256(json.dumps(names).encode("utf-8")).hexdigest() == \
        isolation["excluded_repository_names_sha256"]
    for required in ("pandas", "django", "sympy", "tqdm", "youtube-dl", "requests", "flask"):
        assert required in names
    assert not any(value for key, value in design["leakage"].items() if key != "splits_opened")
    assert design["leakage"]["splits_opened"] == ["train"]
    assert design["starting_state"]["closed_pilots"]["tool_assisted"]["verdict"] == "fail"
    ids = [entry["id"] for entry in design["required_user_decisions"]]
    assert ids == [f"D{i}" for i in range(1, len(ids) + 1)]
    assert [step["step"] for step in design["execution_order"]] == \
        list(range(1, len(design["execution_order"]) + 1))
