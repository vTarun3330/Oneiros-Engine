"""Executable isolation checks for a future repository-native evaluation set.

Disjointness is PROVEN per candidate, not asserted.  A candidate bug is
admissible only if it passes four independent checks against the complete
universe of material that has ever fed any Oneiros split:

1. repository identity - its repository (and any fork parent) is outside the
   union of every repository in upstream BugsInPy, SWE-bench Verified, the
   legacy real-bug directory and the curated examples;
2. patch lineage - its fix commit, buggy commit and normalised patch are not
   those of any known BugsInPy or SWE-bench Verified bug, and its patch is
   not a near-duplicate of one;
3. function near-duplication - its target function is below the frozen
   near-duplicate threshold against every function extracted from the
   upstream MBPP and HumanEval solutions, the BugsInPy code, the SWE-bench
   Verified patches and the permitted train shard;
4. issue lineage - its issue/PR identifier is not a known benchmark instance.

Why this covers protected splits without opening them: every Oneiros split,
the consumed test split included, is built only from MBPP, HumanEval,
BugsInPy, SWE-bench Verified and eight curated examples (corpus manifest
``records_by_source``).  The reference universe below is the complete
UPSTREAM copy of each of those sources, which is a superset of what any split
contains.  A candidate that is disjoint from the upstream superset is disjoint
from every split, and no protected split has to be read to show it.

Similarity reuses the frozen audit (``scripts/audit_cross_split_near_duplicates``):
AST-normalised, docstring-stripped code, exact Jaccard over 5-token shingles,
threshold 0.80, inverted-index candidates.  Nothing is approximated.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from scripts.audit_cross_split_near_duplicates import (
    CANDIDATE_MIN_SHARED, NEAR_DUPLICATE_JACCARD, _index, jaccard, normalise, shingles,
)

ISOLATION_VERSION = "oneiros_repository_isolation_v1"
CORPUS_SOURCES = ("BugsInPy", "SWE-bench Verified", "humaneval", "mbpp",
                  "manual_curated_examples")


def canonical_repository(value: str) -> tuple[str, str]:
    """(owner/name or '', bare name), lower-cased, host and .git stripped."""
    text = str(value or "").strip().lower()
    text = re.sub(r"^(https?://)?(www\.)?github\.com/", "", text).rstrip("/")
    text = re.sub(r"\.git$", "", text)
    parts = [part for part in text.split("/") if part]
    name = parts[-1] if parts else ""
    full = "/".join(parts[-2:]) if len(parts) >= 2 else ""
    return full, name.replace("_", "-")


def normalised_patch(patch: str) -> str:
    """Changed lines only, whitespace-collapsed; headers and context dropped."""
    lines = []
    for line in str(patch or "").splitlines():
        if line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if line.startswith(("+", "-")):
            body = re.sub(r"\s+", " ", line[1:]).strip()
            if body:
                lines.append(line[0] + body)
    return "\n".join(lines)


def patch_hash(patch: str) -> str:
    return hashlib.sha256(normalised_patch(patch).encode("utf-8")).hexdigest()


def _patch_shingles(patch: str) -> frozenset[str]:
    return shingles(normalised_patch(patch).replace("\n", " "))


def extract_functions(source: str) -> list[str]:
    """Every function definition in ``source`` as normalised text.

    Unparseable fragments (patch hunks, partial files) are kept whole so they
    still take part in the comparison rather than silently dropping out.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return [source] if source.strip() else []
    found = [ast.get_source_segment(source, node) or ""
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return [text for text in found if text.strip()] or ([source] if source.strip() else [])


def _code_shingles(code: str) -> frozenset[str]:
    return shingles(normalise(code) or code)


@dataclass
class ReferenceUniverse:
    repository_names: set[str] = field(default_factory=set)
    repository_full: set[str] = field(default_factory=set)
    commits: set[str] = field(default_factory=set)
    patch_hashes: set[str] = field(default_factory=set)
    patch_shingles: dict[str, frozenset[str]] = field(default_factory=dict)
    instance_ids: set[str] = field(default_factory=set)
    functions: dict[str, frozenset[str]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    _postings: dict[str, list[str]] | None = field(default=None, repr=False)

    def postings(self) -> dict[str, list[str]]:
        if self._postings is None:
            self._postings = _index(self.functions.items())
        return self._postings

    def add_repository(self, value: str, source: str) -> None:
        full, name = canonical_repository(value)
        if name:
            self.repository_names.add(name)
        if full:
            self.repository_full.add(full)
        self.sources.setdefault(source, {}).setdefault("repositories", set()).add(name)

    def add_function_source(self, key: str, code: str) -> None:
        for index, function in enumerate(extract_functions(code)):
            items = _code_shingles(function)
            if items:
                self.functions[f"{key}#{index}"] = items

    def summary(self) -> dict[str, Any]:
        return {
            "repository_names": sorted(self.repository_names),
            "repository_full_names": sorted(self.repository_full),
            "known_commits": len(self.commits),
            "known_patches": len(self.patch_hashes),
            "known_instance_ids": len(self.instance_ids),
            "reference_functions": len(self.functions),
            "by_source": {name: {key: (sorted(value) if isinstance(value, set) else value)
                                 for key, value in entry.items()}
                          for name, entry in sorted(self.sources.items())},
        }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reference_universe(root: Path, include_train_shard: bool = True) -> ReferenceUniverse:
    """Load every upstream source that has ever fed an Oneiros split.

    Reads upstream copies and the permitted train shard only.  No protected
    split, no canonical records.json.
    """
    universe = ReferenceUniverse()
    data = root / "data"

    bugsinpy = data / "BugsInPy_repo" / "projects"
    project_files = sorted(bugsinpy.glob("*/project.info"))
    for info in project_files:
        text = info.read_text(encoding="utf-8", errors="replace")
        url = re.search(r'github_url="([^"]+)"', text)
        universe.add_repository(url.group(1) if url else info.parent.name, "BugsInPy")
        universe.add_repository(info.parent.name, "BugsInPy")
    for bug in sorted(bugsinpy.glob("*/bugs/*")):
        info = (bug / "bug.info").read_text(encoding="utf-8", errors="replace") \
            if (bug / "bug.info").exists() else ""
        for key in ("buggy_commit_id", "fixed_commit_id"):
            match = re.search(rf'{key}="([0-9a-f]+)"', info)
            if match:
                universe.commits.add(match.group(1))
        patch_path = bug / "bug_patch.txt"
        if patch_path.exists():
            patch = patch_path.read_text(encoding="utf-8", errors="replace")
            key = f"bugsinpy::{bug.parent.parent.name}::{bug.name}"
            universe.patch_hashes.add(patch_hash(patch))
            universe.patch_shingles[key] = _patch_shingles(patch)
            universe.add_function_source(key, normalised_patch(patch).replace("\n", " "))
            universe.instance_ids.add(key)
    universe.sources["BugsInPy"]["files"] = {"project_info": len(project_files),
                                             "bugs": len(universe.patch_shingles)}

    import pandas as pd
    parquet = data / "swebench_verified_source" / "SWE-bench_Verified.test.parquet"
    frame = pd.read_parquet(parquet, columns=["repo", "instance_id", "base_commit", "patch"])
    for row in frame.itertuples(index=False):
        universe.add_repository(row.repo, "SWE-bench Verified")
        universe.commits.add(str(row.base_commit))
        universe.instance_ids.add(str(row.instance_id))
        universe.patch_hashes.add(patch_hash(row.patch))
        universe.patch_shingles[str(row.instance_id)] = _patch_shingles(row.patch)
        universe.add_function_source(str(row.instance_id),
                                     normalised_patch(row.patch).replace("\n", " "))
    universe.sources["SWE-bench Verified"]["files"] = {"parquet_sha256": _sha(parquet),
                                                       "instances": len(frame)}

    legacy = data / "bugsinpy"
    metadata = legacy / "bugsinpy_metadata.json"
    for entry in json.loads(metadata.read_text(encoding="utf-8")):
        universe.add_repository(entry.get("project", ""), "legacy_real_bugs")
        for key in ("commit_buggy", "commit_fixed"):
            if entry.get(key):
                universe.commits.add(str(entry[key]))
    for path in sorted(legacy.iterdir()):
        if path.is_dir():
            universe.add_repository(path.name, "legacy_real_bugs")
        elif path.suffix == ".py":
            match = re.match(r"bugsinpy_(.+?)_\d+_(buggy|fixed)\.py$", path.name)
            if match:
                universe.add_repository(match.group(1), "legacy_real_bugs")
            universe.add_function_source(f"legacy::{path.name}",
                                         path.read_text(encoding="utf-8", errors="replace"))
    universe.sources["legacy_real_bugs"]["files"] = {"metadata_sha256": _sha(metadata)}

    mbpp = data / "mbpp" / "mbpp_full.jsonl"
    count = 0
    for line in mbpp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            universe.add_function_source(f"mbpp::{entry.get('task_id')}", entry.get("code", ""))
            count += 1
    humaneval = data / "humaneval" / "humaneval_cache.json"
    tasks = json.loads(humaneval.read_text(encoding="utf-8"))
    for entry in tasks:
        universe.add_function_source(f"humaneval::{entry['task_id']}",
                                     entry["prompt"] + entry["canonical_solution"])
    universe.sources["upstream_function_benchmarks"] = {
        "files": {"mbpp_full_sha256": _sha(mbpp), "mbpp_tasks": count,
                  "humaneval_sha256": _sha(humaneval), "humaneval_tasks": len(tasks)}}

    if include_train_shard:
        from harness.corpus_view import load_development_split
        corpus = data / "corpus" / "v4_1_research_hardened_candidate"
        records = load_development_split(corpus, "train", include_excluded=True)
        for record in records:
            for key in ("reference_code", "code_under_test"):
                if record.get(key):
                    universe.add_function_source(f"train::{record['id']}::{key}",
                                                 str(record[key]))
            project = (record.get("provenance") or {}).get("project")
            if project:
                universe.add_repository(str(project), "train_shard")
        universe.sources.setdefault("train_shard", {})["files"] = {
            "development_view_manifest_sha256": _sha(
                corpus / "development_view" / "manifest.json"),
            "records": len(records)}
    return universe


@dataclass(frozen=True)
class CandidateBug:
    repository: str
    fork_parent: str = ""
    fixed_commit: str = ""
    buggy_commit: str = ""
    patch: str = ""
    target_function: str = ""
    issue_id: str = ""


def check_candidate(candidate: CandidateBug, universe: ReferenceUniverse,
                    threshold: float = NEAR_DUPLICATE_JACCARD) -> dict[str, Any]:
    """Every isolation check for one candidate, with the evidence for each."""
    reasons: list[str] = []
    for label, value in (("repository", candidate.repository),
                         ("fork_parent", candidate.fork_parent)):
        if not value:
            continue
        full, name = canonical_repository(value)
        if name in universe.repository_names or (full and full in universe.repository_full):
            reasons.append(f"{label}_in_reference_universe:{full or name}")
    for label, commit in (("fixed_commit", candidate.fixed_commit),
                          ("buggy_commit", candidate.buggy_commit)):
        if commit and commit in universe.commits:
            reasons.append(f"{label}_is_known_benchmark_commit")
    if candidate.issue_id and candidate.issue_id in universe.instance_ids:
        reasons.append("issue_is_known_benchmark_instance")
    nearest_patch = {"reference": None, "jaccard": 0.0}
    if candidate.patch:
        if patch_hash(candidate.patch) in universe.patch_hashes:
            reasons.append("patch_identical_to_known_bug")
        mine = _patch_shingles(candidate.patch)
        for key, items in universe.patch_shingles.items():
            score = jaccard(items, mine)
            if score > nearest_patch["jaccard"]:
                nearest_patch = {"reference": key, "jaccard": round(score, 4)}
        if nearest_patch["jaccard"] >= threshold:
            reasons.append("patch_near_duplicate_of_known_bug")
    nearest_function = {"reference": None, "jaccard": 0.0}
    if candidate.target_function:
        mine = _code_shingles(candidate.target_function)
        postings = universe.postings()
        shared: Counter = Counter()
        for item in mine:
            for key in postings.get(item, ()):
                shared[key] += 1
        for key, count in shared.items():
            if count < CANDIDATE_MIN_SHARED and len(mine) >= CANDIDATE_MIN_SHARED:
                continue
            score = jaccard(universe.functions[key], mine)
            if score > nearest_function["jaccard"]:
                nearest_function = {"reference": key, "jaccard": round(score, 4)}
        if nearest_function["jaccard"] >= threshold:
            reasons.append("function_near_duplicate_of_reference")
    return {"isolation_version": ISOLATION_VERSION, "admissible": not reasons,
            "reasons": reasons, "nearest_patch": nearest_patch,
            "nearest_function": nearest_function, "threshold": threshold}


def corpus_sources_are_covered(manifest: Mapping[str, Any]) -> bool:
    """The corpus draws only from sources whose upstream copies are indexed."""
    return set((manifest.get("records_by_source") or {}).keys()) <= set(CORPUS_SOURCES)
