"""Fail-closed isolation checks for a future repository-native evaluation set.

Claim, stated narrowly: disjointness is ENFORCED against the complete indexed
source universe under the recorded repository, fork, commit, patch, issue and
function-similarity checks.  It is not a claim that no model has ever seen a
repository.

Admission requires, for every candidate:

1. complete, structurally valid EVIDENCE (``evidence_problems``): canonical
   owner/name, a verified repository URL and numeric identity, explicit fork
   status (with a verified parent when a fork), full 40-hex buggy and fixed
   commits, a non-empty normalised patch, a parseable target function with
   enough shingles for a meaningful comparison, an issue/PR identity in the
   same repository, licence evidence, and the target file/module.  Anything
   missing or malformed refuses with ``insufficient_isolation_evidence`` -
   nothing defaults to admission;
2. repository, fork-parent, commit, issue and patch lineage outside the
   universe; and
3. target-function Jaccard below the frozen near-duplicate threshold against
   every indexed reference function.

The universe indexes the full upstream copy of every source that contributed
to any Oneiros split (MBPP, HumanEval, BugsInPy, SWE-bench Verified, the
curated seeds reconstructed from their original definition), the permitted
train shard and the legacy real-bug files.  ``audit_source_coverage`` checks
concrete indexed counts and bound input files per corpus source, not names.

Every isolation record carries the SHA-256 of the reference-universe receipt;
a record made against a different universe is invalid.

Similarity reuses the frozen audit (``scripts/audit_cross_split_near_duplicates``):
AST-normalised, docstring-stripped code, exact Jaccard over 5-token shingles,
threshold 0.80.  No protected split and no canonical records.json is read.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
import warnings

from scripts.audit_cross_split_near_duplicates import (
    CANDIDATE_MIN_SHARED, NEAR_DUPLICATE_JACCARD, _index, jaccard, normalise, shingles,
)

ISOLATION_VERSION = "oneiros_repository_isolation_v2"
CODE_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_SCHEMA = "oneiros_reference_universe_receipt_v1"
CLAIM = ("disjointness is enforced against the complete indexed source universe under "
         "the recorded repository, fork, commit, patch, issue, and function-similarity "
         "checks")

#: Corpus source name (corpus manifest ``records_by_source``) -> universe source.
CORPUS_SOURCE_TO_UNIVERSE = {
    "mbpp": "mbpp", "humaneval": "humaneval", "BugsInPy": "BugsInPy",
    "SWE-bench Verified": "SWE-bench Verified",
    "manual_curated_examples": "manual_curated_examples",
}
#: Universe sources that must be indexed even though no corpus source names
#: them directly: the permitted train view and the legacy real-bug files.
REQUIRED_AUXILIARY_SOURCES = ("train_shard", "legacy_real_bugs")
ADMITTED_LICENCES = ("MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0")
#: A function with fewer shingles cannot be compared meaningfully: short
#: bodies collide with everything or with nothing.
MIN_FUNCTION_SHINGLES = 12
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
OWNER_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38})/[a-z0-9._-]{1,100}$")
ISSUE = re.compile(r"^([a-z0-9][a-z0-9-]{0,38}/[a-z0-9._-]{1,100})#([1-9][0-9]*)$")
INSUFFICIENT = "insufficient_isolation_evidence"


# --- normalisation ----------------------------------------------------------

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
    """Every function definition in ``source``; unparseable fragments kept whole."""
    try:
        with warnings.catch_warnings():
            # Legacy upstream files carry invalid escape sequences; parsing them
            # for indexing must not flood the test output with SyntaxWarnings.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return [source] if source.strip() else []
    found = [ast.get_source_segment(source, node) or ""
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return [text for text in found if text.strip()] or ([source] if source.strip() else [])


def code_shingles(code: str) -> frozenset[str]:
    return shingles(normalise(code) or code)


def fingerprint(items: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


# --- universe ---------------------------------------------------------------

@dataclass
class ReferenceUniverse:
    repository_names: set[str] = field(default_factory=set)
    repository_full: set[str] = field(default_factory=set)
    #: Verified fork relations (child -> parent) of universe repositories.  The
    #: benchmarks record canonical upstreams only, so this is empty unless a
    #: relation is verified; candidates must supply their own fork evidence.
    fork_parents: dict[str, str] = field(default_factory=dict)
    commits: set[str] = field(default_factory=set)
    patch_hashes: set[str] = field(default_factory=set)
    patch_shingles: dict[str, frozenset[str]] = field(default_factory=dict)
    instance_ids: set[str] = field(default_factory=set)
    functions: dict[str, frozenset[str]] = field(default_factory=dict)
    function_source: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    counts: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_repositories: dict[str, set[str]] = field(default_factory=dict)
    _postings: dict[str, list[str]] | None = field(default=None, repr=False)

    def postings(self) -> dict[str, list[str]]:
        if self._postings is None:
            self._postings = _index(self.functions.items())
        return self._postings

    def bind(self, source: str, root: Path, path: Path) -> None:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        self.inputs.setdefault(source, {})[relative] = _sha_file(path)

    def add_repository(self, value: str, source: str) -> None:
        full, name = canonical_repository(value)
        if name:
            self.repository_names.add(name)
            self.source_repositories.setdefault(source, set()).add(name)
        if full:
            self.repository_full.add(full)

    def add_function_source(self, key: str, code: str, source: str) -> int:
        added = 0
        for index, function in enumerate(extract_functions(code)):
            items = code_shingles(function)
            if items:
                name = f"{key}#{index}"
                self.functions[name] = items
                self.function_source[name] = source
                added += 1
        return added


def build_reference_universe(root: Path, include_train_shard: bool = True) -> ReferenceUniverse:
    """Index every upstream source that has ever fed an Oneiros split.

    Reads upstream copies, the curated seed definition in code, the legacy
    real-bug files and the permitted train shard only.  A missing required
    input raises; it is never skipped.
    """
    universe = ReferenceUniverse()
    data = root / "data"

    # BugsInPy: every project.info, bug.info and bug_patch.txt.
    projects = data / "BugsInPy_repo" / "projects"
    project_files = sorted(projects.glob("*/project.info"))
    if not project_files:
        raise FileNotFoundError("upstream BugsInPy project.info files are missing")
    for info in project_files:
        universe.bind("BugsInPy", root, info)
        text = info.read_text(encoding="utf-8", errors="replace")
        url = re.search(r'github_url="([^"]+)"', text)
        universe.add_repository(url.group(1) if url else info.parent.name, "BugsInPy")
        universe.add_repository(info.parent.name, "BugsInPy")
    bugs = sorted(path for path in projects.glob("*/bugs/*") if path.is_dir())
    functions = 0
    for bug in bugs:
        info_path, patch_path = bug / "bug.info", bug / "bug_patch.txt"
        if not info_path.exists() or not patch_path.exists():
            raise FileNotFoundError(f"BugsInPy bug is incomplete: {bug}")
        universe.bind("BugsInPy", root, info_path)
        universe.bind("BugsInPy", root, patch_path)
        info = info_path.read_text(encoding="utf-8", errors="replace")
        for key in ("buggy_commit_id", "fixed_commit_id"):
            match = re.search(rf'{key}="([0-9a-f]+)"', info)
            if match:
                universe.commits.add(match.group(1))
        patch = patch_path.read_text(encoding="utf-8", errors="replace")
        key = f"bugsinpy::{bug.parent.parent.name}::{bug.name}"
        universe.patch_hashes.add(patch_hash(patch))
        universe.patch_shingles[key] = _patch_shingles(patch)
        universe.instance_ids.add(key)
        functions += universe.add_function_source(
            key, normalised_patch(patch).replace("\n", " "), "BugsInPy")
    universe.counts["BugsInPy"] = {"projects": len(project_files), "bugs": len(bugs),
                                   "patches": len(bugs), "functions": functions}

    # SWE-bench Verified: the complete source parquet.
    import pandas as pd
    parquet = data / "swebench_verified_source" / "SWE-bench_Verified.test.parquet"
    universe.bind("SWE-bench Verified", root, parquet)
    frame = pd.read_parquet(parquet, columns=["repo", "instance_id", "base_commit", "patch"])
    functions = 0
    for row in frame.itertuples(index=False):
        universe.add_repository(row.repo, "SWE-bench Verified")
        universe.commits.add(str(row.base_commit))
        universe.instance_ids.add(str(row.instance_id))
        universe.patch_hashes.add(patch_hash(row.patch))
        universe.patch_shingles[str(row.instance_id)] = _patch_shingles(row.patch)
        functions += universe.add_function_source(
            str(row.instance_id), normalised_patch(row.patch).replace("\n", " "),
            "SWE-bench Verified")
    universe.counts["SWE-bench Verified"] = {"instances": len(frame), "patches": len(frame),
                                             "functions": functions}

    # Legacy real-bug files: metadata and every Python source file.
    legacy = data / "bugsinpy"
    metadata = legacy / "bugsinpy_metadata.json"
    universe.bind("legacy_real_bugs", root, metadata)
    entries = json.loads(metadata.read_text(encoding="utf-8"))
    for entry in entries:
        universe.add_repository(entry.get("project", ""), "legacy_real_bugs")
        for key in ("commit_buggy", "commit_fixed"):
            if entry.get(key):
                universe.commits.add(str(entry[key]))
    python_files = sorted(legacy.rglob("*.py"))
    functions = 0
    for path in python_files:
        universe.bind("legacy_real_bugs", root, path)
        match = re.match(r"bugsinpy_(.+?)_\d+_(buggy|fixed)\.py$", path.name)
        if match:
            universe.add_repository(match.group(1), "legacy_real_bugs")
        functions += universe.add_function_source(
            f"legacy::{path.name}", path.read_text(encoding="utf-8", errors="replace"),
            "legacy_real_bugs")
    universe.counts["legacy_real_bugs"] = {"metadata_entries": len(entries),
                                           "python_files": len(python_files),
                                           "functions": functions}

    # MBPP and HumanEval: complete upstream sets.
    mbpp = data / "mbpp" / "mbpp_full.jsonl"
    universe.bind("mbpp", root, mbpp)
    lines = [line for line in mbpp.read_text(encoding="utf-8").splitlines() if line.strip()]
    functions = 0
    for line in lines:
        entry = json.loads(line)
        functions += universe.add_function_source(f"mbpp::{entry.get('task_id')}",
                                                  entry.get("code", ""), "mbpp")
    universe.counts["mbpp"] = {"tasks": len(lines), "functions": functions}
    humaneval = data / "humaneval" / "humaneval_cache.json"
    universe.bind("humaneval", root, humaneval)
    tasks = json.loads(humaneval.read_text(encoding="utf-8"))
    functions = 0
    for entry in tasks:
        functions += universe.add_function_source(
            f"humaneval::{entry['task_id']}", entry["prompt"] + entry["canonical_solution"],
            "humaneval")
    universe.counts["humaneval"] = {"tasks": len(tasks), "functions": functions}

    # Curated seeds: reconstructed from their ORIGINAL definition in code.
    from harness import bugsinpy_loader
    definitions = bugsinpy_loader.CURATED_BUGSINPY_BUGS
    # The definition lives in the code tree, so it is bound relative to it.
    universe.bind("manual_curated_examples", CODE_ROOT, Path(bugsinpy_loader.__file__))
    functions = 0
    for bug in definitions:
        universe.add_repository(bug["project"], "manual_curated_examples")
        for side in ("buggy_code", "fixed_code"):
            functions += universe.add_function_source(
                f"curated::{bug['id']}::{side}", bug[side], "manual_curated_examples")
    # CURATED_BUGSINPY_BUGS is the only generator of curated records
    # (scripts/build_corpus_v1.curated_fixed_bug_records), so indexing every
    # definition covers whichever subset entered any split.  The historical
    # execution filter is NOT re-run: its outcome is timing-sensitive and would
    # make the receipt non-deterministic.
    universe.bind("manual_curated_examples", CODE_ROOT,
                  CODE_ROOT / "scripts" / "build_corpus_v1.py")
    universe.counts["manual_curated_examples"] = {
        "definitions": len(definitions),
        "definition_ids": sorted(f"curated::{bug['id']}" for bug in definitions),
        "definitions_sha256": _sha_json(definitions), "functions": functions}

    if include_train_shard:
        from harness.corpus_view import load_development_split, verify_development_view
        from utils.dataset_identity import dataset_name_from_source
        corpus = data / "corpus" / "v4_1_research_hardened_candidate"
        view = verify_development_view(corpus, ["train"])
        universe.bind("train_shard", root, corpus / "development_view" / "manifest.json")
        universe.bind("train_shard", root,
                      corpus / "development_view" / view["splits"]["train"]["filename"])
        records = load_development_split(corpus, "train", include_excluded=True)
        functions = 0
        for record in records:
            for key in ("reference_code", "code_under_test"):
                if record.get(key):
                    functions += universe.add_function_source(
                        f"train::{record['id']}::{key}", str(record[key]), "train_shard")
            project = (record.get("provenance") or {}).get("project")
            if project:
                universe.add_repository(str(project), "train_shard")
        universe.counts["train_shard"] = {
            "records": len(records), "declared_records": view["splits"]["train"]["record_count"],
            "functions": functions,
            "curated_ids": sorted(str(r["id"]) for r in records if dataset_name_from_source(
                r.get("source") or {}) == "manual_curated_examples")}
    return universe


# --- coverage ----------------------------------------------------------------

def audit_source_coverage(universe: ReferenceUniverse,
                          corpus_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Evidence-based coverage: concrete indexed counts and bound inputs per source."""
    problems: list[str] = []
    declared = dict(corpus_manifest.get("records_by_source") or {})
    per_source: dict[str, Any] = {}
    for source, count in sorted(declared.items()):
        target = CORPUS_SOURCE_TO_UNIVERSE.get(source)
        if target is None:
            problems.append(f"unknown corpus source {source!r}")
            continue
        indexed = universe.counts.get(target)
        bound = universe.inputs.get(target, {})
        if not indexed or not bound:
            problems.append(f"declared source {source!r} is not indexed")
            continue
        if not any(isinstance(value, int) and value > 0 for value in indexed.values()):
            problems.append(f"source {source!r} indexed with zero items")
        per_source[source] = {"declared_records": count, "indexed": indexed,
                              "bound_inputs": len(bound)}
    for source in REQUIRED_AUXILIARY_SOURCES:
        if not universe.counts.get(source) or not universe.inputs.get(source):
            problems.append(f"required source {source!r} is not indexed")
    train = universe.counts.get("train_shard") or {}
    if train and train.get("records") != train.get("declared_records"):
        problems.append("train shard indexed count differs from its view manifest")
    curated = universe.counts.get("manual_curated_examples") or {}
    if "manual_curated_examples" in declared and curated:
        indexed_ids = set(curated.get("definition_ids") or [])
        train_ids = set(train.get("curated_ids") or [])
        if not train_ids <= indexed_ids:
            problems.append("a train curated seed is not indexed from the source definition")
        missing = declared["manual_curated_examples"] - len(train_ids)
        candidates = indexed_ids - train_ids
        if missing > len(candidates):
            problems.append("non-train curated seeds cannot all be accounted for by indexed "
                            "definitions")
        per_source.setdefault("manual_curated_examples", {})["non_train_seed_candidates"] = \
            sorted(candidates)
        per_source["manual_curated_examples"]["non_train_seeds_required"] = missing
    return {"covered": not problems, "problems": problems, "per_source": per_source}


# --- receipt -----------------------------------------------------------------

def reference_universe_receipt(universe: ReferenceUniverse,
                               source_files: Mapping[str, str]) -> dict[str, Any]:
    """Deterministic, hash-bound description of the complete universe."""
    function_fingerprints: dict[str, list[str]] = {}
    for key in sorted(universe.functions):
        function_fingerprints.setdefault(universe.function_source[key], []).append(
            fingerprint(universe.functions[key]))
    for values in function_fingerprints.values():
        values.sort()
    collections = {
        "repository_full_names": sorted(universe.repository_full),
        "repository_names": sorted(universe.repository_names),
        "fork_parent_identities": dict(sorted(universe.fork_parents.items())),
        "commits": sorted(universe.commits),
        "patch_hashes": sorted(universe.patch_hashes),
        "instance_ids": sorted(universe.instance_ids),
        "function_fingerprints": function_fingerprints,
        "input_files": {source: dict(sorted(files.items()))
                        for source, files in sorted(universe.inputs.items())},
        "counts": universe.counts,
    }
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "isolation_version": ISOLATION_VERSION,
        "claim": CLAIM,
        "near_duplicate_threshold": NEAR_DUPLICATE_JACCARD,
        "min_function_shingles": MIN_FUNCTION_SHINGLES,
        "fork_parent_note": ("benchmark repositories are recorded as their canonical "
                             "upstreams; no fork relation among them is asserted offline, so "
                             "every candidate must supply verified fork evidence"),
        "collections": collections,
        "collection_sha256": {name: _sha_json(value) for name, value in collections.items()},
        "source_files_sha256": dict(sorted(source_files.items())),
    }
    body["receipt_sha256"] = _sha_json(body)
    return body


def verify_receipt(receipt: Mapping[str, Any]) -> bool:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return (_sha_json(body) == receipt.get("receipt_sha256") and all(
        _sha_json(receipt["collections"][name]) == digest
        for name, digest in receipt["collection_sha256"].items()))


# --- candidates -----------------------------------------------------------------

@dataclass(frozen=True)
class CandidateBug:
    repository: str = ""
    repository_url: str = ""
    repository_id: str = ""
    fork_status: str = "unknown"          # "not_fork" | "fork"; anything else refuses
    fork_parent: str = ""
    fork_parent_id: str = ""
    buggy_commit: str = ""
    fixed_commit: str = ""
    patch: str = ""
    target_function: str = ""
    target_file: str = ""
    target_module: str = ""
    issue_id: str = ""                    # "owner/name#123"
    licence_spdx: str = ""
    licence_sha256: str = ""


def evidence_problems(candidate: CandidateBug) -> list[str]:
    """Every missing or malformed piece of mandatory evidence (empty = complete)."""
    problems: list[str] = []
    repository = str(candidate.repository or "").strip()
    if not OWNER_NAME.match(repository):
        problems.append("repository_not_canonical_owner_name")
    expected_url = f"https://github.com/{repository}"
    if str(candidate.repository_url).rstrip("/").lower() != expected_url:
        problems.append("repository_url_unverified_or_inconsistent")
    if not str(candidate.repository_id).isdigit():
        problems.append("repository_identity_missing")
    if candidate.fork_status == "fork":
        if not OWNER_NAME.match(str(candidate.fork_parent or "")):
            problems.append("fork_parent_identity_missing")
        if not str(candidate.fork_parent_id).isdigit():
            problems.append("fork_parent_id_missing")
    elif candidate.fork_status != "not_fork":
        problems.append("fork_status_unknown")
    for label in ("buggy_commit", "fixed_commit"):
        if not SHA40.match(str(getattr(candidate, label) or "")):
            problems.append(f"{label}_not_full_sha")
    if candidate.buggy_commit and candidate.buggy_commit == candidate.fixed_commit:
        problems.append("buggy_and_fixed_commit_identical")
    if not normalised_patch(candidate.patch):
        problems.append("patch_empty")
    code = str(candidate.target_function or "")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code) if code.strip() else None
    except (SyntaxError, ValueError):
        tree = None
    if tree is None or not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                               for node in tree.body):
        problems.append("target_function_missing_or_unparseable")
    elif len(code_shingles(code)) < MIN_FUNCTION_SHINGLES:
        problems.append("target_function_too_short_for_comparison")
    if not re.match(r"^[A-Za-z0-9_./-]+\.py$", str(candidate.target_file or "")) or \
            candidate.target_file.startswith(("/", "..")):
        problems.append("target_file_missing")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$",
                    str(candidate.target_module or "")):
        problems.append("target_module_missing")
    issue = ISSUE.match(str(candidate.issue_id or "").lower())
    if not issue:
        problems.append("issue_identity_missing")
    elif issue.group(1) != repository:
        problems.append("issue_repository_mismatch")
    if candidate.licence_spdx not in ADMITTED_LICENCES:
        problems.append("licence_evidence_missing_or_not_admitted")
    if not SHA64.match(str(candidate.licence_sha256 or "")):
        problems.append("licence_file_hash_missing")
    return [f"{INSUFFICIENT}:{problem}" for problem in problems]


def check_candidate(candidate: CandidateBug, universe: ReferenceUniverse,
                    receipt_sha256: str,
                    threshold: float = NEAR_DUPLICATE_JACCARD) -> dict[str, Any]:
    """The isolation record for one candidate.  Admission only if nothing is refused."""
    if not SHA64.match(str(receipt_sha256 or "")):
        raise ValueError("an isolation decision needs the reference-universe receipt hash")
    reasons = evidence_problems(candidate)
    for label, value in (("repository", candidate.repository),
                         ("fork_parent", candidate.fork_parent)):
        if value:
            full, name = canonical_repository(value)
            if name in universe.repository_names or (full and full in universe.repository_full):
                reasons.append(f"{label}_in_reference_universe:{full or name}")
    for label, commit in (("fixed_commit", candidate.fixed_commit),
                          ("buggy_commit", candidate.buggy_commit)):
        if commit and commit in universe.commits:
            reasons.append(f"{label}_is_known_benchmark_commit")
    if candidate.issue_id and candidate.issue_id in universe.instance_ids:
        reasons.append("issue_is_known_benchmark_instance")
    nearest_patch: dict[str, Any] = {"reference": None, "jaccard": 0.0}
    if normalised_patch(candidate.patch):
        if patch_hash(candidate.patch) in universe.patch_hashes:
            reasons.append("patch_identical_to_known_bug")
        mine = _patch_shingles(candidate.patch)
        for key, items in universe.patch_shingles.items():
            score = jaccard(items, mine)
            if score > nearest_patch["jaccard"]:
                nearest_patch = {"reference": key, "jaccard": round(score, 4)}
        if nearest_patch["jaccard"] >= threshold:
            reasons.append("patch_near_duplicate_of_known_bug")
    nearest_function: dict[str, Any] = {"reference": None, "jaccard": 0.0}
    mine = code_shingles(candidate.target_function) if candidate.target_function.strip() \
        else frozenset()
    if mine:
        shared: Counter = Counter()
        for item in mine:
            for key in universe.postings().get(item, ()):
                shared[key] += 1
        for key, count in shared.items():
            if count < CANDIDATE_MIN_SHARED and len(mine) >= CANDIDATE_MIN_SHARED:
                continue
            score = jaccard(universe.functions[key], mine)
            if score > nearest_function["jaccard"]:
                nearest_function = {"reference": key, "jaccard": round(score, 4)}
        if nearest_function["jaccard"] >= threshold:
            reasons.append("function_near_duplicate_of_reference")
    owner = canonical_repository(candidate.repository)[0].split("/")[0]
    organisation_flags = sorted({full for full in universe.repository_full
                                 if owner and full.split("/")[0] == owner})
    return {
        "isolation_version": ISOLATION_VERSION,
        "reference_universe_sha256": receipt_sha256,
        "claim": CLAIM,
        "admissible": not reasons,
        "reasons": reasons,
        "insufficient_evidence": any(reason.startswith(INSUFFICIENT) for reason in reasons),
        "evidence": {item.name: getattr(candidate, item.name) for item in fields(candidate)
                     if item.name not in ("patch", "target_function")},
        "patch_sha256": patch_hash(candidate.patch) if candidate.patch else None,
        "target_function_fingerprint": fingerprint(mine) if mine else None,
        "nearest_patch": nearest_patch,
        "nearest_function": nearest_function,
        "same_organisation_as_excluded": organisation_flags,
        "threshold": threshold,
    }


def isolation_record_is_current(record: Mapping[str, Any], receipt_sha256: str) -> bool:
    """A decision made against a different universe is invalid."""
    return (record.get("isolation_version") == ISOLATION_VERSION
            and record.get("reference_universe_sha256") == receipt_sha256)
