"""Fail-closed isolation checks for a future repository-native evaluation set.

Claim, stated narrowly: disjointness is ENFORCED against the complete indexed
source universe under the recorded repository, fork, commit, patch, issue and
function-similarity checks.  It is not a claim that no model has ever seen a
repository.

Three separate stages decide a candidate:

1. **Evidence schema** (``evidence_problems``): every mandatory field is
   present and well formed.  Anything missing, unknown or malformed refuses
   with ``insufficient_isolation_evidence`` - nothing defaults to admission.
2. **Evidence authentication** (``authentication_problems``): the stored
   repository and issue API responses hash to their recorded SHA-256 and agree
   with the declared identity, fork status, parent and licence; the stored git
   objects hash to their object IDs; the licence file and target file resolve
   through the commit trees to the recorded blobs.  Under the frozen diff
   policy A (``DIFF_POLICY``) the fixed commit's only parent is the buggy
   commit, the authenticated trees show exactly one changed path (the target
   file), and the canonical diff is DERIVED from the two target blobs.  The
   submitted patch is never authoritative: it must carry exactly the derived
   changed lines, and at least one derived hunk must overlap the declared
   target function, whose normalised body must differ between revisions.  The
   frozen temporal rule (``TEMPORAL_RULE``, contamination-risk mitigation only)
   is proved from the authenticated commit timestamps.  All of this is offline
   cross-checking of evidence acquired earlier.
3. **Source-universe overlap** (``overlap_problems``): repository, fork-parent,
   commit and issue lineage outside the universe; patch lineage computed ONLY
   from the derived diff; and target-function Jaccard below the frozen
   near-duplicate threshold against every indexed reference function.

``record_sha256`` on an isolation record detects accidental modification only;
it is a self-hash, not a signature.  Downstream trust comes from
``revalidate_isolation_record``: retain the evidence sidecar, reload the same
frozen universe, rerun ``check_candidate`` and compare the record byte for byte.

Network acquisition of the evidence is a LATER, separate step
(``ACQUISITION_REQUIREMENTS``); nothing here makes a network call.

Decisions are made only against a ``FrozenReferenceUniverse``: a universe
whose recomputed collections, collection hashes, internal receipt hash, input
files and canonical source files were all verified against its receipt when it
was loaded.  The receipt hash on an isolation record is taken from that object;
no caller can supply one.

Similarity reuses the frozen audit (``scripts/audit_cross_split_near_duplicates``):
AST-normalised, docstring-stripped code, exact Jaccard over 5-token shingles,
threshold 0.80.  No protected split and no canonical records.json is read.
"""
from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import warnings

from harness.source_identity import canonical_sha256
from scripts.audit_cross_split_near_duplicates import (
    CANDIDATE_MIN_SHARED, NEAR_DUPLICATE_JACCARD, _index, jaccard, normalise, shingles,
)

ISOLATION_VERSION = "oneiros_repository_isolation_v4"
CODE_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_SCHEMA = "oneiros_reference_universe_receipt_v3"
#: Frozen diff policy A: only a direct-parent fix that changes exactly one file,
#: the target file.  The complete diff is DERIVED from the authenticated buggy
#: and fixed blobs; a submitted patch is never authoritative.
DIFF_POLICY = {
    "id": "A_direct_parent_single_target_file",
    "rule": ("the fixed commit has exactly one parent, the buggy commit; comparing the "
             "authenticated buggy and fixed trees shows exactly one changed path, the target "
             "file; the canonical diff is recomputed from the two target blobs (difflib unified "
             "diff, 3 context lines, a/ and b/ headers); the submitted patch must carry exactly "
             "the same removed and added lines per file; at least one changed hunk overlaps "
             "the declared target function, whose normalised body differs between the buggy "
             "and fixed files"),
    "multi_file_fixes": "refused (policy B is not selected)",
}
#: Frozen temporal rule: contamination-risk MITIGATION only, never proof.
TEMPORAL_CUTOFF_EPOCH = 1735689600  # 2025-01-01T00:00:00Z
TEMPORAL_RULE = {
    "rule": ("the authenticated fixed-commit committer timestamp is on or after "
             "2025-01-01T00:00:00Z"),
    "cutoff_epoch": TEMPORAL_CUTOFF_EPOCH,
    "rejects": ("a missing or unparseable author or committer timestamp; an author timestamp "
                "later than the committer timestamp; a buggy-commit committer timestamp later "
                "than the fixed commit's; a commit timestamp later than the evidence retrieval"),
    "status": "contamination-risk mitigation only; it does not show that no model saw the code",
}
CLAIM = ("disjointness is enforced against the complete indexed source universe under "
         "the recorded repository, fork, commit, patch, issue, and function-similarity "
         "checks")
#: The corpus manifest that decides which sources must be indexed.
CORPUS_MANIFEST = "data/corpus/v4_1_research_hardened_candidate/manifest.json"
#: Every module whose logic changes indexing, source classification,
#: normalisation, curated-seed identification or train-view loading.  The
#: receipt binds each by canonical (LF) SHA-256; loading refuses on any change.
CANONICAL_SOURCES = (
    "harness/repository_isolation.py",
    "harness/source_identity.py",
    "scripts/audit_cross_split_near_duplicates.py",
    "harness/bugsinpy_loader.py",
    "scripts/build_corpus_v1.py",
    "utils/dataset_identity.py",
    "harness/corpus_view.py",
    "harness/corpus.py",
    "harness/function_complexity.py",
    "scripts/build_next_direction_design.py",
)
#: Code modules that are also universe INPUTS, per universe source.
CODE_INPUTS = {
    "manual_curated_examples": ("harness/bugsinpy_loader.py", "scripts/build_corpus_v1.py"),
    "train_shard": ("harness/corpus_view.py", "harness/corpus.py",
                    "harness/function_complexity.py", "utils/dataset_identity.py"),
    "corpus_manifest": ("utils/dataset_identity.py",),
}

#: Corpus source name (corpus manifest ``records_by_source``) -> universe source.
CORPUS_SOURCE_TO_UNIVERSE = {
    "mbpp": "mbpp", "humaneval": "humaneval", "BugsInPy": "BugsInPy",
    "SWE-bench Verified": "SWE-bench Verified",
    "manual_curated_examples": "manual_curated_examples",
}
#: Universe sources that must be indexed even though no corpus source names
#: them directly: the permitted train view, the legacy real-bug files and the
#: corpus manifest itself.
REQUIRED_AUXILIARY_SOURCES = ("train_shard", "legacy_real_bugs", "corpus_manifest")
ADMITTED_LICENCES = ("MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0")
#: A function with fewer shingles cannot be compared meaningfully: short
#: bodies collide with everything or with nothing.
MIN_FUNCTION_SHINGLES = 12
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
OWNER_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38})/[a-z0-9._-]{1,100}$")
ISSUE = re.compile(r"^([a-z0-9][a-z0-9-]{0,38}/[a-z0-9._-]{1,100})#([1-9][0-9]*)$")
NODE_ID = re.compile(r"^[A-Za-z0-9_=+/-]{8,}$")
UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
API = "https://api.github.com/repos/"
INSUFFICIENT = "insufficient_isolation_evidence"

#: What the later, separately authorised acquisition step must store for each
#: candidate.  Nothing in this module fetches it.
ACQUISITION_REQUIREMENTS = {
    "repository_metadata": "GET https://api.github.com/repos/{owner}/{name}: raw response "
                           "bytes, SHA-256, retrieval timestamp (UTC) and ETag; fields "
                           "full_name, id, node_id, html_url, fork, parent{full_name,id,"
                           "node_id}, license.spdx_id are cross-checked",
    "issue_metadata": "GET https://api.github.com/repos/{owner}/{name}/issues/{n} (or "
                      "/pulls/{n}): raw response bytes, SHA-256, timestamp and ETag; "
                      "fields url, number and repository_url are cross-checked",
    "git_objects": "loose-object bodies of the buggy and fixed commits, every tree on "
                   "the paths to the licence file and target file at each revision, and "
                   "the licence and target-file blobs; each is re-hashed to its object ID",
    "network": "acquisition requires separate approval (decision D1); this module is offline",
}


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


def _parse(source: str) -> ast.AST | None:
    try:
        with warnings.catch_warnings():
            # Legacy upstream files carry invalid escape sequences; parsing them
            # for indexing must not flood the test output with SyntaxWarnings.
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def extract_functions(source: str) -> list[str]:
    """Every function definition in ``source``; unparseable fragments kept whole."""
    tree = _parse(source)
    if tree is None:
        return [source] if source.strip() else []
    found = [ast.get_source_segment(source, node) or ""
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return [text for text in found if text.strip()] or ([source] if source.strip() else [])


def code_shingles(code: str) -> frozenset[str]:
    return shingles(normalise(code) or code)


def fingerprint(items: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    """JSON round trip: the form a collection takes once written to a receipt."""
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


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
    #: Data inputs: source -> {path relative to the data root: raw SHA-256}.
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Code inputs: source -> {path relative to the code root: canonical SHA-256}.
    code_inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    counts: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_repositories: dict[str, set[str]] = field(default_factory=dict)
    _postings: Any = field(default=None, repr=False)
    _sealed: bool = field(default=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("a sealed reference universe cannot be modified")
        object.__setattr__(self, name, value)

    def postings(self) -> Mapping[str, Any]:
        if self._postings is None:
            self._postings = _index(self.functions.items())
        return self._postings

    def bind(self, source: str, root: Path, path: Path) -> None:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        self.inputs.setdefault(source, {})[relative] = _sha_file(path)

    def bind_code(self, source: str, code_root: Path, relative: str) -> None:
        self.code_inputs.setdefault(source, {})[relative] = canonical_sha256(code_root / relative)

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

    def sealed(self) -> "ReferenceUniverse":
        """An immutable copy: frozen containers, precomputed postings, no setters."""
        def frozen_map(value):
            return MappingProxyType({key: (frozenset(item) if isinstance(item, set) else
                                           MappingProxyType(dict(item))
                                           if isinstance(item, dict) else item)
                                     for key, item in value.items()})
        copy_ = ReferenceUniverse(
            repository_names=frozenset(self.repository_names),
            repository_full=frozenset(self.repository_full),
            fork_parents=MappingProxyType(dict(self.fork_parents)),
            commits=frozenset(self.commits), patch_hashes=frozenset(self.patch_hashes),
            patch_shingles=MappingProxyType(dict(self.patch_shingles)),
            instance_ids=frozenset(self.instance_ids),
            functions=MappingProxyType(dict(self.functions)),
            function_source=MappingProxyType(dict(self.function_source)),
            inputs=frozen_map(self.inputs), code_inputs=frozen_map(self.code_inputs),
            counts=MappingProxyType(copy.deepcopy(dict(self.counts))),
            source_repositories=frozen_map(self.source_repositories))
        copy_._postings = MappingProxyType({key: tuple(value) for key, value in
                                            _index(self.functions.items()).items()})
        copy_._sealed = True
        return copy_


def build_reference_universe(root: Path, include_train_shard: bool = True,
                             code_root: Path = CODE_ROOT) -> ReferenceUniverse:
    """Index every upstream source that has ever fed an Oneiros split.

    Reads the corpus manifest, upstream copies, the curated seed definition in
    code, the legacy real-bug files and the permitted train shard only.  A
    missing required input raises; it is never skipped.
    """
    universe = ReferenceUniverse()
    data = root / "data"

    # The corpus manifest decides which sources require indexing; it is bound
    # so a changed declaration invalidates the receipt.
    manifest_path = root / CORPUS_MANIFEST
    universe.bind("corpus_manifest", root, manifest_path)
    for relative in CODE_INPUTS["corpus_manifest"]:
        universe.bind_code("corpus_manifest", code_root, relative)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    universe.counts["corpus_manifest"] = {
        "path": CORPUS_MANIFEST, "sha256": _sha_file(manifest_path),
        "records_by_source": dict(sorted((manifest.get("records_by_source") or {}).items()))}

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
    loaded_from = Path(bugsinpy_loader.__file__)
    if canonical_sha256(loaded_from) != canonical_sha256(code_root / CODE_INPUTS[
            "manual_curated_examples"][0]):
        raise ValueError("the imported curated definition differs from the bound code root")
    for relative in CODE_INPUTS["manual_curated_examples"]:
        universe.bind_code("manual_curated_examples", code_root, relative)
    functions = 0
    for bug in definitions:
        universe.add_repository(bug["project"], "manual_curated_examples")
        for side in ("buggy_code", "fixed_code"):
            functions += universe.add_function_source(
                f"curated::{bug['id']}::{side}", bug[side], "manual_curated_examples")
    # CURATED_BUGSINPY_BUGS is the only generator of curated records
    # (scripts/build_corpus_v1.curated_fixed_bug_records), so indexing every
    # definition is a conservative superset of whichever subset entered any
    # split.  The historical execution filter is NOT re-run: its outcome is
    # timing-sensitive and would make the receipt non-deterministic.
    universe.counts["manual_curated_examples"] = {
        "definitions": len(definitions),
        "definition_ids": sorted(f"curated::{bug['id']}" for bug in definitions),
        "definitions_sha256": _sha_json(definitions), "functions": functions}

    if include_train_shard:
        from harness.corpus_view import load_development_split, verify_development_view
        from utils.dataset_identity import dataset_name_from_source
        corpus = root / Path(CORPUS_MANIFEST).parent
        view = verify_development_view(corpus, ["train"])
        universe.bind("train_shard", root, corpus / "development_view" / "manifest.json")
        universe.bind("train_shard", root,
                      corpus / "development_view" / view["splits"]["train"]["filename"])
        for relative in CODE_INPUTS["train_shard"]:
            universe.bind_code("train_shard", code_root, relative)
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

def audit_source_coverage(universe: ReferenceUniverse, root: Path | None = None) -> dict[str, Any]:
    """Evidence-based coverage: concrete indexed counts and bound inputs per source.

    The sources that require indexing come from the corpus manifest BOUND in the
    universe; its hash is recorded, and with ``root`` it is re-verified on disk.
    """
    problems: list[str] = []
    bound_manifest = universe.counts.get("corpus_manifest") or {}
    manifest_sha = bound_manifest.get("sha256")
    if not bound_manifest or (universe.inputs.get("corpus_manifest") or {}).get(
            CORPUS_MANIFEST) != manifest_sha:
        problems.append("corpus manifest is not bound")
    elif root is not None and _sha_file(root / CORPUS_MANIFEST) != manifest_sha:
        problems.append("corpus manifest changed since it was indexed")
    declared = dict(bound_manifest.get("records_by_source") or {})
    if bound_manifest and not declared:
        problems.append("corpus manifest declares no sources")
    per_source: dict[str, Any] = {}
    for source, count in sorted(declared.items()):
        target = CORPUS_SOURCE_TO_UNIVERSE.get(source)
        if target is None:
            problems.append(f"unknown corpus source {source!r}")
            continue
        indexed = universe.counts.get(target)
        bound = {**universe.inputs.get(target, {}), **universe.code_inputs.get(target, {})}
        if not indexed or not bound:
            problems.append(f"declared source {source!r} is not indexed")
            continue
        if not any(isinstance(value, int) and value > 0 for value in indexed.values()):
            problems.append(f"source {source!r} indexed with zero items")
        per_source[source] = {"declared_records": count, "indexed": _plain(dict(indexed)),
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
        per_source.setdefault("manual_curated_examples", {}).update({
            "non_train_seed_candidates": sorted(candidates),
            "non_train_seeds_required": missing,
            "note": "all definitions are indexed: a conservative superset of the seeds "
                    "that entered any split"})
    return {"covered": not problems, "problems": problems,
            "corpus_manifest_sha256": manifest_sha, "per_source": per_source}


# --- receipt -----------------------------------------------------------------

def universe_collections(universe: ReferenceUniverse) -> dict[str, Any]:
    """The receipt's collections, recomputed from a universe."""
    function_fingerprints: dict[str, list[str]] = {}
    for key in sorted(universe.functions):
        function_fingerprints.setdefault(universe.function_source[key], []).append(
            fingerprint(universe.functions[key]))
    for values in function_fingerprints.values():
        values.sort()
    return _plain({
        "repository_full_names": sorted(universe.repository_full),
        "repository_names": sorted(universe.repository_names),
        "fork_parent_identities": dict(sorted(universe.fork_parents.items())),
        "commits": sorted(universe.commits),
        "patch_hashes": sorted(universe.patch_hashes),
        "instance_ids": sorted(universe.instance_ids),
        "function_fingerprints": function_fingerprints,
        "input_files": {source: dict(sorted(files.items()))
                        for source, files in sorted(universe.inputs.items())},
        "code_input_files": {source: dict(sorted(files.items()))
                             for source, files in sorted(universe.code_inputs.items())},
        "counts": {key: dict(value) for key, value in universe.counts.items()},
    })


def reference_universe_receipt(universe: ReferenceUniverse,
                               code_root: Path = CODE_ROOT) -> dict[str, Any]:
    """Deterministic, hash-bound description of the complete universe.

    The canonical source list is fixed (``CANONICAL_SOURCES``); a caller cannot
    choose which sources are bound.
    """
    collections = universe_collections(universe)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "isolation_version": ISOLATION_VERSION,
        "claim": CLAIM,
        "near_duplicate_threshold": NEAR_DUPLICATE_JACCARD,
        "min_function_shingles": MIN_FUNCTION_SHINGLES,
        "diff_policy": DIFF_POLICY,
        "temporal_rule": TEMPORAL_RULE,
        "fork_parent_note": ("benchmark repositories are recorded as their canonical "
                             "upstreams; no fork relation among them is asserted offline, so "
                             "every candidate must supply verified fork evidence"),
        "curated_note": ("all CURATED_BUGSINPY_BUGS definitions are indexed: a conservative "
                         "superset of the curated seeds that entered any split"),
        "collections": collections,
        "collection_sha256": {name: _sha_json(value) for name, value in collections.items()},
        "coverage": audit_source_coverage(universe),
        "source_files_sha256": {relative: canonical_sha256(code_root / relative)
                                for relative in sorted(CANONICAL_SOURCES)},
    }
    body["receipt_sha256"] = _sha_json(body)
    return body


def verify_receipt(receipt: Mapping[str, Any]) -> bool:
    """Internal consistency only: the receipt hash and every collection hash."""
    try:
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        return (_sha_json(body) == receipt.get("receipt_sha256")
                and set(receipt["collection_sha256"]) == set(receipt["collections"])
                and all(_sha_json(receipt["collections"][name]) == digest
                        for name, digest in receipt["collection_sha256"].items()))
    except (KeyError, TypeError, AttributeError):
        return False


class ReferenceUniverseMismatch(ValueError):
    """A universe, its receipt, its inputs or its sources do not agree."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems[:10]) + (" ..." if len(problems) > 10 else ""))
        self.problems = problems


_SEAL = object()


class FrozenReferenceUniverse:
    """A reference universe verified against its receipt at load time.

    Only ``freeze_reference_universe`` / ``load_frozen_reference_universe`` can
    create one.  It carries the sealed universe, a private copy of the verified
    receipt, the internally computed receipt SHA-256 and the verification
    report.  It is immutable.
    """
    __slots__ = ("_universe", "_receipt", "_receipt_sha256", "_verification")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenReferenceUniverse is created only by freeze_reference_universe "
                        "or load_frozen_reference_universe")

    @classmethod
    def _create(cls, seal: object, universe: ReferenceUniverse, receipt: Mapping[str, Any],
                verification: Mapping[str, Any]) -> "FrozenReferenceUniverse":
        if seal is not _SEAL:
            raise TypeError("FrozenReferenceUniverse requires verification")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_universe", universe.sealed())
        object.__setattr__(instance, "_receipt", json.dumps(receipt, sort_keys=True))
        object.__setattr__(instance, "_receipt_sha256", _sha_json(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}))
        object.__setattr__(instance, "_verification", json.dumps(verification, sort_keys=True))
        return instance

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a frozen reference universe is immutable")

    @property
    def universe(self) -> ReferenceUniverse:
        return self._universe

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt)

    @property
    def receipt_sha256(self) -> str:
        return self._receipt_sha256

    @property
    def verification(self) -> dict[str, Any]:
        return json.loads(self._verification)


def freeze_reference_universe(universe: ReferenceUniverse, receipt: Mapping[str, Any], *,
                              root: Path, code_root: Path = CODE_ROOT
                              ) -> FrozenReferenceUniverse:
    """Verify ``universe`` against ``receipt`` and the files on disk; refuse on any mismatch.

    1. recompute the collections and require them to equal the receipt's;
    2. require every collection hash to match;
    3. require the internal receipt hash to verify;
    4. re-hash every data input under ``root``, every code input and every
       canonical source under ``code_root``;
    5. recompute coverage and require it to equal the receipt's and to pass.
    """
    problems: list[str] = []
    for key, expected in (("schema_version", RECEIPT_SCHEMA),
                          ("isolation_version", ISOLATION_VERSION), ("claim", CLAIM),
                          ("near_duplicate_threshold", NEAR_DUPLICATE_JACCARD),
                          ("min_function_shingles", MIN_FUNCTION_SHINGLES),
                          ("diff_policy", _plain(DIFF_POLICY)),
                          ("temporal_rule", _plain(TEMPORAL_RULE))):
        if receipt.get(key) != expected:
            problems.append(f"receipt {key} is not {expected!r}")
    if not verify_receipt(receipt):
        problems.append("receipt internal hash or a collection hash does not verify")
    recomputed = universe_collections(universe)
    stated = receipt.get("collections") or {}
    for name in sorted(set(recomputed) | set(stated)):
        if recomputed.get(name) != stated.get(name):
            problems.append(f"collection {name!r} differs from the receipt")
        elif _sha_json(recomputed[name]) != (receipt.get("collection_sha256") or {}).get(name):
            problems.append(f"collection hash {name!r} differs from the receipt")
    checked_inputs = checked_code = 0
    for source, files in sorted((stated.get("input_files") or {}).items()):
        for relative, digest in sorted(files.items()):
            path = root / relative
            if not path.is_file() or _sha_file(path) != digest:
                problems.append(f"input file changed or missing: {relative}")
            checked_inputs += 1
    for source, files in sorted((stated.get("code_input_files") or {}).items()):
        for relative, digest in sorted(files.items()):
            path = code_root / relative
            if not path.is_file() or canonical_sha256(path) != digest:
                problems.append(f"code input changed or missing: {relative}")
            checked_code += 1
    sources = receipt.get("source_files_sha256") or {}
    if set(sources) != set(CANONICAL_SOURCES):
        problems.append("receipt does not bind exactly the canonical source set")
    for relative, digest in sorted(sources.items()):
        path = code_root / relative
        if not path.is_file() or canonical_sha256(path) != digest:
            problems.append(f"canonical source changed or missing: {relative}")
    coverage = audit_source_coverage(universe, root)
    if not coverage["covered"]:
        problems.append("coverage audit fails: " + "; ".join(coverage["problems"]))
    if _plain(coverage) != receipt.get("coverage"):
        problems.append("coverage differs from the receipt")
    if problems:
        raise ReferenceUniverseMismatch(problems)
    verification = {"collections_match": True, "collection_hashes_match": True,
                    "receipt_hash_verifies": True, "input_files_verified": checked_inputs,
                    "code_inputs_verified": checked_code,
                    "canonical_sources_verified": len(sources), "coverage_covered": True,
                    "corpus_manifest_sha256": coverage["corpus_manifest_sha256"]}
    return FrozenReferenceUniverse._create(_SEAL, universe, receipt, verification)


def load_frozen_reference_universe(root: Path, receipt_path: Path,
                                   code_root: Path = CODE_ROOT,
                                   include_train_shard: bool = True) -> FrozenReferenceUniverse:
    """Rebuild the universe from disk and freeze it against the stored receipt."""
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    universe = build_reference_universe(root, include_train_shard=include_train_shard,
                                        code_root=code_root)
    return freeze_reference_universe(universe, receipt, root=root, code_root=code_root)


# --- git objects (offline verification of acquired evidence) ----------------------

def git_object_id(kind: str, body: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(body)}".encode("ascii") + b"\0" + body).hexdigest()


def parse_commit(body: bytes) -> dict[str, Any]:
    header = body.split(b"\n\n", 1)[0].decode("utf-8", "replace")
    parsed: dict[str, Any] = {"tree": None, "parents": [], "author_epoch": None,
                              "committer_epoch": None}
    for line in header.splitlines():
        key, _, value = line.partition(" ")
        if key == "tree":
            parsed["tree"] = value.strip()
        elif key == "parent":
            parsed["parents"].append(value.strip())
        elif key in ("author", "committer"):
            match = re.search(r" (\d+) [+-]\d{4}$", value)
            parsed[f"{key}_epoch"] = int(match.group(1)) if match else None
    return parsed


def parse_tree(body: bytes) -> dict[str, tuple[str, str]]:
    """{name: (mode, object id)} of one git tree object."""
    entries: dict[str, tuple[str, str]] = {}
    position = 0
    while position < len(body):
        space = body.index(b" ", position)
        nul = body.index(b"\0", space)
        mode = body[position:space].decode("ascii")
        name = body[space + 1:nul].decode("utf-8")
        entries[name] = (mode, body[nul + 1:nul + 21].hex())
        position = nul + 21
    return entries


def resolve_path(objects: Mapping[str, tuple[str, bytes]], tree_id: str, path: str) -> str | None:
    """Blob ID at ``path`` below ``tree_id``, walking only verified tree objects."""
    current = tree_id
    parts = path.split("/")
    for index, part in enumerate(parts):
        kind, body = objects.get(current, ("", b""))
        if kind != "tree":
            return None
        entry = parse_tree(body).get(part)
        if entry is None:
            return None
        mode, current = entry
        is_last = index == len(parts) - 1
        if is_last != (mode != "40000"):
            return None
    return current


class IncompleteTreeEvidence(ValueError):
    """A changed subtree cannot be listed from the supplied tree objects."""


def changed_paths(objects: Mapping[str, tuple[str, bytes]], buggy_tree: str, fixed_tree: str,
                  prefix: str = "") -> list[str]:
    """Every path whose entry differs between two authenticated trees.

    Descends only into subtrees whose IDs differ, so unchanged content needs no
    evidence.  An added or removed subtree is reported as ``path/``.  A differing
    subtree whose object is missing raises: partial evidence is never accepted.
    """
    if buggy_tree == fixed_tree:
        return []
    listings = []
    for tree in (buggy_tree, fixed_tree):
        kind, body = objects.get(tree, ("", b""))
        if kind != "tree":
            raise IncompleteTreeEvidence(prefix or "/")
        listings.append(parse_tree(body))
    old, new = listings
    changed: list[str] = []
    for name in sorted(set(old) | set(new)):
        before, after = old.get(name), new.get(name)
        if before == after:
            continue
        path = prefix + name
        if before and after and before[0] == after[0] == "40000":
            changed += changed_paths(objects, before[1], after[1], path + "/")
        elif (before and before[0] == "40000") or (after and after[0] == "40000"):
            changed.append(path + "/")
        else:
            changed.append(path)
    return changed


def canonical_diff(path: str, buggy_text: str, fixed_text: str) -> str:
    """The frozen, deterministic diff between two authenticated file versions."""
    lines = list(difflib.unified_diff(buggy_text.splitlines(), fixed_text.splitlines(),
                                      f"a/{path}", f"b/{path}", n=3, lineterm=""))
    return "\n".join(lines) + "\n" if lines else ""


def changed_hunks(buggy_text: str, fixed_text: str) -> list[dict[str, Any]]:
    """Non-equal opcodes of the canonical diff, as 1-based line ranges."""
    matcher = difflib.SequenceMatcher(None, buggy_text.splitlines(), fixed_text.splitlines())
    return [{"op": op, "buggy": [i1 + 1, i2], "fixed": [j1 + 1, j2]}
            for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal"]


def parse_unified_diff(patch: str) -> dict[str, dict[str, Counter]] | None:
    """{path: {"removed": Counter, "added": Counter}} of exact changed lines.

    Hunks are read by their header line counts, so a changed line that looks like
    a header cannot hide.  Returns None for a malformed diff.
    """
    files: dict[str, dict[str, Counter]] = {}
    old_path = new_path = None
    lines = str(patch or "").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- "):
            old_path = line[4:].split("\t")[0].strip()
        elif line.startswith("+++ "):
            new_path = line[4:].split("\t")[0].strip()
        elif line.startswith("@@"):
            match = re.match(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
            if not match or new_path is None or old_path is None:
                return None
            path = new_path if new_path != "/dev/null" else old_path
            path = path[2:] if path[:2] in ("a/", "b/") else path
            entry = files.setdefault(path, {"removed": Counter(), "added": Counter()})
            old_left = int(match.group(1)) if match.group(1) is not None else 1
            new_left = int(match.group(2)) if match.group(2) is not None else 1
            while old_left > 0 or new_left > 0:
                index += 1
                if index >= len(lines):
                    return None
                body = lines[index]
                if body.startswith("\\"):
                    continue
                tag, text = (body[:1], body[1:]) if body else (" ", "")
                if tag == " ":
                    old_left, new_left = old_left - 1, new_left - 1
                elif tag == "-":
                    entry["removed"][text.rstrip("\r")] += 1
                    old_left -= 1
                elif tag == "+":
                    entry["added"][text.rstrip("\r")] += 1
                    new_left -= 1
                else:
                    return None
                if old_left < 0 or new_left < 0:
                    return None
        index += 1
    return files


def _function_spans(source: str) -> list[dict[str, Any]]:
    """Every function with its qualified name, line span and normalised source."""
    tree = _parse(source)
    if tree is None:
        return []
    found: list[dict[str, Any]] = []

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = stack + [child.name]
                if not isinstance(child, ast.ClassDef):
                    segment = ast.get_source_segment(source, child) or ""
                    start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    found.append({"qualname": ".".join(qualname), "span": [start,
                                  child.end_lineno], "normalised": normalise(segment)
                                  or segment.strip()})
                visit(child, qualname)
            else:
                visit(child, stack)
    visit(tree, [])
    return found


def _overlaps(hunk: dict[str, Any], span: list[int]) -> bool:
    start, end = span
    first, last = hunk["buggy"]
    if hunk["op"] == "insert":           # inserted after buggy line ``first - 1``
        return start <= first - 1 <= end
    return first <= end and last >= start


# --- candidates -----------------------------------------------------------------

@dataclass(frozen=True)
class ApiResponse:
    """One stored GitHub API response (acquired later; verified offline)."""
    url: str = ""
    body: bytes = b""
    sha256: str = ""
    retrieved_utc: str = ""
    etag: str = ""


@dataclass(frozen=True)
class GitObject:
    """A git loose-object body; its ID is recomputed, never trusted."""
    kind: str = ""
    body: bytes = b""


@dataclass(frozen=True)
class CandidateBug:
    repository: str = ""                  # canonical lower-case owner/name
    repository_url: str = ""
    repository_id: str = ""               # numeric GitHub ID
    repository_node_id: str = ""
    fork_status: str = "unknown"          # "not_fork" | "fork"; anything else refuses
    fork_parent: str = ""
    fork_parent_id: str = ""
    fork_parent_node_id: str = ""
    buggy_commit: str = ""
    fixed_commit: str = ""
    #: Submitted/display patch.  Never authoritative: it must match the diff
    #: derived from the authenticated blobs, and only the derived diff is used.
    patch: str = ""
    target_function: str = ""
    target_file: str = ""
    target_module: str = ""
    target_blob_buggy: str = ""
    target_blob_fixed: str = ""
    issue_id: str = ""                    # "owner/name#123"
    licence_spdx: str = ""
    licence_path: str = ""                # licence file path at the buggy commit
    licence_blob: str = ""
    licence_sha256: str = ""
    repository_metadata: ApiResponse = field(default_factory=ApiResponse)
    issue_metadata: ApiResponse = field(default_factory=ApiResponse)
    git_objects: tuple[GitObject, ...] = ()


def _api_problems(label: str, response: ApiResponse, expected_url: str) -> list[str]:
    problems = []
    if not isinstance(response, ApiResponse):
        return [f"{label}_missing"]
    if str(response.url) != expected_url:
        problems.append(f"{label}_url_missing_or_inconsistent")
    if not isinstance(response.body, bytes) or not response.body:
        problems.append(f"{label}_response_missing")
    if not SHA64.match(str(response.sha256 or "")):
        problems.append(f"{label}_response_hash_missing")
    if not UTC_TIMESTAMP.match(str(response.retrieved_utc or "")):
        problems.append(f"{label}_retrieval_timestamp_missing")
    return problems


def evidence_problems(candidate: CandidateBug) -> list[str]:
    """Stage 1, schema: every missing or malformed piece of mandatory evidence."""
    problems: list[str] = []
    repository = str(candidate.repository or "").strip()
    if not OWNER_NAME.match(repository):
        problems.append("repository_not_canonical_owner_name")
    expected_url = f"https://github.com/{repository}"
    if str(candidate.repository_url).rstrip("/").lower() != expected_url:
        problems.append("repository_url_unverified_or_inconsistent")
    if not str(candidate.repository_id).isdigit():
        problems.append("repository_identity_missing")
    if not NODE_ID.match(str(candidate.repository_node_id or "")):
        problems.append("repository_node_id_missing")
    if candidate.fork_status == "fork":
        if not OWNER_NAME.match(str(candidate.fork_parent or "")):
            problems.append("fork_parent_identity_missing")
        if not str(candidate.fork_parent_id).isdigit():
            problems.append("fork_parent_id_missing")
        if not NODE_ID.match(str(candidate.fork_parent_node_id or "")):
            problems.append("fork_parent_node_id_missing")
    elif candidate.fork_status != "not_fork":
        problems.append("fork_status_unknown")
    for label in ("buggy_commit", "fixed_commit", "target_blob_buggy", "target_blob_fixed",
                  "licence_blob"):
        if not SHA40.match(str(getattr(candidate, label) or "")):
            problems.append(f"{label}_not_full_sha")
    if candidate.buggy_commit and candidate.buggy_commit == candidate.fixed_commit:
        problems.append("buggy_and_fixed_commit_identical")
    if not normalised_patch(candidate.patch):
        problems.append("patch_empty")
    code = str(candidate.target_function or "")
    tree = _parse(code) if code.strip() else None
    if tree is None or not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                               for node in tree.body):
        problems.append("target_function_missing_or_unparseable")
    elif len(code_shingles(code)) < MIN_FUNCTION_SHINGLES:
        problems.append("target_function_too_short_for_comparison")
    for label in ("target_file", "licence_path"):
        value = str(getattr(candidate, label) or "")
        if not re.match(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$", value) or ".." in value.split("/"):
            problems.append(f"{label}_missing")
    if not str(candidate.target_file).endswith(".py"):
        problems.append("target_file_not_python")
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
    problems += _api_problems("repository_metadata", candidate.repository_metadata,
                              f"{API}{repository}")
    number = issue.group(2) if issue else ""
    issue_url = str(getattr(candidate.issue_metadata, "url", ""))
    if issue_url not in (f"{API}{repository}/issues/{number}", f"{API}{repository}/pulls/{number}"):
        issue_url = f"{API}{repository}/issues/{number}"
    problems += _api_problems("issue_metadata", candidate.issue_metadata, issue_url)
    if not candidate.git_objects or not all(
            isinstance(item, GitObject) and item.kind in ("commit", "tree", "blob")
            and isinstance(item.body, bytes) for item in candidate.git_objects):
        problems.append("git_object_evidence_missing")
    return [f"{INSUFFICIENT}:{problem}" for problem in problems]


def _json_body(response: ApiResponse) -> dict[str, Any] | None:
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _utc_epoch(value: str) -> int | None:
    try:
        return int(datetime.strptime(value.split(".")[0].rstrip("Z"), "%Y-%m-%dT%H:%M:%S")
                   .replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def _metadata_problems(candidate: CandidateBug) -> list[str]:
    problems: list[str] = []
    repository = candidate.repository
    meta = candidate.repository_metadata
    if _sha_bytes(meta.body) != meta.sha256:
        problems.append("repository_metadata_hash_mismatch")
    body = _json_body(meta)
    if body is None:
        problems.append("repository_metadata_not_json")
    else:
        queried = meta.url[len(API):]
        if str(body.get("full_name", "")).lower() != queried:
            problems.append("repository_renamed_or_redirected")
        if str(body.get("full_name", "")).lower() != repository:
            problems.append("repository_full_name_mismatch")
        if str(body.get("id")) != str(candidate.repository_id):
            problems.append("repository_id_mismatch")
        if body.get("node_id") != candidate.repository_node_id:
            problems.append("repository_node_id_mismatch")
        if str(body.get("html_url", "")).lower().rstrip("/") != \
                candidate.repository_url.lower().rstrip("/"):
            problems.append("repository_html_url_mismatch")
        fork = body.get("fork")
        if not isinstance(fork, bool) or fork != (candidate.fork_status == "fork"):
            problems.append("fork_status_mismatch")
        parent = body.get("parent")
        if candidate.fork_status == "fork":
            if not isinstance(parent, dict) or (
                    str(parent.get("full_name", "")).lower() != candidate.fork_parent
                    or str(parent.get("id")) != str(candidate.fork_parent_id)
                    or parent.get("node_id") != candidate.fork_parent_node_id):
                problems.append("fork_parent_mismatch")
        elif parent is not None:
            problems.append("undeclared_fork_parent")
        licence = body.get("license")
        if not isinstance(licence, dict) or licence.get("spdx_id") != candidate.licence_spdx:
            problems.append("licence_spdx_mismatch")
    issue = candidate.issue_metadata
    if _sha_bytes(issue.body) != issue.sha256:
        problems.append("issue_metadata_hash_mismatch")
    issue_body = _json_body(issue)
    number = int(candidate.issue_id.rsplit("#", 1)[1])
    if issue_body is None:
        problems.append("issue_metadata_not_json")
    else:
        if issue_body.get("number") != number:
            problems.append("issue_number_mismatch")
        if issue_body.get("url") != issue.url:
            problems.append("issue_url_mismatch")
        if str(issue_body.get("repository_url", "")).lower() != f"{API}{repository}" and not (
                issue.url.endswith(f"/pulls/{number}") and str(
                    ((issue_body.get("base") or {}).get("repo") or {}).get("full_name", "")
                ).lower() == repository):
            problems.append("issue_repository_identity_mismatch")
    return problems


def _temporal_problems(candidate: CandidateBug, commits: dict[str, dict[str, Any]],
                       derived: dict[str, Any]) -> list[str]:
    """The frozen temporal rule, proved from authenticated commit objects."""
    fixed = commits.get("fixed_commit")
    if fixed is None:
        return ["temporal_rule_unprovable_without_fixed_commit"]
    author, committer = fixed["author_epoch"], fixed["committer_epoch"]
    derived["fixed_commit_author_epoch"] = author
    derived["fixed_commit_committer_epoch"] = committer
    if author is None or committer is None:
        return ["fixed_commit_timestamp_missing"]
    problems = []
    if author > committer:
        problems.append("fixed_commit_author_after_committer")
    buggy = commits.get("buggy_commit") or {}
    if buggy.get("committer_epoch") is None:
        problems.append("buggy_commit_timestamp_missing")
    elif buggy["committer_epoch"] > committer:
        problems.append("buggy_commit_committed_after_fix")
    retrieved = [_utc_epoch(candidate.repository_metadata.retrieved_utc),
                 _utc_epoch(candidate.issue_metadata.retrieved_utc)]
    if any(value is None for value in retrieved):
        problems.append("retrieval_timestamp_unparseable")
    elif committer > min(retrieved):
        problems.append("fixed_commit_later_than_evidence_retrieval")
    satisfied = committer >= TEMPORAL_CUTOFF_EPOCH
    derived["temporal_rule_satisfied"] = satisfied and not problems
    if not satisfied:
        problems.append("fixed_commit_before_temporal_cutoff")
    return problems


def authentication_problems(candidate: CandidateBug) -> tuple[list[str], dict[str, Any]]:
    """Stage 2, authentication: cross-check the stored evidence (offline).

    Returns the problems and the facts derived from authenticated evidence,
    including the canonical diff (key ``canonical_diff``), which is the ONLY diff
    later stages use.  Run only on schema-valid candidates.
    """
    problems = _metadata_problems(candidate)
    derived: dict[str, Any] = {"diff_policy": DIFF_POLICY["id"]}

    objects: dict[str, tuple[str, bytes]] = {}
    for item in candidate.git_objects:
        objects[git_object_id(item.kind, item.body)] = (item.kind, item.body)
    commits: dict[str, dict[str, Any]] = {}
    for label in ("buggy_commit", "fixed_commit"):
        kind, body_bytes = objects.get(getattr(candidate, label), ("", b""))
        if kind != "commit":
            problems.append(f"{label}_object_missing")
        else:
            commits[label] = parse_commit(body_bytes)
    if "fixed_commit" in commits:
        parents = commits["fixed_commit"]["parents"]
        if candidate.buggy_commit not in parents:
            problems.append("fixed_commit_parent_is_not_buggy_commit")
        elif len(parents) != 1:
            problems.append("fixed_commit_is_not_a_direct_single_parent_commit")
    problems += _temporal_problems(candidate, commits, derived)

    if "buggy_commit" in commits:
        tree = commits["buggy_commit"]["tree"]
        if resolve_path(objects, tree, candidate.licence_path) != candidate.licence_blob:
            problems.append("licence_blob_not_at_licence_path_in_buggy_tree")
        kind, licence_bytes = objects.get(candidate.licence_blob, ("", b""))
        if kind != "blob" or _sha_bytes(licence_bytes) != candidate.licence_sha256:
            problems.append("licence_file_hash_mismatch")
        if resolve_path(objects, tree, candidate.target_file) != candidate.target_blob_buggy:
            problems.append("target_blob_buggy_not_at_target_file")
    if "fixed_commit" in commits and resolve_path(
            objects, commits["fixed_commit"]["tree"],
            candidate.target_file) != candidate.target_blob_fixed:
        problems.append("target_blob_fixed_not_at_target_file")

    # Complete changed-file list from the authenticated trees (policy A).
    if len(commits) == 2:
        try:
            files = changed_paths(objects, commits["buggy_commit"]["tree"],
                                  commits["fixed_commit"]["tree"])
        except IncompleteTreeEvidence as exc:
            files = None
            problems.append(f"changed_files_not_fully_evidenced:{exc}")
        derived["changed_files"] = files
        if files is not None:
            if candidate.target_file not in files:
                problems.append("target_file_unchanged_by_fix")
            if len(files) > 1:
                problems.append("multi_file_fix_not_admitted_under_single_file_policy")

    buggy_kind, buggy_bytes = objects.get(candidate.target_blob_buggy, ("", b""))
    fixed_kind, fixed_bytes = objects.get(candidate.target_blob_fixed, ("", b""))
    blobs_resolved = not any(problem.startswith(("target_blob_", "buggy_commit_object",
                                                 "fixed_commit_object"))
                             for problem in problems)
    if buggy_kind != "blob" or fixed_kind != "blob" or not blobs_resolved:
        problems.append("target_blob_object_missing_or_unresolved")
        return [f"{INSUFFICIENT}:authentication:{problem}" for problem in problems], derived

    buggy_text = buggy_bytes.decode("utf-8", "replace")
    fixed_text = fixed_bytes.decode("utf-8", "replace")
    diff = canonical_diff(candidate.target_file, buggy_text, fixed_text)
    hunks = changed_hunks(buggy_text, fixed_text)
    derived.update({"canonical_diff": diff,
                    "canonical_diff_sha256": _sha_bytes(diff.encode("utf-8")),
                    "canonical_patch_hash": patch_hash(diff), "changed_hunks": hunks})
    if not diff:
        problems.append("target_file_unchanged_by_fix")

    # The submitted patch must carry exactly the derived changes, per file.
    submitted = parse_unified_diff(candidate.patch)
    expected = parse_unified_diff(diff)
    derived["submitted_patch_sha256"] = _sha_bytes(str(candidate.patch).encode("utf-8"))
    derived["submitted_patch_matches_derived_diff"] = submitted is not None and \
        submitted == expected
    if submitted is None:
        problems.append("submitted_patch_unparseable")
    elif submitted != expected:
        problems.append("submitted_patch_does_not_match_derived_diff")

    # The diff must modify the declared target function.
    wanted = normalise(candidate.target_function) or candidate.target_function.strip()
    matches = [item for item in _function_spans(buggy_text) if item["normalised"] == wanted]
    if len(matches) != 1:
        problems.append("target_function_not_in_buggy_file" if not matches
                        else "target_function_ambiguous_in_buggy_file")
    else:
        target = matches[0]
        fixed_side = [item for item in _function_spans(fixed_text)
                      if item["qualname"] == target["qualname"]]
        derived.update({"target_qualname": target["qualname"],
                        "target_span_buggy": target["span"],
                        "target_span_fixed": fixed_side[0]["span"] if len(fixed_side) == 1
                        else None})
        if not any(_overlaps(hunk, target["span"]) for hunk in hunks):
            problems.append("no_changed_hunk_overlaps_target_function")
        if len(fixed_side) != 1:
            problems.append("target_function_missing_or_ambiguous_in_fixed_file")
        elif fixed_side[0]["normalised"] == target["normalised"]:
            problems.append("declared_target_function_unchanged")
    dotted = candidate.target_file[:-3].replace("/", ".")
    if not (dotted == candidate.target_module or dotted.endswith("." + candidate.target_module)
            or (dotted.endswith(".__init__")
                and dotted[:-9].endswith(candidate.target_module))):
        problems.append("target_module_inconsistent_with_target_file")
    return [f"{INSUFFICIENT}:authentication:{problem}" for problem in problems], derived


def overlap_problems(candidate: CandidateBug, universe: ReferenceUniverse, threshold: float,
                     derived_diff: str | None = None) -> tuple[list[str], dict[str, Any]]:
    """Stage 3, source-universe overlap.

    Patch checks use ONLY ``derived_diff``, the diff recomputed from
    authenticated blobs; without one they are not run (and the candidate is
    already refused by authentication).
    """
    reasons: list[str] = []
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
    patch_checks = "not_run:no_authenticated_diff"
    if derived_diff and normalised_patch(derived_diff):
        patch_checks = "run_on_derived_diff"
        if patch_hash(derived_diff) in universe.patch_hashes:
            reasons.append("patch_identical_to_known_bug")
        mine = _patch_shingles(derived_diff)
        for key, items in universe.patch_shingles.items():
            score = jaccard(items, mine)
            if score > nearest_patch["jaccard"]:
                nearest_patch = {"reference": key, "jaccard": round(score, 4)}
        if nearest_patch["jaccard"] >= threshold:
            reasons.append("patch_near_duplicate_of_known_bug")
    nearest_function: dict[str, Any] = {"reference": None, "jaccard": 0.0}
    mine = code_shingles(candidate.target_function) if str(
        candidate.target_function or "").strip() else frozenset()
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
    return reasons, {"patch_checks": patch_checks, "nearest_patch": nearest_patch,
                     "nearest_function": nearest_function,
                     "target_function_fingerprint": fingerprint(mine) if mine else None,
                     "same_organisation_as_excluded": organisation_flags}


def _evidence_summary(candidate: CandidateBug) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for item in fields(candidate):
        value = getattr(candidate, item.name)
        if item.name in ("patch", "target_function"):
            continue
        if isinstance(value, ApiResponse):
            summary[item.name] = {"url": value.url, "sha256": value.sha256,
                                  "retrieved_utc": value.retrieved_utc, "etag": value.etag}
        elif item.name == "git_objects":
            summary[item.name] = sorted(git_object_id(obj.kind, obj.body) for obj in value
                                        if isinstance(obj, GitObject)
                                        and isinstance(obj.body, bytes))
        else:
            summary[item.name] = value
    return summary


def _record_digest(record: Mapping[str, Any]) -> str:
    return _sha_json({key: value for key, value in record.items() if key != "record_sha256"})


def check_candidate(candidate: CandidateBug, frozen: FrozenReferenceUniverse) -> dict[str, Any]:
    """The isolation record for one candidate.  Admission only if nothing is refused.

    ``frozen`` must be a verified ``FrozenReferenceUniverse``; the receipt hash
    on the record comes from it.  There is no parameter for a receipt hash or a
    threshold: both are fixed by the verified receipt.
    """
    if not isinstance(frozen, FrozenReferenceUniverse):
        raise TypeError("an isolation decision requires a verified FrozenReferenceUniverse; "
                        "a receipt hash cannot be supplied separately")
    threshold = frozen.receipt["near_duplicate_threshold"]
    schema = evidence_problems(candidate)
    if schema:
        authentication, derived = ["not_run:schema_invalid"], {}
    else:
        authentication, derived = authentication_problems(candidate)
    derived_diff = derived.pop("canonical_diff", None)
    overlap, nearest = overlap_problems(candidate, frozen.universe, threshold, derived_diff)
    authentication_failures = [item for item in authentication if not item.startswith("not_run")]
    reasons = schema + authentication_failures + overlap
    record = {
        "isolation_version": ISOLATION_VERSION,
        "reference_universe_sha256": frozen.receipt_sha256,
        "claim": CLAIM,
        "diff_policy": DIFF_POLICY["id"],
        "temporal_rule": TEMPORAL_RULE["rule"],
        "admissible": not reasons,
        "reasons": reasons,
        "insufficient_evidence": bool(schema or authentication_failures),
        "stages": {"schema": schema, "authentication": authentication, "overlap": overlap},
        "authentication_passed": not schema and not authentication_failures,
        "evidence": _evidence_summary(candidate),
        "derived_from_verified_evidence": derived,
        "patch_sha256": derived.get("canonical_patch_hash"),
        **nearest,
        "threshold": threshold,
    }
    record["record_sha256"] = _record_digest(record)
    return record


def isolation_record_is_current(record: Mapping[str, Any],
                                frozen: FrozenReferenceUniverse) -> bool:
    """A record is current only for the verified universe it was made against.

    ``record_sha256`` detects ACCIDENTAL modification only.  It is a self-hash,
    not a digital signature: anyone who edits a record can recompute it.
    Trust in a record comes only from ``revalidate_isolation_record``.
    """
    if not isinstance(frozen, FrozenReferenceUniverse):
        raise TypeError("currency is judged only against a verified FrozenReferenceUniverse")
    return (record.get("isolation_version") == ISOLATION_VERSION
            and record.get("reference_universe_sha256") == frozen.receipt_sha256
            and record.get("record_sha256") == _record_digest(record))


# --- downstream revalidation ------------------------------------------------------

EVIDENCE_SIDECAR_SCHEMA = "oneiros_candidate_evidence_sidecar_v1"


def candidate_to_sidecar(candidate: CandidateBug) -> dict[str, Any]:
    """The complete authenticated evidence, JSON-serialisable (bytes as base64)."""
    def encode(name: str, value: Any) -> Any:
        if isinstance(value, ApiResponse):
            return {"url": value.url, "body_b64": base64.b64encode(value.body).decode("ascii"),
                    "sha256": value.sha256, "retrieved_utc": value.retrieved_utc,
                    "etag": value.etag}
        if name == "git_objects":
            return [{"kind": obj.kind, "body_b64": base64.b64encode(obj.body).decode("ascii")}
                    for obj in value]
        return value
    return {"schema_version": EVIDENCE_SIDECAR_SCHEMA,
            "candidate": {item.name: encode(item.name, getattr(candidate, item.name))
                          for item in fields(candidate)}}


def candidate_from_sidecar(sidecar: Mapping[str, Any]) -> CandidateBug:
    if sidecar.get("schema_version") != EVIDENCE_SIDECAR_SCHEMA:
        raise ValueError("unknown evidence sidecar schema")
    values = dict(sidecar["candidate"])
    for key in ("repository_metadata", "issue_metadata"):
        item = values[key]
        values[key] = ApiResponse(url=item["url"], body=base64.b64decode(item["body_b64"]),
                                  sha256=item["sha256"], retrieved_utc=item["retrieved_utc"],
                                  etag=item["etag"])
    values["git_objects"] = tuple(GitObject(kind=item["kind"],
                                            body=base64.b64decode(item["body_b64"]))
                                  for item in values["git_objects"])
    return CandidateBug(**values)


def canonical_record_bytes(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def revalidate_isolation_record(record: Mapping[str, Any], sidecar: Mapping[str, Any],
                                frozen: FrozenReferenceUniverse) -> dict[str, Any]:
    """Downstream trust: rerun the decision from the retained evidence.

    The stored record is accepted only if ``check_candidate`` on the evidence
    sidecar, against the same verified frozen universe, reproduces it byte for
    byte.  A hand-edited record is rejected even if its ``record_sha256`` was
    recomputed.
    """
    try:
        candidate = candidate_from_sidecar(sidecar)
    except (KeyError, TypeError, ValueError) as exc:
        return {"valid": False, "reason": f"evidence sidecar unusable: {exc}"}
    recomputed = check_candidate(candidate, frozen)
    if canonical_record_bytes(recomputed) != canonical_record_bytes(record):
        differing = sorted(key for key in set(recomputed) | set(record)
                           if recomputed.get(key) != record.get(key))
        return {"valid": False, "reason": "record differs from recomputation",
                "differing_fields": differing}
    return {"valid": True, "admissible": recomputed["admissible"],
            "record_sha256": recomputed["record_sha256"]}
