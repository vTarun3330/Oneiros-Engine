"""Build candidate bugs from acquired evidence and run the full admission pipeline.

For each candidate fix commit, in order:

1. evidence schema validation, 2. offline evidence authentication, 3. exact
policy-A diff validation, 4. temporal validation, 5. source-universe isolation
(1-5 are ``check_candidate`` against the frozen universe), 6. evidence-sidecar
revalidation (the sidecar is reloaded from the content store and the record is
recomputed byte for byte), 7. licence validation (SPDX from the authenticated
repository response must be admitted by D2, and the licence blob's text must
carry that licence's characteristic grant).

Target selection never trusts anything submitted: the target file and the
target function are chosen from the DERIVED diff of authenticated blobs (the
single function whose body changes), and the candidate's patch is that derived
diff.  Selection problems before a candidate can be formed are recorded as
pre-pipeline exclusions with an exact reason.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import re
import textwrap
from typing import Any, Mapping
import warnings

from harness.function_complexity import analyze_function_complexity
from harness.github_acquisition import (
    AcquisitionFailure, ApiClient, ContentStore, RecordingObjects,
)
from harness.repository_isolation import (
    ADMITTED_LICENCES, API, ApiResponse, CandidateBug, FrozenReferenceUniverse, GitObject,
    IncompleteTreeEvidence, candidate_from_sidecar, candidate_to_sidecar, canonical_diff,
    canonical_repository, changed_hunks, changed_paths, check_candidate, code_shingles,
    parse_commit, parse_tree, resolve_path, revalidate_isolation_record,
)
from scripts.audit_cross_split_near_duplicates import NEAR_DUPLICATE_JACCARD, jaccard, normalise

FIX_WORDS = re.compile(r"\b(fix(es|ed|ing)?|bug(fix)?|regression|incorrect(ly)?|wrong(ly)?|"
                       r"crash(es|ed)?|broken|mistake|erroneous)\b", re.IGNORECASE)
NON_CODE_SUBJECT = re.compile(r"^\s*(\[pre-commit\.ci\]|docs?\b|doc:|ci\b|chore|build\b|"
                              r"style\b|typo|test(s)?\b|bump\b|release\b|revert\b|"
                              r"update changelog)", re.IGNORECASE)
PR_SUBJECT = re.compile(r"\(#(\d+)\)\s*$")
ISSUE_REFERENCE = re.compile(r"\b(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#(\d+)",
                             re.IGNORECASE)
MERGE_SUBJECT = re.compile(r"^Merge pull request #(\d+)")
LICENCE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", "LICENCE", "COPYING",
                 "LICENSE-MIT", "LICENSE.MIT")
#: Characteristic grant text per admitted SPDX identifier (normalised whitespace).
LICENCE_MARKERS = {
    "MIT": ("permission is hereby granted, free of charge",),
    "BSD-2-Clause": ("redistribution and use in source and binary forms",),
    "BSD-3-Clause": ("redistribution and use in source and binary forms",
                     "neither the name"),
    "Apache-2.0": ("apache license", "version 2.0"),
    "ISC": ("permission to use, copy, modify, and",),
    "PSF-2.0": ("python software foundation",),
}
BUG_FAMILIES = ("boundary_or_comparison", "none_or_empty_handling", "exception_handling",
                "type_or_conversion", "call_or_argument", "other_logic")


def is_test_path(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    return (any(part in ("test", "tests", "testing") for part in parts[:-1])
            or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def classify_commit(message: str) -> tuple[bool, str, int | None]:
    """(is a fix candidate, reason, linked PR/issue number) from the commit message."""
    subject = message.strip().splitlines()[0] if message.strip() else ""
    if NON_CODE_SUBJECT.match(subject):
        return False, "non_code_commit_subject", None
    if not FIX_WORDS.search(message):
        return False, "not_described_as_a_fix", None
    merge = MERGE_SUBJECT.match(subject)
    if merge:
        return True, "fix_merge_commit", int(merge.group(1))
    match = PR_SUBJECT.search(subject) or ISSUE_REFERENCE.search(message)
    if not match:
        return False, "no_linked_pr_or_issue", None
    return True, "fix_candidate", int(match.group(1))


def _functions(source: str) -> dict[str, dict[str, Any]]:
    """{qualname: {span, segment, normalised, name}} of every function."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    found: dict[str, dict[str, Any]] = {}

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = stack + [child.name]
                if not isinstance(child, ast.ClassDef):
                    segment = ast.get_source_segment(source, child) or ""
                    start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    key = ".".join(qualname)
                    if key not in found:
                        found[key] = {"span": [start, child.end_lineno], "segment": segment,
                                      "normalised": normalise(segment) or segment.strip(),
                                      "name": child.name}
                visit(child, qualname)
            else:
                visit(child, stack)
    visit(tree, [])
    return found


def changed_functions(buggy: str, fixed: str) -> list[str]:
    """Qualified names of buggy-side functions that a hunk overlaps and whose
    normalised body changes (or that disappear) in the fixed file."""
    before, after = _functions(buggy), _functions(fixed)
    hunks = changed_hunks(buggy, fixed)
    changed = []
    for name, item in before.items():
        start, end = item["span"]
        overlaps = any((h["op"] == "insert" and start <= h["buggy"][0] - 1 <= end)
                       or (h["op"] != "insert" and h["buggy"][0] <= end and h["buggy"][1] >= start)
                       for h in hunks)
        if overlaps and (name not in after or after[name]["normalised"] != item["normalised"]):
            changed.append(name)
    # Keep the innermost function only when a nested function and its parent both change.
    return [name for name in changed
            if not any(other != name and other.startswith(name + ".") for other in changed)]


def bug_family(diff: str) -> str:
    """Heuristic label from the derived diff's changed lines (reported as heuristic)."""
    removed = " ".join(line[1:] for line in diff.splitlines()
                       if line.startswith("-") and not line.startswith("---"))
    added = " ".join(line[1:] for line in diff.splitlines()
                     if line.startswith("+") and not line.startswith("+++"))
    both = removed + " " + added
    if re.search(r"\bis (not )?None\b|\bif not\b|\blen\([^)]*\)\s*==\s*0|or \[\]|or \{\}", added) \
            and not re.search(r"\bis (not )?None\b", removed):
        return "none_or_empty_handling"
    if re.search(r"\b(try|except|raise|finally)\b", both):
        return "exception_handling"
    if re.search(r"\b(isinstance|int|str|float|bool|bytes|list|tuple|dict|set)\(", added) and \
            not re.search(r"\b(isinstance|int|str|float|bool|bytes|list|tuple|dict|set)\(",
                          removed):
        return "type_or_conversion"
    comparisons = re.compile(r"(<=|>=|==|!=|<|>|\+ ?1|- ?1)")
    if sorted(comparisons.findall(removed)) != sorted(comparisons.findall(added)):
        return "boundary_or_comparison"
    calls = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\(")
    if calls.findall(removed) and set(calls.findall(removed)) == set(calls.findall(added)):
        return "call_or_argument"
    return "other_logic"


def complexity_tier(target_function: str, name: str) -> dict[str, Any]:
    try:
        return analyze_function_complexity(textwrap.dedent(target_function), name).to_dict()
    except (ValueError, KeyError, LookupError) as exc:
        return {"tier": "unclassified", "error": str(exc)[:200]}


def licence_text_problems(spdx: str, text: str) -> list[str]:
    problems = []
    if spdx not in ADMITTED_LICENCES:
        problems.append("licence_not_admitted_by_d2")
    normalised = re.sub(r"\s+", " ", text.lower())
    markers = LICENCE_MARKERS.get(spdx, ())
    if not markers or not all(marker in normalised for marker in markers):
        problems.append("licence_text_does_not_match_spdx")
    return problems


@dataclass
class RepositoryContext:
    repository: str                 # canonical owner/name
    metadata: ApiResponse
    value: dict[str, Any]
    renames: list[dict[str, str]]


def build_candidate(context: RepositoryContext, commit_oid: str, number: int,
                    objects: RecordingObjects, client: ApiClient) -> tuple[
                        CandidateBug | None, dict[str, Any]]:
    """A candidate from authenticated objects, or (None, {exclusion}) with a reason."""
    info: dict[str, Any] = {"fixed_commit": commit_oid, "linked_number": number}
    fixed = objects.get(commit_oid)
    if fixed is None or fixed[0] != "commit":
        raise AcquisitionFailure("git_object_missing", f"fixed commit {commit_oid}")
    parsed = parse_commit(fixed[1])
    if not parsed["parents"]:
        return None, {**info, "exclusion": "root_commit_has_no_parent"}
    buggy_oid = parsed["parents"][0]
    buggy = objects.get(buggy_oid)
    if buggy is None or buggy[0] != "commit":
        raise AcquisitionFailure("git_object_missing", f"buggy commit {buggy_oid}")
    buggy_tree, fixed_tree = parse_commit(buggy[1])["tree"], parsed["tree"]
    try:
        paths = changed_paths(objects, buggy_tree, fixed_tree)
    except IncompleteTreeEvidence as exc:
        raise AcquisitionFailure("incomplete_tree_evidence", str(exc)) from exc
    info["changed_files"] = paths
    code = [path for path in paths if path.endswith(".py") and not is_test_path(path)]
    info["non_test_python_files"] = code
    if not code:
        return None, {**info, "exclusion": "no_non_test_python_file_changed"}
    target_file = sorted(code)[0]
    buggy_blob = resolve_path(objects, buggy_tree, target_file)
    fixed_blob = resolve_path(objects, fixed_tree, target_file)
    if buggy_blob is None:
        return None, {**info, "exclusion": "target_file_added_by_fix"}
    if fixed_blob is None:
        return None, {**info, "exclusion": "target_file_deleted_by_fix"}
    buggy_text = objects[buggy_blob][1].decode("utf-8", "replace")
    fixed_text = objects[fixed_blob][1].decode("utf-8", "replace")
    functions = changed_functions(buggy_text, fixed_text)
    info["changed_functions"] = functions
    if not functions:
        return None, {**info, "exclusion": "no_existing_function_changed"}
    if len(functions) > 1:
        return None, {**info, "exclusion": "multiple_functions_changed"}
    target = _functions(buggy_text)[functions[0]]
    licence_path = licence_blob = None
    root_entries = parse_tree(objects[buggy_tree][1])
    for name in LICENCE_NAMES:
        if name in root_entries and root_entries[name][0] != "40000":
            licence_path, licence_blob = name, resolve_path(objects, buggy_tree, name)
            break
    if licence_blob is None:
        return None, {**info, "exclusion": "licence_file_missing_at_buggy_commit"}
    licence_bytes = objects[licence_blob][1]
    try:
        issue, _ = client.get(f"{API}{context.repository}/pulls/{number}")
    except AcquisitionFailure as failure:
        if failure.category != "not_found":
            raise
        issue, _ = client.get(f"{API}{context.repository}/issues/{number}")
    spdx = str((context.value.get("license") or {}).get("spdx_id") or "")
    module = target_file[:-3].split("/")[-1]
    if module == "__init__":
        module = target_file[:-3].split("/")[-2] if "/" in target_file else "__init__"
    candidate = CandidateBug(
        repository=context.repository,
        repository_url=f"https://github.com/{context.repository}",
        repository_id=str(context.value.get("id", "")),
        repository_node_id=str(context.value.get("node_id", "")),
        fork_status="fork" if context.value.get("fork") else "not_fork",
        fork_parent=str(((context.value.get("parent") or {}).get("full_name") or "")).lower(),
        fork_parent_id=str((context.value.get("parent") or {}).get("id") or ""),
        fork_parent_node_id=str((context.value.get("parent") or {}).get("node_id") or ""),
        buggy_commit=buggy_oid, fixed_commit=commit_oid,
        patch=canonical_diff(target_file, buggy_text, fixed_text),
        target_function=target["segment"], target_file=target_file, target_module=module,
        target_blob_buggy=buggy_blob, target_blob_fixed=fixed_blob,
        issue_id=f"{context.repository}#{number}", licence_spdx=spdx,
        licence_path=licence_path, licence_blob=licence_blob,
        licence_sha256=hashlib.sha256(licence_bytes).hexdigest(),
        repository_metadata=context.metadata, issue_metadata=issue,
        git_objects=tuple(GitObject(kind, body) for _, (kind, body) in
                          sorted(objects.accessed.items())))
    info.update({"target_file": target_file, "target_qualname": functions[0],
                 "target_name": target["name"], "licence_text": licence_bytes.decode(
                     "utf-8", "replace")})
    return candidate, info


def store_candidate(store: ContentStore, candidate: CandidateBug) -> dict[str, str]:
    """Persist every piece of evidence; returns the content addresses."""
    for item in candidate.git_objects:
        store.put_git(item.kind, item.body)
    return {"repository_metadata": store.put_raw(candidate.repository_metadata.body),
            "issue_metadata": store.put_raw(candidate.issue_metadata.body),
            "sidecar": store.put_json(candidate_to_sidecar(candidate))}


def evaluate_candidate(candidate: CandidateBug, frozen: FrozenReferenceUniverse,
                       store: ContentStore, addresses: Mapping[str, str],
                       licence_text: str) -> dict[str, Any]:
    """Steps 1-7 for one candidate; the record is stored content-addressed."""
    record = check_candidate(candidate, frozen)
    sidecar = json.loads(store.get_raw(addresses["sidecar"]))
    revalidation = revalidate_isolation_record(record, sidecar, frozen)
    reloaded_equal = candidate_from_sidecar(sidecar) == candidate
    licence = licence_text_problems(candidate.licence_spdx, licence_text)
    same_organisation = record.get("same_organisation_as_excluded") or []
    reasons = list(record["reasons"])
    if not revalidation["valid"] or not reloaded_equal:
        reasons.append("sidecar_revalidation_failed")
    reasons += [f"licence:{problem}" for problem in licence]
    if same_organisation:
        reasons.append("same_organisation_as_indexed_repository")
    return {"record_address": store.put_json(record), "record_sha256": record["record_sha256"],
            "check_candidate_admissible": record["admissible"],
            "stages": record["stages"], "reasons": reasons, "admitted": not reasons,
            "revalidation_valid": bool(revalidation["valid"] and reloaded_equal),
            "licence_problems": licence, "same_organisation": same_organisation,
            "derived": record["derived_from_verified_evidence"],
            "patch_sha256": record["patch_sha256"],
            "insufficient_evidence": record["insufficient_evidence"]}


def within_set_duplicates(targets: list[tuple[str, str]]) -> dict[str, str]:
    """{later key: earlier key} for target functions that are near-duplicates."""
    kept: list[tuple[str, frozenset]] = []
    duplicates = {}
    for key, code in targets:
        mine = code_shingles(code)
        match = next((other for other, items in kept if jaccard(items, mine)
                      >= NEAR_DUPLICATE_JACCARD), None)
        if match:
            duplicates[key] = match
        else:
            kept.append((key, mine))
    return duplicates


def repository_screen(repository: str, value: Mapping[str, Any],
                      frozen: FrozenReferenceUniverse) -> list[str]:
    """Repository-level D2 and universe screens, before any commit is mined."""
    problems = []
    spdx = str((value.get("license") or {}).get("spdx_id") or "")
    if spdx not in ADMITTED_LICENCES:
        problems.append(f"licence_not_admitted_by_d2:{spdx or 'none'}")
    universe = frozen.universe
    full, name = canonical_repository(repository)
    if name in universe.repository_names or full in universe.repository_full:
        problems.append("repository_in_reference_universe")
    owner = full.split("/")[0]
    if any(existing.split("/")[0] == owner for existing in universe.repository_full):
        problems.append("same_organisation_as_indexed_repository")
    if value.get("archived"):
        problems.append("repository_archived")
    return problems
