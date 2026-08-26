"""Oneiros adapters for behaviorally verified SWE-bench repository records."""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from harness.bugsinpy_v2 import _changed_definition_context, _test_fragment
from harness.corpus import SCHEMA_VERSION, record_content_hash


def patch_paths(patch: str, python_only: bool = True) -> List[str]:
    """Return deterministic non-deleted target paths from a unified diff."""
    if not patch.strip():
        return []
    paths = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        path = line[4:].split("\t", 1)[0].strip()
        if path == "/dev/null":
            continue
        if path.startswith("b/"):
            path = path[2:]
        path = PurePosixPath(path).as_posix()
        if python_only and not path.endswith(".py"):
            continue
        paths.append(path)
    return sorted(set(paths))


def _compact_source_pair(buggy: str, fixed: str) -> Tuple[str, str]:
    """Use AST-changed definitions when safe, otherwise preserve full sources."""
    try:
        compacted = _changed_definition_context(buggy, fixed)
    except (SyntaxError, ValueError):
        compacted = None
    return compacted if compacted is not None else (buggy, fixed)


def compact_repository_context(
    buggy_sources: Dict[str, str], fixed_sources: Dict[str, str],
) -> Tuple[str, str, List[str]]:
    buggy_parts: List[str] = []
    fixed_parts: List[str] = []
    retained_paths: List[str] = []
    for path in sorted(set(buggy_sources) & set(fixed_sources)):
        buggy = buggy_sources[path]
        fixed = fixed_sources[path]
        if not buggy or not fixed or buggy == fixed:
            continue
        try:
            ast.parse(buggy)
            ast.parse(fixed)
        except SyntaxError:
            continue
        compact_buggy, compact_fixed = _compact_source_pair(buggy, fixed)
        header = f"# File: {path}\n"
        buggy_parts.append(header + compact_buggy.rstrip() + "\n")
        fixed_parts.append(header + compact_fixed.rstrip() + "\n")
        retained_paths.append(path)
    return "\n".join(buggy_parts), "\n".join(fixed_parts), retained_paths


def selector_target(selector: str) -> Optional[Tuple[str, List[str]]]:
    """Parse a pytest-style FAIL_TO_PASS selector into a file and AST names."""
    parts = selector.split("::")
    if len(parts) < 2 or not parts[0].endswith(".py"):
        return None
    names = [re.sub(r"\[.*\]$", "", value) for value in parts[1:] if value]
    if not names:
        return None
    return PurePosixPath(parts[0]).as_posix(), names


def extract_test_fragments(
    base_test_sources: Dict[str, str], patched_test_sources: Dict[str, str],
    fail_to_pass: Iterable[str],
) -> List[str]:
    """Extract complete F2P tests, with changed-definition fallback."""
    fragments: List[str] = []
    for selector in fail_to_pass:
        target = selector_target(selector)
        if target is None:
            continue
        path, names = target
        source = patched_test_sources.get(path)
        if not source:
            continue
        fragment = _test_fragment(source, names)
        if fragment:
            fragments.append(fragment)

    if not fragments:
        for path in sorted(set(base_test_sources) & set(patched_test_sources)):
            before = base_test_sources[path]
            after = patched_test_sources[path]
            if before == after:
                continue
            try:
                changed = _changed_definition_context(before, after)
            except (SyntaxError, ValueError):
                changed = None
            if changed is not None:
                fragments.append(changed[1])

    unique: List[str] = []
    seen = set()
    for fragment in fragments:
        normalized = fragment.strip() + "\n"
        try:
            compile(normalized, "<swebench-test-fragment>", "exec")
        except SyntaxError:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(normalized)
    return unique


def build_repository_record(
    instance: Dict[str, Any], buggy_sources: Dict[str, str],
    fixed_sources: Dict[str, str], base_test_sources: Dict[str, str],
    patched_test_sources: Dict[str, str], verification: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    buggy_code, fixed_code, source_paths = compact_repository_context(
        buggy_sources, fixed_sources,
    )
    if not buggy_code or not fixed_code:
        return None
    fail_to_pass = instance.get("FAIL_TO_PASS") or []
    if isinstance(fail_to_pass, str):
        import json
        fail_to_pass = json.loads(fail_to_pass)
    pass_to_pass = instance.get("PASS_TO_PASS") or []
    if isinstance(pass_to_pass, str):
        import json
        pass_to_pass = json.loads(pass_to_pass)
    fragments = extract_test_fragments(
        base_test_sources, patched_test_sources, fail_to_pass,
    )
    if not fragments:
        return None
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    selector_digest = hashlib.sha256(
        "\n".join(sorted(fail_to_pass)).encode("utf-8")
    ).hexdigest()[:12]
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": f"official-repository::swebench-verified::{instance_id}::{selector_digest}",
        "task_type": "official_repository_swebench_reproduction",
        "language": "python",
        "source": {
            "name": "swebench_verified_official_repository_fragment",
            "upstream": "SWE-bench Verified",
        },
        "group_id": f"project:swebench:{repo.lower()}",
        "code_under_test": buggy_code,
        "reference_code": fixed_code,
        "entry_point": "",
        "specification": instance.get("problem_statement", ""),
        "tests": [{
            "code": fragment,
            "oracle": "fixed_passes_buggy_fails_repository",
            "format": "pytest_fragment",
        } for fragment in fragments],
        "provenance": {
            "official_task_id": instance_id,
            # Use the canonical repository slug for deterministic project-
            # disjoint splitting across BugsInPy and SWE-bench. For example,
            # matplotlib/matplotlib must share the same split as the existing
            # BugsInPy matplotlib family.
            "project": repo.rsplit("/", 1)[-1].lower(),
            "repository": repo,
            "base_commit": instance["base_commit"],
            "environment_setup_commit": instance.get("environment_setup_commit", ""),
            "version": instance.get("version", ""),
            "difficulty": instance.get("difficulty", ""),
            "patched_source_paths": source_paths,
            "test_paths": sorted(patched_test_sources),
            "fail_to_pass": list(fail_to_pass),
            "pass_to_pass": list(pass_to_pass),
            "gold_patch_sha256": hashlib.sha256(
                instance.get("patch", "").encode("utf-8")
            ).hexdigest(),
            "test_patch_sha256": hashlib.sha256(
                instance.get("test_patch", "").encode("utf-8")
            ).hexdigest(),
            "official_test_evidence": verification,
        },
        "quality": {
            "pair_behaviorally_verified": True,
            "official_targeted_test_fixed_pass_buggy_fail": True,
            "execution_mode": "repository_pytest_fragment",
            "oracle": "repository_fixed_vs_buggy",
            "test_count": len(fragments),
        },
    }
    record["content_hash"] = record_content_hash(record)
    return record
