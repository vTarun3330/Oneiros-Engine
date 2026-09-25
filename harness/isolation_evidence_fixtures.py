"""Synthetic, internally consistent candidate evidence for tests and self-checks.

Everything here is fabricated offline: GitHub API responses as JSON bytes and
git loose objects (blobs, trees, commits) built in memory.  It exists to prove
that ``harness.repository_isolation.authentication_problems`` cross-checks the
evidence rather than its format.  It is never a source of real candidates; a
real candidate's evidence comes from the later, separately approved
acquisition step.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from harness.repository_isolation import (
    API, ApiResponse, CandidateBug, GitObject, canonical_diff, git_object_id,
)

LICENCE_TEXT = b"MIT License\n\nCopyright (c) 2025 Example\n"
COMMIT_EPOCH = 1740787200  # 2025-03-01T00:00:00Z
RETRIEVED = "2026-09-25T00:00:00Z"


def _response(url: str, value: dict[str, Any]) -> ApiResponse:
    body = json.dumps(value, sort_keys=True).encode("utf-8")
    return ApiResponse(url=url, body=body, sha256=hashlib.sha256(body).hexdigest(),
                       retrieved_utc=RETRIEVED, etag='"synthetic"')


def _tree(entries: dict[str, tuple[str, str]]) -> bytes:
    """Git tree body; entries {name: (mode, oid)} sorted the way git sorts them."""
    def key(name):
        return name + "/" if entries[name][0] == "40000" else name
    return b"".join(f"{entries[name][0]} {name}".encode("utf-8") + b"\0"
                    + bytes.fromhex(entries[name][1]) for name in sorted(entries, key=key))


def _tree_for(files: dict[str, bytes], objects: list[GitObject]) -> str:
    """Build nested trees for {path: content}; returns the root tree ID."""
    direct: dict[str, tuple[str, str]] = {}
    nested: dict[str, dict[str, bytes]] = {}
    for path, content in files.items():
        head, _, rest = path.partition("/")
        if rest:
            nested.setdefault(head, {})[rest] = content
        else:
            objects.append(GitObject("blob", content))
            direct[head] = ("100644", git_object_id("blob", content))
    for name, children in nested.items():
        direct[name] = ("40000", _tree_for(children, objects))
    body = _tree(direct)
    objects.append(GitObject("tree", body))
    return git_object_id("tree", body)


def _commit(tree: str, parents: list[str], message: str, epoch: int | None,
            author_epoch: int | None = None) -> bytes:
    author = epoch if author_epoch is None else author_epoch
    def stamp(value):
        return "" if value is None else f" {value} +0000"
    lines = [f"tree {tree}", *[f"parent {parent}" for parent in parents],
             f"author Example <dev@example.invalid>{stamp(author)}",
             f"committer Example <dev@example.invalid>{stamp(epoch)}", "", message, ""]
    return "\n".join(lines).encode("utf-8")


def files_from_patch(patch: str, extra: str = "") -> tuple[str, str, str]:
    """(target path, buggy text, fixed text) reconstructed from a one-file diff."""
    path = ""
    buggy: list[str] = []
    fixed: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].split("\t")[0].strip()
            path = path[2:] if path.startswith("b/") else path
        elif line.startswith(("--- ", "diff ", "index ", "@@")):
            continue
        elif line.startswith("-"):
            buggy.append(line[1:])
        elif line.startswith("+"):
            fixed.append(line[1:])
        else:
            buggy.append(line[1:] if line.startswith(" ") else line)
            fixed.append(line[1:] if line.startswith(" ") else line)
    # The hunk is an indented fragment, so it is kept verbatim inside a string
    # literal: the file stays parseable and every patch line is present in it.
    def wrap(lines: list[str]) -> str:
        return f'{extra}\n\n_PATCH_CONTEXT = r"""\n' + "\n".join(lines) + '\n"""\n'
    return path, wrap(buggy), wrap(fixed)


def synthetic_candidate(*, repository: str = "example-org/ranges", buggy_text: str,
                        fixed_text: str, target_function: str, target_file: str,
                        target_module: str, patch: str | None = None,
                        fork_parent: str | None = None, issue_number: int = 1,
                        repository_id: int = 123456,
                        extra_files: dict[str, tuple[str, str]] | None = None,
                        committer_epoch: int | None = COMMIT_EPOCH,
                        author_epoch: int | None = None,
                        buggy_epoch: int | None = COMMIT_EPOCH - 3600,
                        **overrides: Any) -> CandidateBug:
    """A candidate whose every piece of evidence is mutually consistent.

    ``extra_files`` ({path: (buggy text, fixed text)}) adds further files to both
    revisions; the commit timestamps can be set to exercise the temporal rule.
    The default patch is the canonical diff of the target file.
    """
    objects: list[GitObject] = []
    buggy_bytes, fixed_bytes = buggy_text.encode("utf-8"), fixed_text.encode("utf-8")
    extra = extra_files or {}
    buggy_tree = _tree_for({"LICENSE": LICENCE_TEXT, target_file: buggy_bytes,
                            **{p: b.encode("utf-8") for p, (b, _) in extra.items()}}, objects)
    fixed_tree = _tree_for({"LICENSE": LICENCE_TEXT, target_file: fixed_bytes,
                            **{p: f.encode("utf-8") for p, (_, f) in extra.items()}}, objects)
    buggy_body = _commit(buggy_tree, [], "buggy", buggy_epoch)
    buggy_commit = git_object_id("commit", buggy_body)
    fixed_body = _commit(fixed_tree, [buggy_commit], f"fix #{issue_number}", committer_epoch,
                         author_epoch)
    objects += [GitObject("commit", buggy_body), GitObject("commit", fixed_body)]
    if patch is None:
        patch = canonical_diff(target_file, buggy_text, fixed_text)
    node = "R_" + re.sub(r"[^A-Za-z0-9]", "", repository) + "node"
    metadata = {"full_name": repository, "id": repository_id, "node_id": node,
                "html_url": f"https://github.com/{repository}", "fork": fork_parent is not None,
                "license": {"spdx_id": "MIT"}, "url": f"{API}{repository}"}
    parent_fields: dict[str, Any] = {}
    if fork_parent is not None:
        parent_node = "R_" + re.sub(r"[^A-Za-z0-9]", "", fork_parent) + "node"
        metadata["parent"] = {"full_name": fork_parent, "id": 7, "node_id": parent_node}
        parent_fields = {"fork_parent": fork_parent, "fork_parent_id": "7",
                         "fork_parent_node_id": parent_node}
    issue_url = f"{API}{repository}/issues/{issue_number}"
    values: dict[str, Any] = dict(
        repository=repository, repository_url=f"https://github.com/{repository}",
        repository_id=str(repository_id), repository_node_id=node,
        fork_status="fork" if fork_parent is not None else "not_fork", **parent_fields,
        buggy_commit=buggy_commit, fixed_commit=git_object_id("commit", fixed_body),
        patch=patch, target_function=target_function, target_file=target_file,
        target_module=target_module,
        target_blob_buggy=git_object_id("blob", buggy_bytes),
        target_blob_fixed=git_object_id("blob", fixed_bytes),
        issue_id=f"{repository}#{issue_number}", licence_spdx="MIT", licence_path="LICENSE",
        licence_blob=git_object_id("blob", LICENCE_TEXT),
        licence_sha256=hashlib.sha256(LICENCE_TEXT).hexdigest(),
        repository_metadata=_response(f"{API}{repository}", metadata),
        issue_metadata=_response(issue_url, {
            "url": issue_url, "number": issue_number,
            "repository_url": f"{API}{repository}"}),
        git_objects=tuple(objects))
    values.update(overrides)
    return CandidateBug(**values)


def reissue_response(response: ApiResponse, **changes: Any) -> ApiResponse:
    """A response whose JSON body changed and whose hash was recomputed to match.

    Used to prove that authentication checks content, not just the hash."""
    value = json.loads(response.body.decode("utf-8"))
    value.update(changes)
    body = json.dumps(value, sort_keys=True).encode("utf-8")
    return ApiResponse(url=response.url, body=body, sha256=hashlib.sha256(body).hexdigest(),
                       retrieved_utc=response.retrieved_utc, etag=response.etag)
