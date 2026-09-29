"""Verify historical source bindings against the commit that froze them.

A historical artifact records ``canonical_sha256`` values for the sources that
built it. Those values are a claim about the code *at the time*, so the honest
place to check them is the frozen commit, not today's working tree. Checking the
working tree conflates two separate questions:

1. Was the historical artifact built from the source it says it was?
   Answered here by hashing the LF-canonical Git blob at the frozen commit.
2. Is the historical artifact safe to reuse for a new run today?
   Answered by the artifact's own launch guard, which must refuse when the
   current source differs. Nothing here relaxes that guard.

Current-source drift is not hidden. It is accepted as *historical* only when a
versioned drift receipt names every drifted file with its current hash, so a
further unrecorded change still fails. Nothing in this module rewrites a file.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from harness.source_identity import canonical_bytes, canonical_sha256

DRIFT_SCHEMA_VERSION = "oneiros_historical_source_drift_v1"
_VERSION_RE = re.compile(r"_v(\d+)\.json$")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)


def resolve_commit(root: Path, commit: str) -> str:
    """Full SHA of a real commit object, or ValueError."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError(f"not a commit id: {commit!r}")
    result = _git(root, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if result.returncode != 0:
        raise ValueError(f"commit {commit} does not exist in this repository")
    return result.stdout.decode("ascii").strip()


def blob_canonical_sha256(root: Path, commit: str, relative: str) -> Optional[str]:
    """LF-canonical SHA-256 of ``relative`` as stored at ``commit``; None if absent."""
    result = _git(root, "show", f"{commit}:{relative}")
    if result.returncode != 0:
        return None
    return hashlib.sha256(canonical_bytes(result.stdout)).hexdigest()


def verify_at_commit(root: Path, commit: str,
                     bindings: Mapping[str, str]) -> Dict[str, Dict[str, object]]:
    """Per bound file: the recorded hash, the hash at ``commit`` and whether they match."""
    full = resolve_commit(root, commit)
    report = {}
    for relative, expected in sorted(bindings.items()):
        at_commit = blob_canonical_sha256(root, full, relative)
        report[relative] = {"recorded": expected, "at_commit": at_commit,
                            "matches": at_commit == expected}
    return report


def current_drift(root: Path, bindings: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """Bound files whose current canonical hash differs, mapped to that hash."""
    drift = {}
    for relative, expected in sorted(bindings.items()):
        path = root / relative
        current = canonical_sha256(path) if path.exists() else None
        if current != expected:
            drift[relative] = current
    return drift


def changing_commits(root: Path, commit: str, relative: str) -> List[str]:
    """Commits after ``commit`` up to HEAD that touched ``relative``, oldest first."""
    result = _git(root, "log", "--format=%H", "--reverse", f"{commit}..HEAD", "--", relative)
    return result.stdout.decode("ascii").split()


def latest_drift_receipt(directory: Path, stem: str) -> Optional[Path]:
    """Highest-versioned ``<stem>_v<N>.json``; successors never overwrite predecessors."""
    candidates = [(int(m.group(1)), path) for path in directory.glob(f"{stem}_v*.json")
                  if (m := _VERSION_RE.search(path.name))]
    return max(candidates)[1] if candidates else None


def drift_problems(root: Path, receipt: Mapping[str, object],
                   artifacts: Iterable[Mapping[str, object]]) -> List[str]:
    """Everything that stops the historical bindings from being accepted.

    ``artifacts`` are ``{"path", "sha256", "bindings"}``: each historical file,
    its expected byte hash, and the source hashes it records. The drift receipt
    must match those artifacts byte-for-byte, every binding must reproduce at
    the receipt's frozen commit, and every current mismatch must be recorded
    with its exact current hash.
    """
    problems: List[str] = []
    if receipt.get("schema_version") != DRIFT_SCHEMA_VERSION:
        problems.append("drift receipt schema is not recognised")
    commit = str(receipt.get("frozen_source_commit", ""))
    try:
        commit = resolve_commit(root, commit)
    except ValueError as error:
        return problems + [str(error)]
    recorded_drift = {entry["path"]: entry["current_canonical_sha256"]
                      for entry in receipt.get("current_source_drift", [])}
    recorded_artifacts = {entry["path"]: entry["sha256"]
                          for entry in receipt.get("historical_artifacts", [])}
    for artifact in artifacts:
        path = str(artifact["path"])
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if actual != artifact["sha256"] or recorded_artifacts.get(path) != actual:
            problems.append(f"historical artifact bytes differ from the receipt: {path}")
        for relative, report in verify_at_commit(root, commit, artifact["bindings"]).items():
            if not report["matches"]:
                problems.append(f"{path}: {relative} does not reproduce at {commit[:12]}")
        for relative, current in current_drift(root, artifact["bindings"]).items():
            if relative not in recorded_drift:
                problems.append(f"unrecorded source drift: {relative}")
            elif recorded_drift[relative] != current:
                problems.append(f"source drifted beyond the recorded hash: {relative}")
    return problems


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
