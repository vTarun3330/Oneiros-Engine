"""Receipts for acquisition reports (schema v1, used by every report after the
failed policy-A pilot; that historical report is left untouched).

A receipt binds a report to everything needed to reproduce and trust it: git
commit, LF-canonical source-tree identity, dirty-tree status, the design bundle
generation and manifest hash, the frozen reference-universe receipt hash, the
isolation version and diff-policy ID, the candidate-repository list hash, the
acquisition-tool source hashes, the journal hash, content-store verification,
the exact command and configuration, start/end times, API usage, and
protected-data-access EVIDENCE.

Event versus gate: ``protected_data_access`` is the observed EVENT and must be
false; ``no_protected_data_access`` is the pass/fail GATE and must be true.
``validate_receipt`` rejects any contradiction between them, and publication is
refused when the frozen-universe or source identity is absent.

Protected-access evidence comes from a Python audit hook that records every
file open whose path matches a protected pattern (validation, ablation-dev,
consumed test, confirmation, sealed-final data and the canonical corpus
records.json).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping

from harness.source_identity import canonical_sha256

SCHEMA = "oneiros_acquisition_receipt_v1"
PROTECTED_PATTERNS = (
    r"development_view[/\\][^/\\]*(validation|ablation[_-]?dev|test|confirmation)[^/\\]*$",
    r"sealed",
    r"confirmation",
    r"(^|[/\\])records\.json$",
    r"[/\\](validation|test|ablation_dev)\.jsonl?$",
)
REQUIRED_IDENTITY = ("git_commit", "source_tree_sha256", "dirty_tree", "bundle_generation",
                     "bundle_manifest_sha256", "reference_universe_receipt_sha256",
                     "isolation_version", "diff_policy_id", "candidate_repository_list_sha256",
                     "tool_source_sha256", "journal_sha256", "store_verification", "command",
                     "configuration", "start_utc", "end_utc", "api")
EXCLUDED_TREE_PREFIXES = ("results/", "data/", "runs/", "checkpoints/", "logs/")


class ProtectedAccessMonitor:
    """Records opens of protected paths for the rest of the process (audit hook)."""
    _installed = False
    events: list[str] = []
    opens_checked = 0

    @classmethod
    def install(cls) -> None:
        if cls._installed:
            return
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in PROTECTED_PATTERNS]

        def hook(event: str, args: tuple) -> None:
            if event != "open" or not args:
                return
            path = args[0]
            if not isinstance(path, (str, bytes)):
                return
            text = path.decode("utf-8", "replace") if isinstance(path, bytes) else path
            cls.opens_checked += 1
            if any(pattern.search(text) for pattern in patterns):
                cls.events.append(text)

        sys.addaudithook(hook)
        cls._installed = True

    @classmethod
    def mark(cls) -> tuple[int, int]:
        """A position to report evidence from (the hook is process-wide)."""
        return len(cls.events), cls.opens_checked

    @classmethod
    def evidence(cls, since: tuple[int, int] = (0, 0)) -> dict[str, Any]:
        return {"method": "python audit hook on every file open, active for the whole run",
                "installed": cls._installed, "opens_checked": cls.opens_checked - since[1],
                "protected_patterns": list(PROTECTED_PATTERNS),
                "protected_paths_opened": sorted(set(cls.events[since[0]:]))}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout


def source_tree_identity(root: Path) -> dict[str, Any]:
    """Git commit, dirty status and an LF-canonical hash of every tracked source file."""
    files = [path for path in _git(root, "ls-files", "-z").split("\0")
             if path and not path.startswith(EXCLUDED_TREE_PREFIXES)]
    digests = {path: canonical_sha256(root / path) for path in sorted(files)
               if (root / path).is_file()}
    status = _git(root, "status", "--porcelain")
    dirty_paths = [line[3:] for line in status.splitlines() if line.strip()]
    return {"git_commit": _git(root, "rev-parse", "HEAD").strip(),
            "source_tree_sha256": hashlib.sha256(json.dumps(
                digests, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "source_tree_files": len(digests),
            "dirty_tree": bool(dirty_paths),
            "dirty_paths": dirty_paths[:50]}


def validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Every contradiction or missing identity (empty = publishable)."""
    problems = []
    if receipt.get("schema_version") != SCHEMA:
        problems.append("receipt schema_version is not " + SCHEMA)
    identity = receipt.get("identity") or {}
    for key in REQUIRED_IDENTITY:
        if identity.get(key) in (None, "", [], {}) and key != "dirty_tree":
            problems.append(f"identity.{key} is absent")
    if not isinstance(identity.get("dirty_tree"), bool):
        problems.append("identity.dirty_tree must be a boolean")
    for key in ("reference_universe_receipt_sha256", "source_tree_sha256",
                "candidate_repository_list_sha256", "journal_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get(key) or "")):
            problems.append(f"identity.{key} is not a SHA-256")
    events = receipt.get("events") or {}
    gate = receipt.get("gate") or {}
    if events.get("protected_data_access") is not False:
        problems.append("events.protected_data_access must be the observed event and false")
    if events.get("model_called") is not False:
        problems.append("events.model_called must be false")
    if events.get("evaluation_set_created") is not False:
        problems.append("events.evaluation_set_created must be false")
    if "protected_data_access" in gate:
        problems.append("gate must not carry an event field named protected_data_access")
    if gate.get("no_protected_data_access") is not (events.get("protected_data_access") is False):
        problems.append("gate.no_protected_data_access contradicts the observed event")
    evidence = receipt.get("protected_access_evidence") or {}
    if not evidence.get("installed"):
        problems.append("protected-access evidence was not collected")
    elif bool(evidence.get("protected_paths_opened")) != bool(events.get("protected_data_access")):
        problems.append("protected-access evidence contradicts the event")
    for name, value in gate.items():
        if not isinstance(value, bool):
            problems.append(f"gate.{name} must be a boolean")
    return problems


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_sources(root: Path, paths: Iterable[str]) -> dict[str, str]:
    return {path: canonical_sha256(root / path) for path in sorted(paths)}
