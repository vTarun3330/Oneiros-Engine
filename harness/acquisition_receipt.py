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

Protected-access evidence comes from a Python audit hook (``sys.addaudithook``)
that checks every Python-process file open, and every subprocess command,
working directory and path argument, against RESOLVED protected locations
(``ProtectedLocations``): canonical corpus records, split assignments, every
non-train development shard, sealed-final artifacts and the reserved
confirmation IDs.  The hook cannot see opens made inside external subprocesses;
those commands are recorded, and git's local paths are confined to the store.
Schema v1 used substring patterns that both over-matched (a store named
'...confirmation...') and missed val.records.json; v2 replaces them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping

from harness.source_identity import canonical_sha256

SCHEMA = "oneiros_acquisition_receipt_v2"
PROTECTION_POLICY = "oneiros_protected_locations_v2"
REQUIRED_IDENTITY = ("git_commit", "source_tree_sha256", "dirty_tree", "bundle_generation",
                     "bundle_manifest_sha256", "reference_universe_receipt_sha256",
                     "isolation_version", "diff_policy_id", "candidate_repository_list_sha256",
                     "tool_source_sha256", "journal_sha256", "store_verification", "command",
                     "configuration", "start_utc", "end_utc", "api")
EXCLUDED_TREE_PREFIXES = ("results/", "data/", "runs/", "checkpoints/", "logs/")
PROTECTED_LOCATIONS = {
    "canonical_corpus_records": "data/corpus/<version>/records.json",
    "corpus_split_assignment": "data/corpus/<version>/splits.json (conservative)",
    "non_train_development_shard": "data/corpus/<version>/development_view/<split>.records.json "
                                   "for every split except train (val, ablation_dev, ...)",
    "sealed_final": "results/sealed_final* (files, and everything under such directories)",
    "reserved_confirmation": "any file named unopened_confirmation.ids.json",
}


def _norm(path: Any) -> Path | None:
    try:
        text = os.fsdecode(path)
    except TypeError:
        return None
    try:
        resolved = Path(text).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        resolved = Path(os.path.abspath(text))
    return Path(os.path.normcase(str(resolved)))


class ProtectedLocations:
    """Resolved, case-normalised predicate over the ACTUAL protected locations.

    Paths are compared as resolved absolute paths relative to the repository
    root - never by substring - so a harmless name containing words such as
    'confirmation' or 'sealed' is not flagged.
    """

    def __init__(self, root: Path):
        self.root = _norm(root)

    def reason(self, path: Any) -> str | None:
        resolved = _norm(path)
        if resolved is None or self.root is None:
            return None
        try:
            parts = resolved.relative_to(self.root).parts
        except ValueError:
            return None
        if not parts:
            return None
        name = parts[-1]
        if len(parts) >= 2 and parts[0] == "data" and parts[1] == "corpus":
            if len(parts) == 4 and name == "records.json":
                return "canonical_corpus_records"
            if len(parts) == 4 and name == "splits.json":
                return "corpus_split_assignment"
            if len(parts) == 5 and parts[3] == "development_view" and \
                    name.endswith(".records.json") and name != "train.records.json":
                return "non_train_development_shard"
        if len(parts) >= 2 and parts[0] == "results" and parts[1].startswith("sealed_final"):
            return "sealed_final"
        if name == "unopened_confirmation.ids.json":
            return "reserved_confirmation"
        return None


class AuditScope:
    """One process's audit window: from before configuration loading until
    immediately before publication validation."""

    def __init__(self, monitor: type["ProtectedAccessMonitor"]):
        self.monitor = monitor
        self.start_utc = _utc()
        self.events_from = len(monitor.events)
        self.opens_from = monitor.opens_checked
        self.subprocesses_from = len(monitor.subprocesses)
        self.end_utc: str | None = None

    def snapshot(self) -> dict[str, Any]:
        events = self.monitor.events[self.events_from:]
        return {"scope_start_utc": self.start_utc,
                "opens_checked": self.monitor.opens_checked - self.opens_from,
                "subprocesses": len(self.monitor.subprocesses) - self.subprocesses_from,
                "protected_accesses": [dict(item) for item in events]}

    def close(self) -> dict[str, Any]:
        self.end_utc = _utc()
        return {**self.snapshot(), "scope_end_utc": self.end_utc,
                "installed": self.monitor._installed,
                "protection_policy": PROTECTION_POLICY,
                "protected_locations": PROTECTED_LOCATIONS,
                "method": ("sys.addaudithook: every Python-process file open ('open' events) and "
                           "every subprocess launch ('subprocess.Popen' events: executable, "
                           "arguments, working directory) is checked against the resolved "
                           "protected locations; opens performed inside external subprocesses "
                           "are NOT observed, so subprocess commands and their path arguments "
                           "are recorded and git's local paths are confined to the store"),
                "subprocess_commands": [dict(item) for item in
                                        self.monitor.subprocesses[self.subprocesses_from:]]}


class ProtectedAccessMonitor:
    """Process-wide audit hook: protected opens and every subprocess launch."""
    _installed = False
    locations: ProtectedLocations | None = None
    events: list[dict[str, str]] = []
    subprocesses: list[dict[str, Any]] = []
    opens_checked = 0

    @classmethod
    def configure(cls, root: Path) -> None:
        cls.locations = ProtectedLocations(root)

    @classmethod
    def install(cls, root: Path | None = None) -> None:
        if root is not None:
            cls.configure(root)
        if cls._installed:
            return

        def hook(event: str, args: tuple) -> None:
            if event == "open" and args:
                cls.opens_checked += 1
                if cls.locations is not None and isinstance(args[0], (str, bytes, os.PathLike)):
                    reason = cls.locations.reason(args[0])
                    if reason:
                        cls.events.append({"kind": "open", "reason": reason,
                                           "path": os.fsdecode(args[0])})
            elif event == "subprocess.Popen" and len(args) >= 4:
                executable, arguments, cwd = args[0], args[1], args[2]
                if isinstance(arguments, (list, tuple)):
                    argv = [os.fsdecode(item) if isinstance(item, (str, bytes, os.PathLike))
                            else str(item) for item in arguments]
                else:
                    # Windows passes one command-line string: split it, keeping quoted
                    # arguments whole, so every path argument is checked.
                    text = os.fsdecode(arguments)
                    try:
                        argv = [item.strip('"') for item in shlex.split(text, posix=False)]
                    except ValueError:
                        argv = text.split()
                record = {"executable": os.fsdecode(executable) if executable else None,
                          "argv": argv, "cwd": os.fsdecode(cwd) if cwd else os.getcwd()}
                cls.subprocesses.append(record)
                if cls.locations is not None:
                    for item in [record["cwd"], *argv]:
                        reason = cls.locations.reason(item) if item else None
                        if reason:
                            cls.events.append({"kind": "subprocess_argument",
                                               "reason": reason, "path": item})

        sys.addaudithook(hook)
        cls._installed = True

    @classmethod
    def scope(cls) -> AuditScope:
        return AuditScope(cls)

    @classmethod
    def mark(cls) -> tuple[int, int]:
        return len(cls.events), cls.opens_checked

    @classmethod
    def evidence(cls, since: tuple[int, int] = (0, 0)) -> dict[str, Any]:
        return {"method": "python audit hook", "installed": cls._installed,
                "opens_checked": cls.opens_checked - since[1],
                "protected_paths_opened": sorted({e["path"] for e in cls.events[since[0]:]})}


def confine_to_store(path: Path, store_root: Path) -> Path:
    """A local path handed to a subprocess must lie inside the acquisition store
    and must not be a protected location."""
    resolved, root = _norm(path), _norm(store_root)
    if resolved is None or root is None:
        raise ValueError(f"unresolvable subprocess path {path}")
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"subprocess path {path} lies outside the acquisition store") from None
    locations = ProtectedAccessMonitor.locations
    if locations is not None and locations.reason(path):
        raise ValueError(f"subprocess path {path} resolves into a protected location")
    return Path(path)


def _utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    else:
        if not evidence.get("complete"):
            problems.append("protected-access evidence is incomplete (an audit session never "
                            "closed)")
        if not evidence.get("scope_start_utc") or not evidence.get("scope_end_utc"):
            problems.append("protected-access evidence has no audit scope start/end")
        if bool(evidence.get("protected_accesses")) != bool(events.get("protected_data_access")):
            problems.append("protected-access evidence contradicts the event")
    for name, value in gate.items():
        if not isinstance(value, bool):
            problems.append(f"gate.{name} must be a boolean")
    return problems


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_sources(root: Path, paths: Iterable[str]) -> dict[str, str]:
    return {path: canonical_sha256(root / path) for path in sorted(paths)}
