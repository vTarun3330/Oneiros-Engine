"""Portable, self-validating evidence bundles for Git-ignored local evidence.

Raw run evidence (native rehearsal records, GPU-smoke receipts, durable-run status
files) lives under ignored paths and exists only on the GPU machine. A bundle embeds
the scientifically necessary files as text inside ONE tracked JSON file, so another
clone can check a number without the originals.

For each embedded file the bundle records:

* ``original_sha256``: SHA-256 of the ignored original's bytes (checked at build time,
  and again whenever the original is present);
* ``text`` and ``text_sha256``: the published text and its hash;
* ``normalizations``: the LABELS of the deterministic rewrites applied (never the
  private strings themselves), with counts. An empty list means ``text`` is the
  original byte-for-byte, so ``text_sha256 == original_sha256``.

Normalisations only replace machine-private locations (the local repository root,
the WSL host name). They never touch hashes, counts, timings or categories.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

SCHEMA_VERSION = "oneiros_portable_evidence_bundle_v1"
_HOST_RE = re.compile(r'("host": "Linux )(\S+)( )')


def _root_forms(root: Path) -> List[str]:
    plain = str(root)
    forms = [json.dumps(plain)[1:-1], plain, root.as_posix()]
    return sorted(set(forms), key=len, reverse=True)


def normalizer(root: Path) -> Dict[str, Callable[[str], Tuple[str, int]]]:
    """Label -> rewrite. Labels are what a bundle records; the strings stay local."""
    def repository_root(text: str) -> Tuple[str, int]:
        count = 0
        for form in _root_forms(root):
            count += text.count(form)
            text = text.replace(form, "<repo>")
        return text, count

    def wsl_host(text: str) -> Tuple[str, int]:
        return _HOST_RE.subn(r"\1<host>\3", text)

    return {"repository_root": repository_root, "wsl_host": wsl_host}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(root: Path, relative: str, labels: List[str]) -> dict:
    raw = (root / relative).read_bytes()
    text = raw.decode("utf-8")   # CR bytes survive: JSON escapes them inside the string
    rules = normalizer(root)
    applied = []
    for label in labels:
        text, count = rules[label](text)
        applied.append({"label": label, "replacements": count})
    return {"original_path": relative, "original_sha256": hashlib.sha256(raw).hexdigest(),
            "original_bytes": len(raw), "normalizations": applied,
            "text": text, "text_sha256": sha256_text(text)}


def file_problems(root: Path, name: str, entry: Mapping[str, object],
                  private_patterns: List[str]) -> List[str]:
    """Checks that hold on any clone, plus the original when it is present."""
    problems = []
    text = str(entry["text"])
    if sha256_text(text) != entry["text_sha256"]:
        problems.append(f"{name}: embedded text does not match its hash")
    if not entry["normalizations"] or not any(n["replacements"] for n in entry["normalizations"]):
        if entry["text_sha256"] != entry["original_sha256"]:
            problems.append(f"{name}: un-normalised text differs from the original hash")
    for pattern in private_patterns:
        if re.search(pattern, text):
            problems.append(f"{name}: machine-private or secret pattern {pattern!r} present")
    original = root / str(entry["original_path"])
    if original.exists():
        raw = original.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["original_sha256"]:
            problems.append(f"{name}: local original no longer matches the recorded hash")
        else:
            rebuilt = embed(root, str(entry["original_path"]),
                            [n["label"] for n in entry["normalizations"]])
            if rebuilt["text"] != text:
                problems.append(f"{name}: normalising the local original does not reproduce "
                                "the embedded text")
    return problems


#: Scanned in every embedded text. Generic, so the scan itself leaks nothing.
PRIVATE_PATTERNS = [
    r"[A-Za-z]:\\\\Users\\\\",            # JSON-escaped Windows user profile path
    r"[A-Za-z]:\\Users\\",                # raw Windows user profile path
    r"/home/[A-Za-z0-9_.-]+/",
    r"/mnt/[a-z]/Users/",
    r"DESKTOP-[A-Z0-9]{5,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"(?i)authorization:\s*(bearer|token)\s+\S+",
    r"(?i)(api[_-]?key|secret|password|hf_token|github_token)\"?\s*[:=]\s*\"?[A-Za-z0-9_\-]{12,}",
    r"hf_[A-Za-z0-9]{30,}",
]


def bundle_problems(root: Path, bundle: Mapping[str, object]) -> List[str]:
    problems = []
    if bundle.get("schema_version") != SCHEMA_VERSION:
        problems.append("bundle schema is not recognised")
    for name, entry in dict(bundle.get("files", {})).items():
        problems += file_problems(root, name, entry, PRIVATE_PATTERNS)
    return problems


def load_json_text(entry: Mapping[str, object]):
    return json.loads(str(entry["text"]))


def load_jsonl_text(entry: Mapping[str, object]) -> List[dict]:
    return [json.loads(line) for line in str(entry["text"]).splitlines() if line.strip()]


def git_canonical_sha256(root: Path, commit: str, relative: str) -> Optional[str]:
    from harness.historical_source_provenance import blob_canonical_sha256
    return blob_canonical_sha256(root, commit, relative)
