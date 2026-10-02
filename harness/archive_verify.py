"""Portable, read-only verification of external byte-verified archives (stdlib only).

The archive root is ALWAYS supplied by the caller (``--archive-root``); no personal path is
compiled into tracked source. ``verify`` returns the resolved root for local display; tracked
receipts redact it (scripts/v26_archive_verify.py). Fails closed: a missing root, a missing
manifest, a missing archive directory or any byte difference is an error. Never creates, moves, renames or deletes anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable


class ArchiveError(SystemExit):
    """The archive cannot be verified (fail closed)."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_root(archive_root) -> Path:
    if archive_root is None or str(archive_root).strip() == "":
        raise ArchiveError("REFUSED: --archive-root is required")
    root = Path(archive_root).expanduser().resolve()
    if not root.is_dir():
        raise ArchiveError(f"REFUSED: archive root does not exist: {root}")
    return root


def verify(archive_root, manifests: Iterable[Path]) -> Dict:
    root = resolve_root(archive_root)
    out, problems, total, ok = {}, [], 0, 0
    for m in manifests:
        m = Path(m)
        if not m.is_file():
            raise ArchiveError(f"REFUSED: manifest missing: {m}")
        man = json.loads(m.read_text(encoding="utf-8"))
        base = root / man["archive_directory_name"]
        if not base.is_dir():
            raise ArchiveError(f"REFUSED: archive directory missing: {base}")
        good = 0
        for item in man["items"]:
            path = base / item["file"]
            if path.is_file() and _sha(path) == item["sha256"]:
                good += 1
            else:
                problems.append(f"{man['archive_directory_name']}/{item['file']}")
        out[man["archive_directory_name"]] = {"items": len(man["items"]), "byte_verified": good}
        total += len(man["items"])
        ok += good
    return {"archive_root": str(root), "archives": out, "items": total, "byte_verified": ok,
            "problems": problems, "passed": not problems and total > 0}
