"""Portable archive verification: required root, spaces, fail-closed, no personal paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from harness import archive_verify as av

ROOT = Path(__file__).resolve().parent.parent


def _archive(root: Path, name="arc one") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "a.txt").write_bytes(b"hello")
    man = root.parent / f"{name}.json"
    man.write_text(json.dumps({"archive_directory_name": name, "items": [
        {"file": "a.txt", "sha256": hashlib.sha256(b"hello").hexdigest()}]}))
    return man


def test_valid_root_with_spaces(tmp_path):
    root = tmp_path / "my archive root"
    man = _archive(root)
    out = av.verify(str(root), [man])
    assert out["passed"] and out["byte_verified"] == 1
    assert out["archive_root"] == str(root.resolve())


def test_missing_root_fails_closed_and_creates_nothing(tmp_path):
    with pytest.raises(av.ArchiveError, match="does not exist"):
        av.verify(str(tmp_path / "absent"), [])
    assert not (tmp_path / "absent").exists()
    with pytest.raises(av.ArchiveError, match="required"):
        av.resolve_root("")


def test_missing_manifest_or_directory_fails_closed(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    with pytest.raises(av.ArchiveError, match="manifest missing"):
        av.verify(str(root), [tmp_path / "none.json"])
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"archive_directory_name": "gone", "items": []}))
    with pytest.raises(av.ArchiveError, match="directory missing"):
        av.verify(str(root), [man])


def test_byte_difference_fails(tmp_path):
    root = tmp_path / "r"
    man = _archive(root)
    (root / "arc one" / "a.txt").write_bytes(b"changed")
    out = av.verify(str(root), [man])
    assert not out["passed"] and out["problems"] == ["arc one/a.txt"]


def test_no_personal_absolute_path_in_v25_v26_source():
    files = subprocess.run(["git", "ls-files", "scripts/v25_*", "scripts/v26_*", "harness/"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    hits = [f for f in files if (ROOT / f).is_file() and
            "Users\\Student2" in (ROOT / f).read_text(encoding="utf-8", errors="ignore")
            or (ROOT / f).is_file() and "/Users/Student2/" in
            (ROOT / f).read_text(encoding="utf-8", errors="ignore")]
    assert hits == []
