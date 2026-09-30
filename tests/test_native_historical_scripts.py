"""Historical versioned preflight scripts stay byte-for-byte frozen (amendment v2.4 section A).

The blob identities are those of commit 5ca22ee (the last pre-v2.3 commit). Successors are
separate files; a historical script is never renamed, edited or deleted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FROZEN_BLOBS = {
    "scripts/native_generated_tests_preflight_v2.py": "ed9a5c9c76e5d84d98b09f8ba0e4053db3d56e5a",
    "scripts/native_generated_tests_preflight_v2_2.py": "9a5afe38b6e341fd0d74ae0bbd380a7804b1e2bf",
}
SUCCESSORS = ("scripts/native_generated_tests_preflight_v2_3.py",
              "scripts/native_generated_tests_preflight_v2_4.py")


def _git_blob(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.mark.parametrize("rel, blob", sorted(FROZEN_BLOBS.items()))
def test_historical_preflight_script_is_byte_identical(rel, blob):
    assert _git_blob(ROOT / rel) == blob


def test_successors_are_separate_files():
    for rel in SUCCESSORS:
        assert (ROOT / rel).is_file(), rel
    assert len({_git_blob(ROOT / r) for r in (*FROZEN_BLOBS, *SUCCESSORS)}) == 4
