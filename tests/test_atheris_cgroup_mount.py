"""The Atheris harness finds the cgroup-v2 hierarchy on both WSL layouts (regression: WSL kernel
6.x mounts cgroup2 directly at /sys/fs/cgroup, so the hard-coded hybrid path
/sys/fs/cgroup/unified failed every CPU-budgeted search with FileNotFoundError)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import native_generated_tests_atheris_wsl as ath  # noqa: E402


def _only(existing: set):
    def is_file(self):
        return self.as_posix() in existing
    return is_file


@pytest.mark.parametrize("existing, expected", [
    ({"/sys/fs/cgroup/cgroup.controllers"}, "/sys/fs/cgroup"),                  # pure cgroup v2
    ({"/sys/fs/cgroup/unified/cgroup.controllers"}, "/sys/fs/cgroup/unified"),  # hybrid layout
    ({"/sys/fs/cgroup/unified/cgroup.controllers", "/sys/fs/cgroup/cgroup.controllers"},
     "/sys/fs/cgroup/unified"),                       # hybrid wins where both exist (unchanged)
])
def test_cgroup2_hierarchy_is_resolved_on_both_layouts(monkeypatch, existing, expected):
    monkeypatch.setattr(Path, "is_file", _only(existing))
    assert ath._cgroup2_mount().as_posix() == expected


def test_no_cgroup2_keeps_the_historical_path(monkeypatch):
    monkeypatch.setattr(Path, "is_file", _only(set()))
    assert ath._cgroup2_mount().as_posix() == "/sys/fs/cgroup/unified"
