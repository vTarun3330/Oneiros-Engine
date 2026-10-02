"""The v2.7 runner launcher's line-split cache is behaviour-identical to ``ast``."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import v27_runner as vr

ROOT = Path(__file__).resolve().parent.parent
FILES = ["harness/repository_isolation.py", "scripts/run_repository_native_acquisition_pilot.py",
         "engine/sft_trainer.py"]


@pytest.mark.parametrize("rel", FILES)
def test_every_segment_is_identical(rel, monkeypatch):
    source = (ROOT / rel).read_text(encoding="utf-8")
    nodes = [n for n in ast.walk(ast.parse(source)) if hasattr(n, "end_lineno")]
    expected = [(ast.get_source_segment(source, n), ast.get_source_segment(source, n, padded=True))
                for n in nodes]
    monkeypatch.setattr(ast, "_splitlines_no_ff", vr.cached_splitlines)
    got = [(ast.get_source_segment(source, n), ast.get_source_segment(source, n, padded=True))
           for n in nodes]
    assert got == expected and len(nodes) > 100


def test_cache_never_mutates_and_respects_maxlines():
    src = "a\nb\x0cc\nd\n"
    full = vr.cached_splitlines(src)
    assert vr.cached_splitlines(src, maxlines=2) == vr._ORIGINAL(src, maxlines=2)
    assert vr.cached_splitlines(src) is full and full == vr._ORIGINAL(src)
