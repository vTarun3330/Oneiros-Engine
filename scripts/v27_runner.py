"""v2.7 launcher for scripts/run_repository_native_acquisition_pilot.py with ONE behaviour-
identical performance fix (the runner and the hash-bound isolation harness are unchanged).

``ast.get_source_segment`` re-splits the WHOLE source on every call, so extracting each function
of a very large file (e.g. coleifer/peewee's single 8,000-line module) is quadratic and stalled
the acquisition for hours. Here the full line split is computed once per source string and
reused; ``get_source_segment`` only reads that list (it rebinds a slice before inserting), so
every returned segment is byte-identical (tests/test_v27_runner.py).

    python scripts/v27_runner.py --config <acquisition config>
"""
from __future__ import annotations

import ast
from pathlib import Path
import runpy
import sys

RUNNER = Path(__file__).resolve().parent / "run_repository_native_acquisition_pilot.py"
_ORIGINAL = ast._splitlines_no_ff
_CACHE: dict = {}


def cached_splitlines(source, maxlines=None):
    lines = _CACHE.get(source)
    if lines is None:
        if len(_CACHE) >= 64:
            _CACHE.clear()
        lines = _CACHE[source] = _ORIGINAL(source)
    return lines if maxlines is None else lines[:maxlines] if maxlines < len(lines) else lines


def install() -> None:
    ast._splitlines_no_ff = cached_splitlines


if __name__ == "__main__":
    install()
    sys.argv = [str(RUNNER), *sys.argv[1:]]
    runpy.run_path(str(RUNNER), run_name="__main__")
