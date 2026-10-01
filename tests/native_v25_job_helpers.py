"""Test helper: a real v2.5 job built by the v2.5 job builder (no model, whitespace tokens)."""
from __future__ import annotations

import json
from pathlib import Path

from harness import native_generated_test_job_v25 as jobs

BUGGY = '''"""Mod."""
import math

LIMIT = 10


def area(r: float) -> float:
    """Area of a circle of radius r."""
    if r > LIMIT:
        return r
    return math.pi * r
'''
FIXED = BUGGY.replace("return math.pi * r", "return math.pi * r * r")
VERIFIER = {"fixed_source": FIXED, "patch": "+    return math.pi * r * r\n",
            "official_tests": ["from pkg.geo import area\n\ndef test_area():\n"
                               "    assert area(2.0) == 12.566370614359172\n"]}


def dto(i: int) -> dict:
    return {"target_key": f"t{i}", "repository": "o/r", "buggy_commit": "abc",
            "target_file": "src/pkg/geo.py", "qualname": "area"}


def targets(n: int) -> list:
    return [{"dto": dto(i), "buggy_source": BUGGY, "verifier": VERIFIER} for i in range(n)]


def count_words(text: str) -> int:
    return len(text.split())


def v25_job_file(path: Path, n: int = 2, root: Path = None) -> dict:
    root = root or Path(__file__).resolve().parent.parent
    data = jobs.build_job_file(targets(n), count_words, root, purpose="test")
    Path(path).write_text(json.dumps(data, sort_keys=True))
    return data
