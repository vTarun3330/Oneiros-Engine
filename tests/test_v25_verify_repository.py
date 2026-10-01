"""Verifier-side target localisation for repository verification (never shown to tests)."""
from __future__ import annotations

from scripts.v25_verify_repository_wsl import localize

BUGGY = '''import os


class Point:
    def distance(self, p):
        return abs(self.x - p.x)

    def scale(self, k):
        return k


def helper():
    return 1
'''


def test_method_enclosing_the_first_change_is_the_target():
    fixed = BUGGY.replace("return abs(self.x - p.x)", "return abs(self.x - p.x) ** 0.5")
    assert localize(BUGGY, fixed) == "Point.distance"


def test_top_level_function_change():
    assert localize(BUGGY, BUGGY.replace("return 1", "return 2")) == "helper"


def test_module_level_change_is_unlocalizable():
    assert localize(BUGGY, BUGGY.replace("import os", "import sys")) is None


def test_no_change_is_unlocalizable():
    assert localize(BUGGY, BUGGY) is None


def test_decorated_function_counts_its_decorator_lines():
    src = "@deco\ndef f():\n    return 1\n"
    assert localize(src, src.replace("@deco", "@other")) == "f"
