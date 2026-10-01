"""Confirmation-panel lineage and near-duplicate fingerprints."""
from __future__ import annotations

from scripts import v25_confirmation_panel_spec as ps


def test_repository_key_collides_forks_and_urls():
    assert ps.repo_key("https://github.com/django/django") == "django"
    assert ps.repo_key("python-attrs__cattrs.git") == "cattrs"
    assert ps.repo_key("Sympy/SymPy/") == "sympy"


def test_function_fingerprint_ignores_names_docstrings_and_annotations():
    a = 'def area(r: float) -> float:\n    """Doc."""\n    return 3.14 * r * r\n'
    b = "def size(x):\n    return 3.14 * x * x\n"
    c = "def size(x):\n    return 3.14 * x * x * x\n"
    assert ps.function_fingerprint(a) == ps.function_fingerprint(b)
    assert ps.function_fingerprint(b) != ps.function_fingerprint(c)


def test_functions_in_finds_every_definition_and_tolerates_bad_code():
    code = "def f(a):\n    return a\n\nclass K:\n    def g(self, b):\n        return b + 1\n"
    assert len(ps.functions_in(code)) == 2
    assert ps.functions_in("def broken(:") == []


def test_lineage_fingerprint_components():
    lf = ps.lineage_fingerprint("https://github.com/o/Repo", "ABC123", "def f(x):\n    return x\n")
    assert lf["repository"] == "repo" and lf["fix_commit"] == "abc123" and len(lf["function"]) == 64
