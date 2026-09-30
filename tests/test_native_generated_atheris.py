"""Atheris v2 helpers that run in the fuzz/replay children: eligibility and canonical results."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
ath = pytest.importorskip("native_generated_tests_atheris_wsl")
NS: dict = {}
exec(ath.COMMON, NS)


def test_eligibility_handles_every_parameter_kind():
    elig = NS["eligibility"]

    def pos_only(a: int, /, b: int = 1, *, c: str, d: list[int] = None) -> int: ...
    def variadic(*xs: int) -> int: ...
    def kwvar(**kw: int) -> int: ...
    def untyped(x): ...
    def untyped_default(x: int, y=3) -> int: ...
    plan = elig(pos_only, "pos_only")
    assert plan["eligible"] and [(p["name"], p["keyword"]) for p in plan["plan"]] == [
        ("a", False), ("b", False), ("c", True), ("d", True)]
    assert elig(variadic, "variadic")["reason"] == "variadic_signature"
    assert elig(kwvar, "kwvar")["reason"] == "variadic_signature"
    assert elig(untyped, "untyped")["reason"].startswith("unsupported_parameter:x")
    assert [p["name"] for p in elig(untyped_default, "u")["plan"]] == ["x"]

    class Box:
        def total(self, n: int) -> int: ...
    assert elig(Box.total, "Box.total")["reason"] == "instance_method_receiver"


def test_canonical_values_are_stable_and_opaque_values_refuse():
    canonical, Opaque = NS["canonical"], NS["Opaque"]
    assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})
    assert canonical({3, 1, 2}) == canonical({2, 3, 1})
    assert canonical(float("nan")) == ["float", "nan"]
    assert canonical(b"\x00") == ["bytes", "AA=="]
    assert canonical((1, [True, None])) != canonical([1, [True, None]])
    assert canonical(1) != canonical(True) and canonical(1) != canonical(1.0)
    with pytest.raises(Opaque):
        canonical(object())
    with pytest.raises(Opaque):
        canonical([[[[[[[[1]]]]]]]])


def test_budgets_and_parity_constants():
    assert ath.FULL_BUDGET_CPU_SECONDS == 600 and ath.CORPUS_CAP == 2000
    assert ath.PYTHON == "/usr/bin/python3.11"
    assert ath.MODES == ("ordinary", "posthoc", "differential")
    source = (ROOT / "scripts" / "native_generated_tests_atheris_wsl.py").read_text(encoding="utf-8")
    assert '"prlimit", f"--cpu={cpu}"' in source and '"unshare", "-n"' in source
