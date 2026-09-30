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
    assert ath.DESIGN_VERSION == "oneiros_native_generated_tests_atheris_v4"
    assert ath.tolerance(600) == pytest.approx(13.0) and ath.tolerance(20) == pytest.approx(1.4)


def test_v3_isolation_and_aggregate_accounting_are_wired():
    """v2.2 E: one cgroup-v2 group per search, the shared sandbox, /target only."""
    source = (ROOT / "scripts" / "native_generated_tests_atheris_wsl.py").read_text(encoding="utf-8")
    assert '"unshare", "--mount", "--pid", "--net", "--fork", "--mount-proc"' in source
    assert '"SB_PYTHONPATH": f"/target:{ATHERIS_SITE}"' in source
    assert "cgroup.kill" in source and "usage_usec" in source
    assert "worker_launcher" not in source and "PYTHONPATH\": f\"{view}" not in source
    assert ath.INNER.name == "native_sandbox_inner.sh"


def test_end_reasons_are_distinct(tmp_path):
    started = tmp_path / "started"
    sup = {"supervisor_reason": None, "wall_seconds": 10.0}
    assert ath._end_reason(sup, 0, tmp_path, 20) == "infrastructure_failure"   # never started
    started.write_text("")
    assert ath._end_reason(sup, 0, tmp_path, 20) == "completed"
    assert ath._end_reason({**sup, "supervisor_reason": "cpu_budget_exhausted"}, -9, tmp_path,
                           20) == "cpu_budget_exhausted"
    assert ath._end_reason({**sup, "supervisor_reason": "wall_timeout"}, -9, tmp_path,
                           20) == "wall_timeout"
    assert ath._end_reason(sup, 152, tmp_path, 20) == "cpu_budget_exhausted"
    assert ath._end_reason(sup, 1, tmp_path, 20) == "crashed"
    (tmp_path / "worker_lost").write_text("")
    assert ath._end_reason(sup, 3, tmp_path, 20) == "infrastructure_failure"
    assert set(ath.END_REASONS) == {"completed", "cpu_budget_exhausted", "wall_timeout",
                                    "crashed", "infrastructure_failure"}


def test_witnesses_cut_off_by_a_kill_are_dropped(tmp_path):
    import hashlib
    good = b"complete input"
    (tmp_path / hashlib.sha256(good).hexdigest()).write_bytes(good)
    (tmp_path / hashlib.sha256(b"longer original").hexdigest()).write_bytes(b"longer")
    kept, dropped = ath._intact(tmp_path)
    assert [p.read_bytes() for p in kept] == [good] and dropped == 1
    assert ath._intact(tmp_path / "missing") == ([], 0)
