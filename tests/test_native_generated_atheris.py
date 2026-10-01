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


V5_MODULE = """
import enum
import typing


class Color(enum.Enum):
    RED = 1


class Widget:
    pass


def pos_only(a: int, /, b: int = 1, *, c: str, d: list[int] = None) -> int: ...
def variadic(*xs: int) -> int: ...
def kwvar(**kw: int) -> int: ...
def untyped(x): ...
def untyped_default(x: int, y=3) -> int: ...
def union(x: typing.Optional[int], y: typing.Union[str, bytes]) -> int: ...
def anything(x: typing.Any) -> int: ...
def colour(c: Color) -> int: ...
def literal(m: typing.Literal["a", 2]) -> int: ...
def custom(w: Widget) -> int: ...


class Box:
    def total(self, n: int) -> int: ...


class Account:
    def __init__(self, balance: int, owner: str = "x"):
        self.balance = balance

    def withdraw(self, amount: int) -> int: ...


class Locked:
    def __init__(self, w: Widget):
        self.w = w

    def open(self, n: int) -> int: ...
"""


@pytest.fixture
def v5mod(tmp_path, monkeypatch):
    (tmp_path / "v5_adapter_mod.py").write_text(V5_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "v5_adapter_mod"
    sys.modules.pop("v5_adapter_mod", None)


def test_v5_adapter_handles_every_parameter_kind(v5mod):
    elig, resolve = NS["eligibility"], NS["resolve"]
    E = lambda q: elig(resolve(v5mod, q), q, v5mod)          # noqa: E731
    plan = E("pos_only")
    assert plan["eligible"] and [(p["name"], p["keyword"]) for p in plan["plan"]["params"]] == [
        ("a", False), ("b", False), ("c", True), ("d", True)]
    assert E("variadic")["reason"] == "adapter_unsupported:variadic_signature"
    assert E("kwvar")["reason"] == "adapter_unsupported:variadic_signature"
    assert E("untyped")["reason"].startswith("adapter_unsupported:unsupported_parameter:x")
    assert [p["name"] for p in E("untyped_default")["plan"]["params"]] == ["x"]
    assert E("custom")["reason"] == "adapter_unsupported:unsupported_parameter:w:Widget"


def test_v5_adapter_types_union_any_enum_literal(v5mod):
    elig, resolve = NS["eligibility"], NS["resolve"]
    E = lambda q: elig(resolve(v5mod, q), q, v5mod)          # noqa: E731
    u = E("union")["plan"]["params"]
    assert u[0]["spec"] == {"kind": "union", "options": [{"kind": "prim", "type": "int"},
                                                         {"kind": "none"}]}
    assert u[1]["spec"]["kind"] == "union"
    assert E("anything")["plan"]["params"][0]["spec"] == {"kind": "any"}
    assert E("colour")["plan"]["params"][0]["spec"]["members"] == ["RED"]
    assert E("literal")["plan"]["params"][0]["spec"] == {"kind": "literal", "values": ["a", 2]}


def test_v5_receivers_only_through_public_constructors(v5mod):
    elig, resolve = NS["eligibility"], NS["resolve"]
    E = lambda q: elig(resolve(v5mod, q), q, v5mod)          # noqa: E731
    box = E("Box.total")["plan"]
    assert box["receiver"] == {"module": v5mod, "qualname": "Box", "plan": []}
    assert box["method"] == "total" and [p["name"] for p in box["params"]] == ["n"]
    acct = E("Account.withdraw")["plan"]["receiver"]["plan"]
    assert [p["name"] for p in acct] == ["balance", "owner"]
    assert E("Locked.open")["reason"] ==         "adapter_unsupported:receiver_constructor:unsupported_parameter:w:Widget"


def test_v5_values_are_built_from_specs(v5mod):
    value_for = NS["value_for"]

    class FDP:                                    # deterministic stand-in for atheris
        def ConsumeIntInRange(self, lo, hi):
            return lo
        def ConsumeBool(self):
            return True
    f = FDP()
    assert value_for(f, {"kind": "none"}) is None
    assert value_for(f, {"kind": "literal", "values": ["a", 2]}) == "a"
    assert value_for(f, {"kind": "union", "options": [{"kind": "none"},
                                                      {"kind": "prim", "type": "int"}]}) is None
    assert value_for(f, {"kind": "any"}) == -10 ** 6          # first menu entry: int
    member = value_for(f, {"kind": "enum", "module": v5mod, "qualname": "Color",
                           "members": ["RED"]})
    assert member.name == "RED"


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
    assert not hasattr(ath, "PYTHON") and not hasattr(ath, "ATHERIS_SITE")   # v5: per target
    assert ath.ATHERIS_V5_ROOT.as_posix() == "/opt/oneiros_atheris_v5"
    assert ath.MODES == ("ordinary", "posthoc", "differential")
    assert ath.DESIGN_VERSION == "oneiros_native_generated_tests_atheris_v5"
    assert ath.tolerance(600) == pytest.approx(13.0) and ath.tolerance(20) == pytest.approx(1.4)


def test_v3_isolation_and_aggregate_accounting_are_wired():
    """v2.2 E: one cgroup-v2 group per search, the shared sandbox, /target only."""
    source = (ROOT / "scripts" / "native_generated_tests_atheris_wsl.py").read_text(encoding="utf-8")
    assert '"unshare", "--mount", "--pid", "--net", "--fork", "--mount-proc"' in source
    assert """"SB_PYTHONPATH": f"/target:{RUNTIME['overlay']}\"""" in source   # overlay last
    assert 'RUNTIME["python"], "-B"' in source                  # the target's interpreter
    assert "atheris311" not in source and '"/usr/bin/python3.11", "-B"' not in source
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
