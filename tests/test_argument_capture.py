"""Argument-capture pilot: literal-only serialisation and a real pytest capture on a toy target."""
from __future__ import annotations

import enum
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
capture = pytest.importorskip("native_argument_capture_wsl")

LIMITS = capture.LIMITS
NS: dict = {}
exec(capture.COMMON, NS)


class Colour(enum.IntEnum):
    RED = 1


def test_literal_accepts_exact_literals_only():
    literal, Unsupported = NS["literal"], NS["Unsupported"]
    assert literal((1, "a", [2.5, None], {"k": (True, b"x")}), LIMITS) == \
        repr((1, "a", [2.5, None], {"k": (True, b"x")}))
    for bad, reason in ((object(), "type:"), (Colour.RED, "type:"), (float("nan"), "non_finite"),
                        (iter([1]), "type:"), (sys, "type:"), ("x" * 600, "too_large")):
        with pytest.raises(Unsupported, match=reason):
            literal(bad, LIMITS)
    deep = [[[[[[1]]]]]]
    with pytest.raises(Unsupported, match="too_deep"):
        literal(deep, LIMITS)
    with pytest.raises(Unsupported, match="too_many_items"):
        literal(list(range(100)), LIMITS)
    with pytest.raises(Unsupported, match="type:builtins.frozenset"):
        literal(frozenset({1}), LIMITS)
    assert literal({1, 2}, LIMITS) == "{1, 2}"


def test_resolve_reports_the_callable_kind():
    resolve = NS["resolve"]
    assert resolve("json", "dumps")[0] == "function"
    assert resolve("fractions", "Fraction.from_float")[0] == "classmethod"
    assert resolve("datetime", "date.fromisoformat")[0].startswith("unsupported:")  # C-level
    assert resolve("pathlib", "PurePath.joinpath")[0] == "instance_method"


def test_oracle_requires_a_short_fixed_literal_and_a_buggy_difference():
    calls = [{"args": "(1,)", "kwargs": "{}"}] * 4
    fixed = {"results": [["ok", "2"], ["ok", "2"], ["raise", "ValueError"], ["ok", "'x'" * 60]]}
    buggy = {"results": [["ok", "3"], ["ok", "2"], ["ok", "1"], ["ok", "1"]]}
    rows = capture.classify(calls, fixed, buggy)
    assert [r["short_verified_oracle"] for r in rows] == [True, False, False, False]


def test_plugin_captures_real_arguments_only_in_wanted_tests(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "toy.py").write_text(
        "class Box:\n"
        "    def __init__(self, v):\n        self.v = v\n"
        "    def get(self, n):\n        return self.v + n\n"
        "    @classmethod\n    def make(cls, v):\n        return cls(v)\n"
        "def double(x, *rest, scale=1, **extra):\n    x.append(0) if isinstance(x, list) else None\n"
        "    return x\n", encoding="utf-8")
    (package / "test_toy.py").write_text(
        "import pytest\nfrom toy import double, Box\n"
        "@pytest.mark.parametrize('value', [[1, 2], 'ab'])\n"
        "def test_wanted(value):\n    double(value, 3, scale=2, tag='t')\n    double(object())\n"
        "def test_other():\n    double([9])\n"
        "class TestBox:\n    def test_method(self):\n        Box(1).get(2)\n", encoding="utf-8")
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "oneiros_capture.py").write_text(capture.PLUGIN, encoding="utf-8")
    out = tmp_path / "captured.json"
    spec = {"module": "toy", "qualname": "double", "out": str(out), "limits": LIMITS,
            "wanted": ["test_toy::test_wanted[value0]", "test_toy::test_wanted[ab]"]}
    env = dict(os.environ, ONEIROS_CAPTURE_SPEC=json.dumps(spec),
               PYTHONPATH=os.pathsep.join([str(plugin_dir), str(package)]))
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-o", "addopts=", "-p", "oneiros_capture", "test_toy.py"],
                          cwd=package, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-800:]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["kind"] == "function" and data["resolve_error"] is None
    assert data["observed"] == 4                     # 2 wanted tests x 2 calls; test_other ignored
    # Captured at entry, before the body appended 0 to the list.
    assert [c["args"] for c in data["calls"]] == ["([1, 2], 3)", "('ab', 3)"]
    assert data["calls"][0]["kwargs"] == "{'scale': 2, 'tag': 't'}"
    assert any(reason.startswith("type:builtins.object") for reason in data["rejected"])
    method = dict(spec, qualname="Box.get", wanted=["test_toy.TestBox::test_method"])
    env["ONEIROS_CAPTURE_SPEC"] = json.dumps(method)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
                    "addopts=", "-p", "oneiros_capture", "test_toy.py"], cwd=package, env=env,
                   capture_output=True, text=True, timeout=120)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["kind"] == "instance_method" and data["calls"] == []
    assert data["rejected"] == {"instance_method_receiver:Box": 1}


def test_pilot_receipt_totals_rederive_and_admit_nothing():
    receipt_path = ROOT / "results" / "sft_root_cause_phase4_argument_capture_receipt_v1.json"
    if not receipt_path.exists():
        pytest.skip("pilot receipt not yet written")
    import hashlib
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    embedded = receipt["portable_evidence"]["records.jsonl"]
    assert hashlib.sha256(embedded["text"].encode()).hexdigest() == embedded["text_sha256"] \
        == embedded["original_sha256"]
    rows = [json.loads(l) for l in embedded["text"].splitlines() if l.strip()]
    oracles = sum(c["short_verified_oracle"] for r in rows for c in r.get("calls") or [])
    assert oracles == receipt["totals"]["short_verified_oracle"]
    usable = sum(any(c["short_verified_oracle"] for c in r.get("calls") or []) for r in rows)
    assert usable == receipt["qualified_targets_with_a_usable_fixed_call"]["k"]
    assert receipt["isolation_v6_revalidated"] is False and receipt["mass_acquisition"] is False
    for r in rows:
        for c in r.get("calls") or []:
            assert set(c) >= {"args", "kwargs", "fixed", "buggy"} and "source" not in c
