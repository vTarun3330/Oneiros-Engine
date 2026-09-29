"""Receiver-aware capture v2: real pytest capture and replay processes on a toy package."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
rc = pytest.importorskip("native_receiver_capture_wsl")

CORE = textwrap.dedent('''
    class Counter:
        def __init__(self, start, *, step=1):
            self.value = start
            self.step = step
        def advance(self, n):
            return self.value + n * self.step
        @classmethod
        def from_pair(cls, pair):
            return cls(pair[0], step=pair[1])

    class Slotted:
        __slots__ = ("a", "b")
        def __init__(self, a):
            self.a = a
        def total(self, n):
            return self.a + n

    class OpaqueSlots:
        __slots__ = ("h",)
        def __init__(self):
            self.h = len
        def total(self, n):
            return n

    class Holder:
        def __init__(self, items):
            self.items = items
        def size(self, extra):
            self.items.append(extra)
            return len(self.items)

    _CACHE = {}
    class Cached:
        def __init__(self, k):
            self.k = k
        @classmethod
        def get(cls, k):
            if k not in _CACHE:
                _CACHE[k] = cls(k)
            return _CACHE[k]
        def show(self, n):
            return self.k * n

    class Meta(type):
        pass
    class WithMeta(metaclass=Meta):
        def __init__(self, x):
            self.x = x
        def go(self, n):
            return n

    class WithNew:
        def __new__(cls, x):
            return super().__new__(cls)
        def __init__(self, x):
            self.x = x
        def go(self, n):
            return n

    class Opaque:
        def __init__(self, x):
            self.x = x
            self.f = object()
        def go(self, n):
            return n

    class Raises:
        def __init__(self, x):
            raise ValueError("bad constructor")
        def go(self, n):
            return n

    def slow(n):
        while True:
            pass

    def net(n):
        import socket
        socket.create_connection(("example.com", 80))
        return 1

    def boom(n):
        raise ValueError("boom")
''')

TESTS = textwrap.dedent('''
    from toypkg.core import Counter, Slotted, OpaqueSlots, Cached, WithMeta, WithNew, Opaque

    class Sub(Counter):
        pass

    def test_ctor():
        assert Counter(2, step=3).advance(4) == 14

    def test_factory():
        assert Counter.from_pair((1, 2)).advance(5) == 11

    def test_mutated():
        c = Counter(1)
        c.value = 99
        assert c.advance(1) == 100

    def test_nonliteral():
        c = Counter(3, step=len)
        try:
            c.advance(1)
        except TypeError:
            pass

    def test_foreign():
        assert Sub(1).advance(1) == 2

    def test_id_reuse():
        a = Counter(1)
        del a
        b = object.__new__(Counter)
        b.value, b.step = 1, 1
        assert b.advance(1) == 2

    def test_slots():
        assert Slotted(3).total(4) == 7
        assert OpaqueSlots().total(1) == 1

    def test_cached():
        assert Cached.get(2).show(3) == 6
        assert Cached.get(2).show(4) == 8

    def test_ambiguous():
        assert WithMeta(1).go(1) == 1
        assert WithNew(1).go(1) == 1
        assert Opaque(1).go(1) == 1

    def test_not_wanted():
        assert Counter(7).advance(1) == 8
''')


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    (root / "toypkg").mkdir()
    (root / "toypkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "toypkg" / "core.py").write_text(CORE, encoding="utf-8")
    (root / "test_toy.py").write_text(TESTS, encoding="utf-8")
    (root / "oneiros_receiver_capture.py").write_text(rc.PLUGIN, encoding="utf-8")
    (root / "oneiros_receiver_replay.py").write_text(rc.REPLAY, encoding="utf-8")
    return root


def capture(toy, qualname, tests):
    out = toy / f"captured_{qualname}.json"
    spec = {"module": "toypkg.core", "qualname": qualname, "package": "toypkg",
            "wanted": [f"test_toy::{t}" for t in tests], "out": str(out), "limits": rc.LIMITS_V2}
    env = dict(os.environ, ONEIROS_CAPTURE_SPEC=json.dumps(spec), PYTHONPATH=str(toy))
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
                           "addopts=", "-p", "oneiros_receiver_capture", "test_toy.py"],
                          cwd=toy, env=env, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stdout[-1500:]
    return json.loads(out.read_text(encoding="utf-8"))


def replay(toy, qualname, calls, per_call_seconds=5):
    spec = toy / "replay_spec.json"
    spec.write_text(json.dumps({"module": "toypkg.core", "qualname": qualname,
                                "package": "toypkg", "calls": calls,
                                "limits": {**rc.LIMITS_V2,
                                           "per_call_seconds": per_call_seconds}}),
                    encoding="utf-8")
    done = subprocess.run([sys.executable, str(toy / "oneiros_receiver_replay.py"), str(spec)],
                          cwd=toy, env=dict(os.environ, PYTHONPATH=str(toy)),
                          capture_output=True, text=True, timeout=120)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_literal_constructor_with_positional_and_keyword_arguments(toy):
    data = capture(toy, "Counter.advance", ["test_ctor", "test_factory"])
    assert {c["recipe"]["kind"] for c in data["calls"]} == {"constructor"}
    ctor = next(c for c in data["calls"] if c["args"] == "(4,)")
    assert ctor["recipe"]["args"] == "(2,)" and ctor["recipe"]["kwargs"] == "{'step': 3}"
    assert ctor["args"] == "(4,)"
    # The classmethod factory built its instance through the allowlisted constructor,
    # whose recipe (the inner cls(...) call) is the one kept.
    assert data["observed"] == 2 and len(data["calls"]) == 2
    assert "== 14" not in json.dumps(data) and "== 11" not in json.dumps(data)


@pytest.mark.parametrize("test, reason", [
    ("test_mutated", "receiver_mutated"),
    ("test_nonliteral", "receiver_constructor_type:"),
    ("test_foreign", "receiver_without_allowlisted_construction:Sub"),
    ("test_id_reuse", "receiver_without_allowlisted_construction:Counter"),
    ("test_not_wanted", None),
])
def test_unsafe_receivers_are_rejected_with_reasons(toy, test, reason):
    wanted = [test] if reason else []
    data = capture(toy, "Counter.advance", wanted)
    assert data["calls"] == []
    if reason:
        assert any(r.startswith(reason) for r in data["rejected"]), data["rejected"]
    else:
        assert data["observed"] == 0


def test_nonliteral_constructor_argument_is_rejected(toy):
    data = capture(toy, "Counter.advance", ["test_nonliteral"])
    assert any("constructor_type:builtins.builtin_function_or_method" in r
               for r in data["rejected"]), data["rejected"]


def test_slots_supported_and_opaque_slots_rejected(toy):
    ok = capture(toy, "Slotted.total", ["test_slots"])
    assert len(ok["calls"]) == 1 and "slot:a" in ok["calls"][0]["snapshot"]
    bad = capture(toy, "OpaqueSlots.total", ["test_slots"])
    assert bad["calls"] == [] and any("opaque_state" in r for r in bad["rejected"])


def test_cached_factory_is_rejected_after_first_use(toy):
    data = capture(toy, "Cached.show", ["test_cached"])
    assert len(data["calls"]) == 1
    assert any("cached_factory" in r for r in data["rejected"]), data["rejected"]


@pytest.mark.parametrize("qualname", ["WithMeta.go", "WithNew.go", "Opaque.go"])
def test_ambiguous_or_opaque_construction_is_rejected(toy, qualname):
    data = capture(toy, qualname, ["test_ambiguous"])
    assert data["calls"] == []
    assert any(r.startswith("receiver_") for r in data["rejected"]), data["rejected"]


def _ctor(args, kwargs, snapshot_obj_fields, qual="Counter"):
    return {"kind": "constructor", "module": "toypkg.core", "qualname": qual,
            "args": args, "kwargs": kwargs}


def test_replay_reconstructs_independently_and_detects_mismatch(toy):
    data = capture(toy, "Counter.advance", ["test_ctor"])
    call = data["calls"][0]
    out = replay(toy, "Counter.advance", [call, {**call, "snapshot": call["snapshot"] + "x"}])
    assert out["results"] == [["ok", "14"], ["reconstruction_mismatch", None]]
    holder = {"test": "t", "recipe": _ctor("([1],)", "{}", None, "Holder"),
              "snapshot": None, "args": "(5,)", "kwargs": "{}"}
    from_ns = {}
    exec(rc.COMMON + rc.SNAPSHOT, from_ns)

    class Holder:                                  # the snapshot the capture would record
        pass
    fake = Holder()
    fake.items = [1]
    holder["snapshot"] = from_ns["snapshot"](fake, rc.LIMITS_V2)
    twice = replay(toy, "Holder.size", [holder, holder])
    assert twice["results"] == [["ok", "2"], ["ok", "2"]]     # no state shared across calls


def test_replay_failure_modes_are_separated(toy):
    raises = {"test": "t", "recipe": _ctor("(1,)", "{}", None, "Raises"), "snapshot": "",
              "args": "(1,)", "kwargs": "{}"}
    assert replay(toy, "Raises.go", [raises])["results"] == [["constructor_raise", "ValueError"]]
    meta = {**raises, "recipe": _ctor("(1,)", "{}", None, "WithMeta")}
    assert replay(toy, "WithMeta.go", [meta])["results"][0][0] == "reconstruction_refused"
    plain = {"test": "t", "recipe": None, "snapshot": None, "args": "(1,)", "kwargs": "{}"}
    assert replay(toy, "boom", [plain])["results"] == [["raise", "ValueError"]]
    assert replay(toy, "net", [plain])["results"] == [["raise", "NetworkDisabled"]]
    assert replay(toy, "slow", [plain], per_call_seconds=1)["results"] == [["timeout", None]]


def test_classification_requires_determinism_and_a_difference():
    calls = [{"test": "t", "args": "()", "kwargs": "{}", "recipe": None}] * 7
    ok, other, long_ = ["ok", "1"], ["ok", "2"], ["ok", "'" + "x" * 130 + "'"]
    fixed = [{"results": [ok, ok, other, ok, ok, long_, ["timeout", None]]},
             {"results": [ok, other, other, ok, ok, long_, ["timeout", None]]}]
    buggy = [{"results": [other, other, other, ["raise", "E"], ok, other, ok]},
             {"results": [other, other, other, ok, ok, other, ok]}]
    status = [r["status"] for r in rc.classify(calls, fixed, buggy)]
    assert status == ["usable", "nondeterministic_fixed", "no_difference",
                      "nondeterministic_buggy", "no_difference", "fixed_value_too_long",
                      "timeout"]
    crashed = rc.classify(calls[:1], [{"replay_error": "x"}, {"results": [ok]}],
                          [{"results": [other]}] * 2)
    assert crashed[0]["status"] == "replay_process_failure"


def test_no_unsafe_deserialisation_in_the_implementation():
    source = (ROOT / "scripts" / "native_receiver_capture_wsl.py").read_text(encoding="utf-8")
    assert not re.search(r"\b(import|from)\s+(pickle|marshal|dill|shelve|copyreg)\b", source)
    assert not re.search(r"\b(pickle|marshal|dill|shelve)\.(load|loads)\b", source)
    assert "__reduce__" not in source
    assert not re.search(r"(?<!literal_)eval\(", source)
    assert "exec(" not in source


def test_replay_refuses_a_module_imported_from_the_wrong_revision(toy, tmp_path):
    spec = toy / "replay_spec_rev.json"
    plain = {"test": "t", "recipe": None, "snapshot": None, "args": "(1,)", "kwargs": "{}"}
    spec.write_text(json.dumps({"module": "toypkg.core", "qualname": "boom", "package": "toypkg",
                                "calls": [plain], "limits": rc.LIMITS_V2,
                                "expected_root": str(tmp_path)}), encoding="utf-8")
    done = subprocess.run([sys.executable, str(toy / "oneiros_receiver_replay.py"), str(spec)],
                          cwd=toy, env=dict(os.environ, PYTHONPATH=str(toy)),
                          capture_output=True, text=True, timeout=60)
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert "revision_mismatch" in out and "results" not in out


def test_pilot_receipt_applies_the_unchanged_gate_and_admits_nothing():
    receipt_path = ROOT / "results" / "sft_root_cause_phase4_receiver_capture_receipt_v2.json"
    if not receipt_path.exists():
        pytest.skip("pilot receipt not yet written")
    import hashlib
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    gate = receipt["feasibility_gate"]
    assert gate["thresholds"] == {"min_usable_targets": 8, "targets": 24, "min_repositories": 4,
                                  "repositories": 8, "max_repository_share": 0.4}
    usable = receipt["usable_targets"]["k"]
    repos = receipt["usable_repositories"]["k"]
    share = receipt["largest_repository_share"]
    assert gate["checks"] == {"usable_targets": usable >= 8, "repositories": repos >= 4,
                              "max_repository_share": share is not None and share <= 0.4}
    assert receipt["branch"] == ("A" if all(gate["checks"].values()) else "B")
    assert receipt["admitted_to_training"] is False and receipt["mass_acquisition"] is False
    embedded = receipt["portable_evidence"]["records.jsonl"]
    assert hashlib.sha256(embedded["text"].encode()).hexdigest() == embedded["original_sha256"]
    manifest = json.loads((ROOT / receipt["manifest"]["path"]).read_text(encoding="utf-8"))
    v1 = json.loads((ROOT / manifest["identical_to"]["path"]).read_text(encoding="utf-8"))
    assert manifest["targets"] == v1["targets"]
