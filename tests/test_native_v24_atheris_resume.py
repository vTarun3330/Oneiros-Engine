"""Amendment v2.4 E enforcement: Atheris resume validates every retained row with the SAME
authoritative validator as the final full-grid loader (frozen protocol unchanged).

A genuine partial run is produced by the real ``run`` (fake probe and fake ``fuzz`` only), then
corrupted in one way per case. Resume must refuse before any search, leave the results and the
contract byte-for-byte unchanged, and never silently skip or rerun a questionable row. Valid
recorded infrastructure outcomes are retained, not rerun, and classified by the final loader.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import native_atheris_results as ar
from scripts import native_generated_tests_atheris_wsl as ath
from scripts import native_generation_io as gio
from tests import native_analysis_helpers as helpers
from tests.test_native_v23_cohort_pipeline import QUALIFIED, world  # noqa: F401 (fixture)

PLAN_INT_V5 = {"params": [{"name": "x", "spec": {"kind": "prim", "type": "int"}, "keyword": False}],
               "receiver": None, "method": None}

@pytest.fixture(autouse=True)
def _fake_atheris_runtime(monkeypatch):
    """v5 resolves each target's prepared interpreter; test worlds have none."""
    from tests.native_analysis_helpers import fake_runtime
    monkeypatch.setattr(ath, "runtime_for", fake_runtime)


SEEDS = (42, 43, 44)
CELLS = len(QUALIFIED) * len(ar.MODES) * len(SEEDS)


class Crash(RuntimeError):
    pass


def _probe_ok(view, module, qualname):
    return helpers.probe_eligible(view, module, qualname)


def _fuzzer(calls, crash_after=None, special=None):
    def fake(mode, sources, target, info, seed, budget, work):
        if crash_after is not None and len(calls) >= crash_after:
            raise Crash("simulated interruption")
        calls.append((target["module"], mode, seed))
        over = (special or {}).get((len(calls), mode, seed)) or {}
        return helpers.fuzz_evidence(mode, seed, budget, **over)
    return fake


def _run(world, out):
    return ath.run(world["prep"], world["manifest"], out, 600, SEEDS, root=world["tmp"])


@pytest.fixture
def partial(world, monkeypatch):
    """A genuine interrupted run: 10 rows written by the real runner, then a crash."""
    monkeypatch.setattr(ath, "probe", _probe_ok)
    calls = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls, crash_after=10))
    out = world["tmp"] / "atheris"
    with pytest.raises(Crash):
        _run(world, out)
    results = out / "atheris_results.jsonl"
    assert len(results.read_bytes().splitlines()) == 10
    return {"out": out, "results": results, "contract": out / "atheris_contract.json"}


def _rows(path):
    return [json.loads(l) for l in path.read_bytes().decode().splitlines()]


def _dump(rows):
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def _corrupt_rows(fn):
    def apply(data):
        rows = [json.loads(l) for l in data.decode().splitlines()]
        return _dump(fn(rows))
    return apply


def _set(i, **fields):
    def fn(rows):
        rows[i] = {**rows[i], **fields}
        return rows
    return _corrupt_rows(fn)


def _drop(i, field):
    def fn(rows):
        rows[i] = {k: v for k, v in rows[i].items() if k != field}
        return rows
    return _corrupt_rows(fn)


def _unexpected(rows):
    extra = {**rows[0], "seed": 45, "key": rows[0]["key"].rsplit("::", 1)[0] + "::45"}
    return rows + [extra]


def _key_disagrees(rows):
    # the key names a cell that exists but has not run; the fields still say another cell
    t, m, _ = rows[0]["key"].split("::")
    rows[0] = {**rows[0], "key": f"{QUALIFIED[-1]}::{m}::44"}
    return rows


def _ineligible_label(rows):
    rows[3] = {k: rows[3][k] for k in ("key", "contract_sha256", "target_key", "mode", "seed")}
    rows[3].update(eligible=False, status="adapter_unsupported", reason="adapter_unsupported:variadic_signature",
                   kill=False)
    return rows


CORRUPTIONS = {
    "partial_final_line": lambda d: d[:-7],
    "missing_final_newline": lambda d: d[:-1],
    "malformed_json": lambda d: d.replace(d.splitlines()[4], b"{not json", 1),
    "not_an_object": lambda d: d + b"[1, 2, 3]\n",
    "blank_line": lambda d: d + b"\n",
    "duplicate_key": lambda d: d + d.splitlines(keepends=True)[2],
    "unexpected_cell": _corrupt_rows(_unexpected),
    "key_fields_disagree": _corrupt_rows(_key_disagrees),
    "seed_as_string": _set(1, seed="42"),
    "stale_contract": _set(5, contract_sha256="0" * 64),
    "eligibility_disagrees": _corrupt_rows(_ineligible_label),
    "status_reason_disagree": _set(6, reason="adapter_unsupported:variadic_signature"),
    "missing_identity_field": _drop(2, "status"),
    "missing_evidence_field": _drop(7, "end_reason"),
    "missing_cpu_field": _drop(8, "aggregate_cpu_seconds"),
    "confirmations_not_a_list": _set(0, confirmations="none"),
    "confirmation_not_an_object": _set(0, confirmations=[1], witnesses=1),
    "cpu_not_numeric": _set(1, aggregate_cpu_seconds="600"),
    "bool_as_number": _set(2, witnesses=True),
    "flag_not_boolean": _set(3, reached="yes"),
    "unknown_end_reason": _set(4, end_reason="finished"),
    "unknown_field": _set(5, injected=1),
    "budget_differs_from_contract": _set(6, budget_cpu_seconds=60),
    "tolerance_differs_from_contract": _set(7, tolerance_cpu_seconds=2.0),
    "posthoc_field_on_ordinary_row": _set(0, corpus=3),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_resume_refuses_questionable_rows_and_preserves_the_artifact(partial, world,
                                                                     monkeypatch, name):
    results, contract = partial["results"], partial["contract"]
    results.write_bytes(CORRUPTIONS[name](results.read_bytes()))
    before = results.read_bytes(), contract.read_bytes()
    calls = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls))
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(world, partial["out"])
    assert calls == []                                # nothing skipped, nothing rerun
    assert (results.read_bytes(), contract.read_bytes()) == before


def test_results_without_their_contract_are_refused(partial, world, monkeypatch):
    partial["contract"].unlink()
    before = partial["results"].read_bytes()
    calls = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls))
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(world, partial["out"])
    assert calls == [] and partial["results"].read_bytes() == before
    assert not partial["contract"].exists()


def test_a_genuine_partial_run_resumes_only_the_missing_cells(partial, world, monkeypatch):
    kept = partial["results"].read_bytes()
    calls = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls))
    assert _run(world, partial["out"]) == 0
    data = partial["results"].read_bytes()
    assert data.startswith(kept) and len(calls) == CELLS - 10
    keys = [r["key"] for r in _rows(partial["results"])]
    assert len(keys) == len(set(keys)) == CELLS
    prepared = gio.resolve_prep(world["prep"], world["manifest"], world["tmp"])
    loaded = helpers.load_atheris(world, partial["results"], partial["contract"], prepared)
    assert loaded["cells"] == CELLS and loaded["counts"]["usable"] == len(QUALIFIED)


def test_recorded_infrastructure_outcomes_are_retained_not_rerun(world, monkeypatch):
    monkeypatch.setattr(ath, "probe", _probe_ok)
    calls = []
    lost = {"reached": False, "end_reason": "infrastructure_failure", "supervisor_reason": None,
            "exit": 1, "aggregate_cpu_seconds": 0.2, "main_cpu_seconds": 0.2,
            "wall_seconds": 0.3, "within_budget": True}
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls, crash_after=4,
                                             special={(2, "ordinary", 43): lost}))
    out = world["tmp"] / "atheris"
    with pytest.raises(Crash):
        _run(world, out)
    infra = _rows(out / "atheris_results.jsonl")[1]
    assert infra["end_reason"] == "infrastructure_failure" and infra["reached"] is False
    calls2 = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls2))
    assert _run(world, out) == 0
    assert (infra["target_key"] and (QUALIFIED.index(infra["target_key"]) >= 0))
    assert len(calls2) == CELLS - 4                   # the infrastructure row is not rerun
    prepared = gio.resolve_prep(world["prep"], world["manifest"], world["tmp"])
    loaded = helpers.load_atheris(world, out / "atheris_results.jsonl",
                                  out / "atheris_contract.json", prepared)
    target = loaded["targets"][infra["target_key"]]
    assert target["status"] == "infrastructure_excluded"
    assert any("reached" in p for p in target["problems"]["ordinary::43"])
    assert loaded["counts"]["infrastructure_excluded"] == 1


def test_a_structurally_invalid_new_row_is_never_appended(world, monkeypatch):
    monkeypatch.setattr(ath, "probe", _probe_ok)
    calls = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls, special={(3, "ordinary", 44):
                                                            {"confirmations": "broken"}}))
    out = world["tmp"] / "atheris"
    with pytest.raises(SystemExit, match="REFUSED"):
        _run(world, out)
    assert len(_rows(out / "atheris_results.jsonl")) == 2
    rejected = out / "atheris_rejected_rows.jsonl"
    assert json.loads(rejected.read_bytes().decode().splitlines()[0])["row"]["confirmations"] \
        == "broken"


def test_resume_and_final_loader_share_one_validator(partial, world, monkeypatch):
    seen = []
    for module in {ar, ath.verdicts}:                 # WSL runner imports it top-level
        real = module.validate_rows
        monkeypatch.setattr(module, "validate_rows",
                            lambda *a, _real=real, **k: seen.append(k.get("complete"))
                            or _real(*a, **k))
    calls = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls))
    assert _run(world, partial["out"]) == 0
    prepared = gio.resolve_prep(world["prep"], world["manifest"], world["tmp"])
    helpers.load_atheris(world, partial["results"], partial["contract"], prepared)
    assert seen[0] is False and seen[-1] is True


@pytest.mark.parametrize("name", ["confirmations_not_a_list", "cpu_not_numeric",
                                  "unknown_field", "budget_differs_from_contract",
                                  "missing_evidence_field"])
def test_final_loader_refuses_the_same_structural_corruption(partial, world, monkeypatch, name):
    calls = []
    monkeypatch.setattr(ath, "fuzz", _fuzzer(calls))
    assert _run(world, partial["out"]) == 0
    results = partial["results"]
    results.write_bytes(CORRUPTIONS[name](results.read_bytes()))
    prepared = gio.resolve_prep(world["prep"], world["manifest"], world["tmp"])
    with pytest.raises(SystemExit, match="REFUSED"):
        helpers.load_atheris(world, results, partial["contract"], prepared)


def test_eligibility_non_run_rows_carry_no_fabricated_evidence(world):
    contract, prepared, eligibility = helpers.atheris_contract(world, ineligible=[QUALIFIED[1]])
    chash = ar.contract_hash(contract)
    key = f"{QUALIFIED[1]}::ordinary::42"
    row = {"key": key, "contract_sha256": chash, "target_key": QUALIFIED[1], "mode": "ordinary",
           "seed": 42, "eligible": False, "status": "adapter_unsupported",
           "reason": "adapter_unsupported:variadic_signature", "kill": False}
    ok = ar.validate_rows(_dump([row]), contract, qualified=QUALIFIED, seeds=SEEDS, budget=600,
                          complete=False)
    assert list(ok) == [key]
    for bad in ({**row, "kill": True}, {**row, "reached": True}):
        with pytest.raises(SystemExit, match="REFUSED"):
            ar.validate_rows(_dump([bad]), contract, qualified=QUALIFIED, seeds=SEEDS,
                             budget=600, complete=False)
