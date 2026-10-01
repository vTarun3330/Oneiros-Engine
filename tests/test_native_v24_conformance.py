"""Amendment v2.4 implementation conformance (frozen protocol unchanged):

A. live preparation-view verification (one authoritative view-manifest hash; executor over
   the 23 generation targets, Atheris over all 24 qualified targets, before any write);
B. analysis bound to the execution contract and the exact generation cells;
C. the complete engineering gate (receipts, canaries, integrity, explained failures,
   coverage, repositories);
E. Atheris applicability versus infrastructure failure, frozen in the contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from harness import native_launch_gate as gate
from scripts import native_atheris_results as ar
from scripts import native_generated_tests_analyse as an
from scripts import native_generated_tests_atheris_wsl as ath
from scripts import native_generated_tests_execute_wsl as ex
from scripts import native_generation_io as gio
from tests.test_native_v23_cohort_pipeline import (  # noqa: F401  (fixture + helpers)
    COND, EXCLUDED, GENERATED, QUALIFIED, _generate, _stub_sandbox, world)

PLAN_INT_V5 = {"params": [{"name": "x", "spec": {"kind": "prim", "type": "int"}, "keyword": False}],
               "receiver": None, "method": None}

@pytest.fixture(autouse=True)
def _fake_atheris_runtime(monkeypatch):
    """v5 resolves each target's prepared interpreter; test worlds have none."""
    from tests.native_analysis_helpers import fake_runtime
    monkeypatch.setattr(ath, "runtime_for", fake_runtime)


REPO = Path(__file__).resolve().parent.parent


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prep_rows(world):
    return {json.loads(l)["key"]: json.loads(l) for l in world["prep"].read_text().splitlines()}


def _mutate_view(world, key, label):
    view = Path(_prep_rows(world)[key]["views"][label])
    (view / "pkg" / "core.py").write_text("# drifted after preparation\n")


def _executed(world, monkeypatch):
    root = _generate(world)
    _stub_sandbox(monkeypatch)
    out = world["tmp"] / "exec"
    arms = gio.arm_paths(root, None, None, COND)
    assert ex.run(world["prep"], world["manifest"], world["job"], arms, COND, out,
                  root=world["tmp"]) == 0
    return root, out / f"results_{COND}.jsonl", out / f"execute_contract_{COND}.json"


# --- A. live preparation views ------------------------------------------------------------------

def test_view_manifest_hash_is_exactly_the_preparation_algorithm(world):
    for key, row in list(_prep_rows(world).items())[:3]:
        for label in ("buggy", "fixed"):
            view = Path(row["views"][label])
            files = {p.relative_to(view).as_posix(): _sha(p) for p in sorted(view.rglob("*"))
                     if p.is_file()}
            reference = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            assert gio.view_manifest_sha256(view) == reference == row["view_manifest_sha256"][label]


def test_view_manifest_hash_refuses_missing_or_symlinked_views(world, tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        gio.view_manifest_sha256(tmp_path / "absent")
    view = Path(_prep_rows(world)[GENERATED[0]]["views"]["buggy"])
    try:
        os.symlink(view / "pkg" / "core.py", view / "pkg" / "linked.py")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    with pytest.raises(SystemExit, match="symlink"):
        gio.view_manifest_sha256(view)


@pytest.mark.parametrize("label", ["buggy", "fixed"])
def test_executor_refuses_a_drifted_generation_view_before_execution(world, monkeypatch, label):
    root = _generate(world)
    _stub_sandbox(monkeypatch)
    _mutate_view(world, GENERATED[3], label)
    out = world["tmp"] / "exec"
    with pytest.raises(SystemExit, match="changed since preparation"):
        ex.run(world["prep"], world["manifest"], world["job"], gio.arm_paths(root, None, None, COND),
               COND, out, root=world["tmp"])
    assert not (out / f"results_{COND}.jsonl").exists()


def _probe_ok(view, module, qualname):
    return {"eligible": True, "plan": PLAN_INT_V5,
            "returns": "int", "python": "3.11.15"}


@pytest.mark.parametrize("label", ["buggy", "fixed"])
def test_atheris_refuses_drift_in_the_excluded_target_before_any_write(world, monkeypatch, label):
    _mutate_view(world, EXCLUDED, label)
    monkeypatch.setattr(ath, "probe", _probe_ok)
    out = world["tmp"] / "atheris"
    with pytest.raises(SystemExit, match="changed since preparation"):
        ath.run(world["prep"], world["manifest"], out, 600, (42, 43, 44), root=world["tmp"])
    assert not (out / "atheris_contract.json").exists()
    assert not (out / "atheris_results.jsonl").exists()


def test_atheris_verifies_both_revisions_of_all_24_qualified_targets(world, monkeypatch):
    seen = []
    real = gio.view_manifest_sha256
    monkeypatch.setattr(ath.gio, "view_manifest_sha256",
                        lambda view: seen.append(Path(view)) or real(view))
    contract, _, _ = ath.prepare_contract(world["prep"], world["manifest"], 600, (42, 43, 44),
                                          root=world["tmp"], probe_fn=_probe_ok)
    rows = _prep_rows(world)
    expected = {Path(rows[k]["views"][l]) for k in QUALIFIED for l in ("buggy", "fixed")}
    assert expected <= set(seen) and len(expected) == 48
    assert set(contract["live_views"]) == set(QUALIFIED)
    assert contract["live_views"][EXCLUDED] == rows[EXCLUDED]["view_manifest_sha256"]
    assert contract["live_views_sha256"] == gio.contract_sha(contract["live_views"])


# --- E. applicability versus infrastructure failure ----------------------------------------------

@pytest.mark.parametrize("info, status", [
    ({"eligible": True, "plan": PLAN_INT_V5, "returns": None}, "eligible"),
    ({"eligible": False, "reason": "adapter_unsupported:no_signature"}, "adapter_unsupported"),
    ({"eligible": False, "reason": "adapter_unsupported:receiver_constructor:x:Widget"},
     "adapter_unsupported"),
    ({"eligible": False, "reason": "adapter_unsupported:variadic_signature"},
     "adapter_unsupported"),
    ({"eligible": False, "reason": "adapter_unsupported:unsupported_parameter:x:Widget"},
     "adapter_unsupported"),
    # v5: an unprefixed v4 applicability reason is NOT trusted as applicability
    ({"eligible": False, "reason": "variadic_signature"}, "infrastructure_failure"),
    ({"eligible": False, "reason": "import_failure:ImportError"}, "infrastructure_failure"),
    ({"eligible": False, "reason": "runtime_mismatch:SyntaxError"}, "infrastructure_failure"),
    ({"eligible": False, "reason": "atheris_abi_unavailable"}, "infrastructure_failure"),
    ({"eligible": False, "reason": "probe_failure"}, "infrastructure_failure"),
    ({"eligible": False, "reason": "view_missing"}, "infrastructure_failure"),
    ({"eligible": False, "reason": None}, "infrastructure_failure"),
])
def test_probe_classification(info, status):
    assert ar.classify_probe(info) == status


def _probe_mixed(view, module, qualname):
    return {}


def _atheris_world(world, probe_map=None):
    """A v2.4 contract built by the real prepare_contract plus a full 216-cell result grid."""
    probe_map = probe_map or {}

    def probe(view, module, qualname):
        return dict(probe_map.get(module, _probe_ok(view, module, qualname)))
    contract, prepared, eligibility = ath.prepare_contract(
        world["prep"], world["manifest"], 600, (42, 43, 44), root=world["tmp"], probe_fn=probe)
    return contract, prepared, eligibility


def test_eligibility_is_frozen_per_target_in_the_contract(world, monkeypatch):
    rows = _prep_rows(world)
    contract, prepared, eligibility = _atheris_world(world)
    assert set(eligibility) == set(QUALIFIED)
    entry = contract["eligibility"][QUALIFIED[2]]
    for field in ("target_key", "module", "qualname", "status", "reason", "plan", "returns",
                  "views", "prep_record_sha256"):
        assert field in entry, field
    assert entry["views"] == rows[QUALIFIED[2]]["view_manifest_sha256"]
    assert entry["prep_record_sha256"] == gio.contract_sha(rows[QUALIFIED[2]])
    assert contract["eligibility_sha256"] == gio.contract_sha(contract["eligibility"])
    assert contract["prep"] == {"path": prepared["path"], "sha256": prepared["sha256"]}


def _grid(contract, eligibility, mutate=None):
    chash = ar.contract_hash(contract)
    out = []
    for t in QUALIFIED:
        e = eligibility[t]
        for m in ar.MODES:
            for seed in ar.SEEDS:
                base = {"key": f"{t}::{m}::{seed}", "contract_sha256": chash, "target_key": t,
                        "mode": m, "seed": seed, "status": e["status"], "reason": e["reason"]}
                if e["status"] != "eligible":
                    out.append({**base, "eligible": False, "kill": False})
                    continue
                from tests.native_analysis_helpers import fuzz_evidence
                out.append({**base, "eligible": True,
                            **fuzz_evidence(m, seed, end_reason="completed",
                                            supervisor_reason=None, exit=0,
                                            aggregate_cpu_seconds=500.0)})
    return mutate(out) if mutate else out


def _load(world, contract, rows, prepared):
    cpath = world["tmp"] / "atheris_contract.json"
    cpath.write_text(json.dumps(contract, sort_keys=True))
    rpath = world["tmp"] / "atheris_results.jsonl"
    rpath.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ar.load_results(
        rpath, cpath, qualified=QUALIFIED, manifest_sha256=_sha(world["manifest"]),
        prep={"path": prepared["path"], "sha256": prepared["sha256"]},
        prep_rows=prepared["rows"], script_sha256=_sha(REPO / "scripts/native_generated_tests_atheris_wsl.py"),
        inner_sha256=_sha(REPO / "scripts/native_sandbox_inner.sh"),
        verdicts_sha256=_sha(REPO / "scripts/native_atheris_results.py"))


def test_applicability_and_infrastructure_targets_are_reported_separately(world):
    rows = _prep_rows(world)
    probes = {rows[QUALIFIED[0]]["module"]: {"eligible": False, "reason": "adapter_unsupported:variadic_signature"}}
    contract, prepared, eligibility = _atheris_world(world)
    eligibility[QUALIFIED[5]] = {**eligibility[QUALIFIED[5]], "status": "adapter_unsupported",
                                 "reason": "adapter_unsupported:variadic_signature", "plan": None, "returns": None}
    eligibility[QUALIFIED[6]] = {**eligibility[QUALIFIED[6]], "status": "infrastructure_failure",
                                 "reason": "import_failure:ModuleNotFoundError", "plan": None,
                                 "returns": None}
    contract = {**contract, "eligibility": eligibility,
                "eligibility_sha256": gio.contract_sha(eligibility)}
    loaded = _load(world, contract, _grid(contract, eligibility), prepared)
    assert loaded["targets"][QUALIFIED[5]]["status"] == "adapter_unsupported"
    assert loaded["targets"][QUALIFIED[6]]["status"] == "infrastructure_excluded"
    assert loaded["counts"] == {"usable": 22, "adapter_unsupported": 1, "infrastructure_excluded": 1}
    assert probes


def test_fabricated_ineligible_rows_for_an_eligible_target_are_refused(world):
    contract, prepared, eligibility = _atheris_world(world)

    def forge(rows):
        return [{**r, "eligible": False, "status": "adapter_unsupported",
                 "reason": "adapter_unsupported:variadic_signature", "kill": False}
                if r["target_key"] == QUALIFIED[3] else r for r in rows]
    with pytest.raises(SystemExit, match="frozen eligibility"):
        _load(world, contract, _grid(contract, eligibility, forge), prepared)


def test_an_infrastructure_reason_labelled_ineligible_is_refused(world):
    contract, prepared, eligibility = _atheris_world(world)
    eligibility[QUALIFIED[4]] = {**eligibility[QUALIFIED[4]], "status": "adapter_unsupported",
                                 "reason": "probe_failure", "plan": None, "returns": None}
    contract = {**contract, "eligibility": eligibility,
                "eligibility_sha256": gio.contract_sha(eligibility)}
    with pytest.raises(SystemExit, match="eligibility"):
        _load(world, contract, _grid(contract, eligibility), prepared)


def test_contract_views_that_differ_from_preparation_are_refused(world):
    contract, prepared, eligibility = _atheris_world(world)
    live = {**contract["live_views"], EXCLUDED: {"buggy": "0" * 64, "fixed": "0" * 64}}
    contract = {**contract, "live_views": live, "live_views_sha256": gio.contract_sha(live)}
    with pytest.raises(SystemExit, match="live views"):
        _load(world, contract, _grid(contract, eligibility), prepared)


# --- B/C. analysis binding and the complete engineering gate -------------------------------------

def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=1, sort_keys=True) + "\n").encode())
    return path


def valid_receipts(world, drop=(), mutate=None) -> dict:
    """Receipts that validate against the CURRENT repository source."""
    from scripts import native_pipeline_synthetic as ps
    base = world["tmp"] / "receipts"
    receipts = {
        "full_suite": {"schema_version": "oneiros_native_full_suite_receipt_v2", "exit": 0,
                       "failed": 0, "passed": 10, "skipped": 0, "tree_clean_at_start": True,
                       "source_commit": "abc",
                       "executable_tree_sha256": gate.source_identity(REPO)["executable_tree_sha256"]},
        "synthetic_pipeline": {"schema_version": "oneiros_native_pipeline_synthetic_v1",
                               "passed": True,
                               "components_sha256": {c: _sha(REPO / c) for c in ps.COMPONENTS}},
        "sandbox_canaries": {"schema_version": "oneiros_native_sandbox_canaries_v2", "passed": True,
                             "executor_sha256": _sha(REPO / "scripts/native_generated_tests_execute_wsl.py"),
                             "inner_sha256": _sha(REPO / "scripts/native_sandbox_inner.sh"),
                             "prepare_sha256": _sha(REPO / "scripts/native_rehearsal_prepare_wsl.py")},
        "atheris_canaries": {"schema_version": "oneiros_native_atheris_canaries_v4", "passed": True,
                             "design_version": ar.DESIGN_VERSION,
                             "script_sha256": _sha(REPO / "scripts/native_generated_tests_atheris_wsl.py"),
                             "inner_sha256": _sha(REPO / "scripts/native_sandbox_inner.sh"),
                             "verdicts_sha256": _sha(REPO / "scripts/native_atheris_results.py"),
                             "checks": {k: True for k in an.ATHERIS_REQUIRED_CHECKS}},
    }
    if mutate:
        mutate(receipts)
    entries = {}
    for kind, value in receipts.items():
        if kind in drop:
            continue
        path = _write(base / f"{kind}.json", value)
        entries[kind] = {"path": path.relative_to(world["tmp"]).as_posix(), "sha256": _sha(path)}
    return entries


def evidence(world, drop=(), mutate=None, ledger=None, preflight_mutate=None) -> Path:
    receipts = valid_receipts(world, drop, mutate)
    ledger_path = _write(world["tmp"] / "ledger.json",
                         ledger if ledger is not None else
                         {"schema_version": an.LEDGER_SCHEMA, "entries": []})
    rel = lambda p: Path(p).relative_to(world["tmp"]).as_posix()  # noqa: E731
    inputs = {rel(p): _sha(p) for p in (world["manifest"], world["job"], world["prep"], ledger_path)}
    inputs.update({e["path"]: e["sha256"] for e in receipts.values()})
    pre = {"schema_version": gate.PREFLIGHT_SCHEMA, "pipeline_ready": True,
           "job": {"path": rel(world["job"])}, "manifest": {"path": rel(world["manifest"])},
           "inputs": inputs,
           "gate_evidence": {"receipts": receipts,
                             "quarantine_ledger": {"path": rel(ledger_path),
                                                   "sha256": _sha(ledger_path)}}}
    if preflight_mutate:
        preflight_mutate(pre)
    return _write(world["tmp"] / "preflight_v2_4.json", pre)


def analyse(world, results, contract, root, preflight=None, extra=(), out="analysis.json",
            generations=None):
    args = ["analyse", "--manifest", str(world["manifest"]), "--job", str(world["job"]),
            "--prep", str(world["prep"]), "--preflight", str(preflight or evidence(world)),
            "--execution-contract", str(contract), "--results", str(results),
            *(generations or ["--generations", str(root)]), "--condition", COND,
            "--study-mode", "engineering_dress_rehearsal", "--out", str(world["tmp"] / out),
            *extra]
    an.main(args, root=world["tmp"])
    return json.loads((world["tmp"] / out).read_text())


def test_complete_evidence_passes_every_subgate(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    result = analyse(world, results, contract, root)
    gates = result["engineering_gate"]["subgates"]
    assert gates == {"coverage_gate_passed": True, "repository_gate_passed": True,
                     "stage_receipts_gate_passed": True, "canaries_gate_passed": True,
                     "artifact_integrity_gate_passed": True,
                     "unexplained_failures_gate_passed": True}
    assert result["engineering_gate_passed"] is True and "kill_at_8" in result


@pytest.mark.parametrize("kind, subgate", [
    ("full_suite", "stage_receipts_gate_passed"),
    ("synthetic_pipeline", "stage_receipts_gate_passed"),
    ("sandbox_canaries", "canaries_gate_passed"),
    ("atheris_canaries", "canaries_gate_passed"),
])
def test_missing_receipt_fails_its_subgate_and_suppresses_comparisons(world, monkeypatch, kind,
                                                                     subgate):
    root, results, contract = _executed(world, monkeypatch)
    result = analyse(world, results, contract, root, preflight=evidence(world, drop=(kind,)))
    assert result["engineering_gate"]["subgates"][subgate] is False
    assert result["engineering_gate_passed"] is False
    assert "kill_at_8" not in result and result["arm_comparison"].startswith("SUPPRESSED")


@pytest.mark.parametrize("change", [
    lambda r: r["full_suite"].update(failed=1),
    lambda r: r["full_suite"].update(executable_tree_sha256="0" * 64),
    lambda r: r["synthetic_pipeline"].update(passed=False),
    lambda r: r["synthetic_pipeline"]["components_sha256"].update(
        {"scripts/native_generation_io.py": "0" * 64}),
    lambda r: r["sandbox_canaries"].update(executor_sha256="0" * 64),
    lambda r: r["atheris_canaries"].update(schema_version="oneiros_native_atheris_canaries_v2"),
    lambda r: r["atheris_canaries"]["checks"].update(ordinary_different_exceptions_not_a_kill=False),
])
def test_failing_or_unbound_receipts_fail_the_gate(world, monkeypatch, change):
    root, results, contract = _executed(world, monkeypatch)
    result = analyse(world, results, contract, root, preflight=evidence(world, mutate=change))
    assert result["engineering_gate_passed"] is False and "kill_at_8" not in result


def test_receipt_whose_bytes_differ_from_the_preflight_fails(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    pre = evidence(world)
    receipt = world["tmp"] / "receipts" / "sandbox_canaries.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    result = analyse(world, results, contract, root, preflight=pre)
    assert result["engineering_gate"]["subgates"]["canaries_gate_passed"] is False
    assert result["engineering_gate"]["subgates"]["stage_receipts_gate_passed"] is False


def test_unexplained_quarantine_fails_and_a_ledger_entry_explains_it(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    q = root / "base" / "quarantine"
    q.mkdir()
    (q / "123_generations_primary_whole_module_base.jsonl").write_text("stale\n")
    result = analyse(world, results, contract, root)
    assert result["engineering_gate"]["subgates"]["unexplained_failures_gate_passed"] is False
    stale = q / "123_generations_primary_whole_module_base.jsonl"
    explained = {"schema_version": an.LEDGER_SCHEMA, "entries": [{
        "artifact": stale.relative_to(world["tmp"]).as_posix(), "sha256": _sha(stale),
        "reason": "partial line after an interrupted run", "corrective_change": "resume",
        "superseded_by": {"path": results.relative_to(world["tmp"]).as_posix(),
                          "sha256": _sha(results)}}]}
    result = analyse(world, results, contract, root, preflight=evidence(world, ledger=explained),
                     out="b.json")
    assert result["engineering_gate"]["subgates"]["unexplained_failures_gate_passed"] is True


def test_malformed_ledger_entry_fails_the_unexplained_subgate(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    bad = {"schema_version": an.LEDGER_SCHEMA, "entries": [{"artifact": "x", "reason": ""}]}
    result = analyse(world, results, contract, root, preflight=evidence(world, ledger=bad))
    assert result["engineering_gate"]["subgates"]["unexplained_failures_gate_passed"] is False


def test_invalid_preflight_fails_the_stage_receipts_subgate(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    for mutate in (lambda p: p.update(pipeline_ready=False),
                   lambda p: p.update(schema_version="oneiros_native_generated_tests_preflight_v2_3"),
                   lambda p: p["inputs"].update({"results/elsewhere.json": "0" * 64})):
        result = analyse(world, results, contract, root, preflight=evidence(world, preflight_mutate=mutate))
        assert result["engineering_gate"]["subgates"]["stage_receipts_gate_passed"] is False


def _rows(results):
    return [json.loads(l) for l in results.read_text().splitlines()]


def _rewrite(results, rows):
    results.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def test_missing_or_altered_execution_contract_is_refused(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    moved = contract.with_suffix(".bak")
    contract.rename(moved)
    with pytest.raises((an.AnalysisRefused, SystemExit), match="execution contract"):
        analyse(world, results, contract, root)
    data = json.loads(moved.read_text())
    data["limits"] = {**data["limits"], "cpu_seconds": 999}
    contract.write_text(json.dumps(data))
    with pytest.raises((an.AnalysisRefused, SystemExit), match="contract"):
        analyse(world, results, contract, root)


@pytest.mark.parametrize("field, value", [
    ("job_sha256", "0" * 64), ("manifest_sha256", "0" * 64),
    ("prep", {"path": "prep/records.jsonl", "sha256": "0" * 64}),
    ("executor_sha256", "0" * 64), ("telemetry_schema", "v1"),
])
def test_execution_contract_bound_to_other_artifacts_is_refused(world, monkeypatch, field, value):
    root, results, contract = _executed(world, monkeypatch)
    data = json.loads(contract.read_text())
    data[field] = value
    contract.write_text(json.dumps(data))
    chash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    _rewrite(results, [{**r, "contract_sha256": chash} for r in _rows(results)])
    with pytest.raises((an.AnalysisRefused, SystemExit), match=field.split("_")[0]):
        analyse(world, results, contract, root)


def test_swapped_or_altered_generation_artifacts_are_refused(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    (root / "base").rename(root / "tmp")
    (root / "sft").rename(root / "base")
    (root / "tmp").rename(root / "sft")
    with pytest.raises(an.AnalysisRefused, match="missing|swapped"):
        analyse(world, results, contract, root)


def test_generation_file_changed_after_execution_is_refused(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    path = root / "base" / f"generations_{COND}_base.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    c = rows[0]["candidates"][1]
    c["raw"] = c["module"] = "def test_other():\n    assert 2 == 2\n"
    c["raw_sha256"] = c["module_sha256"] = hashlib.sha256(c["raw"].encode()).hexdigest()
    c["fence_stripped"] = False
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    with pytest.raises((an.AnalysisRefused, SystemExit), match="generation"):
        analyse(world, results, contract, root)


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r.update(module_sha256="0" * 64), "module"),
    (lambda r: r["generation"].update(generated_tokens=r["generation"]["generated_tokens"] + 1),
     "telemetry"),
    (lambda r: r["generation"].update(target_seed=1), "telemetry"),
    (lambda r: r["evidence"]["static"].update(counts={"tests": 9, "asserts": 9,
                                                      "target_calls": 0}), "static"),
])
def test_altered_execution_cells_are_refused(world, monkeypatch, mutate, fragment):
    root, results, contract = _executed(world, monkeypatch)
    rows = _rows(results)
    mutate(rows[7])
    _rewrite(results, rows)
    with pytest.raises(an.AnalysisRefused, match=fragment):
        analyse(world, results, contract, root)


def test_internally_consistent_evidence_from_another_candidate_is_refused(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    rows = _rows(results)
    kill = next(i for i, r in enumerate(rows) if r["class"] == "semantic_kill")
    other = next(i for i, r in enumerate(rows) if r["class"] == "pass_both"
                 and r["arm"] == rows[kill]["arm"] and r["target_key"] == rows[kill]["target_key"])
    forged = {**rows[kill], **{k: rows[other][k] for k in ("key", "slot", "seed", "module_sha256",
                                                           "generation")}}
    rows[other] = forged                       # the kill's evidence, the pass candidate's identity
    _rewrite(results, rows)
    with pytest.raises(an.AnalysisRefused, match="candidate"):
        analyse(world, results, contract, root)


def test_missing_duplicate_extra_or_partial_execution_rows_are_refused(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    rows = _rows(results)
    for bad in (rows[:-1], rows + rows[:1],
                rows + [{**rows[0], "key": f"sft::{EXCLUDED}::42::0", "target_key": EXCLUDED}]):
        _rewrite(results, bad)
        with pytest.raises(an.AnalysisRefused):
            analyse(world, results, contract, root)
    _rewrite(results, rows)
    results.write_bytes(results.read_bytes() + b'{"key"')
    with pytest.raises(an.AnalysisRefused, match="partial"):
        analyse(world, results, contract, root)


def test_executor_resume_quarantines_rows_that_do_not_match_their_candidate(world, monkeypatch):
    root, results, contract = _executed(world, monkeypatch)
    rows = _rows(results)
    rows[3]["class"] = rows[3]["classification"]["class"] = "crash_kill"   # forged prior row
    _rewrite(results, rows)
    assert ex.run(world["prep"], world["manifest"], world["job"],
                  gio.arm_paths(root, None, None, COND), COND, results.parent,
                  root=world["tmp"]) == 0
    assert list((results.parent / "quarantine").iterdir())
    assert all(ex.verify_row(r) == [] for r in _rows(results))
