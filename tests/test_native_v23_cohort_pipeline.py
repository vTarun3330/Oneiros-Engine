"""Amendment v2.3 end-to-end pre-GPU pipeline on CPU: 24 qualified targets, a job admitting 23,
one pre-generation overflow exclusion, separate base/ and sft/ generation directories with full
telemetry, 1,104 execution rows, analysis over exactly 23 targets and Atheris joining.

Only the sandboxed candidate execution is stubbed (it needs WSL); generation (mock backend),
generation loading, the executor's cohort/grid/contract logic, the analysis CLI and the
Atheris join are the real code. The old behaviour (iterating all kept targets) fails here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import native_generated_tests_analyse as an
from scripts import native_generated_tests_execute_wsl as ex
from scripts import native_generated_tests_generate as gen
from scripts import native_generation_io as gio

COND = "primary_whole_module"
QUALIFIED = [f"cand:repo{i // 3}/proj@{i:040d}" for i in range(24)]
EXCLUDED = QUALIFIED[17]                       # like tomlkit: overflow, never generated
GENERATED = [k for k in QUALIFIED if k != EXCLUDED]


def _prompt(key: str) -> str:
    return f"### TEST GENERATION TASK\nTarget {key}\n" + "context " * 20


def _job_file(path: Path) -> bytes:
    sealed = [{"target_key": k, "prompt": _prompt(k) + ("x " * 5000 if k == EXCLUDED else ""),
               "condition": "whole_module"} for k in QUALIFIED]
    for s in sealed:
        s["prompt_sha256"] = hashlib.sha256(s["prompt"].encode()).hexdigest()
    fit = gen.sequence_fit(sealed, lambda text: len(text.split()), limit=2048)
    job = gen.build_job(sealed, {s["target_key"]: {"ok": True} for s in sealed}, fit)
    data = (json.dumps({COND: job}, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


@pytest.fixture
def world(tmp_path):
    job_path = tmp_path / "job_v4.json"
    job_bytes = _job_file(job_path)
    targets = [{"key": k, "repository": k.split("/")[0][5:]} for k in QUALIFIED]
    fields = gio.cohort_fields(targets, "job_v4.json", job_bytes, COND,
                               {EXCLUDED: {"prompt_tokens": 5027, "prompt_token_limit": 2048}})
    prep = []
    for k in QUALIFIED:
        views = {}
        for label in ("buggy", "fixed"):
            v = tmp_path / "views" / hashlib.sha256(k.encode()).hexdigest()[:8] / label / "pkg"
            v.mkdir(parents=True)
            (v / "core.py").write_text(f"# {label} {k}\n")
            views[label] = str(v.parent)
        prep.append({"key": k, "views": views, "module": "pkg.core", "qualname": "f",
                     "category": "requalified", "interpreter": "/usr/bin/python3.11",
                     "attestation": {"buggy": "b", "fixed": "f"},
                     "environment_lock": {"freeze_sha256": "e"},
                     "python_path": "/usr/bin/python3.11", "env_dir": "/env",
                     "module_sha256": {"buggy": "b", "fixed": "f"},
                     "view_manifest_sha256": {l: ex.build_view_hash(Path(views[l]))
                                              for l in views}})
    prep_path = tmp_path / "prep" / "records.jsonl"
    prep_path.parent.mkdir()
    prep_path.write_text("".join(json.dumps(r) + "\n" for r in prep))
    manifest = {"kept_targets": QUALIFIED, "targets": targets, **fields,
                "nature": "ENGINEERING DRESS REHEARSAL ONLY",
                "study_mode": "engineering_dress_rehearsal",
                "requalification_records": {
                    "path": "prep/records.jsonl",
                    "sha256": hashlib.sha256(prep_path.read_bytes()).hexdigest()}}
    manifest_path = tmp_path / "manifest_v6.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return {"tmp": tmp_path, "job": job_path, "manifest": manifest_path, "prep": prep_path,
            "job_file_sha256": hashlib.sha256(job_bytes).hexdigest()}


def _identity(world, arm, **over):
    job = json.loads(world["job"].read_text())[COND]
    return {"backend": "mock", "condition": COND, "arm": arm, "source_commit": "abc",
            "job_file_sha256": world["job_file_sha256"], "job_sha256": job["job_sha256"],
            "preflight_sha256": "pre", "authorization_sha256": "auth",
            "model": {"snapshot_manifest_sha256": "m"},
            "adapter_manifest_sha256": None if arm == "base" else "adapter", **over}


def _sft_backend(prompt, n, seed):
    """Mock telemetry; the SFT arm's first candidate is a designed kill."""
    result = gen.mock_backend(prompt, n, seed)
    first = result["batches"][0]["candidates"][0]
    first.update(raw="def test_kill():\n    assert 'KILL'\n", generated_tokens=9, eos_reached=True)
    return result


def _generate(world, root=None, over=None):
    over = over or {}
    root = root or world["tmp"] / "generations"
    job = gen.select_condition(json.loads(world["job"].read_text()), COND)
    for arm, backend in (("base", gen.mock_backend), ("sft", _sft_backend)):
        gen.run(job, arm, COND, root / arm, _identity(world, arm, **over.get(arm, {})), backend)
    return root


def _stub_sandbox(monkeypatch):
    """Only the WSL sandbox process is replaced; execution, canary and evidence are real."""
    from tests.native_fake_sandbox import fake_run_sandboxed
    monkeypatch.setattr(ex, "run_sandboxed", fake_run_sandboxed)


def _execute(world, root, monkeypatch, out=None):
    _stub_sandbox(monkeypatch)
    out = out or world["tmp"] / "exec"
    arms = ex.gio.arm_paths(root, None, None, COND)
    code = ex.run(world["prep"], world["manifest"], world["job"], arms, COND, out,
                  root=world["tmp"])
    return code, out / f"results_{COND}.jsonl"


# --- the full pre-GPU pipeline ------------------------------------------------------------------

def test_24_qualified_23_generated_pipeline_end_to_end(world, monkeypatch):
    from harness.acquisition_receipt import ProtectedAccessMonitor
    ProtectedAccessMonitor.install(Path(__file__).resolve().parent.parent)
    mark = ProtectedAccessMonitor.mark()
    root = _generate(world)
    loaded = _load(world, root)                    # raw hashes and extraction re-verified
    by_cell = {}
    for (arm, target, seed), row in loaded["rows"].items():
        for slot, cand in enumerate(row["candidates"]):
            assert gio.extract(cand["raw"])["module"] == cand["module"]
            by_cell[f"{arm}::{target}::{seed}::{slot}"] = cand
    for arm in ("base", "sft"):                 # the exact proposed separate-arm layout
        lines = (root / arm / f"generations_{COND}_{arm}.jsonl").read_text().splitlines()
        assert len(lines) == 69 and (root / arm / f"contract_{COND}_{arm}.json").is_file()
    code, results = _execute(world, root, monkeypatch)
    rows = [json.loads(l) for l in results.read_text().splitlines()]
    assert code == 0 and len(rows) == 1104 and len({r["key"] for r in rows}) == 1104
    assert {r["target_key"] for r in rows} == set(GENERATED)
    assert not any(r["target_key"] == EXCLUDED for r in rows)
    contract = json.loads((world["tmp"] / "exec" / f"execute_contract_{COND}.json").read_text())
    assert contract["job_file_sha256"] == world["job_file_sha256"]
    assert contract["cohort"]["qualified"] == 24 and len(contract["cohort"]["generation_targets"]) == 23
    assert set(contract["generations_sha256"]) == {"base", "sft"}
    # completion-limit evidence survives generation -> execution (mock: every 4th candidate)
    assert sum(r["generation"]["hit_completion_limit"] for r in rows) == 1104 // 4
    for r in rows:                                  # every execution row is its generation cell
        cand = by_cell[r["key"]]
        assert r["module_sha256"] == cand["module_sha256"]
        assert {k: r["generation"][k] for k in ex.GENERATION_FIELDS} == \
            {k: cand[k] for k in ex.GENERATION_FIELDS}
    assert contract["generation_identity_sha256"] == loaded["identity_sha256"]
    assert contract["telemetry_schema"] == gio.TELEMETRY_SCHEMA
    atheris_path, atheris_contract = _atheris_fixture(world)
    from tests import native_analysis_helpers as helpers
    result = helpers.analyse(world, results, results.parent / f"execute_contract_{COND}.json",
                             root, extra=["--atheris", str(atheris_path),
                                          "--atheris-contract", str(atheris_contract)])
    cohort = result["cohort"]
    assert (cohort["qualified_targets"], cohort["generation_targets"],
            cohort["pre_generation_excluded"]) == (24, 23, 1)
    assert cohort["post_generation_infrastructure_excluded"] == 0
    assert cohort["pre_generation_exclusions"][0]["target_key"] == EXCLUDED
    assert result["grid_cells"] == 1104 and result["eligible_targets"] == 23
    tele = result["generation_telemetry"]
    for arm in ("base", "sft"):
        assert tele[arm]["candidates"] == 552 and tele[arm]["target_seed_latency_seconds"]["rows"] == 69
        assert set(tele[arm]["generated_tokens"]) == {"p50", "p90", "p99", "max"}
    assert tele["base"]["completion_limit_hits"] == 138 and tele["base"]["completion_limit_hit_rate"] == 25.0
    assert tele["sft"]["completion_limit_hits"] == 138
    assert result["fixed_valid_rate"]["completion_limit_hit_rate"] == {"base": 25.0, "sft": 25.0}
    assert result["unique_bugs_killed"] == {"base": 0, "sft": 23}
    assert "SUPPRESSED" in result["decisions"] and "primary" not in result
    # Atheris: 24 rows; the joint comparison is restricted to generated AND eligible targets
    den = result["atheris_denominators"]
    assert {k: v for k, v in den.items() if k != "rule"} == {
        "qualified": 24, "atheris_cells": 216, "atheris_usable": 23, "atheris_ineligible": 1,
        "atheris_infrastructure_excluded": 0, "generation_targets": 23,
        "infrastructure_eligible": 23, "joint": 22}
    assert result["atheris_jointly_eligible"]["atheris_unique_kills"] == {
        "ordinary": 1, "posthoc": 0, "differential": 0}
    assert result["atheris_only_not_generated"]["targets"] == [EXCLUDED]
    assert result["atheris_only_not_generated"]["kill_by_mode"][EXCLUDED]["ordinary"] is True
    assert result["engineering_gate_passed"] is True
    assert result["engineering_gate"]["repositories"] == 8
    assert result["inputs"]["job"] == world["job_file_sha256"]
    for claim in ("root_cause_established", "generalization_established",
                  "sft_benefit_established", "atheris_superiority_established"):
        assert result[claim] is False
    # the excluded target is in no model denominator and never a model failure
    for arm in ("base", "sft"):
        assert result["denominators"][arm]["requested"] == 552
        assert sum(result["denominators"][arm]["classes"].values()) == 552
    assert result["infrastructure"]["excluded_targets"] == []
    assert not ProtectedAccessMonitor.evidence(mark)["protected_paths_opened"]


def test_the_old_all_kept_targets_behaviour_fails(world, monkeypatch):
    """Regression: iterating manifest['kept_targets'] (24) expects 1,152 cells, 48 of which can
    never exist, so both the old executor grid and the old analysis refuse."""
    root = _generate(world)
    code, results = _execute(world, root, monkeypatch)
    rows = [json.loads(l) for l in results.read_text().splitlines()]
    with pytest.raises(an.AnalysisRefused, match="incomplete grid: 48 missing"):
        an.analyse(rows, QUALIFIED, {k: "r" for k in QUALIFIED}, "engineering_dress_rehearsal")
    old_manifest = world["tmp"] / "old_manifest.json"
    old_manifest.write_text(json.dumps({"kept_targets": QUALIFIED,
                                        "targets": json.loads(world["manifest"].read_text())["targets"]}))
    with pytest.raises(SystemExit, match="never inferred from kept targets"):
        gio.resolve_cohort(world["job"], old_manifest)
    with pytest.raises(SystemExit, match="never inferred"):
        ex.run(world["prep"], old_manifest, world["job"], ex.gio.arm_paths(root, None, None, COND),
               COND, world["tmp"] / "exec_old", root=world["tmp"])


def test_explicit_per_arm_paths_are_equivalent(world, monkeypatch):
    root = _generate(world)
    _stub_sandbox(monkeypatch)
    arms = ex.gio.arm_paths(None, root / "base", root / "sft", COND)
    assert ex.run(world["prep"], world["manifest"], world["job"], arms, COND,
                  world["tmp"] / "exec2", root=world["tmp"]) == 0
    with pytest.raises(SystemExit, match="give --generations ROOT or both"):
        ex.gio.arm_paths(root, root / "base", None, COND)
    with pytest.raises(SystemExit, match="are the same"):
        ex.gio.arm_paths(None, root / "base", root / "base", COND)


def test_executor_cli_accepts_the_generation_root(world, monkeypatch):
    root = _generate(world)
    _stub_sandbox(monkeypatch)
    monkeypatch.setattr(ex, "REPO_ROOT", world["tmp"])
    assert ex.main(["run", "--prep", str(world["prep"]), "--manifest", str(world["manifest"]),
                    "--job", str(world["job"]), "--generations", str(root),
                    "--condition", COND, "--out", str(world["tmp"] / "exec3")]) == 0


# --- generation-layout and loader refusals -----------------------------------------------------

def _load(world, root):
    cohort = gio.resolve_cohort(world["job"], world["manifest"])
    return gio.load_arm_generations(gio.arm_paths(root, None, None, COND), cohort)


def test_swapped_arms_are_refused(world):
    root = _generate(world)
    (root / "base").rename(root / "tmp")
    (root / "sft").rename(root / "base")
    (root / "tmp").rename(root / "sft")
    with pytest.raises(SystemExit, match="missing|swapped"):
        _load(world, root)
    for arm, other in (("base", "sft"), ("sft", "base")):   # files renamed to match the dirs
        for kind in ("generations", "contract"):
            ext = "jsonl" if kind == "generations" else "json"
            (root / arm / f"{kind}_{COND}_{other}.{ext}").rename(
                root / arm / f"{kind}_{COND}_{arm}.{ext}")
    with pytest.raises(SystemExit, match="swapped"):
        _load(world, root)


@pytest.mark.parametrize("field, value, fragment", [
    ("preflight_sha256", "other-preflight", "differ beyond the arm"),
    ("source_commit", "other-source", "differ beyond the arm"),
    ("model", {"snapshot_manifest_sha256": "other"}, "differ beyond the arm"),
    ("job_sha256", "0" * 64, "job differs"),
])
def test_arms_from_different_runs_are_refused(world, field, value, fragment):
    root = _generate(world, over={"sft": {field: value}})
    with pytest.raises(SystemExit, match=fragment):
        _load(world, root)


def _rewrite(path: Path, mutate) -> None:
    lines = path.read_text().splitlines()
    path.write_text("".join(l + "\n" for l in mutate(lines)))


@pytest.mark.parametrize("mutate, fragment", [
    (lambda ls: ls[:-1], "incomplete"),
    (lambda ls: ls + ls[:1], "duplicate"),
    (lambda ls: ls + [json.dumps({**json.loads(ls[0]), "key": f"{EXCLUDED}::42",
                                  "target_key": EXCLUDED})], "unexpected"),
    (lambda ls: [json.dumps({**json.loads(ls[0]), "identity_sha256": "stale"})] + ls[1:],
     "identity"),
    (lambda ls: [json.dumps({**json.loads(ls[0]), "wall_seconds": 99.0})] + ls[1:],
     "sum of its batches"),
    (lambda ls: [json.dumps({k: v for k, v in json.loads(ls[0]).items()
                             if k != "prompt_tokens"})] + ls[1:], "missing"),
    (lambda ls: [json.dumps({**json.loads(ls[0]), "target_seed": 1})] + ls[1:],
     "target_seed does not recompute"),
    (lambda ls: [json.dumps({**json.loads(ls[0]), "prompt_tokens": 2049})] + ls[1:],
     "prompt_tokens outside"),
    (lambda ls: [json.dumps({**json.loads(ls[0]), "prompt_sha256": "0" * 64})] + ls[1:],
     "prompt hash differs"),
])
def test_missing_duplicated_extra_stale_or_malformed_rows_are_refused(world, mutate, fragment):
    root = _generate(world)
    _rewrite(root / "base" / f"generations_{COND}_base.jsonl", mutate)
    with pytest.raises(SystemExit, match=fragment):
        _load(world, root)


def test_partial_line_and_missing_contract_are_refused(world):
    root = _generate(world)
    path = root / "sft" / f"generations_{COND}_sft.jsonl"
    path.write_bytes(path.read_bytes() + b'{"key": "x"')
    with pytest.raises(SystemExit, match="partial line"):
        _load(world, root)
    root2 = _generate(world, root=world["tmp"] / "g2")
    (root2 / "base" / f"contract_{COND}_base.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        _load(world, root2)


def test_generation_files_are_never_merged_or_copied(world):
    source = (Path(__file__).resolve().parent.parent / "scripts" /
              "native_generated_tests_execute_wsl.py").read_text(encoding="utf-8")
    run_source = source.split("def run(", 1)[1].split("\ndef ", 1)[0]
    assert "shutil.copy" not in run_source and "kept_targets" not in run_source


# --- preparation binding (amendment v2.4 section C) ---------------------------------------------

def _prep_rows(world):
    return [json.loads(l) for l in world["prep"].read_text().splitlines()]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_exact_declared_preparation_file_is_accepted(world):
    prepared = gio.resolve_prep(world["prep"], world["manifest"], world["tmp"])
    assert len(prepared["rows"]) == 24 and prepared["path"] == "prep/records.jsonl"


@pytest.mark.parametrize("case, fragment", [
    ("copied", "not the manifest-declared"),
    ("outside_root", "not beneath the repository root"),
    ("edited", "differ from the manifest-declared hash"),
    ("stale", "differ from the manifest-declared hash"),
])
def test_copied_edited_stale_or_misplaced_preparation_is_refused(world, case, fragment,
                                                                 tmp_path_factory):
    prep = world["prep"]
    if case == "copied":
        prep = world["tmp"] / "copy_records.jsonl"
        shutil.copy(world["prep"], prep)
    elif case == "outside_root":
        prep = tmp_path_factory.mktemp("elsewhere") / "records.jsonl"
        shutil.copy(world["prep"], prep)
    elif case == "edited":
        rows = _prep_rows(world)
        rows[0]["env_dir"] = "/other"
        _write(prep, rows)
    else:
        prep.write_bytes(prep.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match=fragment):
        gio.resolve_prep(prep, world["manifest"], world["tmp"])


@pytest.mark.parametrize("mutate, fragment", [
    (lambda rows: rows + rows[:1], "duplicate preparation record"),
    (lambda rows: rows[1:], "do not match"),
    (lambda rows: rows + [{**rows[0], "key": "cand:extra/x@1"}], "do not match"),
    (lambda rows: [{**rows[0], "category": "not_stable"}] + rows[1:], "is 'not_stable'"),
    (lambda rows: [{k: v for k, v in rows[0].items() if k != "attestation"}] + rows[1:], "lacks"),
    (lambda rows: [{**rows[0], "views": {"buggy": "/v", "fixed": None}}] + rows[1:], "lacks views"),
    (lambda rows: [{**rows[0], "module_sha256": None}] + rows[1:], "lacks"),
])
def test_wrong_category_duplicate_or_incomplete_preparation_is_refused(world, mutate, fragment):
    _write(world["prep"], mutate(_prep_rows(world)))
    manifest = json.loads(world["manifest"].read_text())       # re-declare the hash so only
    manifest["requalification_records"]["sha256"] = hashlib.sha256(  # the content check fires
        world["prep"].read_bytes()).hexdigest()
    world["manifest"].write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(SystemExit, match=fragment):
        gio.resolve_prep(world["prep"], world["manifest"], world["tmp"])


def test_executor_refuses_an_undeclared_preparation_and_binds_the_declared_one(world, monkeypatch):
    root = _generate(world)
    _stub_sandbox(monkeypatch)
    copy = world["tmp"] / "copy.jsonl"
    shutil.copy(world["prep"], copy)
    arms = ex.gio.arm_paths(root, None, None, COND)
    with pytest.raises(SystemExit, match="not the manifest-declared"):
        ex.run(copy, world["manifest"], world["job"], arms, COND, world["tmp"] / "e",
               root=world["tmp"])
    assert ex.run(world["prep"], world["manifest"], world["job"], arms, COND, world["tmp"] / "e",
                  root=world["tmp"]) == 0
    contract = json.loads((world["tmp"] / "e" / f"execute_contract_{COND}.json").read_text())
    assert contract["prep"] == {"path": "prep/records.jsonl",
                                "sha256": hashlib.sha256(world["prep"].read_bytes()).hexdigest()}


# --- auditable execution evidence (amendment v2.4 section D) ------------------------------------

def _executed_rows(world, monkeypatch):
    root = _generate(world)
    code, results = _execute(world, root, monkeypatch)
    assert code == 0
    return [json.loads(l) for l in results.read_text().splitlines()]


def test_every_row_retains_sanitised_evidence_that_reproduces_its_class(world, monkeypatch):
    rows = _executed_rows(world, monkeypatch)
    kills = [r for r in rows if r["class"] == "semantic_kill"]
    assert len(kills) == 69 and all(r["classification"]["rerun_agrees"] for r in kills)
    for r in rows:
        assert ex.verify_row(r) == [], r["key"]
        reports = r["evidence"]["reports"]
        assert set(reports) >= {"buggy", "fixed"}
        for rep in reports.values():
            assert "tail" not in rep and rep["uid"] == 65534 and rep["raw_report_sha256"]
            assert rep["attestation"]["module_file"].startswith("/target/")
        assert ("rerun_buggy" in reports) == (r["class"] == "semantic_kill")
        assert r["evidence"]["canary"]["ok"] is True
        assert "host-only" not in json.dumps(r)


def _kill(rows):
    return json.loads(json.dumps(next(r for r in rows if r["class"] == "semantic_kill")))


def _pass(rows):
    return json.loads(json.dumps(next(r for r in rows if r["class"] == "pass_both")))


def test_altered_semantic_kill_label_is_refused(world, monkeypatch):
    row = _kill(_executed_rows(world, monkeypatch))
    row["class"] = row["classification"]["class"] = "crash_kill"
    assert any("disagrees with evidence" in p for p in ex.verify_row(row))


def test_altered_crash_evidence_is_refused(world, monkeypatch):
    row = _kill(_executed_rows(world, monkeypatch))
    for key in ("buggy", "rerun_buggy"):
        exc = row["evidence"]["reports"][key]["nodes"][next(iter(
            row["evidence"]["reports"][key]["nodes"]))]["phases"]["call"]["exception"]
        exc.update(assertion=False, target_in_traceback=True)       # evidence now says crash
    assert any("('crash_kill')" in p for p in ex.verify_row(row))


def test_altered_fixed_valid_is_refused(world, monkeypatch):
    row = _pass(_executed_rows(world, monkeypatch))
    row["fixed_valid"] = False
    assert ex.verify_row(row) == ["fixed_valid disagrees with evidence"]


def test_altered_target_reached_is_refused(world, monkeypatch):
    row = _pass(_executed_rows(world, monkeypatch))
    node = next(iter(row["evidence"]["reports"]["fixed"]["nodes"].values()))
    node["reached"] = False
    assert any("('target_not_reached')" in p for p in ex.verify_row(row))


def test_altered_or_missing_rerun_is_refused(world, monkeypatch):
    rows = _executed_rows(world, monkeypatch)
    row = _kill(rows)
    del row["evidence"]["reports"]["rerun_buggy"], row["evidence"]["reports"]["rerun_fixed"]
    problems = ex.verify_row(row)
    assert "kill without retained rerun evidence" in problems
    row = _kill(rows)
    call = next(iter(row["evidence"]["reports"]["rerun_buggy"]["nodes"].values()))["phases"]["call"]
    call.update(outcome="passed", exception=None)                   # rerun no longer agrees
    assert any("('nondeterminism')" in p for p in ex.verify_row(row))


def test_canary_and_missing_evidence_are_refused(world, monkeypatch):
    rows = _executed_rows(world, monkeypatch)
    row = _pass(rows)
    row["canary_failed"] = True
    assert "canary_failed disagrees with the canary evidence" in ex.verify_row(row)
    row = _pass(rows)
    del row["evidence"]
    assert ex.verify_row(row) == ["no retained evidence"]
    row = _pass(rows)
    row["class"] = row["classification"]["class"] = "semantic_kill"
    assert any("disagrees with evidence" in p for p in ex.verify_row(row))


def test_static_evidence_is_checked_against_the_recoverable_module(world, monkeypatch):
    rows = _executed_rows(world, monkeypatch)
    root = world["tmp"] / "generations"
    loaded = _load(world, root)
    row = _pass(rows)
    arm, target, seed, slot = row["key"].split("::")
    module = loaded["rows"][(arm, target, int(seed))]["candidates"][int(slot)]["module"]
    assert ex.verify_row(row, module, "f") == []
    assert "static evidence disagrees with the module" in ex.verify_row(row, "import os\n", "f")


def test_analysis_refuses_a_tampered_execution_row(world, monkeypatch):
    rows = _executed_rows(world, monkeypatch)
    rows[5]["fixed_valid"] = not rows[5]["fixed_valid"]
    results = world["tmp"] / "tampered.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from tests import native_analysis_helpers as helpers
    shutil.copy(world["tmp"] / "exec" / f"execute_contract_{COND}.json",
                world["tmp"] / f"execute_contract_{COND}.json")
    with pytest.raises(an.AnalysisRefused, match="fails verification"):
        helpers.analyse(world, results, world["tmp"] / f"execute_contract_{COND}.json",
                        world["tmp"] / "generations", out="a.json")



# --- Atheris design v4 (amendment v2.4 section E) ----------------------------------------------

from scripts import native_atheris_results as ar  # noqa: E402

RAISE, OTHER, OK = ["raise", "ValueError"], ["raise", "TypeError"], ["ok", ["int", 1]]
KILLED = (EXCLUDED, QUALIFIED[1])


def _file_sha(rel):
    return hashlib.sha256((Path(__file__).resolve().parent.parent / rel).read_bytes()).hexdigest()


def _atheris_contract(world):
    """A real v2.4 contract (live views + frozen eligibility; QUALIFIED[0] is legitimately
    Atheris-ineligible) built by prepare_contract with a fake probe."""
    from tests import native_analysis_helpers as helpers
    return helpers.atheris_contract(world, ineligible=(QUALIFIED[0],))[0]


def _atheris_rows(chash, contract):
    rows = []
    for t in QUALIFIED:
        e = contract["eligibility"][t]
        for m in ar.MODES:
            for seed in ar.SEEDS:
                base = {"key": f"{t}::{m}::{seed}", "contract_sha256": chash, "target_key": t,
                        "mode": m, "seed": seed, "status": e["status"], "reason": e["reason"]}
                if e["status"] != "eligible":
                    rows.append({**base, "eligible": False, "kill": False})
                    continue
                checks = [{"witness": "w0", **ar.judge(m, [RAISE, RAISE], [OK, OK])}] \
                    if (t in KILLED and m == "ordinary") else []
                rows.append({**base, "eligible": True, "reached": True,
                             "views_unchanged": True, "within_budget": True, "cleanup_ok": True,
                             "replay_cleanup_ok": True, "end_reason": "cpu_budget_exhausted",
                             "aggregate_cpu_seconds": 600.4, "replay_errors": 0,
                             "confirmations": checks, "witnesses": len(checks),
                             "confirmed": sum(c["kill"] for c in checks),
                             "kill": any(c["kill"] for c in checks)})
    return rows


def _atheris_fixture(world, mutate=None, contract_mutate=None):
    original = _atheris_contract(world)
    contract = json.loads(json.dumps(original))
    if contract_mutate:
        contract_mutate(contract)
    cpath = world["tmp"] / "atheris_contract.json"
    cpath.write_text(json.dumps(contract, indent=1, sort_keys=True))
    rows = _atheris_rows(ar.contract_hash(original), original)
    if mutate:
        rows = mutate(rows)
    rpath = world["tmp"] / "atheris_results.jsonl"
    rpath.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rpath, cpath


def _load_atheris(world, rpath, cpath):
    from tests import native_analysis_helpers as helpers
    prepared = gio.resolve_prep(world["prep"], world["manifest"], world["tmp"])
    return helpers.load_atheris(world, rpath, cpath, prepared)


@pytest.mark.parametrize("buggy, fixed, error, kill", [
    ([RAISE, RAISE], [OK, OK], None, True),              # buggy raises, fixed returns: kill
    ([RAISE, RAISE], [OTHER, OTHER], None, False),       # ValueError vs TypeError: NOT a kill
    ([RAISE, RAISE], [RAISE, RAISE], None, False),       # same exception on both
    ([RAISE, OK], [OK, OK], None, False),                # unstable replay
    ([], [], "replay_process_failure", False),           # replay error
    ([OK, OK], [RAISE, RAISE], None, False),             # inverted
])
def test_ordinary_kill_rule(buggy, fixed, error, kill):
    assert ar.judge("ordinary", buggy, fixed, error)["kill"] is kill


def test_posthoc_and_differential_semantics_are_unchanged():
    assert ar.judge("posthoc", [RAISE, RAISE], [OTHER, OTHER])["kill"] is True
    assert ar.judge("differential", [OK, OK], [["ok", ["int", 2]]] * 2)["kill"] is True
    assert ar.judge("posthoc", [RAISE, OK], [OK, OK])["kill"] is False


def test_valid_216_cell_grid_loads(world):
    loaded = _load_atheris(world, *_atheris_fixture(world))
    assert loaded["cells"] == 216 and loaded["counts"] == {
        "usable": 23, "atheris_ineligible": 1, "infrastructure_excluded": 0}


@pytest.mark.parametrize("mutate, fragment", [
    (lambda rows: rows[:-1], "grid incomplete"),
    (lambda rows: rows + rows[-1:], "duplicate cell"),
    (lambda rows: rows + [{**rows[-1], "key": f"{rows[-1]['target_key']}::ordinary::45",
                           "seed": 45}], "unexpected cell"),
    (lambda rows: [{**rows[0], "contract_sha256": "stale"}] + rows[1:], "stale"),
    (lambda rows: [{**rows[0], "key": rows[1]["key"]}] + rows[1:], "key/fields disagree"),
    (lambda rows: [{**rows[5], "seed": 43}] + rows[:5] + rows[6:], "key/fields disagree"),
])
def test_structurally_invalid_atheris_grids_are_refused(world, mutate, fragment):
    with pytest.raises(SystemExit, match=fragment):
        _load_atheris(world, *_atheris_fixture(world, mutate=mutate))


def test_partial_or_malformed_atheris_lines_are_refused(world):
    rpath, cpath = _atheris_fixture(world)
    rpath.write_bytes(rpath.read_bytes() + b'{"key"')
    with pytest.raises(SystemExit, match="partial line"):
        _load_atheris(world, rpath, cpath)
    rpath.write_bytes(rpath.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="malformed"):
        _load_atheris(world, rpath, cpath)


@pytest.mark.parametrize("change", [
    lambda c: c.update(budget_cpu_seconds=60), lambda c: c.update(tolerance=50.0),
    lambda c: c.update(corpus_cap=5000), lambda c: c.update(design_version="v3"),
    lambda c: c.update(script_sha256="0" * 64), lambda c: c.update(verdicts_sha256="0" * 64),
    lambda c: c.update(prep={"path": "other.jsonl", "sha256": "0" * 64}),
    lambda c: c.update(manifest_sha256="0" * 64), lambda c: c.update(seeds=[42]),
])
def test_wrong_atheris_contract_is_refused(world, change):
    with pytest.raises(SystemExit, match="contract differs"):
        _load_atheris(world, *_atheris_fixture(world, contract_mutate=change))


VICTIM = QUALIFIED[4]


@pytest.mark.parametrize("field, value, problem", [
    ("within_budget", False, "within_budget is not true"),
    ("aggregate_cpu_seconds", 614.0, "aggregate CPU over the budget"),
    ("cleanup_ok", False, "cleanup_ok is not true"),
    ("replay_cleanup_ok", False, "replay_cleanup_ok is not true"),
    ("replay_errors", 1, "replay errors"),
    ("end_reason", "wall_timeout", "end_reason 'wall_timeout'"),
    ("end_reason", "infrastructure_failure", "end_reason 'infrastructure_failure'"),
    ("end_reason", "crashed", "end_reason 'crashed'"),
    ("reached", False, "reached is not true"),
    ("views_unchanged", False, "views_unchanged is not true"),
    ("kill", True, "kill/confirmed/witnesses disagree with the confirmations"),
    ("confirmations", None, "no confirmation evidence"),
])
def test_unusable_rows_become_explicit_target_infrastructure_exclusions(world, field, value,
                                                                       problem):
    def mutate(rows):
        for r in rows:
            if r["key"] == f"{VICTIM}::posthoc::43":
                r[field] = value
        return rows
    loaded = _load_atheris(world, *_atheris_fixture(world, mutate=mutate))
    status = loaded["targets"][VICTIM]
    assert status["status"] == "infrastructure_excluded"
    assert problem in status["problems"]["posthoc::43"]
    assert loaded["counts"]["infrastructure_excluded"] == 1


def test_different_exceptions_claimed_as_an_ordinary_kill_are_rejected(world):
    def mutate(rows):
        for r in rows:
            if r["key"] == f"{VICTIM}::ordinary::42":
                forged = {"witness": "w", "kill": True, "buggy_all": [RAISE, RAISE],
                          "fixed_all": [OTHER, OTHER]}
                r.update(confirmations=[forged], witnesses=1, confirmed=1, kill=True)
        return rows
    loaded = _load_atheris(world, *_atheris_fixture(world, mutate=mutate))
    problems = loaded["targets"][VICTIM]["problems"]["ordinary::42"]
    assert "a witness verdict disagrees with its confirmations" in problems


def test_atheris_infrastructure_exclusion_is_symmetric_in_the_joint_analysis(world, monkeypatch):
    root = _generate(world)
    code, results = _execute(world, root, monkeypatch)

    def mutate(rows):
        for r in rows:
            if r["key"] == f"{QUALIFIED[1]}::differential::44":
                r["cleanup_ok"] = False
        return rows
    rpath, cpath = _atheris_fixture(world, mutate=mutate)
    from tests import native_analysis_helpers as helpers
    result = helpers.analyse(world, results, results.parent / f"execute_contract_{COND}.json",
                             root, out="a2.json",
                             extra=["--atheris", str(rpath), "--atheris-contract", str(cpath)])
    assert result["atheris_denominators"]["joint"] == 21
    assert list(result["atheris_infrastructure_exclusions"]) == [QUALIFIED[1]]
    # the excluded target's Atheris kill is not silently a non-kill in the joint count
    assert result["atheris_jointly_eligible"]["atheris_unique_kills"]["ordinary"] == 0
    assert result["atheris_jointly_eligible"]["oneiros_unique_kills"]["sft"] == 21


# --- engineering-only enforcement (amendment v2.4 section F) -----------------------------------

def test_rehearsal_manifest_refuses_confirmation_mode(world, monkeypatch):
    root = _generate(world)
    code, results = _execute(world, root, monkeypatch)
    from tests import native_analysis_helpers as helpers
    contract = results.parent / f"execute_contract_{COND}.json"
    with pytest.raises(an.AnalysisRefused, match="refused: the manifest declares "
                                                 "'engineering_dress_rehearsal'"):
        helpers.analyse(world, results, contract, root, out="c.json", study_mode="confirmation")
    manifest = json.loads(world["manifest"].read_text())         # flipping the manifest alone
    manifest["study_mode"] = "confirmation"                      # still needs a frozen,
    world["manifest"].write_text(json.dumps(manifest))           # hash-bound authorisation
    with pytest.raises(an.AnalysisRefused, match="separately frozen confirmation manifest"):
        helpers.analyse(world, results, contract, root, out="c.json", study_mode="confirmation")
    assert not (world["tmp"] / "c.json").exists()
