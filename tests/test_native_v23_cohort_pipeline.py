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
    manifest = {"kept_targets": QUALIFIED, "targets": targets, **fields}
    manifest_path = tmp_path / "manifest_v6.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    prep = []
    for k in QUALIFIED:
        views = {}
        for label in ("buggy", "fixed"):
            v = tmp_path / "views" / hashlib.sha256(k.encode()).hexdigest()[:8] / label / "pkg"
            v.mkdir(parents=True)
            (v / "core.py").write_text(f"# {label} {k}\n")
            views[label] = str(v.parent)
        prep.append({"key": k, "views": views, "module": "pkg.core", "qualname": "f",
                     "python_path": "/usr/bin/python3.11", "env_dir": "/env",
                     "module_sha256": {"buggy": "b", "fixed": "f"},
                     "view_manifest_sha256": {l: ex.build_view_hash(Path(views[l]))
                                              for l in views}})
    prep_path = tmp_path / "records.jsonl"
    prep_path.write_text("".join(json.dumps(r) + "\n" for r in prep))
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
    monkeypatch.setattr(ex, "canary_ok", lambda target, scratch: True)

    def execute(target, module, scratch, enforce_policy=True):
        kill = "KILL" in module
        return {"classification": {"class": "semantic_kill" if kill else "pass_both",
                                   "fixed_all_executed_passed_reached": True,
                                   "rerun_agrees": True if kill else None}}
    monkeypatch.setattr(ex, "execute_candidate", execute)


def _execute(world, root, monkeypatch, out=None):
    _stub_sandbox(monkeypatch)
    out = out or world["tmp"] / "exec"
    arms = ex.gio.arm_paths(root, None, None, COND)
    code = ex.run(world["prep"], world["manifest"], world["job"], arms, COND, out)
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
    atheris = [{"target_key": k, "mode": "ordinary", "seed": 42, "eligible": k != QUALIFIED[0],
                "kill": k in (EXCLUDED, QUALIFIED[1]), "replay_errors": 0} for k in QUALIFIED]
    atheris_path = world["tmp"] / "atheris.jsonl"
    atheris_path.write_text("".join(json.dumps(a) + "\n" for a in atheris))
    out = world["tmp"] / "analysis.json"
    assert an.main(["analyse", "--manifest", str(world["manifest"]), "--job", str(world["job"]),
                    "--results", str(results), "--condition", COND, "--study-mode",
                    "engineering_dress_rehearsal", "--out", str(out),
                    "--atheris", str(atheris_path)]) == 0
    result = json.loads(out.read_text())
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
    assert result["atheris_denominators"] == {
        "atheris_targets": 24, "atheris_eligible_targets": 23, "generation_targets": 23,
        "infrastructure_eligible": 23, "joint": 22,
        "rule": "joint = generation targets AND infrastructure-eligible AND Atheris-eligible"}
    assert result["atheris_jointly_eligible"]["atheris_unique_kills"] == {"ordinary": 1}
    assert result["atheris_only_not_generated"]["targets"] == [EXCLUDED]
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
               COND, world["tmp"] / "exec_old")


def test_explicit_per_arm_paths_are_equivalent(world, monkeypatch):
    root = _generate(world)
    _stub_sandbox(monkeypatch)
    arms = ex.gio.arm_paths(None, root / "base", root / "sft", COND)
    assert ex.run(world["prep"], world["manifest"], world["job"], arms, COND,
                  world["tmp"] / "exec2") == 0
    with pytest.raises(SystemExit, match="give --generations ROOT or both"):
        ex.gio.arm_paths(root, root / "base", None, COND)
    with pytest.raises(SystemExit, match="are the same"):
        ex.gio.arm_paths(None, root / "base", root / "base", COND)


def test_executor_cli_accepts_the_generation_root(world, monkeypatch):
    root = _generate(world)
    _stub_sandbox(monkeypatch)
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
