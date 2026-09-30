"""End-to-end preflight -> authorisation -> launch-gate lifecycle (amendment v2.2 section D).

A temporary git repository with a bare remote plays the project: a green, source-bound
preflight is committed as a receipt-only descendant, an authorisation is added later, and the
read-only gate is evaluated. The mock backend runs the exact proposed generation command.
No model is loaded; model identity is injected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from harness import native_launch_gate as gate
from scripts import native_generated_tests_generate as gen

ROOT = Path(__file__).resolve().parent.parent
MODEL = {"snapshot_manifest_sha256": "snap", "tokenizer_manifest_sha256": "tok",
         "chat_template_sha256": "tmpl"}
ADAPTER = "adapter-manifest-sha"
COND = "primary_whole_module"
OUT_BASE = "results/native_generations/base"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job_file(path: Path, n: int = 2) -> None:
    prompts = [{"target_key": f"t{i}", "prompt": f"prompt {i}", "condition": "whole_module",
                "prompt_sha256": hashlib.sha256(f"prompt {i}".encode()).hexdigest()}
               for i in range(n)]
    fit = gen.sequence_fit(prompts, len)
    job = gen.build_job(prompts, {p["target_key"]: {"ok": True} for p in prompts}, fit)
    path.write_text(json.dumps({COND: job}, sort_keys=True))


def _commit_push(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    _git(repo, "push", "-q", "origin", f"HEAD:refs/heads/{gate.BRANCH}")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def project(tmp_path):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo.mkdir()
    _git(repo, "init", "-q", "-b", gate.BRANCH)
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "remote", "add", "origin", str(remote))
    for rel, text in (("engine/core.py", "X = 1\n"), ("scripts/tool.py", "print(1)\n"),
                      ("config/c.json", "{}\n"), ("tests/test_x.py", "def test(): pass\n"),
                      *((p, f"# {p}\n") for p in gate.PROTOCOL_FILES)):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    (repo / "harness").mkdir()
    shutil.copy(ROOT / "harness/native_generated_test_prompt.py", repo / "harness")
    (repo / "results").mkdir()
    _job_file(repo / "results/job.json")
    source_commit = _commit_push(repo, "source")
    identity = gate.source_identity(repo)
    preflight = {"schema_version": gate.PREFLIGHT_SCHEMA, "pipeline_ready": True,
                 "source": {"commit": source_commit, **{k: identity[k] for k in (
                     "executable_tree_sha256", "protocol_sha256")}},
                 "job": {"path": "results/job.json",
                         "file_sha256": _sha(repo / "results/job.json")},
                 "model": MODEL, "adapter_manifest_sha256": ADAPTER}
    (repo / "results/preflight.json").write_text(json.dumps(preflight, sort_keys=True))
    receipt_commit = _commit_push(repo, "preflight receipt only")
    return {"repo": repo, "remote": remote, "source_commit": source_commit,
            "receipt_commit": receipt_commit, "identity": identity}


def _authorise(p, **override) -> Path:
    repo = p["repo"]
    auth = {"schema_version": gate.AUTH_SCHEMA, "approved_by_user": True,
            "created_utc": "2026-10-01T00:00:00Z", "preflight_path": "results/preflight.json",
            "preflight_sha256": _sha(repo / "results/preflight.json"),
            "source_commit": p["source_commit"],
            "source": {k: p["identity"][k] for k in ("executable_tree_sha256",
                                                     "protocol_sha256")},
            "protocol_sha256": p["identity"]["protocol_sha256"],
            "job_file_sha256": _sha(repo / "results/job.json"),
            "model": MODEL, "adapter_manifest_sha256": ADAPTER,
            "allowed_conditions": [COND], "allowed_arms": ["base"],
            "output_dirs": {"base": OUT_BASE}}
    auth.update(override)
    path = repo / "results/authorization.json"
    path.write_text(json.dumps(auth, sort_keys=True))
    return path


def _evaluate(p, auth=None, arm="base", out=OUT_BASE, job="results/job.json", **kw):
    repo = p["repo"]
    return gate.evaluate(repo, repo / "results/preflight.json", auth, job_path=repo / job,
                         condition=kw.pop("condition", COND), arm=arm, out_dir=repo / out,
                         model_identity=kw.pop("model", lambda: MODEL),
                         adapter_sha256=kw.pop("adapter", lambda: ADAPTER), **kw)


def test_receipt_only_preflight_commit_is_pipeline_ready_and_not_authorised(project):
    result = _evaluate(project, auth=project["repo"] / "results/authorization.json")
    assert result["pipeline_ready"] is True, result["pipeline_problems"]
    assert result["gpu_authorized"] is False and result["launch_ready"] is False
    assert result["authorization_problems"] == ["no authorisation receipt"]
    assert result["head"] == project["receipt_commit"] != project["source_commit"]
    assert result["fetch"]["rc"] == 0 and result["fetch"]["remote_sha"] == result["head"]


def test_later_authorisation_validates_without_changing_the_preflight(project):
    repo = project["repo"]
    before = _sha(repo / "results/preflight.json")
    auth = _authorise(project)
    _commit_push(repo, "authorisation receipt only")          # a second receipt-only descendant
    status_before = _git(repo, "status", "--porcelain", "--untracked-files=all")
    result = _evaluate(project, auth=auth)
    assert result["launch_ready"] is True, result
    assert _sha(repo / "results/preflight.json") == before
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == status_before


def test_exact_proposed_command_reaches_mock_generation(project, monkeypatch):
    repo = project["repo"]
    auth = _authorise(project)
    _commit_push(repo, "authorisation receipt only")
    monkeypatch.setattr(gen, "ROOT", repo)
    monkeypatch.setattr(gen, "model_identity", lambda: MODEL)
    monkeypatch.setattr(gen, "adapter_sha256", lambda: ADAPTER)
    argv = ["run", "--job", str(repo / "results/job.json"), "--condition", COND,
            "--arm", "base", "--out", str(repo / OUT_BASE), "--backend", "mock",
            "--preflight", str(repo / "results/preflight.json"), "--authorization", str(auth)]
    assert gen.main(argv) == 0
    lines = (repo / OUT_BASE / f"generations_{COND}_base.jsonl").read_text().splitlines()
    assert len(lines) == 2 * len(gen.CONTRACT["seeds"])
    with pytest.raises(gen.Refused, match="launch gate not ready"):
        gen.main([*argv[:6], "sft", *argv[7:]])


@pytest.mark.parametrize("override, fragment", [
    ({"schema_version": "oneiros_native_gpu_authorization_v1"}, "authorisation schema"),
    ({"approved_by_user": False}, "explicit approval"),
    ({"preflight_sha256": "0" * 64}, "preflight hash"),
    ({"job_file_sha256": "0" * 64}, "authorisation job"),
    ({"allowed_arms": ["sft"]}, "arm not authorised"),
    ({"allowed_conditions": ["secondary_scaffolded_diagnostic"]}, "condition not authorised"),
    ({"output_dirs": {"base": "results/elsewhere"}}, "output directory"),
    ({"source": {"executable_tree_sha256": "0" * 64, "protocol_sha256": {}}},
     "authorisation source identity"),
    ({"source_commit": "0" * 40}, "authorisation source identity"),
    ({"protocol_sha256": {"docs/x.md": "0"}}, "authorisation protocol"),
    ({"model": {**MODEL, "chat_template_sha256": "other"}}, "authorisation model"),
    ({"adapter_manifest_sha256": "other"}, "authorisation adapter"),
    ({"preflight_path": "results/other_preflight.json"}, "different preflight"),
])
def test_wrong_or_tampered_authorisation_is_rejected(project, override, fragment):
    result = _evaluate(project, auth=_authorise(project, **override))
    assert result["gpu_authorized"] is False and result["launch_ready"] is False
    assert any(fragment in p for p in result["authorization_problems"]), result


@pytest.mark.parametrize("content", ["", "yes", "{}", "[1, 2]", "{not json"])
def test_merely_creating_a_file_at_the_authorisation_path_does_not_authorise(project, content):
    path = project["repo"] / "results/authorization.json"
    path.write_text(content)
    result = _evaluate(project, auth=path)
    assert result["gpu_authorized"] is False and result["launch_ready"] is False


def test_copying_the_preflight_to_the_authorisation_path_does_not_authorise(project):
    repo = project["repo"]
    shutil.copy(repo / "results/preflight.json", repo / "results/authorization.json")
    assert _evaluate(project, auth=repo / "results/authorization.json")["gpu_authorized"] is False


def test_stale_authorisation_after_a_preflight_rerun_is_rejected(project):
    repo = project["repo"]
    auth = _authorise(project)
    pre = json.loads((repo / "results/preflight.json").read_text())
    pre["created_utc"] = "later"
    (repo / "results/preflight.json").write_text(json.dumps(pre, sort_keys=True))
    _commit_push(repo, "rerun preflight")
    result = _evaluate(project, auth=auth)
    assert "preflight hash (stale or tampered)" in result["authorization_problems"]


def test_wrong_job_arm_output_and_condition_requests_are_rejected(project):
    repo = project["repo"]
    auth = _authorise(project)
    _commit_push(repo, "authorisation receipt only")
    _job_file(repo / "results/job_other.json", 3)
    _commit_push(repo, "another job (results only)")
    wrong_job = _evaluate(project, auth=auth, job="results/job_other.json")
    assert "job file differs from the preflight" in wrong_job["pipeline_problems"]
    assert "authorisation job" in wrong_job["authorization_problems"]
    assert "arm not authorised" in _evaluate(project, auth=auth, arm="sft")[
        "authorization_problems"]
    assert "output directory not authorised" in _evaluate(project, auth=auth, out="results/x")[
        "authorization_problems"]
    assert "condition not authorised" in _evaluate(project, auth=auth, condition="other")[
        "authorization_problems"]


def test_dirty_or_untracked_executable_source_is_rejected(project):
    repo = project["repo"]
    auth = _authorise(project)
    _commit_push(repo, "authorisation receipt only")
    (repo / "engine/core.py").write_text("X = 2\n")
    result = _evaluate(project, auth=auth)
    assert "tracked files modified" in result["pipeline_problems"]
    assert "executable source differs from the preflight" in result["pipeline_problems"]
    assert result["launch_ready"] is False
    _git(repo, "checkout", "--", "engine/core.py")
    (repo / "scripts/extra.py").write_text("print('not tracked')\n")
    result = _evaluate(project, auth=auth)
    assert "untracked files in executable directories" in result["pipeline_problems"]


def test_committed_source_change_is_rejected_even_when_pushed(project):
    repo = project["repo"]
    auth = _authorise(project)
    (repo / "scripts/tool.py").write_text("print(2)\n")
    _commit_push(repo, "source change")
    result = _evaluate(project, auth=auth)
    assert "HEAD is not the preflight commit or a receipt-only descendant" in \
        result["pipeline_problems"]
    assert "executable source differs from the preflight" in result["pipeline_problems"]
    assert "authorisation source identity" in result["authorization_problems"]


def test_failed_fetch_and_unpushed_head_are_rejected(project):
    repo = project["repo"]
    auth = _authorise(project)
    _commit_push(repo, "authorisation receipt only")
    _git(repo, "remote", "set-url", "origin", str(project["remote"]) + "_missing")
    result = _evaluate(project, auth=auth)
    assert result["fetch"]["rc"] != 0 and "fresh git fetch failed" in result["pipeline_problems"]
    _git(repo, "remote", "set-url", "origin", str(project["remote"]))
    (repo / "results/note.json").write_text("{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local only")
    result = _evaluate(project, auth=auth)
    assert "HEAD differs from the freshly fetched remote SHA" in result["pipeline_problems"]


def test_red_preflight_model_change_and_adapter_change_are_rejected(project):
    repo = project["repo"]
    auth = _authorise(project, allowed_arms=["base", "sft"],
                      output_dirs={"base": OUT_BASE, "sft": "results/native_generations/sft"})
    _commit_push(repo, "authorisation receipt only")
    moved = _evaluate(project, auth=auth, model=lambda: {**MODEL, "snapshot_manifest_sha256": "x"})
    assert any("model snapshot" in p for p in moved["pipeline_problems"])
    sft = _evaluate(project, auth=auth, arm="sft", out="results/native_generations/sft",
                    adapter=lambda: "changed")
    assert "adapter differs from the preflight" in sft["pipeline_problems"]
    pre = json.loads((repo / "results/preflight.json").read_text())
    (repo / "results/preflight.json").write_text(json.dumps({**pre, "pipeline_ready": False}))
    red = _evaluate(project, auth=auth)
    assert "preflight not pipeline_ready" in red["pipeline_problems"]


def test_gate_module_never_writes():
    source = (ROOT / "harness/native_launch_gate.py").read_text(encoding="utf-8")
    for forbidden in ("write_text", "write_bytes", "open(", "publish", "unlink", "rename"):
        assert forbidden not in source, forbidden


def test_every_preflight_cli_prints_usage_without_side_effects():
    """Regression: the v2.2 rebuild script once had no --help and ran the rebuild instead."""
    import sys
    from scripts import native_generated_tests_preflight_v2_2 as preflight
    before = _git(ROOT, "status", "--porcelain", "--untracked-files=all")
    for script in preflight.CLIS:
        done = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=180)
        assert done.returncode == 0 and "usage" in done.stdout.lower(), script
    assert _git(ROOT, "status", "--porcelain", "--untracked-files=all") == before
