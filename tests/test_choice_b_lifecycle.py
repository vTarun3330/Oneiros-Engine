"""Choice B v2 lifecycle: every legal transition and the important refusals, on a real
throwaway git repository."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from harness import choice_b_lifecycle as lc

ARMS = {"control": "full_completion", "treatment": "value_only"}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message)


class Repo:
    def __init__(self, root: Path):
        self.root = root
        _git(root, "init", "-q")
        (root / ".gitignore").write_text("ckpt/\nraw/\n", encoding="utf-8")
        (root / "src.py").write_text("x = 1\n", encoding="utf-8")
        (root / "split.json").write_text('{"rows": 3}\n', encoding="utf-8")
        (root / "results").mkdir()
        _commit(root, "source")
        self.layout = lc.Layout(
            root=root, preflight="results/cb_v2_preflight.json",
            results={s: f"results/cb_v2_{s}.json" for s in lc.STAGES[1:]},
            result_prefix="results/cb_v2_", source_files=("src.py", "split.json"))
        identity = lc.source_identity(self.layout)
        self.preflight = {
            "schema_version": lc.PREFLIGHT_SCHEMA, "ready": True,
            "git": {"commit": _git(root, "rev-parse", "HEAD")},
            "source_identity": identity, "split_sha256": "s" * 64,
            "evaluation_spec_sha256": "e" * 64, "model": "m", "revision": "r", "seed": 42,
            "arms": {arm: {"objective_mode": mode, "adapter_path": f"ckpt/{arm}/final_adapter",
                           "run_contract": {"arm": arm, "objective_mode": mode},
                           "evaluation_contract_template": {"arm": arm, "decoding": "greedy"}}
                     for arm, mode in ARMS.items()}}
        self.write(self.layout.preflight, self.preflight)
        _commit(root, "preflight")

    def write(self, rel: str, payload) -> None:
        (self.root / rel).write_text(json.dumps(payload, sort_keys=True) + "\n",
                                     encoding="utf-8")

    def artifact(self, stage: str) -> dict:
        arm = {"control_trained": "control", "treatment_trained": "treatment",
               "control_evaluated": "control", "treatment_evaluated": "treatment"}.get(stage)
        body = {**lc.artifact_header(self.layout, self.preflight, stage, arm),
                "status": "complete"}
        if stage.endswith("_trained"):
            adapter = self.root / f"ckpt/{arm}/final_adapter"
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_model.safetensors").write_bytes(f"weights-{arm}".encode())
            body.update(run_contract=self.preflight["arms"][arm]["run_contract"],
                        adapter_path=f"ckpt/{arm}/final_adapter",
                        adapter_sha256=lc.sha256_file(adapter / "adapter_model.safetensors"))
        if stage.endswith("_evaluated"):
            training_rel = self.layout.results[f"{arm}_trained"]
            training = json.loads((self.root / training_rel).read_text(encoding="utf-8"))
            training_sha = lc.sha256_file(self.root / training_rel)
            body.update(training_result_sha256=training_sha,
                        adapter_sha256=training["adapter_sha256"],
                        evaluation_contract_sha256=lc.contract_sha256(
                            lc.evaluation_contract(self.preflight, arm, training, training_sha)))
        if stage == "analysed":
            body["gate_looks"] = {s: lc.sha256_file(self.root / self.layout.results[s])
                                  for s in ("control_evaluated", "treatment_evaluated")}
        return body

    def complete(self, stage: str, commit: bool = True, **override) -> None:
        self.write(self.layout.results[stage], {**self.artifact(stage), **override})
        if commit:
            _commit(self.root, f"result {stage}")

    def refuses(self, action: str, fragment: str) -> None:
        with pytest.raises(lc.LifecycleError, match=fragment):
            lc.verify_action(self.layout, action)


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path)


def test_every_legal_transition_accepts_result_only_commits(repo):
    for action, (required, produces, arm) in lc.ACTIONS.items():
        ctx = lc.verify_action(repo.layout, action)
        assert (ctx["current_stage"], ctx["produces"], ctx["arm"]) == (required, produces, arm)
        repo.complete(produces)
    assert lc.completed_stages(repo.layout) == list(lc.STAGES)
    for action in lc.ACTIONS:
        repo.refuses(action, "already completed")


def test_order_is_enforced_and_evaluation_waits_for_both_arms(repo):
    repo.refuses("train_treatment", "requires stage control_trained")
    repo.refuses("evaluate_control", "requires stage treatment_trained")
    repo.refuses("analyse", "requires stage treatment_evaluated")
    repo.complete("control_trained")
    repo.refuses("evaluate_control", "requires stage treatment_trained")
    repo.refuses("train_control", "already completed")
    repo.complete("treatment_trained")
    repo.complete("control_evaluated")
    repo.refuses("analyse", "requires stage treatment_evaluated")


def test_skipped_stage_result_is_refused(repo):
    repo.write(repo.layout.results["treatment_trained"], {"stage": "treatment_trained"})
    _commit(repo.root, "skip")
    repo.refuses("train_control", "not a prefix")


def test_source_drift_is_refused_even_when_committed(repo):
    repo.complete("control_trained")
    (repo.root / "src.py").write_text("x = 2\n", encoding="utf-8")
    _commit(repo.root, "drift")
    repo.refuses("train_treatment", "source drift")


def test_unknown_dirty_and_untracked_files_are_refused(repo):
    (repo.root / "notes.txt").write_text("scratch\n", encoding="utf-8")
    repo.refuses("train_control", "not clean")
    (repo.root / "notes.txt").unlink()
    repo.complete("control_trained", commit=False)            # an uncommitted result is dirt
    repo.refuses("train_treatment", "not clean")


def test_unknown_result_file_and_unexpected_committed_file_are_refused(repo):
    repo.write("results/cb_v2_extra.json", {"x": 1})
    _commit(repo.root, "unknown result")
    repo.refuses("train_control", "unknown result files")


def test_committed_non_result_change_is_refused(repo):
    (repo.root / "README.md").write_text("changed\n", encoding="utf-8")
    _commit(repo.root, "docs")
    repo.refuses("train_control", "files changed since the preflight")


def test_stale_preflight_is_refused(repo):
    repo.complete("control_trained")
    stale = {**repo.preflight, "seed": 43}
    repo.write(repo.layout.preflight, stale)
    _commit(repo.root, "edit preflight")
    repo.refuses("train_treatment", "preflight_sha256|seed")


@pytest.mark.parametrize("override, fragment", [
    ({"arm": "treatment"}, "arm is"),
    ({"objective_mode": "value_only"}, "objective_mode is"),
    ({"split_sha256": "x" * 64}, "split_sha256 is"),
    ({"preflight_sha256": "x" * 64}, "preflight_sha256 is"),
    ({"source_identity": "x" * 64}, "source_identity is"),
    ({"revision": "main"}, "revision is"),
    ({"run_contract": {"arm": "control", "objective_mode": "value_only"}}, "run contract"),
    ({"adapter_sha256": "0" * 64}, "adapter bytes differ"),
    ({"status": "running"}, "not complete"),
])
def test_bad_training_artifacts_are_refused(repo, override, fragment):
    repo.complete("control_trained", **override)
    repo.refuses("train_treatment", fragment)


def test_altered_adapter_is_refused(repo):
    repo.complete("control_trained")
    (repo.root / "ckpt/control/final_adapter/adapter_model.safetensors").write_bytes(b"tampered")
    repo.refuses("train_treatment", "adapter bytes differ")


def test_cross_arm_gate_look_is_refused(repo):
    repo.complete("control_trained")
    repo.complete("treatment_trained")
    treatment_sha = lc.sha256_file(repo.root / repo.layout.results["treatment_trained"])
    repo.complete("control_evaluated", training_result_sha256=treatment_sha)
    repo.refuses("evaluate_treatment", "different training result")


def test_analysis_with_incomplete_inputs_is_refused(repo):
    for stage in ("control_trained", "treatment_trained", "control_evaluated"):
        repo.complete(stage)
    repo.write(repo.layout.results["analysed"], {"stage": "analysed"})
    _commit(repo.root, "premature analysis")
    repo.refuses("analyse", "not a prefix")


def test_interrupted_stage_resumes_only_under_the_exact_contract(repo, tmp_path):
    ckpt = repo.root / "ckpt" / "control"
    contract = repo.preflight["arms"]["control"]["run_contract"]
    assert lc.check_resume(ckpt, contract) == "fresh"
    (ckpt / "checkpoint-50").mkdir()
    # Interrupted: no result yet, so the same action is still the legal next step.
    assert lc.verify_action(repo.layout, "train_control")["produces"] == "control_trained"
    assert lc.check_resume(ckpt, contract) == "resume"
    with pytest.raises(lc.LifecycleError, match="contract differs"):
        lc.check_resume(ckpt, repo.preflight["arms"]["treatment"]["run_contract"])
    stray = repo.root / "ckpt" / "stray"
    stray.mkdir()
    (stray / "x").write_text("y")
    with pytest.raises(lc.LifecycleError, match="no contract"):
        lc.check_resume(stray, contract)


def test_stage_commands_differ_only_in_the_arm():
    commands = lc.stage_commands("py", "s.py")
    diff = lc.command_difference(commands["train_control"], commands["train_treatment"])
    assert [d[1:] for d in diff] == [("choice_b_v2_train_control", "choice_b_v2_train_treatment"),
                                     ("train_control", "train_treatment")]
