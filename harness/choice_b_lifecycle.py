"""Choice B stage-aware lifecycle: one source identity, exact prior artifacts, no skips.

    preflighted -> control_trained -> treatment_trained
                -> control_evaluated -> treatment_evaluated -> analysed

The v1 guard required a clean tree AND no change since the preflight commit except the
receipt, so the first tracked stage result blocked every later stage. Here:

* a SOURCE IDENTITY (SHA-256 over the canonical hashes of every source, configuration
  and frozen input file) is bound in the preflight and recomputed at every stage; result
  files are not part of it, so result-only commits never change it;
* the tree must be clean at every launch (no dirt is ignored), HEAD must descend from
  the preflight commit, and every file changed since that commit must be the preflight
  itself or the result of a COMPLETED stage - anything else (source, configuration, an
  unknown result, a cross-arm or future-stage result) refuses;
* completed stages must form an exact prefix of the order; the requested action must be
  the next one; a repeated, skipped or out-of-order stage refuses;
* every completed artifact is re-validated against the preflight: schema, stage, arm,
  objective, source identity, preflight/split/evaluation-spec hashes, model, immutable
  revision, seed, run contract, adapter path and the adapter's actual bytes; gate looks
  are also bound to their training result and generation contract.

Both arms must be trained before either gate look (no control outcome can exist while
the treatment is still trainable), and both looks must exist before the analysis.
Checkpoints and raw outputs stay on the GPU system (ignored); only hashes are tracked.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harness.source_identity import canonical_sha256

STAGES = ("preflighted", "control_trained", "treatment_trained", "control_evaluated",
          "treatment_evaluated", "analysed")
ACTIONS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "train_control": ("preflighted", "control_trained", "control"),
    "train_treatment": ("control_trained", "treatment_trained", "treatment"),
    "evaluate_control": ("treatment_trained", "control_evaluated", "control"),
    "evaluate_treatment": ("control_evaluated", "treatment_evaluated", "treatment"),
    "analyse": ("treatment_evaluated", "analysed", None),
}
STAGE_SCHEMAS = {
    "control_trained": "oneiros_choice_b_v2_training_result",
    "treatment_trained": "oneiros_choice_b_v2_training_result",
    "control_evaluated": "oneiros_choice_b_v2_gate_look",
    "treatment_evaluated": "oneiros_choice_b_v2_gate_look",
    "analysed": "oneiros_choice_b_v2_analysis",
}
PREFLIGHT_SCHEMA = "oneiros_choice_b_v2_preflight"
HEADER_KEYS = ("source_identity", "preflight_sha256", "split_sha256",
               "evaluation_spec_sha256", "model", "revision", "seed")


class LifecycleError(RuntimeError):
    """A stage may not run; the message names the exact reason."""


@dataclass(frozen=True)
class Layout:
    root: Path
    preflight: str
    results: Mapping[str, str]            # stage -> tracked result path
    result_prefix: str                     # any other file with this prefix is unknown
    source_files: Tuple[str, ...]          # the source identity covers exactly these
    extra_allowed: Tuple[str, ...] = field(default_factory=tuple)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def source_identity(layout: Layout) -> Dict[str, Any]:
    files = {}
    for rel in sorted(set(layout.source_files)):
        path = layout.root / rel
        if not path.is_file():
            raise LifecycleError(f"source identity: missing file {rel}")
        files[rel] = canonical_sha256(path)
    return {"files": files, "sha256": contract_sha256(files)}


def _git(root: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and done.returncode != 0:
        raise LifecycleError(f"git {' '.join(args)} failed: {done.stderr.strip()[:200]}")
    return done.stdout


def completed_stages(layout: Layout) -> List[str]:
    present = [s for s in STAGES[1:] if (layout.root / layout.results[s]).exists()]
    expected = list(STAGES[1:1 + len(present)])
    if present != expected:
        raise LifecycleError(f"stage results are not a prefix of the order: present {present}")
    return ["preflighted", *present]


def unknown_results(layout: Layout) -> List[str]:
    """Files carrying the result prefix that are not a declared stage result."""
    prefix = layout.result_prefix
    directory = (layout.root / prefix).parent
    stem = Path(prefix).name
    declared = set(layout.results.values()) | {layout.preflight, *layout.extra_allowed}
    found = sorted(p.relative_to(layout.root).as_posix() for p in directory.glob(f"{stem}*"))
    return [rel for rel in found if rel not in declared]


def load_preflight(layout: Layout) -> Dict[str, Any]:
    path = layout.root / layout.preflight
    if not path.exists():
        raise LifecycleError("no preflight v2 receipt")
    preflight = json.loads(path.read_text(encoding="utf-8"))
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA:
        raise LifecycleError("preflight schema is not recognised")
    if preflight.get("ready") is not True:
        raise LifecycleError("preflight is not ready")
    return preflight


def artifact_header(layout: Layout, preflight: Mapping[str, Any], stage: str,
                    arm: Optional[str]) -> Dict[str, Any]:
    """Fields every stage artifact must carry, taken from the preflight."""
    return {"schema_version": STAGE_SCHEMAS[stage], "stage": stage, "arm": arm,
            "objective_mode": preflight["arms"][arm]["objective_mode"] if arm else None,
            "source_identity": preflight["source_identity"]["sha256"],
            "preflight_sha256": sha256_file(layout.root / layout.preflight),
            "split_sha256": preflight["split_sha256"],
            "evaluation_spec_sha256": preflight["evaluation_spec_sha256"],
            "model": preflight["model"], "revision": preflight["revision"],
            "seed": preflight["seed"]}


def evaluation_contract(preflight: Mapping[str, Any], arm: str,
                        training: Mapping[str, Any], training_sha: str) -> Dict[str, Any]:
    """The immutable raw-output contract: preflight template plus the trained adapter."""
    return {**preflight["arms"][arm]["evaluation_contract_template"],
            "adapter_sha256": training["adapter_sha256"],
            "training_result_sha256": training_sha}


def validate_artifact(layout: Layout, preflight: Mapping[str, Any], stage: str,
                      check_adapters: bool = True) -> Dict[str, Any]:
    path = layout.root / layout.results[stage]
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LifecycleError(f"{stage}: unreadable result ({error})") from None
    arm = {"control_trained": "control", "treatment_trained": "treatment",
           "control_evaluated": "control", "treatment_evaluated": "treatment"}.get(stage)
    expected = artifact_header(layout, preflight, stage, arm)
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise LifecycleError(f"{stage}: {key} is {artifact.get(key)!r}, expected {value!r}")
    if stage.endswith("_trained"):
        spec = preflight["arms"][arm]
        if artifact.get("run_contract") != spec["run_contract"]:
            raise LifecycleError(f"{stage}: run contract differs from the preflight")
        if artifact.get("adapter_path") != spec["adapter_path"]:
            raise LifecycleError(f"{stage}: adapter path differs from the preflight")
        if artifact.get("status") != "complete":
            raise LifecycleError(f"{stage}: training result is not complete")
        if check_adapters:
            adapter = layout.root / spec["adapter_path"] / "adapter_model.safetensors"
            if not adapter.exists():
                raise LifecycleError(f"{stage}: adapter file missing")
            if sha256_file(adapter) != artifact.get("adapter_sha256"):
                raise LifecycleError(f"{stage}: adapter bytes differ from the recorded hash")
    if stage.endswith("_evaluated"):
        training_rel = layout.results[f"{arm}_trained"]
        training = json.loads((layout.root / training_rel).read_text(encoding="utf-8"))
        training_sha = sha256_file(layout.root / training_rel)
        if artifact.get("training_result_sha256") != training_sha:
            raise LifecycleError(f"{stage}: bound to a different training result")
        if artifact.get("adapter_sha256") != training["adapter_sha256"]:
            raise LifecycleError(f"{stage}: evaluated a different adapter")
        contract = evaluation_contract(preflight, arm, training, training_sha)
        if artifact.get("evaluation_contract_sha256") != contract_sha256(contract):
            raise LifecycleError(f"{stage}: generation contract differs")
        if artifact.get("status") != "complete":
            raise LifecycleError(f"{stage}: gate look is not complete")
    if stage == "analysed":
        for look in ("control_evaluated", "treatment_evaluated"):
            if artifact.get("gate_looks", {}).get(look) != sha256_file(
                    layout.root / layout.results[look]):
                raise LifecycleError("analysed: bound to different gate looks")
    return artifact


def verify_action(layout: Layout, action: str, *, check_adapters: bool = True
                  ) -> Dict[str, Any]:
    """Refuse unless ``action`` is exactly the next legal stage. Returns the context."""
    if action not in ACTIONS:
        raise LifecycleError(f"unknown action {action!r}")
    required, produces, arm = ACTIONS[action]
    preflight = load_preflight(layout)
    identity = source_identity(layout)
    if identity["sha256"] != preflight["source_identity"]["sha256"]:
        drifted = sorted(k for k, v in identity["files"].items()
                         if preflight["source_identity"]["files"].get(k) != v)
        raise LifecycleError(f"source drift since the preflight: {drifted}")
    dirt = _git(layout.root, "status", "--porcelain", "--untracked-files=all").strip()
    if dirt:
        raise LifecycleError(f"working tree is not clean: {dirt.splitlines()[:5]}")
    commit = preflight["git"]["commit"]
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                              cwd=layout.root, capture_output=True)
    if ancestor.returncode != 0:
        raise LifecycleError("preflight commit is not an ancestor of HEAD")
    unknown = unknown_results(layout)
    if unknown:
        raise LifecycleError(f"unknown result files: {unknown}")
    done = completed_stages(layout)
    changed = set(filter(None, _git(layout.root, "diff", "--name-only", commit, "HEAD")
                         .splitlines()))
    allowed = {layout.preflight} | {layout.results[s] for s in done[1:]}
    if changed - allowed:
        raise LifecycleError(f"files changed since the preflight: {sorted(changed - allowed)}")
    current = done[-1]
    if produces in done:
        raise LifecycleError(f"stage {produces} already completed (repeated stage)")
    if current != required:
        raise LifecycleError(f"{action} requires stage {required}; current stage is {current}")
    for stage in done[1:]:
        validate_artifact(layout, preflight, stage, check_adapters=check_adapters)
    return {"action": action, "arm": arm, "current_stage": current, "produces": produces,
            "preflight": preflight, "head": _git(layout.root, "rev-parse", "HEAD").strip()}


def check_resume(directory: Path, contract: Mapping[str, Any]) -> str:
    """Exact-contract resume for a checkpoint or raw-output directory.

    Returns ``"fresh"`` or ``"resume"``; refuses a directory holding anything without a
    contract, or a contract that differs in any field (cross-arm, changed settings).
    """
    directory = Path(directory)
    contract_path = directory / "contract.json"
    if contract_path.exists():
        stored = json.loads(contract_path.read_text(encoding="utf-8"))
        if stored != dict(contract):
            changed = sorted(k for k in set(stored) | set(contract)
                             if stored.get(k) != contract.get(k))
            raise LifecycleError(f"{directory.name}: contract differs in {changed}")
        return "resume"
    if directory.exists() and any(directory.iterdir()):
        raise LifecycleError(f"{directory.name}: not empty and has no contract")
    directory.mkdir(parents=True, exist_ok=True)
    contract_path.write_bytes((json.dumps(dict(contract), indent=1, sort_keys=True) + "\n")
                              .encode("utf-8"))
    return "fresh"


def stage_commands(python: str, script: str) -> Dict[str, List[str]]:
    """Durable launch commands; the two arms differ only in the arm token of the action."""
    out = {}
    for action in ACTIONS:
        inner = [python, script, "run", action]
        out[action] = [python, "scripts/gpu_run.py", "start", "--name",
                       f"choice_b_v2_{action}", "--", *inner]
    return out


def command_difference(a: Sequence[str], b: Sequence[str]) -> List[Tuple[int, str, str]]:
    if len(a) != len(b):
        return [(-1, str(len(a)), str(len(b)))]
    return [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
