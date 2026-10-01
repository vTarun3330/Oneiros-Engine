"""Read-only launch gate for native generated-test GPU generation (amendment v2.2 section D).

Three distinct states, evaluated WITHOUT writing anything:

``pipeline_ready``
    The immutable pipeline preflight says ``pipeline_ready`` and still describes the current
    checkout: the same canonical executable-source identity, HEAD equal to the preflight's
    commit or a receipt-only descendant of it (amendment v2.3 E: the ONLY files such a
    descendant may change are the preflight receipt and the authorisation receipt), no tracked
    modification, no untracked file in an executable directory, a successful fresh fetch with
    HEAD equal to the fetched remote SHA, the same protocol, job, model and adapter
    identities, and EVERY path in the preflight's ``inputs`` - resolved safely beneath the
    repository root - still exists with the recorded SHA-256.
``gpu_authorized``
    A receipt of schema ``oneiros_native_gpu_authorization_v2`` whose CONTENT binds the exact
    preflight bytes, the source identity, the job/protocol/model/adapter hashes, the condition,
    the arm and the output directory. A file merely existing at the path never authorises.
``launch_ready``
    Both. For the SFT arm the pipeline is additionally ready only when the base arm has
    completed and passes the exact single-arm verifier (amendment v2.4 I.2): every expected
    row and candidate, full telemetry, and the same job, preflight, authorisation, backend,
    model, protocols and generation source; its immutable hashes are returned.

The preflight is never rewritten or re-timestamped here, so an authorisation that names its
hash stays valid for as long as the source, cohort and model are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional

EXECUTABLE_DIRS = ("engine", "harness", "scripts", "config", "tests")
PROTOCOL_FILES = ("docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_1.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_2.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_3.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_4.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5.md",
                  "docs/SFT_ROOT_CAUSE_NATIVE_GENERATED_TEST_PROTOCOL_V2_5_ADDENDUM_1.md")
RECEIPT_ONLY_PREFIX = "results/"
PREFLIGHT_SCHEMA = "oneiros_native_generated_tests_preflight_v2_4"
AUTH_SCHEMA = "oneiros_native_gpu_authorization_v2"
BRANCH = "experiment/research-eval-ablations"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def source_identity(root: Path) -> Dict[str, Any]:
    """Canonical (line-ending-normalised) hash of every tracked executable file plus the
    protocol documents. Independent of the commit that happens to contain them."""
    files = sorted(f for f in git(root, "ls-files", "--", *EXECUTABLE_DIRS).stdout.splitlines()
                   if f and (Path(root) / f).is_file())
    digest = {f: _canonical(Path(root) / f) for f in files}
    protocols = {p: _canonical(Path(root) / p) for p in PROTOCOL_FILES
                 if (Path(root) / p).is_file()}
    body = json.dumps({"files": digest, "protocols": protocols}, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return {"executable_tree_sha256": _sha_bytes(body), "files": len(digest),
            "protocol_sha256": protocols, "dirs": list(EXECUTABLE_DIRS)}


def fresh_fetch(root: Path, remote: str = "origin", branch: str = BRANCH) -> Dict[str, Any]:
    done = git(root, "fetch", "-q", remote, branch)
    remote_sha = None
    if done.returncode == 0:
        remote_sha = git(root, "rev-parse", f"refs/remotes/{remote}/{branch}").stdout.strip() or None
    return {"rc": done.returncode, "remote_sha": remote_sha,
            "stderr_tail": done.stderr.strip()[-200:] if done.returncode else ""}


def receipt_only_descendant(root: Path, base: str, head: str,
                            allowed: Optional[List[str]] = None) -> Dict[str, Any]:
    """HEAD equals ``base``, or descends from it changing only receipts: with ``allowed``,
    exactly those repository-relative paths (launch gate); otherwise anything under
    ``results/`` (used only for the full-suite receipt, whose source identity is also bound)."""
    if head == base:
        return {"ok": True, "relation": "same_commit", "changed": []}
    if git(root, "merge-base", "--is-ancestor", base, head).returncode != 0:
        return {"ok": False, "relation": "not_a_descendant", "changed": []}
    changed = [f for f in git(root, "diff", "--name-only", base, head).stdout.splitlines() if f]
    if allowed is not None:
        bad = [f for f in changed if PurePosixPath(f).as_posix() not in set(allowed)]
    else:
        bad = [f for f in changed
               if not PurePosixPath(f).as_posix().startswith(RECEIPT_ONLY_PREFIX)]
    return {"ok": not bad, "relation": "receipt_only_descendant" if not bad else
            "descendant_with_source_changes", "changed": changed, "non_receipt": bad}


def checkout_state(root: Path, fetch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    fetch = dict(fetch) if fetch is not None else fresh_fetch(root)
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    tracked_dirty = git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    untracked_exec = git(root, "status", "--porcelain", "--untracked-files=all", "--",
                         *EXECUTABLE_DIRS).stdout.strip()
    return {"head": head, "fetch": fetch, "tracked_dirty": bool(tracked_dirty),
            "untracked_executable": [l[3:] for l in untracked_exec.splitlines()][:20],
            "synced": fetch.get("rc") == 0 and fetch.get("remote_sha") == head}


def checkout_problems(state: Mapping[str, Any]) -> List[str]:
    problems = []
    if state["fetch"].get("rc") != 0:
        problems.append("fresh git fetch failed")
    elif not state["synced"]:
        problems.append("HEAD differs from the freshly fetched remote SHA")
    if state["tracked_dirty"]:
        problems.append("tracked files modified")
    if state["untracked_executable"]:
        problems.append("untracked files in executable directories")
    return problems


def relative_to_root(root: Path, path: Optional[Path]) -> Optional[str]:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, TypeError):
        return None


def safe_input(root: Path, rel: Any) -> Optional[Path]:
    """A recorded input path, only if it is relative and stays beneath the repository root."""
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        return None
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    resolved = (Path(root) / rel).resolve()
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError:
        return None
    return resolved


def input_problems(root: Path, pre: Mapping[str, Any]) -> List[str]:
    """Recompute EVERY immutable preflight input hash at launch time."""
    inputs = pre.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return ["preflight records no inputs"]
    problems = []
    for rel, recorded in sorted(inputs.items()):
        path = safe_input(root, rel)
        if path is None:
            problems.append(f"unsafe input path {rel!r}")
        elif not path.is_file():
            problems.append(f"input missing: {rel}")
        elif _sha_bytes(path.read_bytes()) != recorded:
            problems.append(f"input hash changed: {rel}")
    return problems


def _load(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def pipeline_problems(root: Path, preflight_path: Path, *, job_path: Path,
                      identity: Mapping[str, Any], state: Mapping[str, Any],
                      model_identity: Callable[[], Mapping[str, Any]],
                      adapter_sha256: Callable[[], Optional[str]], arm: str,
                      auth_path: Optional[Path] = None) -> List[str]:
    pre = _load(preflight_path)
    if pre is None:
        return ["preflight missing or malformed"]
    problems = []
    if pre.get("schema_version") != PREFLIGHT_SCHEMA:
        problems.append("preflight schema")
    if pre.get("pipeline_ready") is not True:
        problems.append("preflight not pipeline_ready")
    source = pre.get("source") or {}
    if source.get("executable_tree_sha256") != identity["executable_tree_sha256"]:
        problems.append("executable source differs from the preflight")
    if source.get("protocol_sha256") != identity["protocol_sha256"]:
        problems.append("protocol differs from the preflight")
    commit = source.get("commit")
    allowed = [r for r in (relative_to_root(root, preflight_path),
                           relative_to_root(root, auth_path) if auth_path else None) if r]
    if not commit or not receipt_only_descendant(root, commit, state["head"], allowed)["ok"]:
        problems.append("HEAD is not the preflight commit or a receipt-only descendant")
    problems += input_problems(root, pre)
    problems += checkout_problems(state)
    job = pre.get("job") or {}
    if not Path(job_path).is_file() or \
            _sha_bytes(Path(job_path).read_bytes()) != job.get("file_sha256"):
        problems.append("job file differs from the preflight")
    elif Path(job_path).resolve() != (Path(root) / str(job.get("path", ""))).resolve():
        problems.append("job path differs from the preflight")
    if (pre.get("model") or {}) != dict(model_identity()):
        problems.append("model snapshot/tokenizer/chat template differs from the preflight")
    if arm == "sft" and pre.get("adapter_manifest_sha256") != adapter_sha256():
        problems.append("adapter differs from the preflight")
    return problems


def authorization_problems(root: Path, auth_path: Optional[Path], preflight_path: Path, *,
                           identity: Mapping[str, Any], job_path: Path, condition: str,
                           arm: str, out_dir: Path) -> List[str]:
    if auth_path is None or not Path(auth_path).is_file():
        return ["no authorisation receipt"]
    auth = _load(auth_path)
    if auth is None:
        return ["authorisation malformed"]
    pre_bytes = Path(preflight_path).read_bytes() if Path(preflight_path).is_file() else b""
    pre = _load(preflight_path) or {}
    problems = []
    if auth.get("schema_version") != AUTH_SCHEMA:
        problems.append("authorisation schema")
    if not auth.get("approved_by_user") is True or not auth.get("created_utc"):
        problems.append("authorisation lacks explicit approval or timestamp")
    if (Path(root) / str(auth.get("preflight_path", ""))).resolve() != Path(preflight_path).resolve():
        problems.append("authorisation names a different preflight")
    if auth.get("preflight_sha256") != _sha_bytes(pre_bytes):
        problems.append("preflight hash (stale or tampered)")
    if auth.get("source") != {k: identity[k] for k in ("executable_tree_sha256",
                                                       "protocol_sha256")} or \
            auth.get("source_commit") != (pre.get("source") or {}).get("commit"):
        problems.append("authorisation source identity")
    if auth.get("protocol_sha256") != identity["protocol_sha256"]:
        problems.append("authorisation protocol")
    if auth.get("job_file_sha256") != (_sha_bytes(Path(job_path).read_bytes())
                                       if Path(job_path).is_file() else None) or \
            auth.get("job_file_sha256") != (pre.get("job") or {}).get("file_sha256"):
        problems.append("authorisation job")
    if auth.get("model") != pre.get("model"):
        problems.append("authorisation model")
    if auth.get("adapter_manifest_sha256") != pre.get("adapter_manifest_sha256"):
        problems.append("authorisation adapter")
    if condition not in (auth.get("allowed_conditions") or []):
        problems.append("condition not authorised")
    if arm not in (auth.get("allowed_arms") or []):
        problems.append("arm not authorised")
    outputs = auth.get("output_dirs") or {}
    if arm not in outputs or \
            (Path(root) / str(outputs[arm])).resolve() != Path(out_dir).resolve():
        problems.append("output directory not authorised")
    return problems


def base_arm_problems(root: Path, preflight_path: Path, auth_path: Optional[Path], *,
                      backend: str,
                      generation_source: Optional[Callable[[], Any]] = None) -> tuple:
    """The completed base arm, verified exactly before the SFT arm may launch."""
    from scripts.native_generation_io import resolve_cohort, verify_single_arm
    pre = _load(preflight_path) or {}
    job_rel = (pre.get("job") or {}).get("path")
    manifest_rel = (pre.get("manifest") or {}).get("path")
    base_rel = (pre.get("generation_outputs") or {}).get("base")
    if not (job_rel and manifest_rel and base_rel):
        return ["preflight lacks job/manifest/base output paths"], None
    try:
        cohort = resolve_cohort(Path(root) / job_rel, Path(root) / manifest_rel)
        verified = verify_single_arm(Path(root) / base_rel, "base", cohort)
    except SystemExit as exc:
        return [f"base arm not complete and verified: {str(exc)[:200]}"], None
    identity, problems = verified["identity"], []
    if (verified["rows"], verified["candidates"]) != (cohort["expected"]["rows_per_arm"],
                                                     cohort["expected"]["candidates_per_arm"]):
        problems.append("base arm size differs from the frozen cohort")
    if identity.get("preflight_sha256") != _sha_bytes(Path(preflight_path).read_bytes()):
        problems.append("base arm used a different preflight")
    if auth_path is not None and Path(auth_path).is_file() and \
            identity.get("authorization_sha256") != _sha_bytes(Path(auth_path).read_bytes()):
        problems.append("base arm used a different authorisation")
    if identity.get("backend") != backend:
        problems.append("base arm used a different backend")
    if identity.get("model") != pre.get("model"):
        problems.append("base arm used a different model snapshot")
    raw = {p: _sha_bytes((Path(root) / p).read_bytes()) for p in PROTOCOL_FILES
           if (Path(root) / p).is_file()}
    if identity.get("protocol_sha256") != raw:
        problems.append("base arm used different protocols")
    if generation_source is not None and identity.get("source_tree") != generation_source():
        problems.append("base arm used a different generation source")
    summary = {k: verified[k] for k in ("rows", "candidates", "file_sha256", "contract_sha256",
                                        "identity_sha256", "telemetry_schema")}
    return problems, summary


def evaluate(root: Path, preflight_path: Path, auth_path: Optional[Path], *, job_path: Path,
             condition: str, arm: str, out_dir: Path,
             model_identity: Callable[[], Mapping[str, Any]],
             adapter_sha256: Callable[[], Optional[str]],
             fetch: Optional[Mapping[str, Any]] = None, backend: str = "hf",
             generation_source: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
    """Pure check; never writes. ``fetch`` may be injected (tests); otherwise a fresh fetch."""
    root = Path(root)
    identity = source_identity(root)
    state = checkout_state(root, fetch)
    pipe = pipeline_problems(root, Path(preflight_path), job_path=Path(job_path),
                             identity=identity, state=state, model_identity=model_identity,
                             adapter_sha256=adapter_sha256, arm=arm, auth_path=auth_path)
    auth = authorization_problems(root, auth_path, Path(preflight_path), identity=identity,
                                  job_path=Path(job_path), condition=condition, arm=arm,
                                  out_dir=Path(out_dir))
    base_verification = None
    if arm == "sft":                                   # v2.4 I.2: strictly after a verified base
        base, base_verification = base_arm_problems(root, Path(preflight_path), auth_path,
                                                    backend=backend,
                                                    generation_source=generation_source)
        pipe += base
    return {"pipeline_ready": not pipe, "gpu_authorized": not auth,
            "base_arm_verification": base_verification,
            "launch_ready": not pipe and not auth,
            "pipeline_problems": pipe, "authorization_problems": auth,
            "head": state["head"], "fetch": state["fetch"],
            "source": identity["executable_tree_sha256"]}
