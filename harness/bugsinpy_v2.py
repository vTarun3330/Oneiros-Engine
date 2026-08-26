"""Conservative BugsInPy materialization for Oneiros corpus v2.

Patch fragments are never treated as training examples.  A record is accepted
only when the official targeted test passes on the fixed checkout and fails on
the buggy checkout.  Such a task is materialized either as a self-contained
function assertion, or as an explicitly labelled repository pytest fragment
with its affected-file context.  The two execution modes are never conflated.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from baseline.benchmark_runner import kills_mutant
from harness.corpus import (
    SCHEMA_VERSION, record_content_hash, semantic_python,
    write_json as atomic_write_json,
)


@dataclass(frozen=True)
class BugsInPyTask:
    project: str
    bug_id: str
    repository_url: str
    buggy_commit: str
    fixed_commit: str
    test_files: Tuple[str, ...]
    test_command: str
    python_version: str
    bug_dir: Path

    @property
    def id(self) -> str:
        return f"bugsinpy::{self.project}::{self.bug_id}"


def _read_assignments(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def discover_tasks(repository_root: Path) -> List[BugsInPyTask]:
    """Read official task metadata without accepting any code as data."""
    projects_dir = repository_root / "projects"
    tasks: List[BugsInPyTask] = []
    for project_dir in sorted(path for path in projects_dir.iterdir() if path.is_dir()):
        project_info = _read_assignments(project_dir / "project.info")
        repository_url = project_info.get("github_url", "")
        bugs_dir = project_dir / "bugs"
        if not repository_url or not bugs_dir.exists():
            continue
        for bug_dir in sorted((path for path in bugs_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            info = _read_assignments(bug_dir / "bug.info")
            command = (bug_dir / "run_test.sh").read_text(
                encoding="utf-8", errors="replace"
            ).strip() if (bug_dir / "run_test.sh").exists() else ""
            test_files = tuple(
                item for item in info.get("test_file", "").split(";") if item.strip()
            )
            task = BugsInPyTask(
                project=project_dir.name,
                bug_id=bug_dir.name,
                repository_url=repository_url,
                buggy_commit=info.get("buggy_commit_id", ""),
                fixed_commit=info.get("fixed_commit_id", ""),
                test_files=test_files,
                test_command=command,
                python_version=info.get("python_version", ""),
                bug_dir=bug_dir,
            )
            if all((task.buggy_commit, task.fixed_commit, task.test_command)):
                tasks.append(task)
    return tasks


def _run(
    args: Sequence[str], cwd: Optional[Path] = None, timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=str(cwd) if cwd else None, text=True, encoding="utf-8",
        errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, check=False, env=env,
    )


class RepositoryCache:
    """Fetch project histories once and use short-lived detached worktrees."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task: BugsInPyTask) -> Path:
        return self.root / task.project

    def ensure_commit(self, task: BugsInPyTask, commit: str) -> Path:
        path = self._path(task)
        if not path.exists():
            result = _run([
                "git", "clone", "--filter=blob:none", "--no-checkout", task.repository_url, str(path)
            ], timeout=900)
            if result.returncode:
                raise RuntimeError(f"clone failed: {result.stdout[-500:]}")
        exists = _run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=path)
        if exists.returncode:
            fetched = _run(
                ["git", "fetch", "--filter=blob:none", "origin", commit], cwd=path, timeout=900
            )
            if fetched.returncode:
                raise RuntimeError(f"commit fetch failed: {fetched.stdout[-500:]}")
        return path

    def show(self, repository: Path, commit: str, relative_path: str) -> Optional[str]:
        result = _run(["git", "show", f"{commit}:{relative_path}"], cwd=repository)
        return result.stdout if not result.returncode else None

    def worktree(self, repository: Path, commit: str, path: Path) -> None:
        result = _run(["git", "worktree", "add", "--detach", "--force", str(path), commit], cwd=repository, timeout=300)
        if result.returncode:
            raise RuntimeError(f"worktree failed: {result.stdout[-500:]}")

    def remove_worktree(self, repository: Path, path: Path) -> None:
        _run(["git", "worktree", "remove", "--force", str(path)], cwd=repository, timeout=300)


def _command_args(command: str) -> Optional[List[List[str]]]:
    """Parse each simple official test command from ``run_test.sh`` safely.

    BugsInPy frequently stores one targeted test command per line.  Treating
    the entire file as one shell command makes later lines look like arguments
    to the first command (for example, ``pytest ... pytest ...``), producing
    false collection failures.  We do not execute shell syntax: every accepted
    non-comment line is parsed independently and run directly.
    """
    if not command:
        return None
    commands = []
    for raw_line in command.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if any(token in line for token in ("|", ">", "<", "&&", "||", ";", "$", "`")):
            return None
        try:
            args = shlex.split(line, posix=True)
        except ValueError:
            return None
        if not args:
            continue
        if args[0] in {"python", "python3", "python3.6", "python3.7", "python3.8", "python3.9"}:
            args[0] = sys.executable
        elif args[0] in {"pytest", "py.test"}:
            args = [sys.executable, "-m", "pytest", *args[1:]]
        elif (
            args[0] == "tox"
            and len(args) > 1
            and all("::" in target and target.split("::", 1)[0].endswith(".py") for target in args[1:])
        ):
            # A small number of BugsInPy scripts use ``tox <explicit-node>``
            # only as a test-environment wrapper. The isolated environment is
            # already created and audited here, so conservatively translate
            # only explicit Python test node IDs to their direct pytest form.
            # General tox options/environments remain unsupported.
            args = [sys.executable, "-m", "pytest", *args[1:]]
        else:
            # Tools such as tox encapsulate their own environment, which
            # cannot be audited by this direct fixed-pass/buggy-fail runner.
            return None
        commands.append(args)
    return commands or None


def _environment_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _isolated_process_environment() -> Dict[str, str]:
    """Keep Modal/host Python packages out of historical task runtimes."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _requirements_text(path: Path) -> str:
    """Decode official requirements files, including legacy UTF-16 exports."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def _task_requirements(task: BugsInPyTask) -> List[str]:
    path = task.bug_dir / "requirements.txt"
    if not path.exists():
        return []
    return [
        line.strip() for line in _requirements_text(path).splitlines()
        if line.strip() and not line.strip().startswith(("#", "-e", "git+", "pkg-resources"))
        and not any(ord(character) < 32 for character in line.strip())
    ]


def _prepare_environment(
    task: BugsInPyTask, root: Path, runner_python: str,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Create an isolated best-effort environment from official requirements.

    The requested historical Python version is recorded. A task remains
    excluded if that environment cannot demonstrate fixed-pass/buggy-fail.
    """
    env_dir = root / "environment"
    clean_environment = _isolated_process_environment()
    created = _run(
        [runner_python, "-m", "venv", str(env_dir)], timeout=180,
        env=clean_environment,
    )
    if created.returncode:
        return None, {"reason": "venv_creation_failed", "output_tail": created.stdout[-500:]}
    python = _environment_python(env_dir)
    requested_minor = ".".join(task.python_version.split(".")[:2])
    pip_constraint = "pip<22" if requested_minor == "3.6" else (
        "pip<24.1" if requested_minor == "3.7" else "pip<25.1"
    )
    pytest_constraint = "pytest<7" if requested_minor == "3.6" else "pytest<8"
    pip_upgrade = _run(
        [str(python), "-m", "pip", "install", "--upgrade", pip_constraint],
        timeout=300, env=clean_environment,
    )
    if pip_upgrade.returncode:
        return None, {
            "reason": "pip_bootstrap_failed",
            "output_tail": pip_upgrade.stdout[-500:],
        }
    bootstrap = _run(
        [str(python), "-m", "pip", "install", pytest_constraint],
        timeout=300, env=clean_environment,
    )
    if bootstrap.returncode:
        return None, {
            "reason": "test_runner_install_failed",
            "output_tail": bootstrap.stdout[-500:],
        }
    requirements = _task_requirements(task)
    return python, {
        "runner_python": str(python),
        "runner_python_version": _run([str(python), "--version"]).stdout.strip(),
        "requested_python_version": task.python_version,
        "declared_requirements": requirements,
        "installed_requirements": [pip_constraint, pytest_constraint],
        "failed_requirements": [],
    }


def _test_environment(checkout: Path, task: BugsInPyTask) -> Dict[str, str]:
    environment = _isolated_process_environment()
    extra_paths = [str(checkout)]
    # Historical Python repositories commonly keep importable packages under
    # ``src`` or ``lib``.  The official test command runs from the checkout,
    # but those layouts are not automatically importable without installation.
    # Adding only repository-local source roots preserves the test semantics
    # while avoiding an unverified dependency substitution.
    for source_root in (checkout / "src", checkout / "lib"):
        if source_root.is_dir():
            extra_paths.append(str(source_root))
    for test_file in task.test_files:
        parent = checkout / test_file
        if parent.parent.exists():
            extra_paths.append(str(parent.parent))
    environment["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return environment


def _requirement_for_module(module: str, requirements: List[str]) -> str:
    raw_root = module.split(".", 1)[0].lower()
    aliases = {
        # ``attrs`` exposes the import package named ``attr``. Installing the
        # unrelated PyPI package named ``attr`` shadows the correct module and
        # makes historical Black checkouts fail during collection.
        "attr": "attrs",
        "yaml": "pyyaml",
        "dateutil": "python-dateutil",
        "cv2": "opencv-python",
        "ansible": "ansible-base",
        "bs4": "beautifulsoup4",
        "sklearn": "scikit-learn",
        "pil": "pillow",
        "crypto": "pycryptodome",
        # Historical Scrapy imports use module names that differ from their
        # distribution names.
        "openssl": "pyopenssl",
        "pydispatch": "pydispatcher",
        # ``past`` is provided by the ``future`` distribution.
        "past": "future",
        # The import uses an underscore while the distribution uses a dash.
        "requests_async": "requests-async",
        # Generated by setuptools-scm's ``write_to`` hook in historical Black.
        "_black_version": "setuptools-scm",
    }
    expected = aliases.get(raw_root, raw_root.replace("_", "-"))
    for requirement in requirements:
        package = re.split(r"[<>=!~\[\s]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        if package == expected:
            return requirement
    return expected


def _installation_candidates(requirement: str) -> List[str]:
    """Preserve declared pins, with narrow fallbacks for yanked prereleases."""
    normalized = requirement.lower().replace(" ", "")
    replacements = {
        # Each fallback is the corresponding stable release.  It is used only
        # to create a runnable environment; the task is still accepted solely
        # when its official test passes fixed and fails buggy in that exact
        # environment.
        "numpy==1.19.0rc2": "numpy==1.19.0",
        "scipy==1.5.0rc1": "scipy==1.5.0",
        "ansible-base==2.10.0.dev0": "ansible-base==2.10.0",
        "jinja2==3.0.0a1": "jinja2==3.0.0",
        "markupsafe==2.0.0a1": "markupsafe==2.0.0",
        # typed-ast 1.4 has no Python 3.10 wheel and its source build is not
        # compatible with that runtime. 1.5.5 is the final upstream release;
        # the exact fallback remains in the evidence and acceptance still
        # requires the official fixed-pass/buggy-fail contrast.
        "typed-ast==1.4.0": "typed-ast==1.5.5",
        "typed-ast==1.4.1": "typed-ast==1.5.5",
    }
    if normalized in replacements:
        return [requirement, replacements[normalized]]
    return [requirement]


def _run_with_missing_dependency_install(
    command: List[str], cwd: Path, env: Dict[str, str], python: Path,
    requirements: List[str], evidence: Dict[str, Any], timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Install only a declared dependency that prevents test collection."""
    attempted = set()
    result = _run(command, cwd=cwd, timeout=timeout, env=env)
    for _ in range(16):
        missing = re.search(r"No module named ['\"]?([A-Za-z0-9_.-]+)", result.stdout)
        incompatible_import = re.search(
            r"cannot import name ['\"]?[A-Za-z0-9_.-]+['\"]? from ['\"]([A-Za-z0-9_.-]+)",
            result.stdout, flags=re.IGNORECASE,
        )
        required_library = re.search(
            r"The `([A-Za-z0-9_.-]+)` library must be installed",
            result.stdout,
        )
        # Some historical suites make optional test dependencies observable
        # only through framework-level errors rather than ImportError.
        special_dependency = None
        forced_requirement = None
        if "NameError: name 'AioHTTPTestCase' is not defined" in result.stdout:
            # Black imports blackd, aiohttp and aiohttp-cors inside one broad
            # ImportError guard. Install the two declared extras in order.
            special_dependency = (
                "aiohttp-cors"
                if any(item.lower().startswith("aiohttp==") for item in evidence["installed_requirements"])
                else "aiohttp"
            )
        elif re.search(r"unrecognized arguments:.*--cov", result.stdout):
            special_dependency = "pytest-cov"
        elif re.search(r"unrecognized arguments:.*(?:^|\s)-n(?:\s|$)", result.stdout):
            # Keras declares pytest-xdist but the dependency was never reached
            # by the missing-import-only recovery loop.
            special_dependency = "pytest-xdist"
        elif "fixture 'httpbin' not found" in result.stdout:
            special_dependency = "pytest-httpbin"
        elif "fixture 'mocker' not found" in result.stdout:
            special_dependency = "pytest-mock"
        elif "Descriptors cannot not be created directly" in result.stdout:
            # Old generated TensorFlow/Keras protobuf modules require the
            # historical protobuf pin already present in requirements.txt.
            special_dependency = "protobuf"
        elif "'Function' object has no attribute 'get_marker'" in result.stdout:
            # pytest-docker-pexpect 0.9 uses the API removed in pytest 4. The
            # override is evidence-backed and acceptance still requires the
            # official fixed-pass/buggy-fail contrast after the downgrade.
            forced_requirement = "pytest==3.10.1"
        elif "X509_V_FLAG_NOTIFY_POLICY" in result.stdout:
            # pyOpenSSL 19.1.0 is incompatible with a modern cryptography
            # wheel. Scrapy's official requirements pin the matching release.
            forced_requirement = _requirement_for_module("cryptography", requirements)
        dependency = (
            missing.group(1) if missing else
            incompatible_import.group(1) if incompatible_import else
            required_library.group(1) if required_library else
            special_dependency
        )
        if result.returncode == 0 or (not dependency and not forced_requirement):
            return result
        requirement = forced_requirement or _requirement_for_module(dependency, requirements)
        if requirement in attempted:
            return result
        attempted.add(requirement)
        installed = None
        installed_requirement = None
        pip_environment = env.copy()
        pip_environment.pop("PYTHONPATH", None)
        pip_environment["PYTHONNOUSERSITE"] = "1"
        for candidate in _installation_candidates(requirement):
            attempt = _run(
                [str(python), "-m", "pip", "install", candidate],
                timeout=300, env=pip_environment,
            )
            if attempt.returncode == 0:
                installed = attempt
                installed_requirement = candidate
                break
        if installed is None:
            if requirement not in evidence["failed_requirements"]:
                evidence["failed_requirements"].append(requirement)
            return result
        if installed_requirement not in evidence["installed_requirements"]:
            evidence["installed_requirements"].append(installed_requirement)
        if dependency == "_black_version" and (cwd / "setup.py").exists():
            generated = _run(
                [str(python), "setup.py", "egg_info"], cwd=cwd,
                timeout=300, env=pip_environment,
            )
            evidence.setdefault("local_metadata_generation", []).append({
                "checkout": cwd.name,
                "returncode": generated.returncode,
                "command": "setup.py egg_info",
            })
            if generated.returncode:
                return result
        result = _run(command, cwd=cwd, timeout=timeout, env=env)
    return result


def _needs_local_extension_build(result: subprocess.CompletedProcess[str]) -> bool:
    return (
        ("C extension:" in result.stdout and "not built" in result.stdout)
        or "cannot import name 'ft2font'" in result.stdout
    )


def _build_local_extensions(checkout: Path, python: Path) -> subprocess.CompletedProcess[str]:
    """Build a historical source checkout only when its own tests require it."""
    setup = checkout / "setup.py"
    if not setup.exists():
        return subprocess.CompletedProcess([], returncode=1, stdout="setup.py is missing")
    return _run([str(python), "setup.py", "build_ext", "--inplace"], cwd=checkout, timeout=900)


def _run_official_f2p(
    task: BugsInPyTask, repository: Path, timeout: int, prepare_environment: bool,
    runner_python: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Return fixed-pass/buggy-fail evidence using the official test command."""
    commands = _command_args(task.test_command)
    if commands is None:
        return False, {"reason": "unsupported_official_test_command"}
    with tempfile.TemporaryDirectory(prefix=f"oneiros-{task.project}-{task.bug_id}-") as directory:
        root = Path(directory)
        fixed_dir, buggy_dir = root / "fixed", root / "buggy"
        try:
            # Fetch both commits before making either worktree.
            repository = RepositoryCache(repository.parent).ensure_commit(task, task.fixed_commit)
            RepositoryCache(repository.parent).ensure_commit(task, task.buggy_commit)
            cache = RepositoryCache(repository.parent)
            cache.worktree(repository, task.fixed_commit, fixed_dir)
            cache.worktree(repository, task.buggy_commit, buggy_dir)
            for test_file in task.test_files:
                source, destination = fixed_dir / test_file, buggy_dir / test_file
                if source.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            environment_evidence: Dict[str, Any] = {"runner": "current_python"}
            commands = [list(command) for command in commands]
            runner = Path(commands[0][0])
            requirements: List[str] = []
            if prepare_environment:
                environment_python, environment_evidence = _prepare_environment(task, root, runner_python)
                if environment_python is None:
                    return False, {"reason": "environment_setup_failed", "environment": environment_evidence}
                for command in commands:
                    command[0] = str(environment_python)
                environment_evidence["runner"] = "isolated_current_python"
                runner = environment_python
                requirements = _task_requirements(task)

            def run_commands(checkout: Path) -> List[subprocess.CompletedProcess[str]]:
                results = []
                for command in commands:
                    result = _run_with_missing_dependency_install(
                        command, checkout, _test_environment(checkout, task), runner,
                        requirements, environment_evidence, timeout,
                    ) if prepare_environment else _run(
                        command, cwd=checkout, timeout=timeout, env=_test_environment(checkout, task)
                    )
                    results.append(result)
                    if result.returncode:
                        break
                return results

            fixed_results = run_commands(fixed_dir)
            buggy_results = run_commands(buggy_dir)
            if prepare_environment and any(
                _needs_local_extension_build(result)
                for result in [*fixed_results, *buggy_results]
            ):
                environment_evidence["local_extension_build_attempted"] = True
                fixed_build = _build_local_extensions(fixed_dir, runner)
                buggy_build = _build_local_extensions(buggy_dir, runner)
                environment_evidence["fixed_extension_build_returncode"] = fixed_build.returncode
                environment_evidence["buggy_extension_build_returncode"] = buggy_build.returncode
                environment_evidence["fixed_extension_build_output_tail"] = fixed_build.stdout[-4000:]
                environment_evidence["buggy_extension_build_output_tail"] = buggy_build.stdout[-4000:]
                if fixed_build.returncode == 0 and buggy_build.returncode == 0:
                    fixed_results = run_commands(fixed_dir)
                    buggy_results = run_commands(buggy_dir)

            def combined_returncode(results: List[subprocess.CompletedProcess[str]]) -> int:
                return next((result.returncode for result in results if result.returncode), 0)

            def output_tail(results: List[subprocess.CompletedProcess[str]]) -> str:
                output = "\n".join(
                    f"$ {' '.join(command)}\n{result.stdout}"
                    for command, result in zip(commands, results)
                )
                return output[-4000:]

            fixed_returncode = combined_returncode(fixed_results)
            buggy_returncode = combined_returncode(buggy_results)
            evidence = {
                "commands": commands,
                "environment": environment_evidence,
                "fixed_returncode": fixed_returncode,
                "buggy_returncode": buggy_returncode,
                "fixed_command_returncodes": [result.returncode for result in fixed_results],
                "buggy_command_returncodes": [result.returncode for result in buggy_results],
                "fixed_output_tail": output_tail(fixed_results),
                "buggy_output_tail": output_tail(buggy_results),
            }
            return fixed_returncode == 0 and buggy_returncode != 0, evidence
        except subprocess.TimeoutExpired:
            return False, {"reason": "official_test_timeout", "timeout_seconds": timeout}
        except RuntimeError as exc:
            return False, {"reason": "worktree_error", "detail": str(exc)}
        finally:
            cache = RepositoryCache(repository.parent)
            if fixed_dir.exists():
                cache.remove_worktree(repository, fixed_dir)
            if buggy_dir.exists():
                cache.remove_worktree(repository, buggy_dir)


def _patch_paths(task: BugsInPyTask) -> List[str]:
    patch = task.bug_dir / "bug_patch.txt"
    if not patch.exists():
        return []
    paths = []
    for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
        if match and match.group(2).endswith(".py"):
            paths.append(match.group(2))
    return paths


def _top_level_functions(code: str) -> Tuple[ast.Module, Dict[str, ast.AST]]:
    tree = ast.parse(code)
    return tree, {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function_module(code: str, tree: ast.Module, node: ast.AST) -> str:
    """Keep non-relative imports and one top-level function for oracle execution."""
    imports = []
    for item in tree.body:
        if isinstance(item, ast.Import) or (isinstance(item, ast.ImportFrom) and not item.level):
            segment = ast.get_source_segment(code, item)
            if segment:
                imports.append(segment)
    function = ast.get_source_segment(code, node)
    return "\n".join([*imports, function or ""]).strip() + "\n"


def _direct_call_assertions(test_source: str, entry_point: str) -> Iterable[str]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    assertions: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        calls_entry_point = any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == entry_point
            for child in ast.walk(node.test)
        )
        if calls_entry_point:
            # The ingestion runtime is Python 3.8, where ast.unparse is not
            # available.  Keeping the original source also avoids changing a
            # verified assertion's formatting while materialising it.
            assertion = ast.get_source_segment(test_source, node)
            if assertion:
                assertions.append(assertion)
    return assertions


def _record_for_pair(
    task: BugsInPyTask, entry_point: str, buggy_code: str, fixed_code: str,
    tests: List[str], source_path: str, official_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": f"official::{task.project}::{task.bug_id}::{entry_point}",
        "task_type": "official_repository_bug_reproduction",
        "language": "python",
        "source": {"name": "bugsinpy_official_materialized", "upstream": "BugsInPy"},
        "group_id": f"project:bugsinpy:{task.project.lower()}",
        "code_under_test": buggy_code,
        "reference_code": fixed_code,
        "entry_point": entry_point,
        "specification": "",
        "tests": [{"code": test, "oracle": "fails_target_passes_reference"} for test in tests],
        "provenance": {
            "official_task_id": task.id,
            "project": task.project,
            "bug_id": task.bug_id,
            "repository_url": task.repository_url,
            "buggy_commit": task.buggy_commit,
            "fixed_commit": task.fixed_commit,
            "patched_source_path": source_path,
            "official_test_command": task.test_command,
            "official_test_evidence": official_evidence,
        },
        "quality": {
            "pair_behaviorally_verified": True,
            "official_targeted_test_fixed_pass_buggy_fail": True,
            "extracted_assertion_fixed_pass_buggy_fail": True,
            "oracle": "fixed_vs_buggy",
            "test_count": len(tests),
        },
    }
    record["content_hash"] = record_content_hash(record)
    return record


def _pytest_selector_from_parts(parts: Sequence[str]) -> Optional[Tuple[str, List[str]]]:
    """Return ``(test_path, selector_parts)`` from one pytest argv."""
    for part in reversed(parts):
        if "::" not in part:
            continue
        path, *selector = part.split("::")
        if path.endswith(".py") and selector and all(selector):
            return path, selector
    return None


def _pytest_selector_from_test_command(command: str) -> Optional[Tuple[str, List[str]]]:
    """Return ``(test_path, selector_parts)`` for one explicit pytest node id."""
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return None
    return _pytest_selector_from_parts(parts)


def _observed_failing_pytest_selector(
    task: BugsInPyTask, official_evidence: Dict[str, Any],
) -> Optional[Tuple[str, List[str]]]:
    """Select the exact official pytest node observed failing on buggy code."""
    commands = official_evidence.get("commands", [])
    buggy_returncodes = official_evidence.get("buggy_command_returncodes", [])
    for command, returncode in zip(commands, buggy_returncodes):
        if returncode:
            selector = _pytest_selector_from_parts(command)
            if selector:
                return selector
    return _pytest_selector_from_test_command(task.test_command)


def _unittest_target_from_parts(
    task: BugsInPyTask, parts: Sequence[str],
) -> Optional[Tuple[str, List[str]]]:
    """Resolve one normalized ``unittest`` argv to its test source file."""
    if "unittest" not in parts:
        return None
    targets = [part for part in parts if not part.startswith("-") and "." in part]
    if not targets:
        return None

    def find_path(module: str) -> Optional[str]:
        expected = module.replace(".", "/") + ".py"
        for test_file in task.test_files:
            normalized = test_file.replace("\\", "/")
            if normalized.lower() == expected.lower():
                return test_file
        return None

    for target in reversed(targets):
        components = target.split(".")
        # First prefer the common module.TestCase.test_method shape.
        if len(components) >= 3:
            path = find_path(".".join(components[:-2]))
            if path:
                return path, components[-2:]
        # Then support module.test_function targets.
        if len(components) >= 2:
            path = find_path(".".join(components[:-1]))
            if path:
                return path, components[-1:]
    return None


def _unittest_target_from_test_command(task: BugsInPyTask) -> Optional[Tuple[str, List[str]]]:
    """Resolve one configured ``unittest`` dotted target as a fallback."""
    try:
        parts = shlex.split(task.test_command, posix=True)
    except ValueError:
        return None
    return _unittest_target_from_parts(task, parts)


def _observed_failing_unittest_target(
    task: BugsInPyTask, official_evidence: Dict[str, Any],
) -> Optional[Tuple[str, List[str]]]:
    """Select the exact unittest target observed failing on buggy code."""
    commands = official_evidence.get("commands", [])
    buggy_returncodes = official_evidence.get("buggy_command_returncodes", [])
    for command, returncode in zip(commands, buggy_returncodes):
        if returncode:
            target = _unittest_target_from_parts(task, command)
            if target:
                return target
    return _unittest_target_from_test_command(task)


def _source_with_decorators(source: str, node: ast.AST) -> str:
    """Extract an AST node with its decorators, preserving valid source text."""
    lines = source.splitlines(keepends=True)
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(decorator.lineno for decorator in decorators)])
    end = getattr(node, "end_lineno", node.lineno)
    return "".join(lines[start - 1:end]).rstrip() + "\n"


def _class_test_fragment(
    source: str, class_node: ast.ClassDef, target: ast.AST,
    required_method_names: Optional[set[str]] = None,
) -> Optional[str]:
    """Return a compact, syntactically complete selected test-class fragment.

    The official repository command remains the behavioural oracle.  This
    fragment is SFT/DPO supervision only, so retaining every unrelated test in
    a class adds tokens without adding evidence.  Keep the selected test,
    recursively referenced class helpers, common framework lifecycle hooks,
    and class attributes.  This preserves the relevant local context while
    avoiding a silent SFT drop for large unittest classes.
    """
    methods = {
        node.name: node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    needed = {getattr(target, "name", ""), *(required_method_names or set())}
    pending = [target]
    lifecycle_names = {
        "setUp", "tearDown", "setUpClass", "tearDownClass", "get_app",
        "get_handlers", "get_new_ioloop",
    }
    needed.update(name for name in lifecycle_names if name in methods)
    pending.extend(methods[name] for name in needed if name in methods and methods[name] is not target)

    while pending:
        method = pending.pop()
        for node in ast.walk(method):
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"self", "cls"}
                and node.attr in methods
                and node.attr not in needed
            ):
                continue
            needed.add(node.attr)
            pending.append(methods[node.attr])

    lines = source.splitlines(keepends=True)
    decorators = getattr(class_node, "decorator_list", [])
    class_start = min([class_node.lineno, *(decorator.lineno for decorator in decorators)])
    first_body_line = min(node.lineno for node in class_node.body)
    class_header = "".join(lines[class_start - 1:first_body_line - 1])
    selected = []
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in needed:
            selected.append(_source_with_decorators(source, node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Class-level fixtures are small and may be read through self.
            selected.append(_source_with_decorators(source, node))
    if not selected:
        return None
    return class_header + "".join(selected)


def _test_fragment(test_source: str, selector_parts: List[str]) -> Optional[str]:
    """Extract a selected pytest or unittest function with its class wrapper.

    A class method on its own is not valid pytest source, so its whole class is
    retained.  This is intentionally source-only validation: the official F2P
    command remains the behavioural oracle for repository-mode records.
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return None
    parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    target_name = selector_parts[-1]
    candidates = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name
    ]
    if len(candidates) != 1:
        return None
    target = candidates[0]
    parent = parents.get(id(target))
    if isinstance(parent, ast.ClassDef):
        requested_class_name = selector_parts[-2] if len(selector_parts) > 1 else parent.name
        if parent.name == requested_class_name:
            fragment = _class_test_fragment(test_source, parent, target)
            if fragment is None:
                return None
        else:
            # unittest permits selecting a test/helper inherited by a subclass
            # (for example ``Subclass._get_spiderargs``). Keep the defining
            # base fragment plus the selected subclass's attributes, lifecycle
            # hooks and overrides; rejecting this shape previously discarded
            # behaviorally verified repository tasks.
            classes = {
                node.name: node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            }
            requested_class = classes.get(requested_class_name)
            if requested_class is None:
                return None

            def base_names(node: ast.ClassDef) -> set[str]:
                names = set()
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        names.add(base.id)
                    elif isinstance(base, ast.Attribute):
                        names.add(base.attr)
                return names

            def inherits(node: ast.ClassDef, ancestor: str, seen: set[str]) -> bool:
                direct = base_names(node)
                if ancestor in direct:
                    return True
                for name in direct - seen:
                    base = classes.get(name)
                    if base is not None and inherits(base, ancestor, seen | {name}):
                        return True
                return False

            if not inherits(requested_class, parent.name, {requested_class.name}):
                return None
            defining_fragment = _class_test_fragment(test_source, parent, target)
            defining_methods = {
                node.name for node in parent.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            selected_methods = {
                node.name for node in requested_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            selected_fragment = _class_test_fragment(
                test_source,
                requested_class,
                target,
                required_method_names=defining_methods & selected_methods,
            )
            if defining_fragment is None or selected_fragment is None:
                return None
            fragment = defining_fragment.rstrip() + "\n\n" + selected_fragment
    elif len(selector_parts) == 1:
        fragment = _source_with_decorators(test_source, target)
    else:
        return None
    try:
        compile(fragment, "<repository-test-fragment>", "exec")
    except SyntaxError:
        return None
    return fragment


def _named_module_nodes(tree: ast.Module) -> Dict[Tuple[str, str], ast.AST]:
    return {
        (type(node).__name__, node.name): node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _compact_changed_class(
    source: str, class_node: ast.ClassDef, changed_member_names: set[str],
) -> str:
    """Keep a class header, its attributes, and only semantically changed methods."""
    lines = source.splitlines(keepends=True)
    decorators = getattr(class_node, "decorator_list", [])
    class_start = min([class_node.lineno, *(item.lineno for item in decorators)])
    first_body_line = min(item.lineno for item in class_node.body)
    parts = ["".join(lines[class_start - 1:first_body_line - 1])]
    for node in class_node.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            parts.append(_source_with_decorators(source, node))
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in changed_member_names
        ):
            parts.append(_source_with_decorators(source, node))
    if len(parts) == 1:
        indentation = " " * (class_node.col_offset + 4)
        parts.append(f"{indentation}pass\n")
    return "".join(parts)


def _changed_definition_context(
    buggy_source: str, fixed_source: str,
) -> Optional[Tuple[str, str]]:
    """Return parseable imports plus only AST definitions changed by the fix."""
    buggy_tree = ast.parse(buggy_source)
    fixed_tree = ast.parse(fixed_source)
    buggy_nodes = _named_module_nodes(buggy_tree)
    fixed_nodes = _named_module_nodes(fixed_tree)
    changed_keys = {
        key for key in set(buggy_nodes) | set(fixed_nodes)
        if key not in buggy_nodes
        or key not in fixed_nodes
        or ast.dump(buggy_nodes[key], include_attributes=False)
        != ast.dump(fixed_nodes[key], include_attributes=False)
    }
    if not changed_keys:
        return None

    def compact(
        source: str, tree: ast.Module, nodes: Dict[Tuple[str, str], ast.AST],
        opposite: Dict[Tuple[str, str], ast.AST],
    ) -> str:
        parts = [
            _source_with_decorators(source, node)
            for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        for key in sorted(changed_keys):
            node = nodes.get(key)
            if node is None:
                continue
            if isinstance(node, ast.ClassDef):
                other = opposite.get(key)
                own_members = {
                    item.name: item for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                other_members = {
                    item.name: item for item in getattr(other, "body", [])
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                changed_members = {
                    name for name in set(own_members) | set(other_members)
                    if name not in own_members
                    or name not in other_members
                    or ast.dump(own_members[name], include_attributes=False)
                    != ast.dump(other_members[name], include_attributes=False)
                }
                parts.append(_compact_changed_class(source, node, changed_members))
            else:
                parts.append(_source_with_decorators(source, node))
        result = "\n".join(part.rstrip() for part in parts if part.strip()) + "\n"
        ast.parse(result)
        return result

    buggy_context = compact(buggy_source, buggy_tree, buggy_nodes, fixed_nodes)
    fixed_context = compact(fixed_source, fixed_tree, fixed_nodes, buggy_nodes)
    if semantic_python(buggy_context) == semantic_python(fixed_context):
        return None
    return buggy_context, fixed_context


def _repository_context(
    task: BugsInPyTask, cache: RepositoryCache, repository: Path,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Return parseable affected-file excerpts for the buggy/fixed task pair."""
    buggy_parts: List[str] = []
    fixed_parts: List[str] = []
    paths: List[str] = []
    for source_path in _patch_paths(task):
        fixed_source = cache.show(repository, task.fixed_commit, source_path)
        buggy_source = cache.show(repository, task.buggy_commit, source_path)
        if not fixed_source or not buggy_source:
            continue
        try:
            ast.parse(fixed_source)
            ast.parse(buggy_source)
        except SyntaxError:
            continue
        compacted = _changed_definition_context(buggy_source, fixed_source)
        if compacted is None:
            compacted_buggy, compacted_fixed = buggy_source, fixed_source
        else:
            compacted_buggy, compacted_fixed = compacted
        header = f"# File: {source_path}\n"
        fixed_parts.append(header + compacted_fixed.rstrip() + "\n")
        buggy_parts.append(header + compacted_buggy.rstrip() + "\n")
        paths.append(source_path)
    if not paths:
        return None, None, []
    return "\n".join(buggy_parts), "\n".join(fixed_parts), paths


def _repository_record(
    task: BugsInPyTask, cache: RepositoryCache, repository: Path,
    official_evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Materialize an official repository pytest test as a distinct record mode."""
    pytest_target = _observed_failing_pytest_selector(task, official_evidence)
    unittest_target = (
        None if pytest_target
        else _observed_failing_unittest_target(task, official_evidence)
    )
    target = pytest_target or unittest_target
    if target is None:
        return None
    execution_mode = "repository_pytest_fragment" if pytest_target else "repository_unittest_fragment"
    test_format = "pytest_fragment" if pytest_target else "unittest_fragment"
    task_type = (
        "official_repository_pytest_reproduction" if pytest_target
        else "official_repository_unittest_reproduction"
    )
    test_path, selector_parts = target
    if test_path not in task.test_files:
        return None
    test_source = cache.show(repository, task.fixed_commit, test_path)
    if not test_source:
        return None
    fragment = _test_fragment(test_source, selector_parts)
    if not fragment:
        return None
    buggy_code, fixed_code, source_paths = _repository_context(task, cache, repository)
    if not buggy_code or not fixed_code:
        return None
    selector = "::".join([test_path, *selector_parts])
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": f"official-repository::{task.project}::{task.bug_id}::{selector.replace('::', '__')}",
        "task_type": task_type,
        "language": "python",
        "source": {"name": "bugsinpy_official_repository_fragment", "upstream": "BugsInPy"},
        "group_id": f"project:bugsinpy:{task.project.lower()}",
        "code_under_test": buggy_code,
        "reference_code": fixed_code,
        "entry_point": "",
        "specification": "",
        "tests": [{
            "code": fragment,
            "oracle": "fixed_passes_buggy_fails_repository",
            "format": test_format,
        }],
        "provenance": {
            "official_task_id": task.id,
            "project": task.project,
            "bug_id": task.bug_id,
            "repository_url": task.repository_url,
            "buggy_commit": task.buggy_commit,
            "fixed_commit": task.fixed_commit,
            "patched_source_paths": source_paths,
            "official_test_command": task.test_command,
            "test_file": test_path,
            "test_selector": selector,
            "official_test_evidence": official_evidence,
        },
        "quality": {
            "pair_behaviorally_verified": True,
            "official_targeted_test_fixed_pass_buggy_fail": True,
            "execution_mode": execution_mode,
            "oracle": "repository_fixed_vs_buggy",
            "test_count": 1,
        },
    }
    record["content_hash"] = record_content_hash(record)
    return record


def materialize_task(
    task: BugsInPyTask, cache: RepositoryCache, timeout: int = 120,
    prepare_environment: bool = False, runner_python: str = sys.executable,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Materialize one official task or return a precise exclusion reason."""
    try:
        repository = cache.ensure_commit(task, task.fixed_commit)
        cache.ensure_commit(task, task.buggy_commit)
    except RuntimeError as exc:
        return None, None, {"task_id": task.id, "status": "excluded", "reason": "repository_fetch_failed", "detail": str(exc)}

    official_ok, official_evidence = _run_official_f2p(
        task, repository, timeout, prepare_environment=prepare_environment
        , runner_python=runner_python
    )
    if not official_ok:
        return None, None, {"task_id": task.id, "status": "excluded", "reason": "official_f2p_failed", "evidence": official_evidence}

    test_sources = [cache.show(repository, task.fixed_commit, path) for path in task.test_files]
    test_sources = [source for source in test_sources if source]
    for source_path in _patch_paths(task):
        fixed_source = cache.show(repository, task.fixed_commit, source_path)
        buggy_source = cache.show(repository, task.buggy_commit, source_path)
        if not fixed_source or not buggy_source:
            continue
        try:
            fixed_tree, fixed_functions = _top_level_functions(fixed_source)
            buggy_tree, buggy_functions = _top_level_functions(buggy_source)
        except SyntaxError:
            continue
        for entry_point in sorted(set(fixed_functions) & set(buggy_functions)):
            fixed_function = fixed_functions[entry_point]
            buggy_function = buggy_functions[entry_point]
            if ast.dump(fixed_function, include_attributes=False) == ast.dump(buggy_function, include_attributes=False):
                continue
            fixed_code = _function_module(fixed_source, fixed_tree, fixed_function)
            buggy_code = _function_module(buggy_source, buggy_tree, buggy_function)
            killers: List[str] = []
            for test_source in test_sources:
                for assertion in _direct_call_assertions(test_source, entry_point):
                    if kills_mutant(assertion, fixed_code, buggy_code):
                        killers.append(assertion)
            if killers:
                killers = sorted(set(killers))
                record = _record_for_pair(
                    task, entry_point, buggy_code, fixed_code, killers, source_path, official_evidence
                )
                return record, None, {"task_id": task.id, "status": "accepted", "record_id": record["id"], "record_mode": "function_assertion"}
    repository_record = _repository_record(task, cache, repository, official_evidence)
    if repository_record:
        return None, repository_record, {
            "task_id": task.id,
            "status": "accepted",
            "record_id": repository_record["id"],
            "record_mode": repository_record["quality"]["execution_mode"],
        }
    return None, None, {
        "task_id": task.id,
        "status": "excluded",
        "reason": "no_self_contained_assertion_pair_or_repository_fragment",
        "evidence": official_evidence,
    }


def task_digest(tasks: Iterable[BugsInPyTask]) -> str:
    payload = "\n".join(f"{task.id}:{task.buggy_commit}:{task.fixed_commit}" for task in tasks)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)
