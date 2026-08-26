"""Content-addressed provenance for training and evaluation runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import platform
from typing import Dict, Iterable


_SOURCE_DIRECTORIES = (
    "baseline", "config", "engine", "harness", "metrics", "scripts", "tests", "utils",
)
_ROOT_FILES = ("requirements.txt", "pytest.ini")


def _included_files(project_root: Path) -> Iterable[Path]:
    for directory in _SOURCE_DIRECTORIES:
        root = project_root / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {
                ".py", ".json", ".toml", ".yaml", ".yml", ".ini",
            }:
                yield path
    for name in _ROOT_FILES:
        path = project_root / name
        if path.is_file():
            yield path


def source_tree_sha256(project_root: Path) -> str:
    """Hash executable source/configuration independent of Git availability."""
    project_root = Path(project_root).resolve()
    digest = hashlib.sha256()
    for path in sorted(set(_included_files(project_root)), key=lambda item: item.as_posix()):
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reproducibility_manifest(
    project_root: Path, model_name: str, model_revision: str,
) -> Dict[str, object]:
    """Return immutable identities stored beside every new run artifact."""
    project_root = Path(project_root).resolve()
    requirements = project_root / "requirements.txt"
    critical_packages = (
        "torch", "transformers", "peft", "trl", "datasets", "bitsandbytes",
        "accelerate", "sentence-transformers", "sentencepiece", "protobuf", "faiss-cpu",
    )
    runtime_dependencies = {}
    for package in critical_packages:
        try:
            runtime_dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime_dependencies[package] = "not-installed"
    return {
        "schema_version": "1",
        "source_tree_sha256": source_tree_sha256(project_root),
        "dependency_spec_sha256": file_sha256(requirements) if requirements.exists() else "missing",
        "model_name": model_name,
        "model_revision": model_revision,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "runtime_dependencies": runtime_dependencies,
    }
