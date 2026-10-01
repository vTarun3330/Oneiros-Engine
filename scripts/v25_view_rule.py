"""v2.5 view rule v2 (stdlib only; used by the v2.5 native environment builder).

Identical to the frozen executor's ``build_view`` (which is not modified) except for ONE
exception: a directory named ``testing`` is kept when
1. it is a Python package (has ``__init__.py``) INSIDE a top-level import package of the
   import root (``sympy/testing``, never a top-level ``testing/``), and
2. the packaging metadata of that same revision ships it: the dotted name appears literally
   in ``setup.py`` / ``setup.cfg`` / ``pyproject.toml``, or packaging uses ``find_packages`` /
   ``packages = find:`` / ``[tool.setuptools.packages.find]`` and does not mention the dotted
   name in an exclude.
Inside a kept ``testing`` package, test files (``test_*.py``, ``*_test.py``, ``conftest.py``)
and ``test``/``tests`` directories are still stripped, as everywhere else. Official test files
are separately proven absent by ``official_files_absent``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

VERSION = "oneiros_v25_view_rule_v2"
TEST_DIRS = {"test", "tests", "testing"}
STRIP_ALWAYS = {"test", "tests"}
OTHER_SKIP = {".git", "__pycache__", ".tox", ".nox", ".venv", "venv", "build", "dist", ".eggs",
              ".mypy_cache", ".pytest_cache", "docs", "doc"}
PACKAGING = ("setup.py", "setup.cfg", "pyproject.toml")


def is_test_file(name: str) -> bool:
    return name == "conftest.py" or (name.startswith("test_") and name.endswith(".py")) \
        or name.endswith("_test.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


VERSION_V3 = "oneiros_v26_view_rule_v3"


def _contains_test_files(directory: Path) -> bool:
    return any(p.is_file() and is_test_file(p.name) for p in directory.rglob("*.py"))


def shipped_testing_packages(checkout: Path, root_rel: str,
                             names: tuple = ("testing",)) -> dict:
    """{dotted name: evidence} for every ``<pkg>/.../<name>`` package the revision ships.
    v2: names = ("testing",). v3 adds "test": a ``<pkg>/test`` package is kept only when it
    contains NO test file at any depth (runtime test API such as ``django.test``, never a
    project test suite such as ``tornado/test``)."""
    source_root = checkout / root_rel if root_rel else checkout
    meta = {n: (checkout / n).read_text(encoding="utf-8", errors="replace")
            for n in PACKAGING if (checkout / n).is_file()}
    text = "\n".join(meta.values())
    finds = re.search(r"find_packages|find_namespace_packages|packages\s*=\s*find:|"
                      r"tool\.setuptools\.packages\.find", text) is not None
    found = {}
    for top in sorted(p for p in source_root.iterdir()
                      if p.is_dir() and (p / "__init__.py").exists()):
        for d in sorted(x for name in names for x in top.rglob(name)):
            rel = d.relative_to(source_root)
            if not d.is_dir() or not (d / "__init__.py").exists() or \
                    any(part in OTHER_SKIP | STRIP_ALWAYS for part in rel.parts[:-1]) or \
                    rel.parts[-1] in OTHER_SKIP:
                continue
            if rel.parts[-1] in STRIP_ALWAYS and _contains_test_files(d):
                continue
            # every ancestor must be a package too (importable as a dotted name)
            if not all((source_root / Path(*rel.parts[:i]) / "__init__.py").exists()
                       for i in range(1, len(rel.parts))):
                continue
            dotted = ".".join(rel.parts)
            literal = re.search(rf"['\"]{re.escape(dotted)}['\"]", text) is not None
            excluded = re.search(rf"exclude[^\n]*{re.escape(dotted)}", text) is not None
            if not excluded and (literal or finds):
                found[dotted] = {"evidence": "literal" if literal else "find_packages",
                                 "metadata": sorted(meta)}
    return found


def build_view_v3(checkout: Path, root_rel: str, dest: Path, extra: dict | None = None) -> dict:
    """v2 plus shipped, test-free ``<pkg>/test`` runtime packages."""
    return build_view_v2(checkout, root_rel, dest, extra, names=("testing", "test"),
                         version=VERSION_V3)


def build_view_v2(checkout: Path, root_rel: str, dest: Path, extra: dict | None = None,
                  names: tuple = ("testing",), version: str = VERSION) -> dict:
    source_root = checkout / root_rel if root_rel else checkout
    shipped = shipped_testing_packages(checkout, root_rel, names)
    keep = {tuple(d.split(".")) for d in shipped}
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    tops = []
    for entry in sorted(source_root.iterdir()):
        if entry.is_dir() and entry.name not in TEST_DIRS | OTHER_SKIP and \
                (entry / "__init__.py").exists():
            tops.append(entry)
        elif entry.is_file() and entry.suffix == ".py" and not is_test_file(entry.name) and \
                entry.name not in ("setup.py", "noxfile.py", "tasks.py", "conftest.py"):
            tops.append(entry)

    def skipped(rel: Path) -> bool:
        for i, part in enumerate(rel.parts[:-1] if rel.suffix else rel.parts):
            if tuple(rel.parts[:i + 1]) in keep:
                continue
            if part in OTHER_SKIP or part in STRIP_ALWAYS:
                return True
            if part == "testing" and tuple(rel.parts[:i + 1]) not in keep:
                return True
        return False
    for top in tops:
        if top.is_file():
            shutil.copy2(top, dest / top.name)
            continue
        for path in sorted(top.rglob("*")):
            rel = path.relative_to(source_root)
            if path.is_dir() or skipped(rel) or is_test_file(path.name) or \
                    path.name.startswith(".git"):
                continue
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / rel)
    for rel, data in (extra or {}).items():
        target = dest / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    files = {p.relative_to(dest).as_posix(): _sha(p) for p in sorted(dest.rglob("*"))
             if p.is_file()}
    return {"files": len(files), "view_rule": version, "kept_testing_packages": shipped,
            "manifest_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode())
            .hexdigest(), "tops": [t.name for t in tops]}


def official_files_absent(view: Path, root_rel: str, official_paths) -> dict:
    """Every official test file (selector paths, test-patch files) is absent from the view."""
    present = []
    prefix = (root_rel + "/") if root_rel else ""
    for rel in official_paths:
        rel = rel.split("::")[0]
        inside = rel[len(prefix):] if prefix and rel.startswith(prefix) else rel
        if (view / inside).exists():
            present.append(rel)
    return {"ok": not present, "present": present}
