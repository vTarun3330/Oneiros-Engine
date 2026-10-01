"""Atheris v5 read-only overlays (protocol v2.5 C.1-C.2; WSL, root; stdlib only).

One overlay per CPython minor version, holding ONLY the official Atheris 2.3.0 wheel (no
dependencies, no setuptools/pip/.pth that could shadow a target environment's packages). An
overlay is appended to the sandbox PYTHONPATH after /target; the target's own locked
environment supplies everything else. Each overlay is made read-only and its file manifest is
hashed. A Python version without an Atheris 2.3.0 wheel has no overlay: its targets are the
infrastructure category ``atheris_abi_unavailable`` (never adapter-unsupported, never a
non-kill).

    wsl -u root -- bash scripts/wsl_native_python.sh scripts/v25_atheris_overlay_wsl.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path("/opt/oneiros_atheris_v5")
VERSIONS = ("3.8", "3.9", "3.10", "3.11", "3.12")
ATHERIS = "atheris==2.3.0"


def overlay_dir(version: str) -> Path:
    return ROOT / f"cp{version.replace('.', '')}"


def manifest(path: Path) -> str:
    files = {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(path.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def build(version: str) -> dict:
    dest = overlay_dir(version)
    if dest.exists():
        subprocess.run(["chmod", "-R", "u+w", str(dest)], check=False)
        shutil.rmtree(dest)
    done = subprocess.run(["uv", "pip", "install", "-q", "--target", str(dest), "--no-deps",
                           "--only-binary", ":all:", "--python-version", version, ATHERIS],
                          capture_output=True, text=True)
    if done.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        return {"version": version, "available": False,
                "category": "atheris_abi_unavailable", "detail": done.stderr[-200:]}
    (dest / ".lock").unlink(missing_ok=True)            # uv's install lock, not a package file
    # every file must be listed in Atheris's own wheel RECORD (its sanitizer .so files and the
    # libFuzzer archive are part of the wheel); nothing else may be present
    record = (dest / "atheris-2.3.0.dist-info" / "RECORD").read_text(encoding="utf-8")
    listed = {line.split(",")[0] for line in record.splitlines() if line}
    extra = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts
                   and p.relative_to(dest).as_posix() not in listed)
    if extra:
        return {"version": version, "available": False, "category": "overlay_contaminated",
                "extra": extra}
    digest = manifest(dest)
    subprocess.run(["chmod", "-R", "a-w", str(dest)], check=True)
    return {"version": version, "available": True, "path": str(dest),
            "contents": sorted(p.name for p in dest.iterdir()), "manifest_sha256": digest}


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = [build(v) for v in VERSIONS]
    (ROOT / "overlays.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
