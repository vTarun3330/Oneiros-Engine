"""Provision isolated legacy CPython runtimes for BugsInPy ingestion.

The installers are placed under the project data directory and are not added
to PATH. These end-of-life interpreters are used only for offline historical
benchmark reproduction, never for Oneiros itself.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = ROOT / "data" / "bugsinpy_v2_ingestion" / "runtimes"
RELEASES = {
    "3.6": "3.6.8",
    "3.7": "3.7.9",
    "3.8": "3.8.10",
}


def provision(version: str) -> Path:
    if version not in RELEASES:
        raise ValueError(f"Unsupported runtime family {version}; choose one of {sorted(RELEASES)}")
    release = RELEASES[version]
    target = RUNTIME_ROOT / f"python{version}" / "python.exe"
    if target.exists():
        return target
    if sys.platform != "win32":
        raise RuntimeError("This provisioning helper currently supports Windows only.")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    installer = RUNTIME_ROOT / f"python-{release}-amd64.exe"
    url = f"https://www.python.org/ftp/python/{release}/python-{release}-amd64.exe"
    if not installer.exists():
        print(f"Downloading CPython {release} from python.org...", flush=True)
        urllib.request.urlretrieve(url, installer)
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    print(f"Installer SHA-256 ({release}): {digest}")
    target.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        "/quiet", "InstallAllUsers=0", f"TargetDir={target.parent}",
        "PrependPath=0", "Include_launcher=0", "Include_test=0",
    ]
    result = subprocess.run([str(installer), *arguments], check=False)
    if result.returncode or not target.exists():
        raise RuntimeError(f"CPython {release} installation failed with exit code {result.returncode}.")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision project-local legacy Python runtimes")
    parser.add_argument("--versions", default="3.6,3.7,3.8", help="Comma-separated major.minor runtime families")
    args = parser.parse_args()
    for version in [item.strip() for item in args.versions.split(",") if item.strip()]:
        executable = provision(version)
        print(f"Ready: {executable}")


if __name__ == "__main__":
    main()
