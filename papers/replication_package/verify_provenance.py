"""Cross-platform verification for the planned anonymous artifact.

The same script works in this source checkout and in an assembled archive.  It
uses ``provenance.json`` as the source of truth instead of assuming the POSIX
``sha256sum`` command is installed.  With ``--write-sha256sums`` it also emits
the conventional checksum file after (and only after) every declared digest
has been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_declared_path(declared: str) -> Path | None:
    """Find a declared file in the checkout or a flat assembled archive."""
    candidates = (
        HERE.parent.parent / declared,
        HERE / declared,
        HERE / Path(declared).name,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-sha256sums", action="store_true")
    args = parser.parse_args(argv)

    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    contents = provenance["contents"]
    expected_count = provenance["verification"]["expected_file_count"]
    if len(contents) != expected_count:
        raise SystemExit(
            f"REFUSED: provenance declares {len(contents)} files, expected {expected_count}"
        )

    verified: list[tuple[str, str]] = []
    failures: list[str] = []
    for name, entry in contents.items():
        declared = entry["path"]
        path = resolve_declared_path(declared)
        if path is None:
            failures.append(f"{name}: missing {declared}")
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            failures.append(
                f"{name}: digest mismatch for {declared}: {actual} != {entry['sha256']}"
            )
            continue
        verified.append((actual, path.name))
        print(f"OK  {name}: {declared}")

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 2

    if args.write_sha256sums:
        target = HERE / "SHA256SUMS"
        target.write_text(
            "".join(f"{digest}  {name}\n" for digest, name in verified),
            encoding="utf-8",
        )
        print(f"written: {target}")
    print(f"verified: {len(verified)}/{expected_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
