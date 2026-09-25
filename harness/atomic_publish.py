"""Crash-safe publication of generated artifacts.

Two mechanisms, with honestly stated guarantees:

``publish_file_atomically``
    ONE file.  Staged beside its target, verified, fsynced, then promoted with a
    single ``os.replace``.  A crash leaves either the old or the new file.

``publish_bundle`` / ``read_current_bundle``
    SEVERAL files that must be read together.  Each publication writes a new,
    immutable generation directory::

        <store>/generations/<generation id>/<files...>
        <store>/generations/<generation id>/MANIFEST.json
        <store>/CURRENT                      (small pointer file)

    Files and the manifest are written into a private ``.partial-*`` directory,
    fsynced, re-read and verified, and only then renamed to their final
    generation directory; the pointer is replaced last, in one ``os.replace``.
    The generation ID is derived from the manifest's SHA-256, and an existing
    generation is never overwritten.  Readers follow the pointer, require the
    manifest hash and every listed file hash to match, and reject unlisted
    files.  So a crash at any point leaves readers with either the complete old
    generation or the complete new one - never a mixture.

Durability note: on POSIX the directories are fsynced too.  Windows offers no
directory fsync; there the ordering relies on NTFS metadata journaling, and a
crash can at worst lose the newest pointer update (leaving the complete old
generation current), never produce a mixed generation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping

BUNDLE_SCHEMA = "oneiros_artifact_bundle_v1"
POINTER = "CURRENT"
MANIFEST = "MANIFEST.json"


class PublicationRefused(RuntimeError):
    """A gate failed; nothing was published."""


class BundleUnreadable(RuntimeError):
    """The current bundle is missing or does not verify."""


def _checkpoint(point: str) -> None:
    """Publication points; tests replace this to simulate a crash at each one."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return  # Windows: no directory handles; NTFS journals metadata.
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def _write_synced(path: Path, data: bytes) -> None:
    with open(path, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


# --- one file ------------------------------------------------------------------------

def publish_file_atomically(target: Path, data: bytes,
                            verify: Callable[[Path], None] | None = None) -> None:
    """Per-file atomic: a crash leaves the old or the new file, nothing else."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".staged",
                                    dir=target.parent)
    staged = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if staged.read_bytes() != data:
            raise PublicationRefused(f"staged bytes differ for {target}")
        if verify is not None:
            verify(staged)
        _checkpoint("file_staged")
        os.replace(staged, target)
        _fsync_directory(target.parent)
    except Exception:
        staged.unlink(missing_ok=True)
        raise


# --- bundles -------------------------------------------------------------------------

def _manifest_bytes(files: Mapping[str, bytes], previous: str | None) -> bytes:
    manifest = {"schema_version": BUNDLE_SCHEMA, "previous_generation": previous,
                "files": {name: {"sha256": _sha(data), "bytes": len(data)}
                          for name, data in sorted(files.items())}}
    return (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _generation_id(manifest: bytes) -> str:
    return _sha(manifest)[:24]


def _read_pointer(store: Path) -> dict | None:
    path = store / POINTER
    if not path.exists():
        return None
    try:
        pointer = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BundleUnreadable(f"pointer is not JSON: {exc}") from exc
    if pointer.get("schema_version") != BUNDLE_SCHEMA:
        raise BundleUnreadable("pointer schema differs")
    return pointer


def verify_generation(store: Path, generation: str, manifest_sha256: str) -> dict[str, bytes]:
    """Every file of one generation, verified against its manifest; raises otherwise."""
    directory = Path(store) / "generations" / generation
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise BundleUnreadable(f"generation {generation} has no manifest")
    manifest_bytes = manifest_path.read_bytes()
    if _sha(manifest_bytes) != manifest_sha256 or _generation_id(manifest_bytes) != generation:
        raise BundleUnreadable(f"manifest of {generation} does not match the pointer")
    manifest = json.loads(manifest_bytes)
    listed = manifest.get("files") or {}
    present = {path.name for path in directory.iterdir()} - {MANIFEST}
    if present != set(listed):
        raise BundleUnreadable(f"generation {generation} files differ from its manifest")
    files: dict[str, bytes] = {}
    for name, entry in listed.items():
        data = (directory / name).read_bytes()
        if _sha(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise BundleUnreadable(f"{name} in {generation} does not match its manifest")
        files[name] = data
    return files


def read_current_bundle(store: Path) -> dict:
    """The complete, verified generation the pointer selects."""
    store = Path(store)
    pointer = _read_pointer(store)
    if pointer is None:
        raise BundleUnreadable(f"no published bundle in {store}")
    files = verify_generation(store, pointer["generation"], pointer["manifest_sha256"])
    return {"generation": pointer["generation"], "manifest_sha256": pointer["manifest_sha256"],
            "files": files}


def publish_bundle(store: Path, files: Mapping[str, bytes],
                   verify: Callable[[Mapping[str, Path]], None] | None = None) -> str:
    """Publish ``files`` ({name: bytes}) as one new generation; return its ID.

    ``verify`` receives {name: staged path} and raises to refuse publication.
    """
    store = Path(store)
    generations = store / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    if any(name == MANIFEST or "/" in name or "\\" in name or name.startswith(".")
           for name in files):
        raise PublicationRefused("invalid bundle file name")
    pointer = _read_pointer(store)
    previous = pointer["generation"] if pointer else None
    manifest = _manifest_bytes(files, previous)
    generation = _generation_id(manifest)
    final = generations / generation
    partial = Path(tempfile.mkdtemp(prefix=".partial-", dir=generations))
    staged_pointer: Path | None = None
    try:
        for name, data in sorted(files.items()):
            _write_synced(partial / name, data)
            _checkpoint(f"staged_file:{name}")
        _write_synced(partial / MANIFEST, manifest)
        _checkpoint("staged_manifest")
        _fsync_directory(partial)
        _checkpoint("staged_directory_synced")
        for name, data in files.items():
            if (partial / name).read_bytes() != data:
                raise PublicationRefused(f"staged bytes differ for {name}")
        if verify is not None:
            verify({name: partial / name for name in files})
        if final.exists():
            # Identical content is already an accepted generation: never overwrite it.
            verify_generation(store, generation, _sha(manifest))
            shutil.rmtree(partial)
        else:
            os.rename(partial, final)
            _fsync_directory(generations)
        _checkpoint("generation_renamed")
        pointer_bytes = (json.dumps({"schema_version": BUNDLE_SCHEMA, "generation": generation,
                                     "manifest_sha256": _sha(manifest)}, indent=1,
                                    sort_keys=True) + "\n").encode("utf-8")
        handle, name = tempfile.mkstemp(prefix=".CURRENT.", suffix=".staged", dir=store)
        os.close(handle)
        staged_pointer = Path(name)
        _write_synced(staged_pointer, pointer_bytes)
        _checkpoint("pointer_staged")
        os.replace(staged_pointer, store / POINTER)
        _fsync_directory(store)
        _checkpoint("pointer_replaced")
    except Exception:
        # An ordinary failure cleans up; a crash (process death) leaves only an
        # ignored .partial directory or staged pointer, never a mixed generation.
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        if staged_pointer is not None:
            staged_pointer.unlink(missing_ok=True)
        raise
    return generation
