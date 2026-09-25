"""All-or-nothing publication of generated artifacts.

Artifacts are staged as temporary files beside their targets, the staged bytes
are re-read and verified, and only then promoted with ``os.replace`` (atomic
per file).  If a verification or any promotion fails, every target that was
already promoted is restored to its previous bytes and every staged file is
removed, so the previously accepted artifacts are preserved.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping


class PublicationRefused(RuntimeError):
    """A gate failed; nothing was published."""


def _stage(target: Path, data: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".staged",
                                    dir=target.parent)
    with os.fdopen(handle, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return Path(name)


def publish_atomically(artifacts: Mapping[Path, bytes],
                       verify: Callable[[Mapping[Path, Path]], None] | None = None) -> None:
    """Stage, verify, then promote ``artifacts`` ({target: bytes}) in the given order.

    ``verify`` receives {target: staged path} and raises to refuse publication.
    """
    staged: dict[Path, Path] = {}
    previous: dict[Path, bytes | None] = {}
    promoted: list[Path] = []
    try:
        for target, data in artifacts.items():
            staged[Path(target)] = _stage(Path(target), data)
        for target, path in staged.items():
            if path.read_bytes() != artifacts[target]:
                raise PublicationRefused(f"staged bytes differ for {target}")
        if verify is not None:
            verify(dict(staged))
        for target in staged:
            previous[target] = target.read_bytes() if target.exists() else None
        for target, path in staged.items():
            os.replace(path, target)
            promoted.append(target)
    except BaseException:
        for target in reversed(promoted):
            if previous[target] is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(_stage(target, previous[target]), target)
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
