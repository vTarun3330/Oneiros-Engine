"""Source identity that survives a checkout.

A source hash exists to answer one question: was the code that produced this
number the code we think it was. On one machine, hashing raw file bytes answers
it perfectly. Across machines it does not, and the failure is silent.

On this Windows system ``scripts/train_on_dataset.py`` has CRLF line endings
while Git stores it with LF. A fresh clone on Linux, or a change to
``core.autocrlf``, produces a file with identical program semantics and a
different SHA-256. Anyone re-deriving the hash from the repository would find a
mismatch and have to decide whether the runner had really changed - which is
exactly the judgement call a provenance hash is supposed to remove.

So two hashes are recorded, and they answer different questions:

``raw_sha256``
    The bytes that were actually on disk on the machine that ran the job. This
    is the stronger claim about *this* execution: it is what the interpreter
    opened. It is not portable and is not meant to be.

``canonical_sha256``
    SHA-256 over the same content with line endings normalized to LF. Two files
    that differ only in line endings share this hash; any change to a character
    of source changes it. This is the claim that can be re-derived anywhere.

``git_blob_sha1``
    Git's own object identity for the canonical content, computed offline:
    ``sha1(b"blob <len>\\0" + canonical_bytes)``. Recorded because it can be
    checked against the repository with ``git rev-parse HEAD:<path>`` by anyone,
    without trusting this module's arithmetic.

Neither replaces the other, and nothing here rewrites a file. Normalizing the
working tree to fix the portability problem would change the raw hashes that
every existing receipt in this project was computed against, which would break
the frozen development contract to tidy up a reporting concern. The fix belongs
in what is recorded, not in the files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

#: Bump when the canonicalization rule changes. Recorded beside every hash so a
#: reader never has to infer which rule produced a digest.
HASH_SCHEME_VERSION = "oneiros_portable_source_identity_v1"

#: What the scheme does, in the receipt itself, so it is not folklore.
HASH_SCHEME_DESCRIPTION = (
    "canonical_sha256 = SHA-256 over the file's bytes with CRLF and lone CR "
    "converted to LF. No other transformation: encoding is untouched, a byte "
    "order mark is preserved, and trailing whitespace is preserved. "
    "git_blob_sha1 = SHA-1 over b'blob <len>\\0' + the same canonical bytes, "
    "which is Git's object identity and is checkable with "
    "'git rev-parse HEAD:<path>'."
)

#: Every source that decides what an evaluation measures or how it is scored.
#: Keyed by role rather than by path, because the receipt should say what a file
#: does and not only where it sits.
EVALUATION_DEFINING_SOURCES: Dict[str, str] = {
    "evaluation_entrypoint": "scripts/train_on_dataset.py",
    "evaluator": "metrics/research_evaluation.py",
    "candidate_policy": "harness/candidate_policy.py",
    "timeout_policy_safe_execution": "harness/safe_execution.py",
    "prompt_builder": "engine/test_generation_prompt.py",
    "prompt_budget": "engine/prompt_budget.py",
    "generator": "engine/generator.py",
    "model_runtime": "engine/model_runtime.py",
}

#: The adapter-resolution logic is deliberately absent from the table above.
#: It is a fragment inside the evaluation entrypoint rather than a file, so it
#: is hashed from ``inspect.getsource`` at runtime and recorded separately.
#: Listing it here against some file's path would have been a tidy-looking lie.
ADAPTER_RESOLUTION_IS_A_CODE_FRAGMENT = True


def canonical_bytes(data: bytes) -> bytes:
    """Return ``data`` with every line ending as LF.

    CRLF first, then any surviving lone CR, so old-Mac endings do not survive
    as a silent difference between two otherwise identical files.
    """
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def raw_sha256(path) -> str:
    """SHA-256 of the bytes on this machine's disk."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(path) -> str:
    """SHA-256 of the LF-normalized content; equal across checkouts."""
    return hashlib.sha256(canonical_bytes(Path(path).read_bytes())).hexdigest()


def canonical_text_sha256(text: str) -> str:
    """Canonical hash of source held as a string rather than a file.

    Used for the adapter-resolution fragment, which is read out of the running
    module with ``inspect.getsource`` and so never exists as a file of its own.
    """
    return hashlib.sha256(canonical_bytes(text.encode("utf-8"))).hexdigest()


def git_blob_sha1(path) -> str:
    """Git's object id for the canonical content, computed without Git."""
    content = canonical_bytes(Path(path).read_bytes())
    header = f"blob {len(content)}".encode("utf-8") + b"\0"
    return hashlib.sha1(header + content).hexdigest()


def source_identity(root, relative_path: str) -> Dict[str, object]:
    """Both hashes for one file, plus enough context to interpret them."""
    path = Path(root) / relative_path
    return {
        "path": relative_path,
        "raw_sha256": raw_sha256(path),
        "canonical_sha256": canonical_sha256(path),
        "git_blob_sha1": git_blob_sha1(path),
        "raw_and_canonical_agree": raw_sha256(path) == canonical_sha256(path),
        "bytes": path.stat().st_size,
    }


def evaluation_source_identities(root) -> Dict[str, Dict[str, object]]:
    """Identity for every evaluation-defining source, keyed by role."""
    return {
        role: source_identity(root, relative)
        for role, relative in sorted(EVALUATION_DEFINING_SOURCES.items())
    }


def scheme_block(root) -> Dict[str, object]:
    """The whole binding, ready to drop into a contract or a receipt."""
    identities = evaluation_source_identities(root)
    return {
        "hash_scheme_version": HASH_SCHEME_VERSION,
        "hash_scheme": HASH_SCHEME_DESCRIPTION,
        "sources": identities,
        "portable_hashes_differ_from_raw": sorted(
            role for role, item in identities.items()
            if not item["raw_and_canonical_agree"]
        ),
    }
