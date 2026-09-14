"""One explicit model-identity block, written the same way by every receipt.

An audit found the identity was present but scattered: Arm A carried the model
name and revision under ``tokenization`` and its source tree under
``run_identity``, while Arm B carried neither its own source tree nor its own
tokenizer identity and described the model only by pointing at Arm A. A reader
had to know where to look and to trust a link, which is not the same as a
receipt stating what it used.

Scattering is also how the revision defect survived: three inline resolvers
disagreed with each other for weeks because no single place stated the answer.
So this module builds the block, and both preflights call it rather than each
assembling their own.

Nothing here resolves anything. It records what the resolver returned, and
refuses to record a value that is not an immutable identity.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

SCHEMA = "oneiros_model_identity_v1"

#: A revision identifies a model only if it names one immutable commit.
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")

#: References that move. Recording one of these is recording nothing.
MOVING_REFS = frozenset({"main", "master", "latest", "head", "HEAD",
                         "refs/heads/main", "dev", "", "None", "null"})


def is_immutable_revision(revision: object) -> bool:
    """True only for a full 40-character lowercase commit SHA."""
    return bool(IMMUTABLE_REVISION.match(str(revision or "")))


def describe_revision(revision: object) -> str:
    """Why a revision is not acceptable, for an error a human can act on."""
    text = str(revision or "")
    if not text or text == "None":
        return "empty"
    if text.lower() in MOVING_REFS:
        return f"the moving reference {text!r}"
    if re.fullmatch(r"[0-9a-fA-F]{4,39}", text):
        return f"the shortened SHA {text!r} ({len(text)} of 40 characters)"
    if IMMUTABLE_REVISION.match(text):
        return "immutable"
    return f"the non-immutable value {text!r}"


def build(*, model_name: str, model_revision: str,
          tokenizer_name: str, tokenizer_revision: str,
          source_tree_sha256: str, protocol_name: str, protocol_sha256: str,
          ) -> dict[str, Any]:
    """The block a receipt records verbatim."""
    return {
        "schema_version": SCHEMA,
        "base_model_name": model_name,
        "base_model_revision": model_revision,
        "base_model_revision_is_immutable": is_immutable_revision(model_revision),
        "selection_tokenizer_name": tokenizer_name,
        "selection_tokenizer_revision": tokenizer_revision,
        "selection_tokenizer_revision_is_immutable":
            is_immutable_revision(tokenizer_revision),
        "model_and_tokenizer_identical": (
            model_name == tokenizer_name and model_revision == tokenizer_revision
        ),
        "source_tree_sha256": source_tree_sha256,
        "successor_protocol_name": protocol_name,
        "successor_protocol_sha256": protocol_sha256,
        # What a training command launched from this receipt must load. Stated
        # rather than implied, so a later run can be checked against it.
        "expected_training_model_identity": {
            "model_name": model_name,
            "model_revision": model_revision,
            "tokenizer_name": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
        },
    }


def problems(identity: dict[str, Any] | None, *, label: str) -> list[str]:
    """Every way an identity block fails to identify anything."""
    found: list[str] = []
    if not identity:
        return [f"{label} records no model identity block at all"]

    if identity.get("schema_version") != SCHEMA:
        found.append(
            f"{label} model identity schema is "
            f"{identity.get('schema_version')!r}, expected {SCHEMA!r}")

    for field in ("base_model_name", "selection_tokenizer_name"):
        if not identity.get(field):
            found.append(f"{label} records no {field}")

    for field in ("base_model_revision", "selection_tokenizer_revision"):
        value = identity.get(field)
        if not is_immutable_revision(value):
            found.append(
                f"{label} {field} is {describe_revision(value)}, not a full "
                "40-character immutable snapshot SHA")

    if not identity.get("source_tree_sha256"):
        found.append(
            f"{label} records no source_tree_sha256, so it cannot say which "
            "source produced it")

    if not identity.get("successor_protocol_sha256"):
        found.append(f"{label} records no successor_protocol_sha256")

    if identity.get("model_and_tokenizer_identical") is not True:
        found.append(
            f"{label} model and selection tokenizer identities differ; the "
            "tokenizer decides supervision eligibility and must be the "
            "model's own")
    return found


def disagreements(arm_a: dict[str, Any], arm_b: dict[str, Any]) -> list[str]:
    """Fields the two arms must agree on exactly, or they are not one test."""
    found: list[str] = []
    for field in ("base_model_name", "base_model_revision",
                  "selection_tokenizer_name", "selection_tokenizer_revision",
                  "successor_protocol_name", "successor_protocol_sha256"):
        if arm_a.get(field) != arm_b.get(field):
            found.append(
                f"{field} differs between the arms: "
                f"{arm_a.get(field)!r} vs {arm_b.get(field)!r}")
    return found


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
