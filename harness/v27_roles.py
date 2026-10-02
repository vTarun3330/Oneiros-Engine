"""v2.7 role isolation (stdlib only). Every record carries an explicit role; roles never mix.

- confirmation records may never enter SFT, replay, relearning or any training corpus;
- training-expansion records may never enter the confirmation panel;
- a record without role metadata, or a set mixing roles, fails closed;
- repository identity is the owner-independent key, so aliases / renames / forks of one
  project collide and cannot bypass isolation.
"""
from __future__ import annotations

from typing import Iterable, Mapping

CONFIRMATION = "CONFIRMATION_ONLY"
TRAINING = "TRAINING_EXPANSION"
ROLES = (CONFIRMATION, TRAINING)


class RoleViolation(ValueError):
    """A record would cross a role boundary."""


def repository_key(name: str) -> str:
    name = str(name).lower().rstrip("/").removesuffix(".git").split("github.com/")[-1]
    return name.replace("__", "/").split("/")[-1]


def _role(record: Mapping) -> str:
    role = record.get("role")
    if role not in ROLES:
        raise RoleViolation(f"missing or unknown role metadata: {role!r}")
    return role


def admit_to_training(records: Iterable[Mapping],
                      confirmation_keys: Iterable[str] = ()) -> list:
    """Refuse any confirmation record (by role OR by repository key / alias)."""
    blocked = {repository_key(k) for k in confirmation_keys}
    out = []
    for r in records:
        if _role(r) != TRAINING:
            raise RoleViolation(f"confirmation record offered to training: {r.get('id')}")
        if repository_key(r.get("repository", "")) in blocked:
            raise RoleViolation(f"repository of the confirmation pool (alias/rename): "
                                f"{r.get('repository')}")
        out.append(r)
    return out


def admit_to_confirmation(records: Iterable[Mapping],
                          training_keys: Iterable[str] = ()) -> list:
    blocked = {repository_key(k) for k in training_keys}
    out = []
    for r in records:
        if _role(r) != CONFIRMATION:
            raise RoleViolation(f"training record offered to the confirmation panel: "
                                f"{r.get('id')}")
        if repository_key(r.get("repository", "")) in blocked:
            raise RoleViolation(f"repository of the training pools: {r.get('repository')}")
        out.append(r)
    return out


def candidate_overlap(candidates: Iterable[Mapping], *, training_fingerprints: Iterable[str],
                      prior_commits: Iterable[str], training_lineages: Iterable[str] = ()
                      ) -> list:
    """Post-acquisition candidate-level overlap (function fingerprint, commit, lineage)."""
    fps, commits, lin = set(training_fingerprints), set(prior_commits), set(training_lineages)
    hits = []
    for c in candidates:
        why = [k for k, ok in (("function_fingerprint", c.get("function_fingerprint") in fps),
                               ("commit", c.get("fixed_commit") in commits
                                or c.get("buggy_commit") in commits),
                               ("lineage", c.get("lineage") in lin)) if ok]
        if why:
            hits.append({"id": c.get("id"), "overlap": why})
    return hits
