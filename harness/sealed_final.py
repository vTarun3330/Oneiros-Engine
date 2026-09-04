"""One-time authorized access to the sealed final-test split.

The sealed split is the only measurement in this project that cannot be
repeated: once it has been looked at, every later decision is contaminated by
having seen it.  So the guard here is not about secrecy, it is about ordering.
Nothing may be tuned after the split is opened, which means everything a result
depends on has to be frozen *before* it is opened.

Three properties are enforced structurally rather than by convention:

1. **Everything is frozen first.** A final bundle names the adapter, base model,
   corpus and split hashes, prompt builder, budgets, candidate count, seeds,
   sampling, timeouts, evaluator, baselines, and the selection rule. Its hash
   covers all of it, so a later edit to any one of them invalidates the bundle.

2. **Authorization is one-time and explicit.** A token is issued against one
   bundle hash. Reusing a spent token is refused, which prevents the split from
   being opened twice with a "small fix" in between.

3. **Development commands cannot reach the split at all.** They ask through the
   same guard and are refused without a token, so accidental access fails
   loudly instead of silently returning sealed records.

Every attempt - allowed or refused - is appended to an audit log.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SEALED_FINAL_SCHEMA_VERSION = "oneiros_sealed_final_v1"
SEALED_SPLIT = "test"

#: Everything that must be pinned before the sealed split may be opened.
#: A missing field is a refusal, not a warning: an unfrozen knob is exactly the
#: thing that would let a result be tuned after the fact.
REQUIRED_BUNDLE_FIELDS = (
    "adapter_path",
    "adapter_source_tree_sha256",
    "base_model_name",
    "base_model_revision",
    "corpus_version",
    "corpus_records_sha256",
    "split_ids_sha256",
    "prompt_schema_version",
    "prompt_budgets",
    "candidates_per_target",
    "seeds",
    "sampling",
    "timeout_seconds",
    "evaluator_version",
    "baseline_versions",
    "checkpoint_selection_rule",
)


class SealedAccessError(RuntimeError):
    """Raised whenever the sealed split is reached for without authorization."""


@dataclass(frozen=True)
class FinalBundle:
    """The complete frozen configuration a final measurement depends on."""

    fields: dict[str, Any]
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def missing_fields(self) -> list[str]:
        return [
            name for name in REQUIRED_BUNDLE_FIELDS
            if self.fields.get(name) in (None, "", [], {})
        ]

    def sha256(self) -> str:
        payload = json.dumps(
            {name: self.fields.get(name) for name in sorted(REQUIRED_BUNDLE_FIELDS)},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bundle_sha256"] = self.sha256()
        payload["missing_fields"] = self.missing_fields()
        payload["frozen"] = not self.missing_fields()
        return payload


@dataclass(frozen=True)
class Authorization:
    token: str
    bundle_sha256: str
    issued_utc: str
    issued_by: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def issue_authorization(
    bundle: FinalBundle, issued_by: str, reason: str,
) -> Authorization:
    """Mint a one-time token for one fully frozen bundle."""
    missing = bundle.missing_fields()
    if missing:
        raise SealedAccessError(
            "refusing to authorize a sealed-final run: these are not frozen: "
            + ", ".join(missing)
        )
    if not issued_by.strip() or not reason.strip():
        raise SealedAccessError("authorization requires an issuer and a reason")
    return Authorization(
        token=uuid.uuid4().hex,
        bundle_sha256=bundle.sha256(),
        issued_utc=_utc(),
        issued_by=issued_by.strip(),
        reason=reason.strip(),
    )


class SealedFinalGuard:
    """The single gate through which the sealed split may be read."""

    def __init__(self, state_path: Path, audit_log_path: Path) -> None:
        self.state_path = Path(state_path)
        self.audit_log_path = Path(audit_log_path)

    # -- audit ------------------------------------------------------------
    def _append_audit(self, entry: dict[str, Any]) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"utc": _utc(), **entry}, sort_keys=True) + "\n")

    def audit_entries(self) -> list[dict[str, Any]]:
        if not self.audit_log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.audit_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # -- token state ------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"spent_tokens": [], "runs": []}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    def register(self, authorization: Authorization) -> None:
        state = self._state()
        state.setdefault("issued", []).append(authorization.to_dict())
        self._write_state(state)
        self._append_audit({
            "event": "authorization_issued",
            "bundle_sha256": authorization.bundle_sha256,
            "issued_by": authorization.issued_by,
        })

    # -- the gate ---------------------------------------------------------
    def open_sealed_split(
        self,
        split: str,
        bundle: FinalBundle,
        token: str | None,
        caller: str,
    ) -> None:
        """Authorize one read of the sealed split, or refuse and say why."""
        if split != SEALED_SPLIT:
            return  # development splits need no authorization

        def refuse(reason: str) -> None:
            self._append_audit({
                "event": "sealed_access_refused",
                "caller": caller, "reason": reason,
                "bundle_sha256": bundle.sha256(),
            })
            raise SealedAccessError(reason)

        if not token:
            refuse(
                f"{caller} requested the sealed split with no authorization token"
            )
        missing = bundle.missing_fields()
        if missing:
            refuse(
                "sealed access requires a fully frozen bundle; not frozen: "
                + ", ".join(missing)
            )

        state = self._state()
        issued = {item["token"]: item for item in state.get("issued", [])}
        record = issued.get(token)
        if record is None:
            refuse(f"{caller} presented an unknown authorization token")
        if token in set(state.get("spent_tokens", [])):
            refuse(
                "this authorization has already been spent; the sealed split is "
                "one-time and cannot be reopened with the same token"
            )
        if record["bundle_sha256"] != bundle.sha256():
            refuse(
                "the frozen bundle changed after this token was issued "
                f"({record['bundle_sha256'][:12]} -> {bundle.sha256()[:12]}); "
                "anything tuned after authorization invalidates the measurement"
            )

        state.setdefault("spent_tokens", []).append(token)
        state.setdefault("runs", []).append({
            "utc": _utc(), "caller": caller, "bundle_sha256": bundle.sha256(),
        })
        self._write_state(state)
        self._append_audit({
            "event": "sealed_access_granted",
            "caller": caller, "bundle_sha256": bundle.sha256(),
        })


def refuse_sealed_split_for_development(split: str, command: str) -> None:
    """Hard refusal used by ordinary development entry points.

    Development commands never have a token, so routing them through the full
    guard would only produce a confusing error. This says the real thing.
    """
    if str(split) == SEALED_SPLIT:
        raise SealedAccessError(
            f"{command} may not read the sealed '{SEALED_SPLIT}' split. "
            "Sealed measurement runs through the authorized final evaluator."
        )
