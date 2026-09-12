"""Which protocol produced this number, and may these two be compared?

Section 6.2 established that the legacy parser and the successor protocol are
not two measurements of one quantity - they are two different quantities under
the same field names. On the same retained generations, kill@8 reads 0.7159
under the legacy first-assertion parser and 0.5480 under the successor, because
the legacy parser discarded everything after the first assertion and so acted
as a repair step.

That leaves a trap. Legacy artifacts predate the protocol field entirely, so
they record NOTHING, and an absent field reads as "no reason for concern"
rather than as "this is the legacy protocol". Subtracting a successor number
from a legacy one produces a plausible-looking delta of roughly seventeen
points that measures the parser rather than the model.

This module makes the absence explicit and refuses the subtraction.

THE PROTOCOL OF RECORD. Every result committed in this project - the arm table,
the twelve relearning seeds, the Atheris comparison - was computed under the
legacy parser, so LEGACY remains the protocol of record for the existing
baseline and is not being restated. The successor is the honest description of
what the model emits and is the protocol of record for NEW work. Neither claim
is weakened by the other, and the only thing forbidden is mixing them in one
comparison.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

LEGACY = "first_assertion"
SUCCESSOR = "whole_output"

PROTOCOLS = (LEGACY, SUCCESSOR)

#: What the committed baseline was measured under. Stated rather than inferred.
PROTOCOL_OF_COMMITTED_BASELINE = LEGACY

#: What a new measurement should use. The successor scores the whole output, so
#: it describes what the model actually produced rather than what the parser
#: rescued from it.
PROTOCOL_OF_RECORD_FOR_NEW_WORK = SUCCESSOR


def protocol_of(payload: Mapping[str, Any]) -> str:
    """The protocol this artifact was produced under.

    An artifact with no ``candidate_parse_mode`` predates the field, which
    means it was produced by the legacy first-assertion parser. Returning
    "unknown" here would be more cautious and less true: the absence is
    evidence, not a gap.
    """
    profile = payload.get("evaluation_profile") or {}
    mode = str(profile.get("candidate_parse_mode") or "").strip()
    if not mode:
        return LEGACY
    if mode not in PROTOCOLS:
        raise ValueError(
            f"artifact records an unknown candidate_parse_mode {mode!r}; "
            f"expected one of {PROTOCOLS}"
        )
    return mode


def describe(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = protocol_of(payload)
    profile = payload.get("evaluation_profile") or {}
    return {
        "protocol": protocol,
        "recorded_explicitly": bool(profile.get("candidate_parse_mode")),
        "is_committed_baseline_protocol": protocol == PROTOCOL_OF_COMMITTED_BASELINE,
        "scores": (
            "the first assert statement in each output" if protocol == LEGACY
            else "every assertion in each output, conjunctively"
        ),
    }


def assert_comparable(payloads: Iterable[Mapping[str, Any]],
                      labels: Iterable[str] | None = None) -> str:
    """Refuse a comparison that spans protocols, and say which is which.

    Returns the single shared protocol so a caller can record it.
    """
    payloads = list(payloads)
    names = list(labels) if labels is not None else [
        str(p.get("artifact") or p.get("adapter") or f"artifact_{i}")
        for i, p in enumerate(payloads)
    ]
    protocols = [protocol_of(payload) for payload in payloads]
    if len(set(protocols)) > 1:
        detail = ", ".join(
            f"{name} = {protocol}" for name, protocol in zip(names, protocols))
        raise SystemExit(
            "refusing to compare artifacts measured under different evaluation "
            f"protocols ({detail}). The legacy parser scores the first "
            "assertion and the successor scores every assertion, so the "
            "difference between them measures the parser, not the model. "
            "Re-measure one side under the other's protocol first."
        )
    return protocols[0] if protocols else LEGACY
