"""Tie a locked-validation run to the receipt that froze it.

A frozen preflight is only worth something if the run that follows is the run
it described. Until now the link was procedural: a receipt was written, and
then a command was typed which was believed to match it. Nothing checked.

This module makes the link mechanical. The generated commands carry the
receipt's path and SHA-256, and the runner refuses to start unless the receipt
on disk is that receipt, it says it is ready, and the arm being launched is the
arm it froze - same model revision, same adapter bytes, same split, same
protocol, same resolved generation settings, same evaluation sources.

One asymmetry has to be handled honestly rather than papered over. The receipt
records the commit whose code it froze. Committing the receipt itself then
advances HEAD, so at launch time HEAD is necessarily a later commit than the
one written inside the receipt. Requiring them to be equal would be requiring a
contradiction, and the usual workaround - regenerating the receipt until the
numbers agree - just hides the sequence.

So both are recorded under names that say what they are, and neither is
required to equal the other. The real equivalence test is the evaluation-source
blob identities: if every source file Git-hashes to what the receipt recorded,
the code is the code, whatever a results-only commit did to HEAD afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from harness.source_identity import (
    EVALUATION_DEFINING_SOURCES, HASH_SCHEME_VERSION, canonical_sha256,
    git_blob_sha1, raw_sha256,
)

SCHEMA_VERSION = "oneiros_locked_validation_binding_v1"

#: Settings the receipt froze that the runtime must still agree with. Compared
#: by value, so a silent default change is a refusal rather than a footnote.
BOUND_SETTINGS = (
    "seed",
    "candidates_per_function",
    "candidate_parse_mode",
    "retain_raw_output",
    "temperature",
    "top_p",
    "generation_completion_token_limit",
    "prompt_token_limit",
    "max_sequence_tokens",
)


def _sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_receipt(receipt_path, expected_sha256: str) -> tuple[Dict[str, Any], List[str]]:
    """Read the frozen receipt, refusing anything that is not exactly it."""
    problems: List[str] = []
    path = Path(receipt_path)
    if not path.is_file():
        return {}, [f"frozen preflight receipt not found: {path}"]
    actual = _sha256_of(path)
    expected = str(expected_sha256 or "").strip().lower()
    if not expected:
        return {}, ["--frozen-preflight-receipt requires --expected-preflight-receipt-sha256"]
    if actual != expected:
        return {}, [
            "frozen preflight receipt hash mismatch; refusing before launch.\n"
            f"  receipt : {path}\n"
            f"  expected: {expected}\n"
            f"  found   : {actual}"
        ]
    try:
        return json.loads(path.read_text(encoding="utf-8")), problems
    except json.JSONDecodeError as exc:
        return {}, [f"frozen preflight receipt is not valid JSON: {exc}"]


def _arm_for(receipt: Dict[str, Any], run_name: str):
    for key, arm in (receipt.get("arms") or {}).items():
        if arm.get("run_name") == run_name:
            return key, arm
    return None, None


def verification_problems(
    *,
    root,
    receipt: Dict[str, Any],
    run_name: str,
    phase: str,
    evaluation_split: str,
    model_revision: str,
    adapter_sha256: str | None,
    protocol_name: str,
    protocol_sha256: str,
    resolved_settings: Dict[str, Any],
) -> List[str]:
    """Everything that must hold before a locked-validation arm may start."""
    problems: List[str] = []
    root = Path(root)

    if receipt.get("ready_to_launch") is not True:
        problems.append("frozen preflight receipt is not marked ready_to_launch")
    if receipt.get("preflight_problems"):
        problems.append(
            f"frozen preflight receipt records unresolved problems: "
            f"{receipt['preflight_problems']}")

    # ---- the sealed test stays closed ------------------------------------
    sealed = receipt.get("sealed_final_test") or {}
    if sealed.get("accessed") is not False:
        problems.append("frozen preflight receipt does not record the sealed test as unopened")
    if evaluation_split == "test" or receipt.get("protocol", {}).get("final_test_measurement") is not False:
        problems.append("locked validation must not be a sealed-test measurement")
    corpus = receipt.get("corpus") or {}
    if "test" in (corpus.get("included_splits") or []):
        problems.append("frozen preflight receipt describes a corpus view that includes the sealed split")

    # ---- the arm being launched is an arm the receipt froze --------------
    key, arm = _arm_for(receipt, run_name)
    if arm is None:
        problems.append(
            f"run name {run_name!r} is not an arm in the frozen preflight receipt; "
            f"known arms: {sorted((a.get('run_name') or '?') for a in (receipt.get('arms') or {}).values())}")
        return problems

    expected_phase = "sft_eval" if arm.get("adapter_sha256_expected") else "base_eval"
    if phase != expected_phase:
        problems.append(
            f"arm {key} was frozen as --phase {expected_phase}, launched as --phase {phase}")

    if arm.get("model_revision") != model_revision:
        problems.append(
            f"base model revision differs from the receipt: receipt "
            f"{arm.get('model_revision')!r}, launching {model_revision!r}")

    expected_adapter = arm.get("adapter_sha256_expected")
    if expected_adapter != adapter_sha256:
        problems.append(
            f"adapter SHA-256 differs from the receipt: receipt {expected_adapter!r}, "
            f"launching {adapter_sha256!r}")

    # ---- protocol and split ----------------------------------------------
    protocol = receipt.get("protocol") or {}
    if protocol.get("evaluation_split") != evaluation_split:
        problems.append(
            f"evaluation split differs from the receipt: receipt "
            f"{protocol.get('evaluation_split')!r}, launching {evaluation_split!r}")
    if protocol.get("protocol_name") != protocol_name:
        problems.append(
            f"protocol name differs from the receipt: receipt "
            f"{protocol.get('protocol_name')!r}, launching {protocol_name!r}")
    if protocol.get("protocol_sha256") != protocol_sha256:
        problems.append("protocol sha256 differs from the receipt")

    # ---- resolved generation settings ------------------------------------
    frozen_settings = receipt.get("resolved_generation_settings") or {}
    for name in BOUND_SETTINGS:
        want = frozen_settings.get(name)
        got = resolved_settings.get(name)
        if want != got:
            problems.append(
                f"resolved generation setting {name} differs from the receipt: "
                f"receipt {want!r}, launching {got!r}")

    # ---- evaluation sources are the sources the receipt froze ------------
    frozen_sources = ((receipt.get("source_identity") or {}).get("sources")) or {}
    scheme = (receipt.get("source_identity") or {}).get("hash_scheme_version")
    if scheme != HASH_SCHEME_VERSION:
        problems.append(
            f"source-identity scheme differs: receipt {scheme!r}, "
            f"runtime {HASH_SCHEME_VERSION!r}")
    for role, relative in sorted(EVALUATION_DEFINING_SOURCES.items()):
        frozen = frozen_sources.get(role)
        if not frozen:
            problems.append(f"frozen preflight receipt has no identity for {role}")
            continue
        path = root / relative
        if not path.is_file():
            problems.append(f"evaluation source is missing: {relative}")
            continue
        if git_blob_sha1(path) != frozen.get("git_blob_sha1"):
            problems.append(
                f"{relative} Git blob identity differs from the receipt: receipt "
                f"{frozen.get('git_blob_sha1')}, current {git_blob_sha1(path)}")
        if canonical_sha256(path) != frozen.get("canonical_sha256"):
            problems.append(
                f"{relative} canonical source hash differs from the receipt")
    return problems


def contract_block(
    *,
    root,
    receipt_path,
    receipt_sha256: str,
    receipt: Dict[str, Any],
    git_head_at_launch: str,
    adapter_sha256: str | None,
    model_revision: str,
) -> Dict[str, Any]:
    """What a locked-validation run contract records about its own binding.

    The per-source raw, canonical and blob hashes are not repeated here: they
    already sit in the contract's ``source_identity`` block, and two copies of
    the same fact in one document is two facts that can disagree.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_preflight_receipt_path": str(receipt_path),
        "frozen_preflight_receipt_sha256": receipt_sha256,
        "git_head_at_launch": git_head_at_launch,
        "git_commit_recorded_in_preflight_receipt": (
            (receipt.get("source_identity") or {}).get("git_commit")
            or receipt.get("reproducibility", {}).get("git_commit")),
        "adapter_sha256": adapter_sha256,
        "base_model_revision": model_revision,
        "source_identity_scheme_version": HASH_SCHEME_VERSION,
        "source_identity_is_recorded_in": "run_contract.source_identity",
        "why_the_two_commits_differ": (
            "The preflight receipt records the commit whose code it froze. "
            "Committing that receipt advances HEAD, so git_head_at_launch is "
            "necessarily a later commit than the one named inside the receipt. "
            "They are recorded separately and are NOT required to be equal. "
            "The equivalence rule is the evaluation-source Git blob identity: "
            "every source listed in run_contract.source_identity must hash to "
            "what the receipt recorded, and the runner refuses to start if any "
            "one of them does not. A results-only commit moves HEAD without "
            "moving a single evaluation-defining blob, and must not be read as "
            "code drift."
        ),
        "verified_before_cuda_corpus_or_output_directories": True,
    }
