"""v2.5 generation job builder and validator (``pytest_module_v1`` only).

The ONLY way a v2.5 generation prompt is made: every item is produced here by calling
``harness.native_generated_test_prompt_v25.build_prompt`` on the permitted buggy-side view, then
- refused if the prompt is not a pytest_module_v1 prompt or carries a retired label
  (``pytest_fragment`` / ``assert_statement``);
- scanned by the independent leakage scanner against verifier-only material (fixed source,
  gold patch, official tests, issue text): any fixed line, patch line, official test line or
  hidden expected literal refuses the prompt; missing verifier material refuses (fail closed);
- token-counted on the CHAT-TEMPLATED prompt; it must fit the 2,048 prompt-token limit and,
  with the 1,024 completion budget, the 3,072 sequence limit. Nothing is ever truncated.

The job file binds the job schema, the builder version, the canonical hashes of the v2.5
builder and of the permitted-view code it reuses, the scanner version, and per item: output
type, prompt and view hashes, the target/import identity and the prompt token count.
``validate_job_file`` (called by the generator before any run) refuses a v2.4 or other job, a
job whose builder hash differs from the current builder, or any item that is not a sealed
pytest_module_v1 prompt; so changing the builder invalidates the job (and, through the source
identity, the preflight, the authorisation and resume).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from harness import native_generated_test_prompt_v25 as v25
from harness.native_generated_test_leakage import SCANNER_VERSION, scan
from harness.native_generated_test_prompt import permitted_view

JOB_SCHEMA = "oneiros_native_generated_tests_job_v25"
CONDITION = "primary_whole_module"
PROMPT_TOKEN_LIMIT = 2048
COMPLETION_TOKENS = 1024
SEQUENCE_LIMIT = 3072
RETIRED_LABELS = ("pytest_fragment", "assert_statement")
BUILDER_FILES = {"v25_pytest_module_v1": "harness/native_generated_test_prompt_v25.py",
                 "permitted_view": "harness/native_generated_test_prompt.py"}
VERIFIER_FIELDS = ("fixed_source", "patch", "official_tests")


class JobRefused(SystemExit):
    """A v2.5 job that must not be built or run."""


def _canonical(path: Path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def builder_identity(root: Path) -> Dict[str, Any]:
    return {"builder_version": v25.BUILDER_VERSION, "output_type": v25.OUTPUT_TYPE,
            "builder_sha256": {k: _canonical(Path(root) / rel) for k, rel in
                               BUILDER_FILES.items()},
            "scanner_version": SCANNER_VERSION}


def prompt_problems(prompt: str) -> List[str]:
    """Structural refusals of a model-visible prompt (independent of the leakage scan)."""
    problems = [f"retired label {label}" for label in RETIRED_LABELS if label in prompt]
    if f"Expected test format: {v25.OUTPUT_TYPE}" not in prompt or \
            not prompt.rstrip().endswith(v25.OUTPUT_TYPE):
        problems.append("not a pytest_module_v1 prompt")
    return problems


def build_item(dto: Mapping[str, Any], buggy_source: str,
               verifier: Optional[Mapping[str, Any]],
               count_tokens: Callable[[str], int]) -> Dict[str, Any]:
    """One target -> {"item": ...} or {"refused": ...}; every applicable reason recorded."""
    key = dto.get("target_key")
    try:
        built = v25.build_prompt(dto, buggy_source)
        view = permitted_view(dto, buggy_source)
    except Exception as exc:                                  # noqa: BLE001 - builder refusal
        return {"refused": {"target_key": key, "reasons": ["prompt_refused"],
                            "details": {"prompt_refused": f"{type(exc).__name__}: {exc}"[:200]}}}
    prompt = built["prompt"]
    reasons, details = [], {}
    found = prompt_problems(prompt)
    if found:
        reasons.append("prompt_refused")
        details["prompt_refused"] = found
    if _sha(prompt) != built["prompt_sha256"]:
        reasons.append("seal_mismatch")
    if verifier is None or any(f not in verifier for f in VERIFIER_FIELDS):
        reasons.append("leakage")
        details["leakage"] = ["verifier material missing (fail closed)"]
    else:
        result = scan({"prompt": prompt, "prompt_sha256": built["prompt_sha256"]},
                      {**verifier, "buggy_source": buggy_source})
        if not result["ok"]:
            reasons.append("leakage")
            details["leakage"] = result["reasons"]
    tokens = int(count_tokens(prompt))
    if tokens > PROMPT_TOKEN_LIMIT or tokens + COMPLETION_TOKENS > SEQUENCE_LIMIT:
        reasons.append("sequence_overflow")
        details["sequence_overflow"] = {"prompt_tokens": tokens,
                                        "prompt_token_limit": PROMPT_TOKEN_LIMIT,
                                        "sequence_limit": SEQUENCE_LIMIT}
    if reasons:
        return {"refused": {"target_key": key, "reasons": reasons, "details": details},
                "prompt_tokens": tokens}
    return {"item": {
        "target_key": key, "condition": "whole_module", "prompt": prompt,
        "prompt_sha256": built["prompt_sha256"], "output_type": v25.OUTPUT_TYPE,
        "builder_version": v25.BUILDER_VERSION, "view_sha256": built["view_sha256"],
        "buggy_source_sha256": built["buggy_source_sha256"],
        "target": {"repository": view["dto"]["repository"],
                   "buggy_commit": view["dto"]["buggy_commit"],
                   "target_file": view["dto"]["target_file"],
                   "qualname": view["dto"]["qualname"],
                   "import": f"from {view['module']} import {view['import_name']}"},
        "prompt_tokens": tokens}}


def chat_token_counter(base_model: str, revision: str) -> Callable[[str], int]:
    """Exact count of the CHAT-TEMPLATED prompt with the frozen base tokenizer (CPU, local
    files only; no model is loaded) - the same text the generator's backend tokenises."""
    import transformers
    from engine.test_generation_prompt import format_chat_prompt
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, revision=revision,
                                                           local_files_only=True)

    def count(prompt: str) -> int:
        return len(tokenizer(format_chat_prompt(tokenizer, prompt),
                             add_special_tokens=False)["input_ids"])
    return count


def items_sha(items: Sequence[Mapping[str, Any]]) -> str:
    """Same formula as the generator's ``contract_sha({"items": items})``."""
    return hashlib.sha256(json.dumps({"items": list(items)}, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def build_job_file(targets: Sequence[Mapping[str, Any]], count_tokens: Callable[[str], int],
                   root: Path, *, purpose: str) -> Dict[str, Any]:
    """``targets``: rows with ``dto``, ``buggy_source`` and ``verifier`` (verifier-only)."""
    items, refused, measured = [], [], {}
    for t in targets:
        out = build_item(t["dto"], t["buggy_source"], t.get("verifier"), count_tokens)
        tokens = out.get("prompt_tokens", (out.get("item") or {}).get("prompt_tokens"))
        if tokens is not None:
            measured[t["dto"]["target_key"]] = tokens
        if "item" in out:
            items.append(out["item"])
        else:
            refused.append(out["refused"])
    items.sort(key=lambda i: i["target_key"])
    refused.sort(key=lambda r: r["target_key"])
    keys = [i["target_key"] for i in items] + [r["target_key"] for r in refused]
    if len(keys) != len(set(keys)):
        raise JobRefused("REFUSED: duplicate target keys")
    job = {"items": items, "refused": refused,
           "sequence_fit": {"prompt_token_limit": PROMPT_TOKEN_LIMIT,
                            "completion_tokens": COMPLETION_TOKENS,
                            "sequence_limit": SEQUENCE_LIMIT, "truncation": "never",
                            "prompt_tokens": dict(sorted(measured.items()))},
           "job_sha256": items_sha(items)}
    return {"schema_version": JOB_SCHEMA, "purpose": purpose,
            "builder": builder_identity(root), CONDITION: job}


def validate_job_file(job_file: Mapping[str, Any], condition: str,
                      root: Path) -> Dict[str, Any]:
    """The generator's gate: only a current v2.5 job runs. Returns the condition's job."""
    if not isinstance(job_file, Mapping) or job_file.get("schema_version") != JOB_SCHEMA:
        raise JobRefused("REFUSED: not a v2.5 job (schema "
                         f"{(job_file or {}).get('schema_version')!r}); v2.4 and older jobs "
                         "are retired")
    if job_file.get("builder") != builder_identity(root):
        raise JobRefused("REFUSED: the job was built by a different prompt builder, permitted "
                         "view or scanner; rebuild the job")
    if condition != CONDITION or condition not in job_file:
        raise JobRefused(f"REFUSED: condition {condition!r} not in the v2.5 job")
    job = job_file[condition]
    if items_sha(job.get("items") or []) != job.get("job_sha256"):
        raise JobRefused("REFUSED: job items do not match their hash")
    problems = []
    for item in job["items"]:
        p = prompt_problems(item.get("prompt", ""))
        if _sha(item.get("prompt", "")) != item.get("prompt_sha256"):
            p.append("seal mismatch")
        if item.get("output_type") != v25.OUTPUT_TYPE or \
                item.get("builder_version") != v25.BUILDER_VERSION:
            p.append("item not built by the v2.5 builder")
        if not isinstance(item.get("prompt_tokens"), int) or \
                item["prompt_tokens"] > PROMPT_TOKEN_LIMIT:
            p.append("prompt token fit not recorded or exceeded")
        if p:
            problems.append(f"{item.get('target_key')}: {p}")
    if problems or not job["items"]:
        raise JobRefused(f"REFUSED: invalid v2.5 job items: {problems[:3] or 'no items'}")
    return dict(job)
