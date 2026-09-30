"""Independent post-sealing leakage scanner (protocol v2, section 6).

Runs AFTER a prompt is sealed, in a separate step with verifier-only material. The prompt
builder never sees this material. Each check compares normalised prompt text with content
that must not appear:

* ``fixed_line``: a line that exists in the fixed target file but not in the buggy one;
* ``patch_line``: an added line of the buggy->fixed patch, when a patch is supplied;
* ``official_test_line``: a non-import line of an official test that is not in the
  buggy source;
* ``expected_literal``: a literal compared in an official test assertion that does not
  occur in the buggy source;
* ``issue_text`` and ``commit_message``: sentences or lines of the issue/PR text or the fix
  commit message.

v2 (amendment v2.2 section A): an issue-text LINE is exempt only when the whole normalised
line exactly equals a whole normalised line of the permitted buggy source (exact-line
provenance; never a substring). Every other check is unchanged.

Any hit refuses the prompt (fail closed), with its reasons recorded. Malformed or
seal-mismatched input refuses.
"""
from __future__ import annotations

import ast
import hashlib
import re
from typing import Any, Dict, Iterable, List, Mapping

SCANNER_VERSION = "oneiros_native_generated_test_leakage_v2"
MIN_LINE = 12
MIN_TEST_LINE = 20
MIN_SENTENCE = 40
MIN_COMMIT_LINE = 20
MIN_LITERAL = 6


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _lines(text: str) -> List[str]:
    return [_norm(l) for l in (text or "").splitlines() if _norm(l)]


def _assert_literals(test_source: str) -> List[str]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            for side in [node.test.left, *node.test.comparators]:
                try:
                    value = ast.literal_eval(side)
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    continue
                text = repr(value)
                if len(text) >= MIN_LITERAL:
                    out.append(text)
    return out


def scan(sealed: Mapping[str, Any], verifier: Mapping[str, Any]) -> Dict[str, Any]:
    """``verifier`` keys: buggy_source, fixed_source, patch, official_tests (list),
    issue_text, commit_message (any may be empty)."""
    prompt = sealed.get("prompt") if isinstance(sealed, Mapping) else None
    seal = sealed.get("prompt_sha256") if isinstance(sealed, Mapping) else None
    if not isinstance(prompt, str) or not isinstance(seal, str) or             not isinstance(verifier, Mapping):
        return {"ok": False, "reasons": ["malformed scanner input"],
                "scanner_version": SCANNER_VERSION}
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != seal:
        return {"ok": False, "reasons": ["prompt does not match its seal"],
                "scanner_version": SCANNER_VERSION}
    prompt_norm = _norm(prompt)
    prompt_lines = set(_lines(prompt))
    buggy = set(_lines(verifier.get("buggy_source", "")))
    buggy_text = verifier.get("buggy_source", "") or ""
    reasons: List[str] = []

    def hit(kind: str, items: Iterable[str]) -> None:
        for item in items:
            reasons.append(f"{kind}: {item[:80]}")

    fixed_only = [l for l in _lines(verifier.get("fixed_source", ""))
                  if l not in buggy and len(l) >= MIN_LINE]
    hit("fixed_line", [l for l in fixed_only if l in prompt_lines])
    added = [_norm(l[1:]) for l in (verifier.get("patch") or "").splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    hit("patch_line", [l for l in added if len(l) >= MIN_LINE and l not in buggy
                       and l in prompt_lines])
    for test in verifier.get("official_tests") or []:
        test_lines = [l for l in _lines(test) if len(l) >= MIN_TEST_LINE and l not in buggy
                      and not l.startswith(("import ", "from "))]
        hit("official_test_line", [l for l in test_lines if l in prompt_lines])
        hit("expected_literal", [lit for lit in _assert_literals(test)
                                 if lit not in buggy_text and lit in prompt])
    issue_lines = [l for l in (verifier.get("issue_text") or "").splitlines()
                   if _norm(l) not in buggy]
    sentences = [_norm(s) for l in issue_lines for s in re.split(r"(?<=[.!?])\s+", l)]
    hit("issue_text", [s for s in sentences if len(s) >= MIN_SENTENCE and s in prompt_norm])
    commit = [l for l in _lines(verifier.get("commit_message", "")) if len(l) >= MIN_COMMIT_LINE]
    hit("commit_message", [l for l in commit if l in prompt_lines or l in prompt_norm])
    return {"ok": not reasons, "reasons": sorted(set(reasons)),
            "scanner_version": SCANNER_VERSION, "prompt_sha256": sealed["prompt_sha256"]}
