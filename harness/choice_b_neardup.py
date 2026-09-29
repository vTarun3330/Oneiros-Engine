"""Outcome-blind near-duplicate audit between Choice B training and gate groups.

Group and record disjointness do not rule out the same task appearing in two semantic
groups (e.g. an MBPP problem re-posed in HumanEval, or a duplicated upstream item). For
every training-group record and every gate-group record this compares:

* ``reference_clone``: identical AST of the reference implementation after removing
  docstrings and renaming every identifier in order of first appearance;
* ``reference_similar``: Jaccard >= 0.8 of the anonymised token 4-gram sets;
* ``specification_similar``: Jaccard >= 0.8 of lower-cased word 3-gram sets (both
  specifications at least 8 words).

Exclusion rule (deterministic, outcome-blind, decided before any Choice B outcome): a
TRAINING group with any flagged record is removed from training in full. The gate is
never changed. Only code and specification text is read - no model output.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from typing import Any, Dict, Iterable, List, Mapping, Sequence

THRESHOLD = 0.8
RULE = ("remove from TRAINING every group containing a record whose reference is an "
        "anonymised-AST clone of, or whose anonymised token 4-grams or specification word "
        "3-grams have Jaccard >= 0.8 with, any record of any gate group; the gate is fixed")


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and \
                isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]


def anonymised_ast(code: str) -> str | None:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return None
    _strip_docstrings(tree)
    names: Dict[str, str] = {}
    for node in ast.walk(tree):
        for attr in ("id", "arg", "name", "attr"):
            value = getattr(node, attr, None)
            if isinstance(value, str):
                setattr(node, attr, names.setdefault(value, f"v{len(names)}"))
    return ast.dump(tree, annotate_fields=False)


def token_shingles(code: str, n: int = 4) -> frozenset:
    tokens: List[str] = []
    names: Dict[str, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code or "").readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            if tok.type == tokenize.NAME and tok.string not in KEYWORDS:
                tokens.append(names.setdefault(tok.string, f"v{len(names)}"))
            elif tok.type == tokenize.STRING:
                tokens.append("<s>")
            else:
                tokens.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return frozenset()
    return frozenset(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def word_shingles(text: str, n: int = 3) -> frozenset:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(words) < 8:
        return frozenset()
    return frozenset(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


KEYWORDS = frozenset(__import__("keyword").kwlist) | {"True", "False", "None"}


def _profiles(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One profile per distinct (reference, specification) text, keyed by record id."""
    out = {}
    cache: Dict[tuple, Dict[str, Any]] = {}
    for r in records:
        key = (r.get("reference_code") or "", r.get("specification") or "")
        if key not in cache:
            cache[key] = {"ast": anonymised_ast(key[0]), "tokens": token_shingles(key[0]),
                          "spec": word_shingles(key[1])}
        out[r["id"]] = {**cache[key], "group_id": r["group_id"], "text_key": key}
    return out


def audit(train_records: Sequence[Mapping[str, Any]],
          gate_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train, gate = _profiles(train_records), _profiles(gate_records)
    gate_texts: Dict[tuple, Dict[str, Any]] = {}
    for rid, p in gate.items():
        gate_texts.setdefault(p["text_key"], {**p, "records": []})["records"].append(rid)
    gate_ast = {}
    for p in gate_texts.values():
        if p["ast"]:
            gate_ast.setdefault(p["ast"], []).append(p)
    flags: List[Dict[str, Any]] = []
    checked: Dict[tuple, List[Dict[str, Any]]] = {}
    for rid, p in sorted(train.items()):
        if p["text_key"] not in checked:
            hits = []
            for g in gate_ast.get(p["ast"], []) if p["ast"] else []:
                hits.append({"reason": "reference_clone", "gate_record": g["records"][0],
                             "gate_group": g["group_id"], "score": 1.0})
            for g in gate_texts.values():
                ref = jaccard(p["tokens"], g["tokens"])
                if ref >= THRESHOLD and not (p["ast"] and p["ast"] == g["ast"]):
                    hits.append({"reason": "reference_similar", "gate_record": g["records"][0],
                                 "gate_group": g["group_id"], "score": round(ref, 4)})
                spec = jaccard(p["spec"], g["spec"])
                if spec >= THRESHOLD:
                    hits.append({"reason": "specification_similar",
                                 "gate_record": g["records"][0], "gate_group": g["group_id"],
                                 "score": round(spec, 4)})
            checked[p["text_key"]] = hits
        for hit in checked[p["text_key"]]:
            flags.append({"train_record": rid, "train_group": p["group_id"], **hit})
    excluded = sorted({f["train_group"] for f in flags})
    return {"rule": RULE, "threshold": THRESHOLD,
            "train_records": len(train), "gate_records": len(gate),
            "distinct_train_texts": len(checked), "distinct_gate_texts": len(gate_texts),
            "flags": flags, "flag_reasons": _count(f["reason"] for f in flags),
            "excluded_training_groups": excluded}


def _count(items: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))
