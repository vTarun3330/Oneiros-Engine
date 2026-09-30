"""Scrub and check small receipts before they are tracked (amendment v2.4 section J).
Standard library only (used on Windows and inside WSL).

``scrub`` replaces absolute user-home, cache and root-home prefixes with placeholders;
``findings`` reports anything that must never be published: remaining user or cache paths,
credential-shaped strings, and protected-split markers. A tracked receipt must have none.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, List

_SEP = r"(?:\\\\|\\|/)+"
_SEG = r"[^\\/\"'\s:]+"
REPLACEMENTS = (
    (re.compile(r"[A-Za-z]:" + _SEP + r"Users" + _SEP + _SEG, re.I), "<user-home>"),
    (re.compile(r"/mnt/[a-z]/Users/" + _SEG, re.I), "<user-home>"),
    (re.compile(r"/home/" + _SEG), "<user-home>"),
    (re.compile(r"/root(?=/)"), "<root-home>"),
    (re.compile(r"(?:<user-home>|<root-home>)?(?:\\\\|\\|/)*\.cache(?:\\\\|\\|/)[^\s\"']*"),
     "<cache>"),
)
FORBIDDEN = (
    ("user path", re.compile(r"[A-Za-z]:" + _SEP + r"Users" + _SEP, re.I)),
    ("user path", re.compile(r"/mnt/[a-z]/Users/", re.I)),
    ("user path", re.compile(r"/home/[^\s\"'/]+")),
    ("cache path", re.compile(r"\.cache(?:\\\\|\\|/)")),
    ("credential", re.compile(r"(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]{16,}")),
    ("credential", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("credential", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("credential", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("protected data", re.compile(r"locked_val|sealed_final", re.I)),
)


def scrub(text: str) -> str:
    for pattern, placeholder in REPLACEMENTS:
        text = pattern.sub(placeholder, text)
    return text


def scrub_json(value: Any) -> Any:
    return json.loads(scrub(json.dumps(value)))


def findings(text: str) -> List[str]:
    return sorted({f"{kind}: {m.group(0)[:40]}" for kind, pattern in FORBIDDEN
                   for m in pattern.finditer(text)})


def check_file(path: Path) -> List[str]:
    return findings(Path(path).read_text(encoding="utf-8"))
