"""Structural validation for the four LaTeX drafts.

No LaTeX toolchain is available on this machine, so nothing here is a
substitute for compiling. What it does check, it checks properly:

  1. every \\begin{env} has a matching \\end{env}, in order;
  2. braces balance outside comments;
  3. every \\cite key resolves to an entry in the venue's references.bib;
  4. no identity-revealing string survives (double-anonymous review);
  5. no prohibited claim appears (see claims_traceability.md);
  6. every numeric value in the prose appears in the evidence ledger;
  7. no TODO marker leaks into a paper source.
"""
from __future__ import annotations

import re
import os
import sys
from pathlib import Path

# Resolve from this file so the validation package works on the GPU host, the
# laptop, CI, and an extracted replication bundle without rewriting a username
# or checkout path.
ROOT = Path(__file__).resolve().parent.parent
VENUES = ["saner_rene_2027", "icst_2027", "saner_short_2027", "cain_2027"]

#: Strings that would break double-anonymous review.
IDENTITY = [
    "github.com", "gitlab", "bitbucket", "@gmail", "@student", ".edu",
    "\\author{\\IEEEauthorblockN{A", "acknowledg", "Acknowledg",
    "funded by", "grant no", "our university", "our institution",
]

# Identity-specific terms must not be embedded in a validator that accompanies
# a double-anonymous artifact: the denylist would disclose the very identity it
# is meant to catch.  Internal validation can add private terms without writing
# them to a source file, for example as a pipe-separated environment value.
IDENTITY.extend(
    term for term in os.getenv("ONEIROS_PAPER_PRIVATE_IDENTITY_TERMS", "").split("|")
    if term
)

#: Claims that must never appear, per claims_traceability.md.
PROHIBITED = [
    "beats atheris", "outperforms atheris", "better than atheris",
    "sft generalizes", "sft generalises",
    "better than the base model", "superior to the base",
    "final-test result of", "on the sealed test split we",
    "proves there is no effect", "no difference between",
    "shows no effect",
    # Imprecise phrasings corrected in the readiness pass. These are matched
    # against WHITESPACE-COLLAPSED text: an earlier line-based grep reported
    # them absent while three instances survived, wrapped across a line break.
    "selected nothing", "selects nothing", "selecting nothing",
    "no number of any kind",
    "kill@k is exploitable", "is also exploitable",
    "ai use must be disclosed in the review form",
]

#: Every number the drafts are allowed to state, from evidence_ledger.md.
ALLOWED_NUMBERS = {
    # protocol
    "0.7", "0.9", "42", "8", "1024", "3072", "2", "1.5",
    # development panel
    "0.605166", "0.690037", "0.695572", "0.658672",
    "328", "374", "377", "357", "542", "594", "52",
    "0.5634", "0.6454", "0.6499", "0.7275", "0.6556", "0.7328",
    "0.6178", "0.6973",
    "0.983395", "0.948570", "0.565498", "0.997463", "0.975092",
    "0.458487", "0.998155", "0.993081", "0.455028", "0.973478",
    "0.448801", "3.1365", "44", "61", "17", "0.118", "9.04",
    # locked panel
    "0.594452", "0.628798", "450", "476", "757",
    "0.5591", "0.6289", "0.5938", "0.6625",
    "0.987285", "0.928336", "0.452114", "0.988771", "0.942206",
    "0.333388", "675", "566", "109",
    "3.4346", "114", "88", "26", "362", "193", "202",
    "0.0783", "0.01", "3.0", "15", "11.8726", "0.1486", "1.3870",
    # rehearsal
    "4336", "4,336", "271", "577.753", "0", "55", "72", "223", "43",
    "2.6",
    # "SHA-256" is an algorithm name, not a measurement
    "256",
    # checkpoint identifiers (labels, not measurements)
    "150", "431",
    # structural
    "1", "3", "4", "5", "6", "10", "12", "18", "20", "95",
}

NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        idx, esc = None, False
        for i, ch in enumerate(line):
            if ch == "\\":
                esc = not esc
            elif ch == "%" and not esc:
                idx = i
                break
            else:
                esc = False
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


def check(venue: str) -> list[str]:
    problems: list[str] = []
    tex = ROOT / venue / "main.tex"
    bib = ROOT / venue / "references.bib"
    raw = tex.read_text(encoding="utf-8")
    body = strip_comments(raw)
    # LaTeX thousands separator: 4{,}336 is one number, not two.
    body = body.replace("{,}", ",")

    # 1. environments
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack:
                problems.append(f"\\end{{{env}}} with no open environment")
            elif stack[-1] != env:
                problems.append(f"\\end{{{env}}} closes \\begin{{{stack[-1]}}}")
                stack.pop()
            else:
                stack.pop()
    for env in stack:
        problems.append(f"unclosed environment: {env}")

    # 2. braces
    depth = 0
    for i, ch in enumerate(body):
        if ch == "{" and (i == 0 or body[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or body[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                problems.append("unbalanced closing brace")
                break
    if depth > 0:
        problems.append(f"{depth} unclosed brace(s)")

    # 3. citations resolve
    keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
    cited = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        for k in m.group(1).split(","):
            k = k.strip().lstrip("%").strip()
            if k:
                cited.add(k)
    for k in sorted(cited - keys):
        problems.append(f"\\cite{{{k}}} has no entry in references.bib")

    # 4. anonymity
    low = body.lower()
    for s in IDENTITY:
        if s.lower() in low:
            problems.append(f"ANONYMITY: identity-revealing string {s!r}")

    # 5. prohibited claims, over WHITESPACE-COLLAPSED text so a phrase cannot
    #    hide by wrapping across a line break.
    flat = " ".join(body.lower().split())
    for s in PROHIBITED:
        if s in flat:
            problems.append(f"PROHIBITED CLAIM (source): {s!r}")

    # 5b. the same sweep over the compiled PDF, which is what reviewers read.
    pdf = ROOT / venue / "main.pdf"
    if pdf.is_file():
        try:
            from pypdf import PdfReader
            text = " ".join((pg.extract_text() or "")
                            for pg in PdfReader(str(pdf)).pages)
            flat_pdf = " ".join(text.lower().split())
            for s in PROHIBITED:
                if s in flat_pdf:
                    problems.append(f"PROHIBITED CLAIM (compiled PDF): {s!r}")
            for s in IDENTITY:
                if s.lower() in flat_pdf:
                    problems.append(f"ANONYMITY (compiled PDF): {s!r}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"could not sweep the compiled PDF: {exc!r}")

    # 6. numbers -- over prose and tables only; TikZ geometry is excluded
    prose = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                   " ", body, flags=re.S)
    prose = re.sub(r"p\{[^}]*\}", " ", prose)
    for m in NUM_RE.finditer(prose):
        tok = m.group(0)
        if tok in ALLOWED_NUMBERS:
            continue
        if tok.replace(",", "") in ALLOWED_NUMBERS:
            continue
        # LaTeX/TikZ geometry, versions and dates are not evidential claims.
        ctx = prose[max(0, m.start() - 70):m.start()]
        if any(t in ctx for t in ("tikzpicture", "draw", "node", "fill",
                                  "circle", "foreach", "documentclass",
                                  "usepackage", "columnwidth", "pt}", "cm",
                                  "\\y", "\\lx", "IEEE", "width", "height",
                                  "sep", "minimum", "definecolor", "2027",
                                  "2026", "compat")):
            continue
        problems.append(f"UNVERIFIED NUMBER {tok!r} near: ...{ctx[-45:].strip()}")

    # 7. TODO markers
    if "TODO" in raw:
        problems.append("a TODO marker survives in the paper source")

    return problems


def main() -> int:
    total = 0
    for venue in VENUES:
        problems = check(venue)
        status = "PASS" if not problems else f"{len(problems)} problem(s)"
        print(f"\n=== {venue}: {status} ===")
        for p in problems[:40]:
            print(f"  - {p}")
        total += len(problems)
    print(f"\n{'ALL STRUCTURAL CHECKS PASSED' if not total else f'{total} problem(s) total'}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
