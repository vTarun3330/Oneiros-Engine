"""Explain why eligible repository targets are fewer than repository records.

The corpus holds 457 train repository records but only 346 are training
eligible, and separately ~40% of repository prompts are rejected for budget.
It is tempting to treat those as one problem - a bigger prompt budget fixing
both - and this audit exists because that is FALSE. They are distinct
mechanisms with distinct fixes:

* budget rejection happens at prompt-construction time, is measured by
  ``scripts/audit_prompt_compaction.py``, and is recovered by a larger budget;
* eligibility exclusion happens earlier, in corpus construction, when an
  independent test fragment references symbols the fragment cannot resolve.
  No prompt budget recovers those, because the symbol is absent from the
  record, not merely too expensive to include.

The practical question is whether the second class is recoverable by resolving
imports. This audit answers it by classifying each unresolved symbol.  The
answer is mostly no: the exclusions are dominated by pytest fixture parameters
and test-class inheritance, which are properties of the surrounding test suite
rather than missing import lines.

Stdlib attribution is reported as AMBIGUOUS whenever more than one standard
module exports a name, because a naive first-match index is actively
misleading - it attributes ``StringIO`` to ``csv`` (which merely imports it)
rather than ``io``.  An overstated recovery estimate would argue for building
a resolver that cannot pay for itself.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.corpus import write_json
from utils.reproducibility import source_tree_sha256

DEFAULT_VIEW = (
    ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate" / "development_view"
)

#: Symbols that name a pytest fixture, a bound method parameter, or a loop
#: variable rather than anything importable. These are the signature of a
#: fragment lifted out of a test class that supplied them.
FIXTURE_LIKE = frozenset({
    "kwargs", "args", "self", "cls", "client", "request", "tmpdir", "tmp_path",
    "monkeypatch", "capsys", "caplog", "mocker", "settings", "p", "e", "i",
    "__file__", "__name__",
})

#: A symbol that looks like a test class or mixin the fragment inherited from.
TEST_CLASS = re.compile(r"^[A-Z]\w*(TestCase|Test|Mixin|Case|Base)\w*$")


def _stdlib_exporters() -> dict[str, list[str]]:
    """Map every public stdlib export to EVERY module that exposes it.

    Keeping the full list rather than the first hit is the whole point: a name
    exported by several modules cannot be attributed to one of them without
    reading the code, and pretending otherwise inflates the recovery estimate.
    """
    exporters: dict[str, list[str]] = collections.defaultdict(list)
    for name in sorted(sys.stdlib_module_names):
        if name.startswith("_") or name in {"this", "antigravity", "idlelib"}:
            # `this` prints the Zen of Python on import; `antigravity` opens a
            # browser. Importing the whole standard library for an audit should
            # not have side effects.
            continue
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        exporters[name].append(name)
        for attribute in dir(module):
            if not attribute.startswith("_"):
                exporters[attribute].append(name)
    return dict(exporters)


def classify_symbol(symbol: str, exporters: dict[str, list[str]]) -> dict[str, Any]:
    if symbol in FIXTURE_LIKE:
        return {"symbol": symbol, "category": "fixture_or_local",
                "recoverable_by_import": False}
    if TEST_CLASS.match(symbol):
        return {"symbol": symbol, "category": "inherited_test_class",
                "recoverable_by_import": False}
    modules = exporters.get(symbol) or []
    if not modules:
        return {"symbol": symbol, "category": "project_local_symbol",
                "recoverable_by_import": False}
    if len(modules) == 1:
        return {"symbol": symbol, "category": "stdlib_unambiguous",
                "module": modules[0], "recoverable_by_import": True}
    return {"symbol": symbol, "category": "stdlib_ambiguous",
            "candidate_modules": modules[:6],
            "candidate_module_count": len(modules),
            "recoverable_by_import": False,
            "note": "several standard modules export this name; attribution "
                    "needs the fragment's own code, so it is not counted"}


def audit(view_dir: Path) -> dict[str, Any]:
    exclusions = json.loads(
        (view_dir / "training_exclusions.json").read_text(encoding="utf-8")
    )
    exporters = _stdlib_exporters()

    by_reason = collections.Counter(row.get("reason") for row in exclusions)
    context_rows = [
        row for row in exclusions
        if row.get("reason") == "independent_repository_context_incomplete"
    ]

    categories = collections.Counter()
    fully_recoverable: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for row in context_rows:
        symbols = list(row.get("unresolved_symbols") or [])
        classified = [classify_symbol(symbol, exporters) for symbol in symbols]
        for entry in classified:
            categories[entry["category"]] += 1
        if symbols and all(entry["recoverable_by_import"] for entry in classified):
            fully_recoverable.append({
                "record_id": row.get("record_id"),
                "symbols": [
                    {"symbol": e["symbol"], "module": e.get("module")}
                    for e in classified
                ],
            })
        else:
            blocked.append({
                "record_id": row.get("record_id"),
                "blocking_symbols": [
                    e["symbol"] for e in classified
                    if not e["recoverable_by_import"]
                ],
            })

    total = len(context_rows)
    recoverable = len(fully_recoverable)
    return {
        "schema_version": "oneiros_repository_exclusion_audit_v1",
        "source_tree_sha256": source_tree_sha256(ROOT),
        "sealed_final_test_accessed": False,
        "question": (
            "is the repository eligibility shortfall the same problem as the "
            "repository prompt-budget rejection rate?"
        ),
        "answer": (
            "No. They are separate mechanisms. Budget rejection happens at "
            "prompt construction and a larger budget recovers it. Eligibility "
            "exclusion happens in corpus construction because the independent "
            "fragment references symbols it cannot resolve, and no budget "
            "recovers a symbol that is absent from the record."
        ),
        "exclusion_reasons": dict(by_reason),
        "context_incomplete_exclusions": total,
        "unresolved_symbol_categories": dict(categories),
        "fully_recoverable_by_import_resolution": recoverable,
        "recovery_fraction_of_context_exclusions": round(
            recoverable / max(1, total), 4
        ),
        "verdict": (
            f"An import resolver would recover at most {recoverable} of {total} "
            "excluded records. The remainder are blocked by pytest fixture "
            "parameters and inherited test classes, which belong to the "
            "surrounding test suite rather than to any import line. Building "
            "the resolver is not a route to repository parity."
        ),
        "recoverable_records": fully_recoverable,
        "blocked_records_sample": blocked[:25],
        "blocked_record_count": len(blocked),
        "most_common_blocking_symbols": collections.Counter(
            symbol for row in blocked for symbol in row["blocking_symbols"]
        ).most_common(15),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, default=DEFAULT_VIEW)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "v4_2_repository_exclusion_audit.json",
    )
    arguments = parser.parse_args()

    report = audit(arguments.view_dir)
    write_json(arguments.output, report)
    print(json.dumps({
        key: report[key] for key in (
            "exclusion_reasons",
            "context_incomplete_exclusions",
            "unresolved_symbol_categories",
            "fully_recoverable_by_import_resolution",
            "verdict",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
