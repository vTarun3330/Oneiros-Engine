"""Pin the exclusion audit's classification, because it argues against work.

This audit's output is a decision not to build an import resolver. A silent
regression that reclassified fixture parameters as importable would reverse
that decision on false evidence, so the categories are tested directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_repository_exclusions import (
    TEST_CLASS, audit, classify_symbol,
)


EXPORTERS = {
    "gettempdir": ["tempfile"],
    "StringIO": ["csv", "io", "email"],
    "connect": ["sqlite3"],
}


def test_fixture_parameters_are_never_import_recoverable():
    for symbol in ("kwargs", "args", "client", "self", "__file__"):
        entry = classify_symbol(symbol, EXPORTERS)
        assert entry["category"] == "fixture_or_local"
        assert entry["recoverable_by_import"] is False


def test_inherited_test_classes_are_never_import_recoverable():
    for symbol in ("TestDataMixin", "WebSocketBaseTestCase", "RequestTest"):
        assert TEST_CLASS.match(symbol)
        entry = classify_symbol(symbol, EXPORTERS)
        assert entry["category"] == "inherited_test_class"
        assert entry["recoverable_by_import"] is False


def test_a_name_many_modules_export_is_not_attributed_to_one():
    """StringIO is exported by csv and email as well as io.

    A first-match index attributes it to whichever module sorts first, which
    is how an earlier pass reported it as coming from ``csv``. Ambiguity is
    reported, not guessed.
    """
    entry = classify_symbol("StringIO", EXPORTERS)
    assert entry["category"] == "stdlib_ambiguous"
    assert entry["recoverable_by_import"] is False
    assert entry["candidate_module_count"] == 3


def test_an_unambiguous_stdlib_symbol_is_recoverable():
    entry = classify_symbol("gettempdir", EXPORTERS)
    assert entry["category"] == "stdlib_unambiguous"
    assert entry["module"] == "tempfile"
    assert entry["recoverable_by_import"] is True


def test_a_project_local_symbol_is_not_recoverable():
    entry = classify_symbol("openapi_schema", EXPORTERS)
    assert entry["category"] == "project_local_symbol"
    assert entry["recoverable_by_import"] is False


def test_a_record_is_recoverable_only_when_every_symbol_is(tmp_path):
    """One blocking symbol blocks the record.

    Partial resolution does not make a fragment executable, so a record with
    one unambiguous stdlib import and one fixture parameter must be counted
    as blocked rather than recovered.
    """
    (tmp_path / "training_exclusions.json").write_text(json.dumps([
        {"record_id": "all-stdlib", "reason":
         "independent_repository_context_incomplete",
         "unresolved_symbols": ["gettempdir"]},
        {"record_id": "mixed", "reason":
         "independent_repository_context_incomplete",
         "unresolved_symbols": ["gettempdir", "kwargs"]},
        {"record_id": "other", "reason": "canonical_retained_training_excluded",
         "unresolved_symbols": []},
    ]), encoding="utf-8")

    report = audit(tmp_path)
    assert report["context_incomplete_exclusions"] == 2
    assert report["fully_recoverable_by_import_resolution"] == 1
    assert report["blocked_record_count"] == 1
    assert report["exclusion_reasons"]["canonical_retained_training_excluded"] == 1
    assert report["sealed_final_test_accessed"] is False


def test_the_audit_states_the_two_mechanisms_are_distinct(tmp_path):
    """The headline conclusion is asserted, not left to prose drift.

    Conflating budget rejection with eligibility exclusion is the specific
    error this artifact exists to prevent.
    """
    (tmp_path / "training_exclusions.json").write_text("[]", encoding="utf-8")
    report = audit(tmp_path)
    assert report["answer"].startswith("No.")
    assert "budget" in report["answer"]
