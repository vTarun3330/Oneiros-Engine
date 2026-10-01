"""Phase 6A recovery rule: only an unused forbidden import is removed; the policy, the sandbox,
assertions and expected values are never touched."""
from __future__ import annotations

from scripts import v25_module_conversion as mc
from scripts import v25_policy_recovery as pr

FRAG = "def test_area():\n    assert area(2) == 4\n"


def ctx(header: str) -> str:
    return f"{mc.HEADER_MARK}\n{header}\n"


def test_unused_header_import_is_removed_and_body_kept():
    row = pr.audit_row(FRAG, ctx("import io\nimport os, json\nfrom pkg.geo import area"), "area")
    assert row["category"] == "recoverable:unused_forbidden_import_removed"
    assert row["removed_aliases"] == ["io", "os"]
    assert "import io" not in row["after_module"] and "import json" in row["after_module"]
    assert row["after_module"].endswith(FRAG)
    assert row["before_sha256"] != row["after_sha256"]


def test_a_used_forbidden_module_is_never_removed():
    frag = "def test_area():\n    assert area(io.StringIO('x')) == 4\n"
    row = pr.audit_row(frag, ctx("import io\nfrom pkg.geo import area"), "area")
    assert row["category"] == "unrecoverable:forbidden_module_used_by_test"


def test_a_converter_inserted_forbidden_import_counts_as_used():
    frag = "def test_area():\n    assert area(os.sep) == 4\n"
    row = pr.audit_row(frag, ctx("from pkg.geo import area"), "area")
    assert row["category"] == "unrecoverable:forbidden_module_used_by_test"


def test_imports_inside_the_test_are_never_touched():
    frag = "def test_area():\n    import os\n    assert area(2) == 4\n"
    row = pr.audit_row(frag, ctx("from pkg.geo import area"), "area")
    assert row["category"] == "unrecoverable:forbidden_import_inside_test"


def test_unused_module_level_fragment_import_is_removable():
    frag = "import pickle\n\n\ndef test_area():\n    assert area(2) == 4\n"
    row = pr.audit_row(frag, ctx("from pkg.geo import area"), "area")
    assert row["category"] == "recoverable:unused_forbidden_import_removed"
    assert "pickle" not in row["after_module"]
    assert "assert area(2) == 4" in row["after_module"]


def test_test_support_imports_stay_rejected():
    row = pr.audit_row(FRAG, ctx("import os\nfrom tests.utils import helper\n"
                                 "from pkg.geo import area"), "area")
    assert row["category"] == "unrecoverable:test_support_import_required"


def test_forbidden_names_are_never_recovered():
    frag = "def test_area():\n    assert area(open) == 4\n"
    row = pr.audit_row(frag, ctx("import os\nfrom pkg.geo import area"), "area")
    assert row["category"] == "unrecoverable:forbidden_name_or_attribute_used"
