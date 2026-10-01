"""v2.5 stage-1 r2 corrections: fail-closed corpus access audit, train-only complexity,
canonical duplicate identities, repeatability (not validation) receipts."""
from __future__ import annotations

import pytest

from scripts import v25_converted_corpus as cc

C = cc.CORPUS_REL


@pytest.mark.parametrize("rel", [
    f"{C}/development_view/complexity_manifest.json",
    f"{C}/development_view/val.records.json",
    f"{C}/development_view/ablation_dev.records.json",
    f"{C}/records.json", f"{C}/splits.json", f"{C}/external_eval_index.json",
    "results/sealed_final_state.json",
    "results/v4_3_execution_supervision_v1/unopened_confirmation.ids.json",
])
def test_non_train_metadata_and_protected_files_are_forbidden(rel):
    assert cc.forbidden_reason(rel) is not None


@pytest.mark.parametrize("rel", sorted(cc.ALLOWED_CORPUS_FILES))
def test_exactly_the_train_only_files_are_allowed(rel):
    assert cc.forbidden_reason(rel) is None


def test_the_audit_hook_blocks_and_records_a_forbidden_open(monkeypatch):
    monkeypatch.setattr(cc, "OPENED", [])
    monkeypatch.setattr(cc, "VIOLATIONS", [])
    path = cc.ROOT / C / "development_view" / "complexity_manifest.json"
    with pytest.raises(PermissionError, match="non_train_or_combined"):
        cc._audit("open", (str(path), "r", 0))
    receipt = cc.access_receipt()
    assert receipt["passed"] is False and receipt["combined_complexity_manifest_opened"]


def test_the_audit_ignores_unrelated_files(monkeypatch):
    monkeypatch.setattr(cc, "OPENED", [])
    monkeypatch.setattr(cc, "VIOLATIONS", [])
    cc._audit("open", (str(cc.ROOT / "scripts" / "x.py"), "r", 0))
    assert cc.access_receipt() == {**cc.access_receipt(), "opened": [], "passed": True}


def test_prepare_installs_the_audit_before_loading_the_corpus():
    import inspect
    source = inspect.getsource(cc.prepare)
    assert source.index("install_audit()") < source.index("load_development_split")
    assert "load_complexity_index" not in inspect.getsource(cc)


def test_complexity_is_computed_from_train_code_only():
    assert cc.train_complexity("def f(x):\n    return x\n", "f") == "simple"
    assert cc.train_complexity("not python(", "f") == "unscorable"


def test_canonical_identity_ignores_formatting_and_near_identity_ignores_constants():
    a = cc.canonical_ids("from t import f\n\n\ndef test_f():\n    assert f(1) == 2\n")
    b = cc.canonical_ids("from t import f\ndef test_f():\n    assert f(1)==2  # same\n")
    c = cc.canonical_ids("from t import f\n\n\ndef test_f():\n    assert f(3) == 9\n")
    assert a["raw"] != b["raw"] and a["canonical_ast"] == b["canonical_ast"]
    assert a["canonical_ast"] != c["canonical_ast"]
    assert a["near_duplicate_cluster"] == c["near_duplicate_cluster"]


def test_receipt_requires_two_identical_repeatability_runs(tmp_path):
    (tmp_path / "converted_candidates.jsonl").write_text("")
    (tmp_path / "verification_run1.jsonl").write_text("a\n")
    (tmp_path / "verification_run2.jsonl").write_text("b\n")
    with pytest.raises(SystemExit, match="repeatability"):
        cc.receipt(tmp_path)
