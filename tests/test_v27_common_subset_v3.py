"""v2.7 Phase 6b successor v3: eligibility recomputed from committed per-seed evidence, refusal
of inconsistent or tampered evidence, receipt immutability and the bound historical source."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v3 = _load("v27_freeze3", "scripts/v27_common_subset_freeze_v3.py")
v2 = _load("v27_freeze2b", "scripts/v27_common_subset_freeze_v2.py")
BUNDLE = ROOT / "results/sft_root_cause_v27_phase6_evidence_bundle_v3.json"
RECEIPT = ROOT / "results/sft_root_cause_v27_common_subset_r4_v3.json"


def _bundle():
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def _run_with(tmp_path, monkeypatch, bundle):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setattr(v3, "BUNDLE", str(path))
    return v3.main(["--out", str(tmp_path / "out.json")])


def test_clean_clone_reproduction_with_recomputed_eligibility(tmp_path):
    out = tmp_path / "subset.json"
    assert v3.main(["--out", str(out)]) == 0
    fresh = json.loads(out.read_text(encoding="utf-8"))
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for key in ("sets", "fuzzable_ids_sha256", "native_executable_ids_sha256", "waterfall",
                "exclusions", "composition", "statistics"):
        assert fresh[key] == committed[key], key
    covered = [r for r in fresh["targets"] if r["adapter_covered"]]
    assert covered and all(r["eligibility_recomputed_from_evidence"] for r in covered)


def test_a_stored_flag_that_disagrees_with_the_evidence_is_refused(tmp_path, monkeypatch):
    b = _bundle()
    t = next(t for t in b["targets"] if "pipdeptree" in t["target_id"])
    t["atheris_v2"]["eligible_for_fuzzing"] = True
    t["atheris_v2"]["reason"] = None
    with pytest.raises(SystemExit, match="disagree"):
        _run_with(tmp_path, monkeypatch, b)


def test_tampered_seed_evidence_is_refused(tmp_path, monkeypatch):
    b = _bundle()
    t = next(t for t in b["targets"] if "stamina" in t["target_id"])
    runs = t["atheris_v2_structured_runs"]
    for label in runs:
        runs[label][1] = copy.deepcopy(runs[label][0])      # pretend the jitter vanished
    with pytest.raises(SystemExit, match="disagree"):
        _run_with(tmp_path, monkeypatch, b)


def test_missing_evidence_for_a_covered_target_is_refused(tmp_path, monkeypatch):
    b = _bundle()
    t = next(t for t in b["targets"] if "prompt-toolkit" in t["target_id"])
    t["atheris_v2_structured_runs"] = None
    with pytest.raises(SystemExit, match="per-seed evidence"):
        _run_with(tmp_path, monkeypatch, b)


@pytest.mark.parametrize("module, rel", [
    ("v3", "results/sft_root_cause_v27_common_subset_r4_v3.json"),
    ("v2", "results/sft_root_cause_v27_common_subset_r4_v2.json")])
def test_accepted_receipts_are_never_overwritten(module, rel):
    before = (ROOT / rel).read_bytes()
    with pytest.raises(SystemExit, match="never overwritten"):
        {"v3": v3, "v2": v2}[module].main(["--out", rel])
    assert (ROOT / rel).read_bytes() == before


def test_historical_discordance_is_bound_to_its_source_receipt():
    h = v3.historical_discordance()
    assert (h["gained"], h["lost"], h["paired"]) == (114, 88, 757)
    stats = json.loads(RECEIPT.read_text(encoding="utf-8"))["statistics"]
    assert stats["historical_discordance"]["source"]["sha256"] == h["source"]["sha256"]


def test_statistical_wording_is_derived_from_counts():
    s = json.loads(RECEIPT.read_text(encoding="utf-8"))["statistics"]
    assert f"(n={s['native_executable']['n']})" in s["wording"]
    assert "fuzzable subset (n=3): significance at 0.05 is mathematically impossible" in s["wording"]


def test_bundle_v3_is_lf_and_redacted():
    import receipt_sanitize
    data = BUNDLE.read_bytes()
    assert b"\r\n" not in data and receipt_sanitize.check_file(BUNDLE) == []
    pip = next(t for t in _bundle()["targets"] if "pipdeptree" in t["target_id"])
    seed = pip["atheris_v2_structured_runs"]["buggy"][0]["seeds"][0]
    assert seed["outcome_kind"] == "system_exit" and seed["outcome_detail"] == '"1"'
