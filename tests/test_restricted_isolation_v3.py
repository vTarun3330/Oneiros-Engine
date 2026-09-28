"""Restricted-data isolation for the Phase 3 v3 / Phase 4 redesign code paths.

Static evidence: no new source names a restricted split or protected location.
Runtime evidence: the rescoring loader refuses any record shard other than train.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import fixed_input_rescoring

ROOT = Path(__file__).resolve().parent.parent
NEW_SOURCES = [
    "harness/probe_statistics.py", "harness/fixed_input_rescoring.py",
    "harness/objective_masking.py", "harness/power_simulation.py",
    "scripts/analyze_fixed_input_probe_v3.py", "scripts/analyze_capacity_probe_v3.py",
    "scripts/correct_phase3_addendum.py", "scripts/phase4_power_analysis.py",
    "scripts/update_phase4_pointers.py", "scripts/phase3_v3_preflight.py",
]
RESTRICTED = ["val.records.json", "ablation_dev.records.json", "test.records.json",
              "sealed_final", "unopened_confirmation", "splits.json",
              "aprime_confirmation_pilot", "aprime_fresh_confirmation_pilot"]


@pytest.mark.parametrize("source", NEW_SOURCES)
def test_no_new_source_names_a_restricted_location(source):
    text = (ROOT / source).read_text(encoding="utf-8")
    assert not [token for token in RESTRICTED if token in text], source


def test_the_rescoring_loader_refuses_a_non_train_shard(monkeypatch, tmp_path):
    design = json.loads((ROOT / fixed_input_rescoring.DESIGN_3A).read_text(encoding="utf-8"))
    design["record_shard"]["path"] = design["record_shard"]["path"].replace("train", "val")
    fake = tmp_path / "design.json"
    fake.write_text(json.dumps(design), encoding="utf-8")
    monkeypatch.setattr(fixed_input_rescoring, "DESIGN_3A", str(fake))
    with pytest.raises(SystemExit, match="only the train shard"):
        fixed_input_rescoring.load_inputs(("base",))
