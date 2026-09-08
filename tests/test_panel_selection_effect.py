"""Pin the selection-effect measurement.

ablation_dev chose the checkpoints, so a gain measured on it is not a
generalisation estimate. The size of that bias is now a measured number rather
than a caution, and anything that quotes ablation_dev depends on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_selection_and_locked_panels import compare

REPORT = ROOT / "results" / "v4_2_panel_selection_effect.json"


def _write(path: Path, rate: float, records: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "evaluation_split_records": records,
        "kill_at_k": {"8": {"rate": rate}},
    }), encoding="utf-8")


def test_the_inflation_factor_is_the_ratio_of_mean_gains(tmp_path):
    for seed in (42, 43):
        _write(tmp_path / f"local_base_qwen_adev_seed{seed}"
               / f"base_validation_ablation-dev_seed_{seed}.json", 0.50)
        _write(tmp_path / "arm" / f"sft_validation_ablation-dev_seed_{seed}.json", 0.60)
        _write(tmp_path / f"local_base_qwen_val_seed{seed}"
               / f"base_validation_standard_seed_{seed}.json", 0.50)
        _write(tmp_path / "arm" / f"sft_validation_standard_seed_{seed}.json", 0.55)

    report = compare("arm", "test-arm", [42, 43], results=tmp_path)

    assert report["panels"]["ablation_dev"]["mean_gain"] == 0.1
    assert report["panels"]["val"]["mean_gain"] == 0.05
    assert report["selection_panel_inflation_factor"] == 2.0


def test_a_seed_missing_one_panel_is_skipped_not_half_counted(tmp_path):
    _write(tmp_path / "local_base_qwen_val_seed42"
           / "base_validation_standard_seed_42.json", 0.50)
    _write(tmp_path / "arm" / "sft_validation_standard_seed_42.json", 0.55)

    report = compare("arm", "test-arm", [42], results=tmp_path)

    assert report["panels"]["ablation_dev"]["seeds"] == []
    assert report["panels"]["ablation_dev"]["mean_gain"] is None
    assert report["selection_panel_inflation_factor"] is None, (
        "an inflation factor computed from one panel is not a comparison"
    )


def test_the_committed_finding_still_shows_selection_inflation():
    if not REPORT.exists():
        return
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["sealed_final_test_accessed"] is False
    assert report["selection_panel_inflation_factor"] > 1.0, (
        "ablation_dev no longer overstates the arm; every ablation_dev figure "
        "in the writeup carries this caveat and it must be re-derived"
    )
    assert report["per_seed_ranges_disjoint"] is True, (
        "the two panels' per-seed gain ranges now overlap, so the separation "
        "is no longer clean and the claim needs weakening"
    )
