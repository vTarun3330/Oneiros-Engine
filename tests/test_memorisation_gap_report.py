"""Pin the gap report to the artifacts, so a stale number cannot survive.

The first version of this report was typed by hand. That is how it came to
quote a base control gap in prose while the arms table was built from a
different run, and why adding an arm meant editing JSON. Every assertion here
is about DERIVATION: the report must say what the files say.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_memorisation_gap_report import ARMS, BASE, TRAIN_FILE, VAL_FILE, build

REPORT = ROOT / "results" / "v4_2_memorisation_gap.json"


def _evaluation(rate: float, records: int) -> str:
    return json.dumps({
        "evaluation_split_records": records,
        "kill_at_k": {"8": {"rate": rate, "functions": int(rate * records)}},
    })


def _tree(tmp_path: Path, arms: dict[str, tuple[float, float]]) -> Path:
    for key, relative in BASE.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        rate = 0.70 if key == "train" else 0.60
        path.write_text(_evaluation(rate, 900 if key == "train" else 757),
                        encoding="utf-8")
    for name, (train, val) in arms.items():
        run = tmp_path / ARMS[name]
        run.mkdir(parents=True, exist_ok=True)
        (run / TRAIN_FILE).write_text(_evaluation(train, 900), encoding="utf-8")
        (run / VAL_FILE).write_text(_evaluation(val, 757), encoding="utf-8")
    return tmp_path


def test_the_gap_is_reported_as_excess_over_the_base_control(tmp_path):
    """A raw gap conflates overfitting with the panels differing in difficulty."""
    report = build(_tree(tmp_path, {"full_density": (0.80, 0.62)}))
    assert report["base_control"]["gap_points"] == 10.0
    arm = report["arms"]["full_density"]
    assert arm["gap_points"] == 18.0
    assert arm["excess_gap_over_base_control_points"] == 8.0


def test_an_arm_missing_one_evaluation_is_named_not_half_reported(tmp_path):
    """Reporting a train number with no val partner invents a gap of zero."""
    tree = _tree(tmp_path, {"full_density": (0.80, 0.62)})
    (tree / ARMS["relearning"]).mkdir(parents=True, exist_ok=True)
    (tree / ARMS["relearning"] / TRAIN_FILE).write_text(
        _evaluation(0.83, 900), encoding="utf-8")

    report = build(tree)
    assert "relearning" not in report["arms"]
    assert "relearning" in report["arms_missing_an_evaluation"]


def test_the_prose_answer_quotes_the_measured_base_control(tmp_path):
    """The prose drifted from the table once; it is now formatted from it."""
    report = build(_tree(tmp_path, {"full_density": (0.80, 0.62)}))
    assert "0.7000 on train against 0.6000" in report["answer"]
    assert "10.0 point gap" in report["answer"]


def test_the_committed_report_matches_the_committed_artifacts():
    """Guards against a hand-edit, and against an arm's file being replaced."""
    if not REPORT.exists():
        return
    committed = json.loads(REPORT.read_text(encoding="utf-8"))
    rebuilt = build(ROOT / "results")
    assert committed["arms"] == rebuilt["arms"], (
        "the committed gap report no longer matches the evaluation files it "
        "claims to summarise; rebuild it rather than editing it"
    )
    assert committed["base_control"]["gap_points"] == \
        rebuilt["base_control"]["gap_points"]


def test_every_arm_is_measured_on_the_same_seed_and_panels():
    """A gap across different panels or seeds is not a gap."""
    if not REPORT.exists():
        return
    committed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert TRAIN_FILE.endswith("seed_42.json") and VAL_FILE.endswith("seed_42.json")
    targets = {(a["train_targets"], a["val_targets"])
               for a in committed["arms"].values()}
    assert len(targets) == 1, f"arms measured on different panels: {targets}"


def test_no_arm_is_claimed_to_generalise_better_than_the_untrained_base():
    """Every measurement so far says SFT ADDS gap; a claim otherwise is a bug."""
    if not REPORT.exists():
        return
    committed = json.loads(REPORT.read_text(encoding="utf-8"))
    for name, arm in committed["arms"].items():
        assert arm["excess_gap_over_base_control_points"] > 0, (
            f"{name} reports a smaller gap than the untrained base; that would "
            "reverse the headline finding and needs checking, not publishing"
        )
