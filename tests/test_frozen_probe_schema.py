"""The v1 probe schema identity is frozen: the code must still describe the v1 cohort exactly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harness.fixed_input_probe import (
    CORRECTED_LEVEL_LABELS, LEVEL_DEFINITIONS, PROBE_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "results" / "sft_root_cause_phase3a_cohort.json"
FROZEN_SHA256 = "1316956a66f718ec1b900438132eb0ae3bc7e8e6da718a58b0b5fcc18d718ce7"


def test_the_frozen_cohort_is_unchanged():
    assert hashlib.sha256(COHORT.read_bytes()).hexdigest() == FROZEN_SHA256


def test_v1_schema_metadata_in_code_matches_the_frozen_cohort():
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    assert PROBE_SCHEMA_VERSION == cohort["schema_version"] == "oneiros_fixed_input_probe_v1"
    assert LEVEL_DEFINITIONS == cohort["level_definitions"]


def test_corrected_labels_live_outside_the_v1_schema():
    assert CORRECTED_LEVEL_LABELS["A3"] != LEVEL_DEFINITIONS["A3"]
    assert "oracle-derived" in CORRECTED_LEVEL_LABELS["A3"]
    assert "complete fixed implementation" in CORRECTED_LEVEL_LABELS["A4"]
