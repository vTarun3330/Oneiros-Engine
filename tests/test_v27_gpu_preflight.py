"""v2.7 Phase 8: the repository-native generation job/cohort and the CPU-only GPU preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import native_generated_test_job_v25 as jobs  # noqa: E402
from scripts import v27_gpu_preflight as pf  # noqa: E402
from scripts.native_generation_io import resolve_cohort  # noqa: E402

JOB = ROOT / pf.JOB
MANIFEST = ROOT / pf.MANIFEST
SUBSET = json.loads((ROOT / pf.SUBSET).read_text(encoding="utf-8"))


def test_job_is_a_valid_current_v25_job_with_no_refusals():
    job = jobs.validate_job_file(json.loads(JOB.read_text(encoding="utf-8")), pf.CONDITION, ROOT)
    assert len(job["items"]) == 29 and job["refused"] == []
    assert max(i["prompt_tokens"] for i in job["items"]) + jobs.COMPLETION_TOKENS \
        <= jobs.SEQUENCE_LIMIT


def test_cohort_is_exactly_the_native_executable_set_of_the_frozen_panel():
    cohort = resolve_cohort(JOB, MANIFEST, pf.CONDITION)
    ids = cohort["generation"]
    assert ids == SUBSET["sets"]["native_executable"]
    assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == \
        SUBSET["native_executable_ids_sha256"]
    assert cohort["expected"]["rows_per_arm"] == 87
    assert cohort["expected"]["candidates_per_arm"] == 696


def test_manifest_binds_the_panel_and_keeps_policy_exclusions_outside_the_cohort():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["panel"]["sha256"] == hashlib.sha256((ROOT / pf.PANEL).read_bytes()).hexdigest()
    assert m["panel"]["targets"] == 34 and len(m["panel_policy_exclusions"]) == 5
    assert not {e["target_key"] for e in m["panel_policy_exclusions"]} & \
        set(m["generation_targets"])
    assert m["atheris_independent"] is True and m["training_prohibited"] is True


def test_cohort_never_equals_the_atheris_subset():
    assert set(SUBSET["sets"]["fuzzable"]) < set(SUBSET["sets"]["native_executable"])


def test_checks_record_failures():
    c = pf.Checks()
    c.add("a", True, {})
    c.add("b", False, {"why": 1})
    assert c.failed == ["b"] and c.items["b"]["evidence"] == {"why": 1}


def test_preflight_is_written_once(tmp_path):
    existing = tmp_path / "pre.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED"):
        pf.main(["--out", str(existing)])


ADAPTERS = ROOT / "checkpoints/local_sft_armA_baseline_successor_s42/sft_adapter"


@pytest.mark.skipif(not ADAPTERS.is_dir(), reason="local model checkpoints are not in git")
def test_both_historical_adapters_resolve_from_receipts():
    from scripts import native_generated_tests_generate as gen
    checks = pf.Checks()
    arms = pf.resolve_arms(checks, gen)
    assert not checks.failed, checks.items
    assert arms["sft"]["adapter_sha256"].startswith("e67dd599")
    assert arms["relearn"]["adapter_sha256"].startswith("af262f4f")
    assert arms["sft"]["adapter_dir"] == gen.CONTRACT["sft_adapter"]
