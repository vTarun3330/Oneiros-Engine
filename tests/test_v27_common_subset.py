"""v2.7 Phase 6b: Atheris eligibility classification and the common-subset freeze (exact McNemar
power, model-independent membership, reproducible ID hash)."""
from __future__ import annotations

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


freeze = _load("v27_freeze", "scripts/v27_common_subset_freeze.py")
elig = _load("v27_elig", "scripts/v27_atheris_eligibility_wsl.py")


def test_exact_mcnemar_values():
    assert freeze.mcnemar_p(0, 0) == 1.0
    assert freeze.mcnemar_p(5, 0) == pytest.approx(0.0625)      # 2 * 0.5**5
    assert freeze.mcnemar_p(3, 3) == 1.0
    assert freeze.mcnemar_p(10, 0) == pytest.approx(2 / 1024)


def test_five_targets_can_never_reach_significance():
    assert freeze.smallest_attainable_p(5) > 0.05
    assert freeze.power(5, 0.267, 0.2, 0.05) == 0.0
    assert freeze.mde(5, 0.267, 0.05) is None


def test_power_grows_with_effect_and_sample():
    assert freeze.power(34, 0.267, 0.2, 0.05) > freeze.power(34, 0.267, 0.1, 0.05)
    assert freeze.power(60, 0.267, 0.2, 0.05) > freeze.power(30, 0.267, 0.2, 0.05)


@pytest.mark.parametrize("reason, cls", [
    ("adapter_unsupported:receiver_constructor:variadic_signature", "adapter"),
    ("import_failure:ModuleNotFoundError", "dependency"),
    ("atheris_abi_unavailable", "environment"),
    ("revision_plan_mismatch", "target"),
    ("revision_load_mismatch:adapter_unsupported:x", "target"),
    ("seed_invocation_failure:replay_process_failure", "harness"),
    (None, None),
])
def test_probe_failure_classes(reason, cls):
    assert elig.failure_class(reason) == cls


def test_seed_input_is_fixed():
    assert len(elig.SEED_INPUT) == 256
    assert elig.SEED_INPUT == elig.SEED_INPUT[:32] * 8


RECEIPT = ROOT / "results/sft_root_cause_v27_common_subset_r4.json"


def test_provisional_v1_receipt_is_internally_consistent():
    """v1 (PROVISIONAL, superseded by v2) used git-ignored inputs; on any clone its committed
    receipt must still be self-consistent: the ID hash matches its ID list, membership matches
    its per-target rows and the panel is never redefined."""
    import hashlib
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    ids = committed["common_subset_ids"]
    assert hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == \
        committed["common_subset_ids_sha256"]
    assert sorted(t["target_id"] for t in committed["targets"] if t["in_common_subset"]) == ids
    assert len(committed["targets"]) == 34
