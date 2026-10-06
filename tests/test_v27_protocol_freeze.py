"""v2.7 frozen confirmation protocol: the 70-repository pool, its roles and the KEEP_ALL_70
earlier-contact decision cannot change without a new versioned protocol file and receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FROZEN = {
    "docs/repository_native_v27_confirmation_repositories.json":
        "0018660e5c169f3302d31d3d1fe3f5892cb12639b242dc39d7aa84140f649ef8",
    "docs/repository_native_v27_confirmation_repositories_v2.json":
        "77c677f2ed40d2e6096b9e2f4c1236b53cc5402733913ddcf0bc4806cc85e2e2",
    "results/sft_root_cause_v27_confirmation_role_registry.json":
        "19c4c973e1bd140ac607fbf7c7cb24820068c91d837fe9d3a7c56c39a5703423",
    "results/sft_root_cause_v27_pool_decision.json":
        "160e618e7e5f813721b5ea111c497e5cda5f6697055d8a13874d350e52b76a42",
    "results/sft_root_cause_v27_phase1_overlap_erratum.json":
        "51bdda3ef7ce37d911d2de347b3ed346e7beba95f763224e81709c4055fe27b9",
    "docs/repository_native_v27_previously_scanned_minus_permitted.json":
        "afcf6cbd6b2c5ff78d05677138e49ce207fec9f688751199efb2f2a4709a8630",
    "docs/repository_native_v27_confirmation_acquisition_v3.json":
        "30439dc1eb89943651c69e77a7dff7deddd50bd76aa28b2169b831e0c74d7ae0",
    "docs/repository_native_v27_confirmation_acquisition_v4.json":
        "d52be1343296efe54828e396f136ff4521a1ce41a483feb00a568be5954c20e0",
    "docs/repository_native_v27_confirmation_panel_gate.json":
        "757fe23691ad5594355b47101d48569103768318f483f88028a6f4b3d681dcee",
}


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@pytest.mark.parametrize("rel", sorted(FROZEN))
def test_frozen_protocol_file_is_unchanged(rel):
    """Changing any of these requires a NEW versioned file (and a new pin here), never an edit."""
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == FROZEN[rel]


def test_pool_list_registry_and_decision_agree():
    listed = load("docs/repository_native_v27_confirmation_repositories_v2.json")["repositories"]
    reg = load("results/sft_root_cause_v27_confirmation_role_registry.json")
    names = [r if isinstance(r, str) else r["repository"] for r in listed]
    assert len(names) == len(set(names)) == 70
    assert {r["repository"] for r in reg["repositories"]} == set(names)
    decision = load("results/sft_root_cause_v27_pool_decision.json")
    permitted = decision["permitted_repositories"]
    assert decision["decision"] == "KEEP_ALL_70" and len(set(permitted)) == 14
    assert set(permitted) <= set(names)
    erratum = decision["erratum"]
    assert hashlib.sha256((ROOT / erratum["path"]).read_bytes()).hexdigest() == erratum["sha256"]
    reuse = decision["reuse_exclusion_file"]
    assert hashlib.sha256((ROOT / reuse["path"]).read_bytes()).hexdigest() == reuse["sha256"]


def test_acquisition_config_uses_the_frozen_list_only():
    cfg = load("docs/repository_native_v27_confirmation_acquisition_v3.json")
    flat = json.dumps(cfg)
    assert "repository_native_v27_confirmation_repositories_v2.json" in flat
    assert "training_expansion" not in flat
