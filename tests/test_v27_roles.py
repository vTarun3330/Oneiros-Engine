"""v2.7 role isolation: fail closed on crossing, mixing, missing roles and alias bypass."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import v27_roles as roles

ROOT = Path(__file__).resolve().parent.parent


def rec(i, role, repo="o/r"):
    return {"id": f"x{i}", "role": role, "repository": repo}


def test_confirmation_records_cannot_enter_training():
    with pytest.raises(roles.RoleViolation, match="offered to training"):
        roles.admit_to_training([rec(0, roles.TRAINING), rec(1, roles.CONFIRMATION)])


def test_training_records_cannot_enter_the_panel():
    with pytest.raises(roles.RoleViolation, match="offered to the confirmation panel"):
        roles.admit_to_confirmation([rec(0, roles.TRAINING)])


@pytest.mark.parametrize("bad", [{"id": "a"}, {"id": "a", "role": None},
                                 {"id": "a", "role": "both"}])
def test_missing_or_unknown_role_fails(bad):
    with pytest.raises(roles.RoleViolation, match="role metadata"):
        roles.admit_to_training([bad])
    with pytest.raises(roles.RoleViolation, match="role metadata"):
        roles.admit_to_confirmation([bad])


@pytest.mark.parametrize("alias", ["fork-owner/altair", "https://github.com/Altair-Viz/Altair.git",
                                   "renamed__altair"])
def test_aliases_and_renames_cannot_bypass_isolation(alias):
    with pytest.raises(roles.RoleViolation, match="alias/rename"):
        roles.admit_to_training([rec(0, roles.TRAINING, alias)],
                                confirmation_keys=["altair-viz/altair"])
    with pytest.raises(roles.RoleViolation, match="training pools"):
        roles.admit_to_confirmation([rec(0, roles.CONFIRMATION, alias)],
                                    training_keys=["other/altair"])


def test_candidate_overlap_is_rechecked_after_acquisition():
    cands = [{"id": "a", "function_fingerprint": "fp1", "fixed_commit": "c1", "lineage": "L1"},
             {"id": "b", "function_fingerprint": "fp9", "fixed_commit": "c9", "lineage": "L9"},
             {"id": "c", "function_fingerprint": "fp8", "buggy_commit": "c2", "lineage": "L2"}]
    hits = roles.candidate_overlap(cands, training_fingerprints={"fp1"}, prior_commits={"c2"},
                                   training_lineages={"L1"})
    assert {h["id"]: h["overlap"] for h in hits} == {
        "a": ["function_fingerprint", "lineage"], "c": ["commit"]}


def test_frozen_registry_is_70_confirmation_only_and_disjoint_from_training():
    reg = json.loads((ROOT / "results/sft_root_cause_v27_confirmation_role_registry.json")
                     .read_text(encoding="utf-8"))
    part = json.loads((ROOT / "results/sft_root_cause_v25_universe_partition.json")
                      .read_text(encoding="utf-8"))
    repos = reg["repositories"]
    assert len(repos) == 70 and all(r["role"] == roles.CONFIRMATION and r["training_prohibited"]
                                    for r in repos)
    keys = {roles.repository_key(r["repository"]) for r in repos}
    assert not keys & set(part["training_expansion_pool"])
    assert roles.admit_to_confirmation(repos, training_keys=part["training_expansion_pool"])
