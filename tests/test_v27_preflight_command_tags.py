"""Stored preflight commands derive every tagged identifier from --tag (r6 provenance fix)."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import v27_gpu_preflight_r5 as pf  # noqa: E402

ARMS = ["base", "sft", "relearn"]
R5 = "results/sft_root_cause_v27_gpu_preflight_r5.json"
R6 = "results/sft_root_cause_v27_gpu_preflight_r6.json"
R7 = "results/sft_root_cause_v27_gpu_preflight_r7.json"


def prep() -> str:
    manifest = json.loads((ROOT / pf.MANIFEST).read_text(encoding="utf-8"))
    return manifest["requalification_records"]["path"]


def stored(receipt: dict) -> dict:
    c = receipt["commands"]
    return {"generate": c["generate"], "gates": c["read_only_gate_before_each_arm"],
            "execute": c["execute_after_all_three_arms"], "analyse": c["analyse_after_execution"]}


def all_text(cmds: dict) -> str:
    return " ".join([*cmds["generate"].values(), *cmds["gates"].values(), cmds["execute"],
                     cmds["analyse"]])


def test_default_r5_commands_are_byte_identical_to_the_committed_r5_receipt():
    receipt = json.loads((ROOT / R5).read_text(encoding="utf-8"))
    assert pf.stored_commands("r5", R5, ARMS, prep()) == stored(receipt)
    assert pf.command_tag_problems("r5", stored(receipt), R5) == []


def test_default_r5_wrapper_identity_is_unchanged():
    gen = pf.stored_commands("r5", R5, ARMS, prep())["generate"]
    for arm in ARMS:
        assert gen[arm].startswith(".venv-gpu/Scripts/python.exe scripts/gpu_run.py start "
                                   f"--name v27r5_generate_{arm} --exclusive-key "
                                   "v27r5_generation -- ")


def test_r7_uses_only_r7_wrapper_names_and_exclusive_key():
    gen = pf.stored_commands("r7", R7, ARMS, prep())["generate"]
    for arm in ARMS:
        assert f"--name v27r7_generate_{arm} " in gen[arm]
        assert "--exclusive-key v27r7_generation " in gen[arm]
        assert set(re.findall(r"v27(r\d+)_generat", gen[arm])) == {"r7"}


@pytest.mark.parametrize("stale", ["v27r5", "v27r6"])
def test_no_stale_wrapper_identifier_in_r7_generate_commands(stale):
    gen = pf.stored_commands("r7", R7, ARMS, prep())["generate"]
    assert not any(stale in c for c in gen.values())


def test_r7_paths_run_names_and_key_are_mutually_consistent():
    cmds = pf.stored_commands("r7", R7, ARMS, prep())
    t = pf.tagged_paths("r7")
    assert t == {"OUT_ROOT": "results/sft_root_cause/v27_generations_r7",
                 "EXEC_OUT": "results/sft_root_cause/v27_execution_r7",
                 "ANALYSIS_OUT": "results/sft_root_cause_v27_analysis_r7.json",
                 "AUTH": "results/sft_root_cause_v27_gpu_authorization_r7.json",
                 "RUN_NAME_PREFIX": "v27r7_generate_", "EXCLUSIVE_KEY": "v27r7_generation"}
    for arm in ARMS:
        g, gate = cmds["generate"][arm], cmds["gates"][arm]
        for c in (g, gate):
            assert f"--out {t['OUT_ROOT']}/{arm}" in c and f"--authorization {t['AUTH']}" in c
            assert f"--preflight {R7}" in c
    assert f"--generations {t['OUT_ROOT']} " in cmds["execute"]
    assert cmds["execute"].endswith(f"--out {t['EXEC_OUT']}")
    assert f"--generations {t['OUT_ROOT']} " in cmds["analyse"]
    assert f"{t['EXEC_OUT']}/results_primary_whole_module.jsonl" in cmds["analyse"]
    assert cmds["analyse"].endswith(f"--out {t['ANALYSIS_OUT']}")
    assert pf.command_tag_problems("r7", cmds, R7) == []
    tags = set(pf.TAG_TOKEN.findall(all_text(cmds).replace(pf.JOB, "").replace(pf.MANIFEST, "")
                                    .replace(prep(), "")))
    assert tags == {"r7"}


def test_frozen_inputs_are_not_retagged():
    cmds = pf.stored_commands("r7", R7, ARMS, prep())
    for c in cmds["generate"].values():
        assert f"--job {pf.JOB} " in c
    assert f"--manifest {pf.MANIFEST} " in cmds["execute"] and f"--prep {prep()} " in cmds["execute"]


def test_guard_detects_the_r6_receipt_defect():
    receipt = json.loads((ROOT / R6).read_text(encoding="utf-8"))
    problems = pf.command_tag_problems("r6", stored(receipt), R6)
    assert any("v27r5_generate_base is tagged r5, not r6" in p for p in problems)
    assert any("v27r5_generation is tagged r5, not r6" in p for p in problems)
    assert any("missing --exclusive-key v27r6_generation" in p for p in problems)


@pytest.mark.parametrize("old,new", [
    ("--name v27r7_generate_sft", "--name v27r6_generate_sft"),
    ("--exclusive-key v27r7_generation", "--exclusive-key v27r5_generation"),
    ("v27_generations_r7/sft", "v27_generations_r6/sft"),
    ("authorization_r7.json", "authorization_r6.json"),
])
def test_guard_refuses_any_single_stale_identifier(old, new):
    cmds = pf.stored_commands("r7", R7, ARMS, prep())
    assert old in cmds["generate"]["sft"]
    cmds["generate"]["sft"] = cmds["generate"]["sft"].replace(old, new)
    assert pf.command_tag_problems("r7", cmds, R7)


def test_guard_refuses_stale_execution_or_analysis_paths():
    cmds = pf.stored_commands("r7", R7, ARMS, prep())
    cmds["execute"] = cmds["execute"].replace("v27_execution_r7", "v27_execution_r6")
    assert pf.command_tag_problems("r7", cmds, R7)
    cmds = pf.stored_commands("r7", R7, ARMS, prep())
    cmds["analyse"] = cmds["analyse"].replace("analysis_r7.json", "analysis_r5.json")
    assert pf.command_tag_problems("r7", cmds, R7)


def test_non_alphanumeric_tag_refused():
    with pytest.raises(SystemExit):
        pf.stored_commands("r7/../x", R7, ARMS, prep())
