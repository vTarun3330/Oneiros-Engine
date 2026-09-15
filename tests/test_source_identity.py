"""A source hash that changes on checkout is a false alarm generator.

Three evaluation-defining files are CRLF in this working tree and LF in Git:
the entrypoint, the evaluator and the generator. Their raw SHA-256 therefore
depends on which machine cloned the repository, while the code is identical.
A provenance hash exists to remove that judgement call, so recording only the
raw digest would hand it straight back.

The fix is in what gets recorded, never in the files. Normalizing the working
tree would change the raw hashes that every frozen receipt in this project was
computed against - rewriting evidence to tidy up a reporting problem. So both
families are recorded and these tests pin what each one promises.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from harness.source_identity import (
    EVALUATION_DEFINING_SOURCES, HASH_SCHEME_DESCRIPTION, HASH_SCHEME_VERSION,
    canonical_bytes, canonical_sha256, canonical_text_sha256,
    evaluation_source_identities, git_blob_sha1, raw_sha256, scheme_block,
    source_identity,
)

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_LF = b"def f():\n    return 1\n\n\nclass K:\n    pass\n"
SAMPLE_CRLF = b"def f():\r\n    return 1\r\n\r\n\r\nclass K:\r\n    pass\r\n"
SAMPLE_CR = b"def f():\r    return 1\r\r\rclass K:\r    pass\r"


def write(tmp_path, name, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ------------------------------------- the property the whole module exists for

def test_crlf_and_lf_differ_raw_but_agree_canonically(tmp_path):
    lf = write(tmp_path, "lf.py", SAMPLE_LF)
    crlf = write(tmp_path, "crlf.py", SAMPLE_CRLF)
    assert raw_sha256(lf) != raw_sha256(crlf), "the raw hashes must still distinguish them"
    assert canonical_sha256(lf) == canonical_sha256(crlf)


def test_lone_cr_endings_also_canonicalise(tmp_path):
    lf = write(tmp_path, "lf.py", SAMPLE_LF)
    cr = write(tmp_path, "cr.py", SAMPLE_CR)
    assert raw_sha256(lf) != raw_sha256(cr)
    assert canonical_sha256(lf) == canonical_sha256(cr)


def test_a_real_content_change_changes_the_portable_hash(tmp_path):
    original = write(tmp_path, "a.py", SAMPLE_LF)
    before = canonical_sha256(original)
    original.write_bytes(SAMPLE_LF.replace(b"return 1", b"return 2"))
    assert canonical_sha256(original) != before


def test_a_one_character_change_changes_the_portable_hash(tmp_path):
    a = write(tmp_path, "a.py", b"x = 1\n")
    b = write(tmp_path, "b.py", b"x = 2\n")
    assert canonical_sha256(a) != canonical_sha256(b)


def test_whitespace_is_not_swallowed(tmp_path):
    """Only line endings are normalized; nothing else is forgiven."""
    a = write(tmp_path, "a.py", b"x = 1\n")
    b = write(tmp_path, "b.py", b"x  = 1\n")
    c = write(tmp_path, "c.py", b"x = 1   \n")
    assert len({canonical_sha256(a), canonical_sha256(b), canonical_sha256(c)}) == 3


def test_canonical_bytes_is_idempotent():
    once = canonical_bytes(SAMPLE_CRLF)
    assert canonical_bytes(once) == once
    assert b"\r" not in once


def test_a_byte_order_mark_is_preserved_not_stripped(tmp_path):
    """Documented behaviour: a BOM is a real difference, not a line ending."""
    plain = write(tmp_path, "p.py", SAMPLE_LF)
    bom = write(tmp_path, "b.py", b"\xef\xbb\xbf" + SAMPLE_LF)
    assert canonical_sha256(plain) != canonical_sha256(bom)


# ------------------------------------------------ agreement with Git itself

def _is_dirty(relative: str) -> bool:
    """True while the working copy of ``relative`` differs from the index/HEAD."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", relative],
        capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return bool(out)


@pytest.mark.parametrize("role,relative", sorted(EVALUATION_DEFINING_SOURCES.items()))
def test_committed_git_content_matches_the_portable_identity(role, relative):
    """Computed offline, checked against what Git actually stores.

    This is the claim that makes the canonical hash trustworthy: it is not a
    private convention, it reproduces Git's own object identity. Skipped while
    a file is mid-edit, because "the working copy differs from HEAD" is then
    the expected state and not the defect this guards against.
    """
    stored = subprocess.run(
        ["git", "rev-parse", f"HEAD:{relative}"],
        capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not stored:
        pytest.skip(f"{relative} is not committed at HEAD")
    if _is_dirty(relative):
        pytest.skip(f"{relative} has uncommitted changes")
    assert git_blob_sha1(ROOT / relative) == stored, relative


def test_the_canonical_hash_matches_the_committed_blob_content():
    """Independently: hash the bytes Git hands back, not the working file."""
    checked = 0
    for relative in sorted(EVALUATION_DEFINING_SOURCES.values()):
        if _is_dirty(relative):
            continue
        blob = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            capture_output=True, cwd=ROOT)
        if blob.returncode != 0:
            continue
        expected = hashlib.sha256(canonical_bytes(blob.stdout)).hexdigest()
        assert canonical_sha256(ROOT / relative) == expected, relative
        checked += 1
    if checked == 0:
        pytest.skip("every evaluation-defining source is mid-edit")


def test_the_crlf_files_are_exactly_the_ones_we_think(tmp_path):
    """If this list shrinks, the portability problem partly went away; if it
    grows, a new file acquired CRLF. Either way it should be noticed."""
    block = scheme_block(ROOT)
    assert set(block["portable_hashes_differ_from_raw"]) == {
        "evaluation_entrypoint", "evaluator", "generator"}


# ------------------------------------------------------------ the recorded block

def test_every_evaluation_defining_source_is_bound():
    identities = evaluation_source_identities(ROOT)
    assert set(identities) == set(EVALUATION_DEFINING_SOURCES)
    for role, item in identities.items():
        assert len(item["raw_sha256"]) == 64
        assert len(item["canonical_sha256"]) == 64
        assert len(item["git_blob_sha1"]) == 40
        assert item["bytes"] > 0


@pytest.mark.parametrize("role", [
    "evaluation_entrypoint", "evaluator", "candidate_policy",
    "timeout_policy_safe_execution", "prompt_builder", "prompt_budget",
    "generator", "model_runtime",
])
def test_the_required_roles_are_all_present(role):
    assert role in EVALUATION_DEFINING_SOURCES


def test_the_scheme_is_named_and_described():
    block = scheme_block(ROOT)
    assert block["hash_scheme_version"] == HASH_SCHEME_VERSION
    assert "CRLF" in block["hash_scheme"]
    assert "git rev-parse" in HASH_SCHEME_DESCRIPTION


def test_the_raw_hash_is_still_the_hash_of_the_file_on_disk():
    """The portable hash is an addition, not a replacement."""
    for role, relative in EVALUATION_DEFINING_SOURCES.items():
        path = ROOT / relative
        identity = source_identity(ROOT, relative)
        assert identity["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_adapter_resolution_is_not_pretended_to_be_a_file():
    """It lives inside the entrypoint; listing it as a path would be a lie."""
    assert "adapter_resolution" not in EVALUATION_DEFINING_SOURCES
    assert "harness/source_identity.py" not in EVALUATION_DEFINING_SOURCES.values()


def test_a_code_fragment_canonicalises_like_a_file():
    crlf_text = SAMPLE_CRLF.decode("utf-8")
    lf_text = SAMPLE_LF.decode("utf-8")
    assert canonical_text_sha256(crlf_text) == canonical_text_sha256(lf_text)
    assert canonical_text_sha256(lf_text) != canonical_text_sha256(lf_text + "\n# x\n")


# -------------------------------------------- the runtime still checks raw bytes

def test_the_run_contract_records_both_families():
    from scripts import train_on_dataset as trainer
    contract = trainer._adapter_evaluation_context(
        "fingerprint", "base_model", "a" * 64, "val", None, "b" * 64, 700)["run_contract"]
    assert contract["runner_source_sha256"] == raw_sha256(ROOT / "scripts/train_on_dataset.py")
    binding = contract["source_identity"]
    assert binding["hash_scheme_version"] == HASH_SCHEME_VERSION
    assert set(binding["sources"]) == set(EVALUATION_DEFINING_SOURCES)
    entry = binding["sources"]["evaluation_entrypoint"]
    assert entry["raw_sha256"] != entry["canonical_sha256"], "this file is CRLF here"


def test_the_adapter_hash_check_is_raw_and_happens_before_loading(tmp_path):
    """Weights are binary: canonicalizing them would be meaningless and unsafe."""
    from scripts import train_on_dataset as trainer
    d = tmp_path / "sft_adapter"
    d.mkdir()
    payload = b"\x00\x01\r\n\x02binary weights\r\n"
    (d / "adapter_model.safetensors").write_bytes(payload)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    raw = hashlib.sha256(payload).hexdigest()
    _path, resolved = trainer.resolve_external_adapter(d, raw)
    assert resolved == raw
    # the canonicalized digest must NOT be accepted for weights
    canonical = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    assert canonical != raw
    with pytest.raises(RuntimeError, match="Adapter hash mismatch"):
        trainer.resolve_external_adapter(d, canonical)


def test_the_preflight_still_verifies_the_local_raw_adapter_hash(monkeypatch):
    """Behaviourally: a wrong expected hash must still refuse the run.

    Asserted by running the preflight rather than grepping it, so renaming a
    local variable cannot silently retire the check.
    """
    from scripts import preflight_locked_validation as pf
    if not (ROOT / pf.ARM_A_431_ADAPTER).is_file():
        pytest.skip("Arm A checkpoint not present on this machine")
    clean: list[str] = []
    pf.collect(clean)
    assert not [p for p in clean if "adapter hash" in p]

    monkeypatch.setattr(pf, "ARM_A_431_ADAPTER_SHA256", "d" * 64)
    problems: list[str] = []
    built = pf.collect(problems)
    assert any("adapter hash does not match" in p for p in problems)
    assert built["arms"]["B_arm_a_checkpoint_431"]["adapter_sha256_found"] == raw_sha256(
        ROOT / pf.ARM_A_431_ADAPTER)


# --------------------------------- historical artifacts keep their own scheme

HISTORICAL = {
    "results/v4_2_frozen_development_evaluation_receipt.json":
        "062774028693cfcbf5346eafeed232027748720940af7a47b5ac1a591055c4fe",
}


@pytest.mark.parametrize("relative,expected", sorted(HISTORICAL.items()))
def test_historical_receipts_are_byte_for_byte_untouched(relative, expected):
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_the_frozen_development_receipt_still_uses_the_raw_only_scheme():
    """It predates this module and must not be retrofitted."""
    path = ROOT / "results/v4_2_frozen_development_evaluation_receipt.json"
    if not path.exists():
        pytest.skip("frozen receipt absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert "source_identity" not in receipt
    assert "hash_scheme_version" not in receipt
    assert "contract_source_hashes" in receipt


def test_historical_raw_hashes_still_verify_against_this_working_tree():
    """Raw-only hashes remain correct here; they were never wrong, just local."""
    path = ROOT / "results/v4_2_frozen_development_evaluation_receipt.json"
    if not path.exists():
        pytest.skip("frozen receipt absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    files = {
        "evaluator": "metrics/research_evaluation.py",
        "candidate_policy": "harness/candidate_policy.py",
        "timeout_policy_safe_execution": "harness/safe_execution.py",
        "prompt_builder": "engine/test_generation_prompt.py",
        "prompt_budget": "engine/prompt_budget.py",
        "generator": "engine/generator.py",
        "model_runtime": "engine/model_runtime.py",
    }
    for key, relative in files.items():
        assert raw_sha256(ROOT / relative) == receipt["contract_source_hashes"][key], relative


def test_the_development_selection_receipt_is_not_retrofitted():
    path = ROOT / "results/v4_2_development_selection_receipt.json"
    if not path.exists():
        pytest.skip("selection receipt absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert "source_identity" not in receipt
    assert receipt["decision"]["selected_development_sft_candidate"] == "Arm A checkpoint 431"


def test_development_result_artifacts_carry_no_new_scheme():
    """Four completed arms; none may gain a field after the fact."""
    artifacts = [
        "results/local_base_qwen_ablationdev_successor_s42_v3/base_validation_ablation-dev_parse-whole-output_completion1024_seed_42.json",
        "results/local_sft_armA_baseline_successor_s42/sft_validation_ablation-dev_parse-whole-output_completion1024_seed_42.json",
        "results/local_sft_armB_o1_nocollide_s42/sft_validation_ablation-dev_parse-whole-output_completion1024_seed_42.json",
        "results/local_eval_armA_ckpt150_matched/sft_validation_ablation-dev_parse-whole-output_completion1024_seed_42.json",
    ]
    seen = 0
    for relative in artifacts:
        path = ROOT / relative
        if not path.exists():
            continue
        seen += 1
        contract = json.loads(path.read_text(encoding="utf-8"))["run_contract"]
        assert "source_identity" not in contract, relative
        assert "runner_source_sha256" not in contract, relative
        assert "evaluator_source_sha256" in contract, relative
    if seen == 0:
        pytest.skip("no development artifacts present")
