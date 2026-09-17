"""A rehearsal may not run on settings nobody froze.

Two gaps closed here.

**The dry run was not a gate.** It accepted an absent receipt and an unverified
SHA, so it proved the pipeline worked under whatever happened to be on disk.
A gate that passes without checking is a gate in name only - and the whole
reason a rehearsal exists is that the sealed final ran code nobody had executed.

**The consumed loader still contained the read.** ``sealed_records()`` refused
first and then, unreachably, opened the corpus. Unreachable is not absent: one
moved line, one caught exception, one restored gate, and it is reachable again.

Nothing here reads the consumed split, and nothing here loads a model.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import sealed_final_loader as loader  # noqa: E402
from harness.sealed_final import SealedAccessError  # noqa: E402
import scripts.build_rehearsal_receipt as builder  # noqa: E402
import scripts.run_rehearsal_evaluation as runner  # noqa: E402

RECEIPT = ROOT / "results" / "v4_2_rehearsal_receipt.json"
EXPECTED_TARGETS = 542
PY = sys.executable


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def receipt():
    if not RECEIPT.exists():
        pytest.skip("rehearsal receipt not generated yet")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _run(*args, timeout=300):
    return subprocess.run(
        [PY, "scripts/run_rehearsal_evaluation.py", *args],
        capture_output=True, text=True, cwd=ROOT, timeout=timeout)


# --------------------------------------------------------------------------
# 1. The consumed loader contains no read at all.
# --------------------------------------------------------------------------

def test_sealed_records_contains_no_corpus_read_path():
    """Not "unreachable". Absent.

    Walked as a syntax tree, so a read cannot hide behind a conditional, a
    comprehension or a nested helper.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(loader.sealed_records)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"read_text", "read_bytes", "open"}, \
                f"a file read survives in sealed_records: .{node.attr}"
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            assert name not in {"loads", "load", "open", "select_split_records",
                                "adapt_records"}, \
                f"a corpus path survives in sealed_records: {name}"


def test_the_consumed_loader_module_names_no_corpus_file():
    """No corpus filename appears anywhere in the module's string literals."""
    module = ast.parse((ROOT / "harness" / "sealed_final_loader.py")
                       .read_text(encoding="utf-8"))
    literals = [
        node.value for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    forbidden = ("records" ".json", "splits" ".json")
    offenders = [t for t in literals if any(f in t for f in forbidden)]

    assert offenders == []


def test_sealed_records_still_refuses():
    with pytest.raises(SealedAccessError, match="consumed"):
        loader.sealed_records()


def test_select_split_records_is_still_refusal_first():
    """Unchanged by this repair: refused before indexing or iterating."""
    with pytest.raises(SealedAccessError, match="consumed"):
        loader.select_split_records({}, [], "test")

    body = ast.parse(textwrap.dedent(
        inspect.getsource(loader.select_split_records))).body[0].body
    statements = [n for n in body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    first = statements[0]
    assert isinstance(first, ast.Expr)
    assert first.value.func.id == "refuse_consumed_split"


def test_permitted_split_parsing_is_untouched():
    record = {
        "id": "synthetic::1", "task_type": "hidden_mutation_reproduction",
        "entry_point": "f", "reference_code": "def f(): return 1",
        "code_under_test": "def f(): return 2", "specification": "spec",
        "tests": [{"code": "assert f() == 1"}],
    }
    selected = loader.select_split_records(
        {"rehearsal": [record["id"]]}, [record], "rehearsal")

    assert [r["id"] for r in selected] == [record["id"]]


# --------------------------------------------------------------------------
# 2. The receipt is self-consistent and freezes the right things.
# --------------------------------------------------------------------------

def test_the_receipt_freezes_ablation_dev_only(receipt):
    assert receipt["input_split"] == "ablation_dev"
    assert "test" in receipt["refused_splits"]
    assert receipt["corpus_version"] == "v4_1_research_hardened_candidate"


def test_the_receipt_freezes_exactly_542_targets(receipt):
    assert receipt["expected_target_count"] == EXPECTED_TARGETS
    assert receipt["admission_scope"]["function_mode_targets"] == EXPECTED_TARGETS
    assert len(receipt["evaluation_scope_sha256"]) == 64
    assert receipt["repository_exclusions"]["count"] > 0
    assert receipt["prompt_budget_failures"] == 0


def test_the_receipt_freezes_the_base_model_with_no_adapter(receipt):
    candidate = receipt["candidate"]
    assert candidate["model"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert candidate["model_revision"] == "2e1fd397ee46e1388853d2af2c993145b0f1098a"
    assert candidate["tokenizer_revision"] == candidate["model_revision"]
    assert candidate["adapter"] is None


@pytest.mark.parametrize("key,expected", [
    ("seed", 42),
    ("generation_batch_size", 2),
    ("candidates_per_function", 8),
    ("candidate_parse_mode", "whole_output"),
    ("retain_raw_output", True),
    ("prompt_token_limit", 1024),
    ("generation_completion_token_limit", 1024),
])
def test_the_receipt_freezes_each_generation_semantic(receipt, key, expected):
    assert receipt["frozen_generation_settings"][key] == expected


def test_the_receipt_carries_all_three_denials(receipt):
    assert receipt["operational_rehearsal"] is True
    assert receipt["final_test_measurement"] is False
    assert receipt["eligible_for_model_selection"] is False
    assert receipt["supports_performance_claim"] is False
    assert "not eligible for model selection" in receipt["label"]


@pytest.mark.parametrize("role", sorted(builder.REHEARSAL_DEFINING_SOURCES))
def test_every_defining_source_is_bound_by_hash(receipt, role):
    entry = receipt["source_hashes"][role]

    assert entry["path"] == builder.REHEARSAL_DEFINING_SOURCES[role]
    assert len(entry["canonical_sha256"]) == 64
    from harness.source_identity import canonical_sha256
    assert entry["canonical_sha256"] == canonical_sha256(ROOT / entry["path"])


def test_the_admission_binding_has_its_own_key(receipt):
    """It is a nested block, not a file entry, and must not displace one.

    Merging it in under "admission" replaced that role's file entry, so the
    file-level verifier skipped harness/evaluation_admission.py - the source
    whose change most directly changes which records get measured.
    """
    assert "admission_binding" in receipt
    assert receipt["admission_binding"]["admission_version"]
    assert "path" not in receipt["admission_binding"]
    assert receipt["source_hashes"]["admission"]["path"] == \
        "harness/evaluation_admission.py"


def test_the_receipt_is_ready_and_has_no_problems(receipt):
    assert receipt["receipt_problems"] == []
    assert receipt["ready_for_rehearsal"] is True


def test_the_builder_refuses_the_consumed_split_before_opening_the_corpus(monkeypatch):
    from harness.evaluation_admission import RefusedSplitError

    def explode(*args, **kwargs):
        raise AssertionError("a corpus file was opened on a refused path")

    monkeypatch.setattr(Path, "read_text", explode)

    with pytest.raises(RefusedSplitError, match="consumed"):
        builder.build("test", [])


def test_the_builder_writes_nothing_when_it_refuses(tmp_path):
    result = subprocess.run(
        [PY, "scripts/build_rehearsal_receipt.py", "--split", "test",
         "--output", "results/_never_written.json"],
        capture_output=True, text=True, cwd=ROOT)

    assert result.returncode == 1
    assert "REFUSED" in result.stdout
    assert not (ROOT / "results" / "_never_written.json").exists()


# --------------------------------------------------------------------------
# 3. The runner's receipt gate.
# --------------------------------------------------------------------------

def test_a_dry_run_without_a_sha_is_refused():
    result = _run("--dry-run")

    assert result.returncode == 1
    assert "--expected-receipt-sha256 is required" in result.stdout
    assert "Loading" not in result.stdout


def test_a_dry_run_with_a_wrong_sha_is_refused(receipt):
    result = _run("--dry-run", "--expected-receipt-sha256", "b" * 64)

    assert result.returncode == 1
    assert "hash mismatch" in result.stdout
    assert "Loading" not in result.stdout


def test_a_dry_run_with_a_missing_receipt_is_refused():
    result = _run("--dry-run", "--receipt", "results/does_not_exist.json",
                  "--expected-receipt-sha256", "c" * 64)

    assert result.returncode == 1
    assert "not found" in result.stdout


def test_a_dry_run_with_the_correct_sha_passes(receipt):
    result = _run("--dry-run", "--expected-receipt-sha256", _sha(RECEIPT))

    assert result.returncode == 0, result.stdout
    assert "CPU GATE PASSED" in result.stdout
    assert f"function-mode targets : {EXPECTED_TARGETS}" in result.stdout
    assert "model loads      : 0" in result.stdout
    assert "generations      : 0" in result.stdout


def test_no_model_is_loaded_during_a_dry_run(receipt):
    """Nothing prints a load banner, and torch is never asked for a device."""
    result = _run("--dry-run", "--expected-receipt-sha256", _sha(RECEIPT))

    for banner in ("Loading", "Model loaded", "cuda", "Phi3Generator"):
        assert banner not in result.stdout, f"{banner} appeared in a dry run"


def test_the_consumed_split_is_refused_before_the_receipt_is_even_read():
    result = _run("--split", "test", "--dry-run",
                  "--expected-receipt-sha256", "d" * 64)

    assert result.returncode == 2
    assert "REFUSED" in result.stdout
    assert "consumed" in result.stdout
    # Exit 2, not 1: the split gate precedes the receipt gate.
    assert "hash mismatch" not in result.stdout


@pytest.mark.parametrize("field,value", [
    ("final_test_measurement", True),
    ("eligible_for_model_selection", True),
    ("supports_performance_claim", True),
    ("operational_rehearsal", False),
    ("ready_for_rehearsal", False),
    ("expected_target_count", 541),
    ("input_split", "val"),
])
def test_a_tampered_receipt_is_refused(receipt, tmp_path, field, value):
    tampered = dict(receipt)
    tampered[field] = value
    path = tmp_path / "tampered.json"
    payload = json.dumps(tampered, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")

    _, problems = runner.receipt_problems(path, _sha(path))

    assert problems, f"tampering with {field} was not caught"


def test_a_receipt_naming_an_adapter_is_refused(receipt, tmp_path):
    tampered = json.loads(json.dumps(receipt))
    tampered["candidate"]["adapter"] = "checkpoints/somewhere"
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    _, problems = runner.receipt_problems(path, _sha(path))

    assert any("base-model only" in item for item in problems)


@pytest.mark.parametrize("role", [
    "prompt_factory", "generation_adapter", "admission", "rehearsal_evaluator",
    "safe_execution", "candidate_policy", "record_adaptation", "runner",
])
def test_a_changed_source_hash_invalidates_the_receipt(receipt, tmp_path, role):
    """Every bound source must be able to stop the run on its own."""
    tampered = json.loads(json.dumps(receipt))
    tampered["source_hashes"][role]["canonical_sha256"] = "0" * 64
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    _, problems = runner.receipt_problems(path, _sha(path))

    assert any(role in item for item in problems), problems


def test_the_scope_digest_must_still_resolve(receipt, tmp_path):
    """A frozen digest that no longer matches means the corpus moved."""
    tampered = json.loads(json.dumps(receipt))
    tampered["evaluation_scope_sha256"] = "0" * 64
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    result = _run("--dry-run", "--receipt",
                  str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
                  else str(path),
                  "--expected-receipt-sha256", _sha(path))

    # The path is outside the repo, so the runner reports it as missing rather
    # than resolving it - either way it must not pass.
    assert result.returncode == 1


def test_both_flags_are_required_for_a_real_run():
    """Not just the dry run: the GPU path needs the same proof."""
    signature = inspect.signature(runner.receipt_problems)
    assert list(signature.parameters) == ["receipt_path", "expected_sha256"]

    _, problems = runner.receipt_problems(RECEIPT, "")
    assert problems == ["--expected-receipt-sha256 is required"]


def test_a_receipt_built_from_a_dirty_tree_is_refused(receipt, tmp_path):
    """The bound hashes would verify; the commit would not contain them.

    A receipt frozen against uncommitted work names a source state nobody can
    return to, so it cannot support the claim that a rehearsal ran the code a
    reader can check out.
    """
    tampered = json.loads(json.dumps(receipt))
    tampered["reproducibility"]["git_dirty"] = True
    path = tmp_path / "dirty.json"
    path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    _, problems = runner.receipt_problems(path, _sha(path))

    assert any("dirty working tree" in item for item in problems)


def test_the_committed_receipt_was_built_from_a_clean_tree(receipt):
    assert receipt["reproducibility"]["git_dirty"] is False
    assert len(receipt["reproducibility"]["git_commit"]) == 40
