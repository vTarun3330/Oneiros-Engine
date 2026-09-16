"""The two receipt defects a read-only audit of v5 found, pinned.

Neither defect was in the measurement logic. Both were in the artifact that
describes it - which is the part an operator actually reads before spending the
one authorization that cannot be taken back.

**Defect 1.** ``scripts/preflight_sealed_final.py`` hardcoded
``results/v4_2_sealed_final_executable_receipt_v4.json`` into every receipt's
``exact_command``. The schema version moved to v5; that literal did not. So the
v5 receipt instructed its operator to present the v4 file, which the entrypoint
refuses by schema version. It failed closed, but the one-time instruction was
still wrong, and the only test covering ``exact_command`` asserted index 1 and
never looked at index 3.

**Defect 2.** The frozen bundle carried ``git rev-parse HEAD`` - a 40-hex SHA-1
commit id - in a field named ``adapter_source_tree_sha256``. The name asserted a
hash function and an input the value never had.

These tests exist because both defects survived four receipt generations and a
full suite. What they check is not that the current values are right, but that
the shapes that made them wrong cannot come back.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECEIPT = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v6.json"
PREFLIGHT = ROOT / "scripts" / "preflight_sealed_final.py"
ENTRYPOINT = ROOT / "scripts" / "run_sealed_final_test.py"

#: Every receipt generation that must never again be presentable.
SUPERSEDED = (
    "oneiros_sealed_final_readiness_v1",
    "oneiros_sealed_final_readiness_v2",
    "oneiros_sealed_final_readiness_v3",
    "oneiros_sealed_final_readiness_v4",
    "oneiros_sealed_final_readiness_v5",
)

GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preflight():
    return _load("_preflight_sealed_final_v6", PREFLIGHT)


@pytest.fixture(scope="module")
def entry():
    return _load("_run_sealed_final_test_v6", ENTRYPOINT)


@pytest.fixture(scope="module")
def receipt():
    if not RECEIPT.exists():
        pytest.skip("v6 receipt not generated yet")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def _walk(node, path=""):
    """Yield (dotted_path, value) for every scalar in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")
    else:
        yield path, node


# --------------------------------------------------------------------------
# Defect 1: exact_command must name the receipt it is actually about.
# --------------------------------------------------------------------------

def test_a_receipt_written_anywhere_names_that_exact_path(preflight, tmp_path):
    """The defect in one line: the command named a file it was not.

    Generating to an arbitrary path and reading the command back is the only
    check that cannot pass by coincidence. A hardcoded literal fails it for
    every path except the one that happens to be hardcoded.
    """
    destination = "results/generated/at_an_arbitrary_path_receipt.json"
    built = preflight.collect([], destination)

    command = built["exact_command"]
    assert command[command.index("--executable-receipt") + 1] == destination


@pytest.mark.parametrize("destination", [
    "results/v4_2_sealed_final_executable_receipt_v6.json",
    "results/elsewhere/another_name.json",
    "scratch/deeply/nested/receipt.json",
])
def test_the_command_follows_the_output_path_wherever_it_goes(preflight, destination):
    built = preflight.collect([], destination)
    command = built["exact_command"]

    assert command[command.index("--executable-receipt") + 1] == destination


def test_the_command_never_carries_a_hardcoded_superseded_version(preflight):
    """v4 and v5 are refused by schema. A command naming one is dead on arrival."""
    for destination in ("results/a.json", "results/b.json"):
        rendered = " ".join(preflight.collect([], destination)["exact_command"])
        for stale in ("_receipt_v4.json", "_receipt_v5.json", "_receipt_v3.json",
                      "v4_2_sealed_final_readiness_receipt.json"):
            assert stale not in rendered, f"{stale} reappeared in exact_command"


def test_the_preflight_source_holds_no_hardcoded_receipt_path():
    """Belt and braces: the literal that caused this must not exist at all.

    A future edit could reintroduce the hardcode somewhere the parametrised
    tests above do not reach, so the source itself is checked.
    """
    source = PREFLIGHT.read_text(encoding="utf-8")
    for stale in ("v4_2_sealed_final_executable_receipt_v3.json",
                  "v4_2_sealed_final_executable_receipt_v4.json",
                  "v4_2_sealed_final_executable_receipt_v5.json"):
        # The supersedes list legitimately names these as evidence; the command
        # builder must not. Check the function that builds it, not the file.
        builder = source.split("exact_command = [")[1].split("]")[0]
        assert stale not in builder


def test_the_generated_receipt_names_itself(receipt):
    command = receipt["exact_command"]
    named = command[command.index("--executable-receipt") + 1]

    assert named == "results/v4_2_sealed_final_executable_receipt_v6.json"
    assert Path(named).name == RECEIPT.name


def test_the_command_carries_every_piece_authorization_needs(receipt):
    command = receipt["exact_command"]

    assert command[1] == "scripts/run_sealed_final_test.py"
    assert "--executable-receipt" in command
    assert "--expected-receipt-sha256" in command
    assert "--authorization-token" in command
    assert "--i-understand-this-is-one-time-and-irreversible" in command

    sha_placeholder = command[command.index("--expected-receipt-sha256") + 1]
    token_placeholder = command[command.index("--authorization-token") + 1]
    # Placeholders, not values. A receipt that contained a real token would be
    # a token issued by a preflight, which is exactly what must not happen.
    assert sha_placeholder.startswith("<") and sha_placeholder.endswith(">")
    assert token_placeholder.startswith("<") and token_placeholder.endswith(">")
    assert receipt["authorization_token_issued"] is False


# --------------------------------------------------------------------------
# Defect 2: a Git commit id must not be labelled as a SHA-256.
# --------------------------------------------------------------------------

def test_no_sha256_field_anywhere_holds_a_git_commit(receipt):
    """The general form of the defect, across the whole receipt.

    Checking only the one field that was wrong would leave the next one free to
    be wrong the same way.
    """
    head = _git("rev-parse", "HEAD")
    offenders = []
    for path, value in _walk(receipt):
        if not path.split(".")[-1].split("[")[0].endswith("_sha256"):
            continue
        if not isinstance(value, str):
            continue
        if value == head or (GIT_COMMIT_RE.match(value) and len(value) == 40):
            offenders.append((path, value))

    assert offenders == [], f"Git commit ids labelled as SHA-256: {offenders}"


def test_every_sha256_field_is_a_sha256(receipt):
    bad = []
    for path, value in _walk(receipt):
        if not path.split(".")[-1].split("[")[0].endswith("_sha256"):
            continue
        if not isinstance(value, str) or value.startswith("<"):
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            bad.append((path, value))

    assert bad == [], f"fields named _sha256 that are not 64 hex chars: {bad}"


def test_the_git_commit_field_is_named_for_what_it_holds(receipt):
    fields = receipt["frozen_bundle"]["fields"]

    assert "adapter_source_tree_sha256" not in fields
    assert "candidate_source_tree_git_commit" in fields
    assert GIT_COMMIT_RE.match(fields["candidate_source_tree_git_commit"])


def test_the_git_commit_field_equals_the_commit_the_preflight_ran_at(preflight):
    built = preflight.collect([], "results/probe.json")
    fields = built["frozen_bundle"]["fields"]

    assert fields["candidate_source_tree_git_commit"] == _git("rev-parse", "HEAD")
    assert built["reproducibility"]["git_commit"] == _git("rev-parse", "HEAD")


def test_the_renamed_field_is_still_required_to_freeze_a_bundle():
    """Renaming must not quietly drop the field from the frozen contract."""
    from harness.sealed_final import REQUIRED_BUNDLE_FIELDS, FinalBundle

    assert "candidate_source_tree_git_commit" in REQUIRED_BUNDLE_FIELDS
    assert "adapter_source_tree_sha256" not in REQUIRED_BUNDLE_FIELDS

    fields = dict(json.loads(RECEIPT.read_text(encoding="utf-8"))["frozen_bundle"]["fields"]) \
        if RECEIPT.exists() else None
    if fields is None:
        pytest.skip("v6 receipt not generated yet")
    del fields["candidate_source_tree_git_commit"]
    assert "candidate_source_tree_git_commit" in FinalBundle(fields).missing_fields()


# --------------------------------------------------------------------------
# Schema: v1 through v5 are refused, v6 is required.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("superseded", SUPERSEDED)
def test_every_earlier_schema_is_refused(entry, superseded, tmp_path, receipt):
    stale = dict(receipt)
    stale["schema_version"] = superseded
    path = tmp_path / "stale.json"
    payload = json.dumps(stale, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")

    _, problems = entry.receipt_problems(path, entry.sha256_file(path))

    assert problems, f"{superseded} was not refused"
    assert any(superseded in item for item in problems)


def test_v6_is_the_only_accepted_schema(entry, receipt):
    assert entry.REQUIRED_SCHEMA_VERSION == "oneiros_sealed_final_readiness_v6"
    assert set(entry.REFUSED_SCHEMA_VERSIONS) == set(SUPERSEDED)
    assert receipt["schema_version"] == "oneiros_sealed_final_readiness_v6"

    _, problems = entry.receipt_problems(RECEIPT, entry.sha256_file(RECEIPT))
    assert problems == []


def test_an_unknown_future_schema_is_also_refused(entry, tmp_path, receipt):
    """Refusal must not be a blocklist with a hole in it."""
    odd = dict(receipt)
    odd["schema_version"] = "oneiros_sealed_final_readiness_v99"
    path = tmp_path / "future.json"
    path.write_text(json.dumps(odd, indent=2) + "\n", encoding="utf-8")

    _, problems = entry.receipt_problems(path, entry.sha256_file(path))

    assert any("v99" in item for item in problems)


def test_the_superseded_v5_receipt_is_refused_as_it_sits_on_disk(entry):
    """Not a synthetic mutation: the real committed file, presented as-is."""
    v5 = ROOT / "results" / "v4_2_sealed_final_executable_receipt_v5.json"
    if not v5.exists():
        pytest.skip("v5 receipt absent")

    _, problems = entry.receipt_problems(v5, entry.sha256_file(v5))

    assert any("oneiros_sealed_final_readiness_v5" in item for item in problems)


# --------------------------------------------------------------------------
# No receipt may be selected by omission for a real run.
# --------------------------------------------------------------------------

def test_a_real_run_must_name_its_receipt(entry, capsys):
    """A default receipt is a receipt nobody chose.

    The stale file a version bump leaves behind is precisely the one a default
    would have selected, so an irreversible run has to say which it means.
    """
    code = entry.main([
        "--expected-receipt-sha256", "0" * 64,
        "--authorization-token", "irrelevant-nothing-is-presented",
        "--i-understand-this-is-one-time-and-irreversible",
    ])

    assert code == 2
    output = capsys.readouterr().out
    assert "No token was presented" in output
    assert "missing: --executable-receipt" in output


def test_the_fallback_used_by_check_only_points_at_v6(entry):
    assert entry.EXECUTABLE_RECEIPT == \
        "results/v4_2_sealed_final_executable_receipt_v6.json"


# --------------------------------------------------------------------------
# The command in the receipt, with the real SHA, must actually pass.
# --------------------------------------------------------------------------

def test_the_receipts_own_command_passes_check_only(receipt):
    """The end-to-end form of defect 1.

    Take the command the receipt prints, substitute the real SHA for its
    placeholder, drop the token and acknowledgement, add --check-only, and run
    it. Under v5 this would have loaded the v4 receipt and been refused.

    This runs the full pre-authorization path including the model smoke, so it
    is skipped unless the GPU environment is present. Nothing is authorized:
    --check-only returns before the guard is constructed.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU available for the pre-authorization smoke")

    command = list(receipt["exact_command"])
    sha_index = command.index("--expected-receipt-sha256") + 1
    command[sha_index] = __import__("hashlib").sha256(
        RECEIPT.read_bytes()).hexdigest()

    token_index = command.index("--authorization-token")
    del command[token_index:token_index + 2]
    command.remove("--i-understand-this-is-one-time-and-irreversible")
    command.append("--check-only")

    interpreter = ROOT / command[0]
    result = subprocess.run(
        [str(interpreter) if interpreter.exists() else sys.executable, *command[1:]],
        capture_output=True, text=True, cwd=ROOT, timeout=1800)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRE-AUTHORIZATION CHECKS PASSED" in result.stdout
    assert "No token was presented" in result.stdout
    assert not (ROOT / "results" / "sealed_final_state.json").exists()
    assert not (ROOT / "results" / "sealed_final_run_state.json").exists()


def test_check_only_on_the_receipt_writes_no_sealed_state(receipt):
    """Structural proof, without needing a GPU: the guard is unreachable.

    --check-only returns at its own branch, which sits above the phase that
    constructs SealedFinalGuard. That ordering is what makes the claim true,
    so the ordering is what is asserted.
    """
    source = ENTRYPOINT.read_text(encoding="utf-8")
    check_only_return = source.index("if args.check_only:")
    guard_built = source.index("guard = SealedFinalGuard(")
    token_presented = source.index("guard.open_sealed_split(")

    assert check_only_return < guard_built < token_presented


# --------------------------------------------------------------------------
# Scope B is unchanged by this repair.
# --------------------------------------------------------------------------

def test_scope_b_survives_the_v6_repair(receipt):
    scope = receipt["baseline_scope"]

    assert scope["scope"] == "oneiros_immutable_base_only_no_comparative_baseline"
    assert scope["baselines_bundled"] == []
    assert scope["comparative_claims_supported"] is False
    assert receipt["final_candidate"]["adapter"] is None

    blob = json.dumps(receipt)
    for reintroduced in ("v4_2_atheris_tasks_val.manifest.json",
                         "v4_2_baseline_bundle_val.json"):
        assert reintroduced not in blob


def test_the_v6_receipt_opened_nothing(receipt):
    assert receipt["sealed_split_accessed"] is False
    assert receipt["sealed_records_read"] == 0
    assert receipt["sealed_ids_enumerated"] == 0
    assert receipt["sealed_payload_hashed"] is False
    assert receipt["authorization_token_issued"] is False
