"""v2.7 Phase 6b successor: meaningful-fuzzability decision, exit-evidence-based classification,
portable evidence and a clean-clone reproduction of the adapter-covered descriptive subset."""
from __future__ import annotations

import copy
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


probe = _load("v27_elig2", "scripts/v27_atheris_eligibility_v2_wsl.py")
freeze = _load("v27_freeze2", "scripts/v27_common_subset_freeze_v2.py")
RECEIPT = ROOT / "results/sft_root_cause_v27_common_subset_r4_v2.json"
BUNDLE = ROOT / "results/sft_root_cause_v27_phase6_evidence_bundle_v2.json"


def seed(built=True, invoked=True, consumed=8, outcome=("ok", ["int", 1]), receiver=None):
    return {"receiver_built": receiver, "arguments_built": built, "consumed_bytes": consumed,
            "target_invoked": invoked, "outcome": list(outcome)}


def runs(*seeds_per_run, normal=True):
    return [{"normal_exit": normal, "result": {"seeds": list(s)} if normal else None,
             "process": {"returncode": 0 if normal else 1}} for s in seeds_per_run]


def both(seeds, seeds2=None, normal=True):
    return {"buggy": runs(seeds, seeds2 or seeds, normal=normal),
            "fixed": runs(seeds, seeds2 or seeds, normal=normal)}


def test_fuzz_consuming_deterministic_target_is_fuzzable():
    d = probe.evaluate(both([seed(), seed(consumed=3)]))
    assert d["eligible_for_fuzzing"] and d["consumes_fuzz_input"] and d["deterministic"]


def test_zero_input_target_is_never_fuzzable():
    d = probe.evaluate(both([seed(consumed=0)]))
    assert not d["eligible_for_fuzzing"] and d["reason"] == "zero_input_invocable"
    assert d["failure_class"] == "zero_input"


def test_nondeterminism_is_detected_across_repetitions():
    d = probe.evaluate(both([seed(outcome=("ok", ["float", "1.0"]))],
                            [seed(outcome=("ok", ["float", "2.0"]))]))
    assert d["reason"] == "nondeterministic_invocation" and not d["eligible_for_fuzzing"]


def test_system_exit_is_a_structured_target_outcome():
    d = probe.evaluate(both([seed(consumed=0, outcome=("system_exit", "1"))]))
    assert d["reason"] == "target_terminates_interpreter:system_exit"
    assert d["failure_class"] == "target"


def test_process_failure_stays_a_harness_failure_never_reinterpreted():
    d = probe.evaluate(both([seed()], normal=False))
    assert d["reason"] == "invocation_process_failure" and d["failure_class"] == "harness"


@pytest.mark.parametrize("s, reason", [
    (seed(built=False, invoked=False, outcome=("build_error", "TypeError")),
     "receiver_or_arguments_never_built"),
    (seed(invoked=False), "target_never_entered"),
    (seed(receiver=False, built=False, invoked=False, outcome=("build_error", "ValueError")),
     "receiver_or_arguments_never_built"),
])
def test_build_errors_and_unreached_targets_never_qualify(s, reason):
    d = probe.evaluate(both([s]))
    assert not d["eligible_for_fuzzing"] and d["reason"] == reason


def test_one_fuzzable_seed_suffices_but_both_revisions_must_qualify():
    good = [seed(built=False, invoked=False, outcome=("build_error", "X")), seed()]
    assert probe.evaluate(both(good))["eligible_for_fuzzing"]
    mixed = {"buggy": runs(good, good), "fixed": runs([seed(invoked=False)], [seed(invoked=False)])}
    assert not probe.evaluate(mixed)["eligible_for_fuzzing"]


def test_scrub_removes_host_paths():
    text = "/root/oneiros_v27_prep/r4/x/views/buggy/a.py /tmp/oneiros_ath_work_1/c.py " \
           "/mnt/c/Users/Someone/x"
    assert "Someone" not in probe.scrub(text) and "/root/oneiros_v27_prep/r4" not in \
        probe.scrub(text)


def test_bundle_is_committed_and_redacted():
    sys.path.insert(0, str(ROOT / "scripts"))
    import receipt_sanitize
    assert BUNDLE.is_file()
    assert receipt_sanitize.check_file(BUNDLE) == []


def test_clean_clone_reproduction_of_the_committed_subset(tmp_path):
    out = tmp_path / "subset.json"
    assert freeze.main(["--out", str(out)]) == 0
    fresh = json.loads(out.read_text(encoding="utf-8"))
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for key in ("fuzzable_ids_sha256", "native_executable_ids_sha256", "sets", "waterfall",
                "exclusions", "composition"):
        assert fresh[key] == committed[key], key
    assert len(fresh["targets"]) == 34
    assert all(r["consumes_fuzz_input"] and r["deterministic"] and r["target_invoked"]
               for r in fresh["targets"] if r["eligible_for_fuzzing"])
    assert committed["supersedes"]["status"].startswith("PROVISIONAL")


def test_tampered_bundle_is_refused(tmp_path, monkeypatch):
    data = json.loads(BUNDLE.read_text(encoding="utf-8"))
    bad = copy.deepcopy(data)
    bad["targets"] = bad["targets"][:-1]
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(freeze, "BUNDLE", str(path))
    with pytest.raises(SystemExit, match="REFUSED"):
        freeze.main(["--out", str(tmp_path / "x.json")])


def test_statistics_wording_is_exact():
    s = json.loads(RECEIPT.read_text(encoding="utf-8"))["statistics"]
    assert s["native_executable"]["significance_possible_at_0.01"] is True
    assert s["native_executable"]["mde_pp_80pct_power_alpha_0.01"] is None
    assert s["fuzzable_subset"]["significance_possible_at_0.05"] is False
    assert "PLANNING SENSITIVITY" in s["assumption_note"]
    assert s["repository_clustering"]["native_executable_repositories"] == 14
