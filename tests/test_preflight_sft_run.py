from scripts.preflight_sft_run import preflight_gates_pass
from scripts.train_on_dataset import supervision_exclusion_summary


def test_preflight_readiness_requires_every_declared_gate():
    passing = {
        "corpus_verified": True,
        "zero_sequence_overflows": True,
        "real_fraction_target_reached": True,
    }
    assert preflight_gates_pass(passing)

    failing = {**passing, "real_fraction_target_reached": False}
    assert not preflight_gates_pass(failing)


def test_preflight_readiness_fails_closed_for_missing_gate_set():
    assert not preflight_gates_pass({})


def test_verified_supervision_exclusion_summary_is_ordered_and_deterministic():
    first = supervision_exclusion_summary(["record-b", "record-a", "record-b"])
    second = supervision_exclusion_summary(["record-b", "record-a", "record-b"])

    assert first == second
    assert first["count"] == 2
    assert first["record_ids"] == ["record-b", "record-a"]
    assert len(first["record_ids_sha256"]) == 64
    assert not first["canonical_records_modified"]
