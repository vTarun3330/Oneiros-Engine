import sys
from types import SimpleNamespace

import pytest

from scripts import modal_train_failover as failover


def test_monthly_usage_accepts_current_lowercase_modal_billing_schema(monkeypatch):
    monkeypatch.setattr(
        failover.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='[{"cost":"1.25"},{"Cost":"2.50"}]',
            stderr="",
        ),
    )

    assert failover.monthly_usage("profile") == 3.75


def test_credit_failure_classifier_is_conservative():
    assert failover.is_credit_failure("Workspace budget exceeded", 1.0, 0.25)
    assert failover.is_credit_failure("unrelated error", 0.20, 0.25)
    assert not failover.is_credit_failure("CUDA out of memory", 5.0, 0.25)
    assert not failover.is_credit_failure("dataset fingerprint mismatch", 5.0, 0.25)


def test_dry_run_chooses_profile_with_enough_estimated_credit(monkeypatch, capsys):
    usage = {"primary": 25.3, "backup": 10.0}
    monkeypatch.setattr(failover, "monthly_usage", lambda profile: usage[profile])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modal_train_failover.py",
            "--profiles",
            "primary",
            "backup",
            "--estimated-cost",
            "8",
            "--dry-run",
            "--",
            "--run-name",
            "v3_full_sft_test",
            "--phase",
            "sft",
        ],
    )

    assert failover.main() == 0
    assert "selected first profile: backup" in capsys.readouterr().out


def test_fresh_is_forbidden_before_any_billing_call(monkeypatch):
    monkeypatch.setattr(
        failover,
        "monthly_usage",
        lambda profile: pytest.fail("billing should not be queried"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modal_train_failover.py",
            "--",
            "--run-name",
            "unsafe_reset",
            "--fresh",
        ],
    )

    with pytest.raises(ValueError, match="forbid --fresh"):
        failover.main()


def test_underpowered_sft_smoke_is_rejected_before_billing(monkeypatch):
    monkeypatch.setattr(
        failover,
        "monthly_usage",
        lambda profile: pytest.fail("billing should not be queried"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modal_train_failover.py",
            "--",
            "--run-name",
            "underpowered_smoke",
            "--phase",
            "sft",
            "--max-pairs",
            "64",
            "--sft-epochs",
            "1",
            "--sft-batch-size",
            "1",
            "--sft-max-real-repeats",
            "8",
        ],
    )

    with pytest.raises(ValueError, match="underpowered"):
        failover.main()


def test_bounded_sft_upper_bound_allows_a_larger_diagnostic_smoke():
    assert failover.validate_bounded_sft_monitor_capacity(
        max_pairs=800,
        epochs=1,
        batch_size=1,
        max_real_repeats=8,
    ) >= 100


def test_terminal_only_integration_allows_one_declared_monitor_checkpoint():
    assert failover.validate_bounded_sft_monitor_capacity(
        max_pairs=32,
        epochs=1,
        batch_size=1,
        max_real_repeats=8,
        minimum_checkpoints=1,
    ) == 48
    with pytest.raises(ValueError, match="underpowered"):
        failover.validate_bounded_sft_monitor_capacity(
            max_pairs=32,
            epochs=1,
            batch_size=1,
            max_real_repeats=8,
            minimum_checkpoints=2,
        )
