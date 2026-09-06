"""Pin the regularisation overrides for the memorisation arm.

The measured obstacle is a 19-point gap - 82.9% on train against 63.7% on
locked validation - which says the model fits its training targets rather than
acquiring a transferable skill. Testing that directly needs LoRA dropout and
weight decay varied while everything else is held fixed.

The property that matters most here is the NEGATIVE one: a run that does not
pass these flags must be identical to every result already reported, not
merely similar. An override that silently changed the default would
retroactively invalidate the base, historical, full-density and relearning
arms, all of which are paired against each other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import model_config
from engine.sft_trainer import OneirosSFTTrainer


def _trainer(**kwargs) -> OneirosSFTTrainer:
    return OneirosSFTTrainer(output_dir=ROOT / "checkpoints" / "_unit_test", **kwargs)


def test_omitting_the_flags_reproduces_the_frozen_configuration():
    trainer = _trainer()
    assert trainer.lora_dropout == model_config.lora_dropout
    assert trainer.lora_dropout == 0.05
    assert trainer.weight_decay == 0.0


def test_lora_dropout_override_is_honoured():
    assert _trainer(lora_dropout=0.15).lora_dropout == 0.15


def test_weight_decay_override_is_honoured():
    assert _trainer(weight_decay=0.01).weight_decay == 0.01


def test_zero_is_an_override_not_a_missing_value():
    """0.0 must not be confused with 'unset'.

    ``lora_dropout or default`` would silently restore 0.05 when a run asked
    for no dropout at all, which is the one value a regularisation sweep most
    obviously wants at its lower end.
    """
    trainer = _trainer(lora_dropout=0.0)
    assert trainer.lora_dropout == 0.0
    assert trainer.lora_dropout != model_config.lora_dropout


def test_the_two_knobs_are_independent():
    trainer = _trainer(lora_dropout=0.2)
    assert trainer.lora_dropout == 0.2
    assert trainer.weight_decay == 0.0, "setting dropout must not move decay"

    trainer = _trainer(weight_decay=0.05)
    assert trainer.weight_decay == 0.05
    assert trainer.lora_dropout == model_config.lora_dropout


@pytest.mark.parametrize("value", [0.0, 0.05, 0.1, 0.15, 0.3])
def test_overrides_are_coerced_to_float(value):
    trainer = _trainer(lora_dropout=value, weight_decay=value)
    assert isinstance(trainer.lora_dropout, float)
    assert isinstance(trainer.weight_decay, float)


def test_the_driver_exposes_both_flags_and_defaults_them_to_none():
    """A default of None, not a number, is what preserves the frozen run.

    argparse defaulting to 0.05 would look equivalent but would stamp an
    explicit value into every run fingerprint, making old and new runs
    compare unequal for no real difference.
    """
    import scripts.train_on_dataset as driver

    assert driver.LORA_DROPOUT_OVERRIDE is None
    assert driver.WEIGHT_DECAY_OVERRIDE is None

    source = (ROOT / "scripts" / "train_on_dataset.py").read_text(encoding="utf-8")
    assert '"--lora-dropout", type=float, default=None' in source
    assert '"--weight-decay", type=float, default=None' in source
    assert "lora_dropout=LORA_DROPOUT_OVERRIDE" in source
    assert "weight_decay=WEIGHT_DECAY_OVERRIDE" in source
