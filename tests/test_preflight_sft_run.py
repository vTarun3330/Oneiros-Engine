from pathlib import Path

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


def test_preflight_and_production_accept_the_same_checkpoint_interval():
    """Preflight takes --checkpoint-steps, so production must take it too.

    Section 46 requires the planned monitor schedule to match runtime
    behaviour. If only preflight could set the interval, it could plan
    checkpoints at 18 and 36 while the trainer silently used its hardcoded
    50 and evaluated only the terminal step.
    """
    from scripts import preflight_sft_run, train_on_dataset

    preflight_source = Path(preflight_sft_run.__file__).read_text(encoding="utf-8")
    trainer_source = Path(train_on_dataset.__file__).read_text(encoding="utf-8")

    assert '"--checkpoint-steps"' in preflight_source
    assert '"--sft-checkpoint-steps"' in trainer_source
    # The trainer must actually consume the override, not just accept it.
    assert "SFT_CHECKPOINT_STEPS_OVERRIDE" in trainer_source
    assert trainer_source.count("SFT_CHECKPOINT_STEPS_OVERRIDE") >= 4


def test_checkpoint_interval_override_drives_the_planned_schedule():
    """A 200-pair run must be able to schedule more than a terminal check."""
    from engine.sft_trainer import plan_sft_optimizer_schedule

    default_interval = plan_sft_optimizer_schedule(
        unique_examples=570, num_epochs=1, batch_size=1,
        warmup_steps=25, checkpoint_steps=50,
    )
    tighter_interval = plan_sft_optimizer_schedule(
        unique_examples=570, num_epochs=1, batch_size=1,
        warmup_steps=25, checkpoint_steps=18,
    )

    # Same training work either way; only the monitor cadence changes.
    assert default_interval["planned_optimizer_steps"] == tighter_interval[
        "planned_optimizer_steps"
    ]
    # The default interval exceeds the run length, collapsing to terminal-only.
    assert default_interval["effective_checkpoint_steps"] == default_interval[
        "planned_optimizer_steps"
    ]
    assert tighter_interval["effective_checkpoint_steps"] == 18


def test_supervision_eligibility_uses_the_run_s_own_tokenizer_by_default():
    """Selecting data with one tokenizer while training with another is a bug.

    A base-model override reached the model-loading path but not the three
    tokenizer sites that decide supervision eligibility, so a Qwen run selected
    its training records under Phi-3 tokenization and disagreed with its own
    preflight (540 planned examples vs 553 actual).
    """
    from scripts import train_on_dataset as trainer

    original_name = trainer.BASE_MODEL_NAME_OVERRIDE
    original_selection = trainer.SFT_SELECTION_TOKENIZER_NAME_OVERRIDE
    try:
        trainer.BASE_MODEL_NAME_OVERRIDE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        trainer.SFT_SELECTION_TOKENIZER_NAME_OVERRIDE = None
        name, revision = trainer.resolved_selection_tokenizer_identity()
        assert name == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        assert revision == "main"

        # A controlled comparison may pin both arms to one tokenizer, and that
        # pinning has to show up in the training scope, not be silent.
        trainer.SFT_SELECTION_TOKENIZER_NAME_OVERRIDE = (
            "microsoft/Phi-3-mini-4k-instruct"
        )
        pinned_name, _ = trainer.resolved_selection_tokenizer_identity()
        assert pinned_name == "microsoft/Phi-3-mini-4k-instruct"
        scope = trainer.sft_training_scope(200, True, None)
        assert "selection_tokenizer=microsoft/Phi-3-mini-4k-instruct" in scope

        # The canonical case must not gain a field, so old fingerprints hold.
        trainer.BASE_MODEL_NAME_OVERRIDE = None
        trainer.SFT_SELECTION_TOKENIZER_NAME_OVERRIDE = None
        assert "selection_tokenizer=" not in trainer.sft_training_scope(200, True, None)
    finally:
        trainer.BASE_MODEL_NAME_OVERRIDE = original_name
        trainer.SFT_SELECTION_TOKENIZER_NAME_OVERRIDE = original_selection


def test_no_tokenizer_site_hardcodes_the_canonical_model():
    """Every tokenizer construction must go through the resolver."""
    from scripts import train_on_dataset

    source = Path(train_on_dataset.__file__).read_text(encoding="utf-8")
    from_pretrained_blocks = source.count("AutoTokenizer.from_pretrained(")
    resolver_uses = source.count("resolved_selection_tokenizer_identity()")
    assert from_pretrained_blocks >= 3
    # One resolver call per tokenizer site, plus the definition and the scope use.
    assert resolver_uses >= from_pretrained_blocks
