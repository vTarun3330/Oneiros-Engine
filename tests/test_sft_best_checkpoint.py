from types import SimpleNamespace

from engine.sft_trainer import SFTCheckpointMonitorCallback


class _SavingObject:
    def save_pretrained(self, path):
        from pathlib import Path

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "saved.txt").write_text("saved\n", encoding="utf-8")


def test_improved_checkpoint_is_persisted_outside_rolling_trainer_checkpoints(tmp_path):
    def monitor(step, model, tokenizer):
        return {
            "checkpoint_step": step,
            "function_validation_killed": 201,
            "function_kill_rate": 0.402,
            "improved": True,
            "should_stop": False,
        }

    tokenizer = _SavingObject()
    callback = SFTCheckpointMonitorCallback(monitor, tokenizer, planned_steps=500)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path / "sft_tmp")),
        SimpleNamespace(global_step=100),
        control,
        model=_SavingObject(),
    )

    best = tmp_path / "sft_validation_best" / "checkpoint-100"
    assert callback.best_adapter_path == str(best)
    assert (best / "saved.txt").exists()
    assert (best / "validation_metrics.json").exists()
    assert control.should_training_stop is False


def test_resumed_monitor_keeps_previously_preserved_best_adapter():
    def monitor(step, model, tokenizer):
        return {
            "checkpoint_step": step,
            "function_validation_killed": 275,
            "function_kill_rate": 0.55,
            "improved": False,
            "should_stop": False,
        }

    monitor.initial_best_adapter_path = "preserved/checkpoint-100"
    monitor.initial_best_metrics = {
        "checkpoint_step": 100,
        "function_validation_killed": 280,
        "function_kill_rate": 0.56,
    }
    callback = SFTCheckpointMonitorCallback(monitor, _SavingObject(), planned_steps=500)

    assert callback.best_adapter_path == "preserved/checkpoint-100"
    assert callback.best_metrics["function_validation_killed"] == 280


def test_resume_backfills_a_saved_checkpoint_with_missing_validation(tmp_path):
    calls = []

    def monitor(step, model, tokenizer):
        calls.append(step)
        return {
            "checkpoint_step": step,
            "function_validation_killed": 279,
            "function_kill_rate": 0.558,
            "improved": False,
            "should_stop": False,
        }

    monitor.completed_checkpoint_metrics = {
        100: {
            "checkpoint_step": 100,
            "function_validation_killed": 280,
            "function_kill_rate": 0.56,
            "improved": True,
            "should_stop": False,
        }
    }
    callback = SFTCheckpointMonitorCallback(monitor, _SavingObject(), planned_steps=500)
    control = SimpleNamespace(should_training_stop=False)

    callback.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path / "sft_tmp")),
        SimpleNamespace(global_step=200),
        control,
        model=_SavingObject(),
    )

    assert calls == [200]
    assert [item["checkpoint_step"] for item in callback.history] == [100, 200]
    assert control.should_training_stop is False


def test_resume_reuses_completed_checkpoint_validation_without_rerunning(tmp_path):
    calls = []

    def monitor(step, model, tokenizer):
        calls.append(step)
        raise AssertionError("completed checkpoint validation must be reused")

    monitor.completed_checkpoint_metrics = {
        200: {
            "checkpoint_step": 200,
            "function_validation_killed": 275,
            "function_kill_rate": 0.55,
            "improved": False,
            "should_stop": False,
        }
    }
    callback = SFTCheckpointMonitorCallback(monitor, _SavingObject(), planned_steps=500)
    control = SimpleNamespace(should_training_stop=False)

    callback.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path / "sft_tmp")),
        SimpleNamespace(global_step=200),
        control,
        model=_SavingObject(),
    )

    assert calls == []
    assert [item["checkpoint_step"] for item in callback.history] == [200]


def test_terminal_step_not_divisible_by_save_interval_is_evaluated_once(tmp_path):
    calls = []

    def monitor(step, model, tokenizer):
        calls.append(step)
        return {"checkpoint_step": step, "improved": False, "should_stop": False}

    callback = SFTCheckpointMonitorCallback(monitor, _SavingObject(), planned_steps=143)
    args = SimpleNamespace(output_dir=str(tmp_path / "sft_tmp"))
    control = SimpleNamespace(should_training_stop=False)
    callback.on_save(args, SimpleNamespace(global_step=50), control, model=_SavingObject())
    callback.on_save(args, SimpleNamespace(global_step=100), control, model=_SavingObject())
    callback.on_train_end(args, SimpleNamespace(global_step=143), control, model=_SavingObject())
    callback.on_train_end(args, SimpleNamespace(global_step=143), control, model=_SavingObject())

    assert calls == [50, 100, 143]
    assert [item["checkpoint_step"] for item in callback.history] == [50, 100, 143]


def test_terminal_step_divisible_by_save_interval_is_not_duplicated(tmp_path):
    calls = []

    def monitor(step, model, tokenizer):
        calls.append(step)
        return {"checkpoint_step": step, "improved": False, "should_stop": False}

    callback = SFTCheckpointMonitorCallback(monitor, _SavingObject(), planned_steps=100)
    args = SimpleNamespace(output_dir=str(tmp_path / "sft_tmp"))
    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=100)
    callback.on_save(args, state, control, model=_SavingObject())
    callback.on_train_end(args, state, control, model=_SavingObject())

    assert calls == [100]


def test_resumed_terminal_checkpoint_is_reused(tmp_path):
    calls = []

    def monitor(step, model, tokenizer):
        calls.append(step)
        return {"checkpoint_step": step, "improved": False, "should_stop": False}

    monitor.completed_checkpoint_metrics = {
        143: {"checkpoint_step": 143, "improved": True, "should_stop": False}
    }
    callback = SFTCheckpointMonitorCallback(monitor, _SavingObject(), planned_steps=143)
    args = SimpleNamespace(output_dir=str(tmp_path / "sft_tmp"))
    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=143)
    callback.on_train_begin(args, state, control, model=_SavingObject())
    callback.on_train_end(args, state, control, model=_SavingObject())

    assert calls == []
    assert [item["checkpoint_step"] for item in callback.history] == [143]


def test_terminal_checkpoint_can_become_best_adapter(tmp_path):
    def monitor(step, model, tokenizer):
        return {
            "checkpoint_step": step,
            "function_kill_rate": 0.61,
            "improved": step == 143,
            "should_stop": False,
        }

    callback = SFTCheckpointMonitorCallback(monitor, _SavingObject(), planned_steps=143)
    args = SimpleNamespace(output_dir=str(tmp_path / "sft_tmp"))
    control = SimpleNamespace(should_training_stop=False)
    callback.on_train_end(
        args, SimpleNamespace(global_step=143), control, model=_SavingObject()
    )

    assert callback.best_metrics["checkpoint_step"] == 143
    assert callback.best_adapter_path.endswith("checkpoint-143")
