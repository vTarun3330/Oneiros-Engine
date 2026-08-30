from scripts.v4_1_ready import build_execution_queue, integration_commands


def _all_command_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_command_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_command_strings(child)


def test_gpu_ready_queue_never_emits_a_final_test_command():
    queue = build_execution_queue()
    commands = list(_all_command_strings(queue["stages"]))

    assert queue["safety"]["gpu_auto_launch"] is False
    assert queue["safety"]["final_test_command_emitted"] is False
    assert not any("--confirm-final-test" in command for command in commands)
    assert not any("--evaluation-split test" in command for command in commands)
    assert not any("--phase dpo_eval" in command for command in commands)


def test_integration_queue_has_distinct_fresh_and_resume_commands():
    commands = integration_commands()

    assert "--fresh" in commands["fresh"]
    assert "--fresh" not in commands["resume"]
    assert "--evaluation-split ablation_dev" in commands["fresh"]
    assert "--sft-min-monitor-checkpoints 1" in commands["fresh"]
    assert "--max-pairs 32" in commands["fresh"]
