"""Phase 4 objective masking through the PRODUCTION SFT data path.

Real pinned Qwen2.5-Coder-1.5B tokenizer (local files only), the real
``OneirosSFTTrainer.prepare_dataset``, the real ``CompletionOnlyDataCollator`` and a
real TRL ``SFTTrainer`` dataloader built around a tiny randomly initialised Qwen2
model on CPU (no weights, no GPU, no generation).  All records are synthetic.
"""
from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from engine.sft_trainer import (   # noqa: E402
    IGNORE_INDEX, CompletionOnlyDataCollator, ObjectiveAlignmentError, SFTDataPoint,
    value_token_labels,
)
from harness.objective_masking import (   # noqa: E402
    build_example, completion_text, dataset_trainer, make_datapoint, prepare_arm,
    validate_manifest, verify_pair,
)

REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
RECORD = {"id": "toy::1", "group_id": "toy-group", "entry_point": "f",
          # the reference must not be a substring of the buggy code (probe leak guard)
          "reference_code": "def f(x):\n    return x * 1\n",
          "code_under_test": "def f(x):\n    return x + 1\n",
          "specification": ("Return x unchanged. Notes mention 42, -7, None, True, "
                            "[1, 2, 3] and 'naïve' so values also occur in the prompt.")}
VALUES = [
    ("f(3)", "42"), ("f(3)", "-7"), ("f(3)", "3.25"), ("f(3)", "True"), ("f(3)", "None"),
    ("f('a')", "'it\\'s \\n ok'"), ("f('a')", "'naïve – 東京'"), ("f('a')", "''"),
    ("f([1])", "[1, 2, 3]"), ("f((1,))", "(1, 'b')"), ("f({})", "{'k': [1, {'z': None}]}"),
    ("f(9)", "[[1, 2], [3, [4, 5]], (6, 7)]"),
    ("f(123456789)", "123456789123456789"),     # multi-token value
    ("g(42, '42')", "42"),                        # the value also appears inside the call
    ("h([1, 2, 3])", "[1, 2, 3]"),                # the call argument equals the value
]


@pytest.fixture(scope="module")
def tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", revision=REVISION, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --- per-example: only the labels differ ------------------------------------------------------

@pytest.mark.parametrize("call, value", VALUES)
def test_production_arms_differ_only_in_labels(tokenizer, call, value):
    ex = build_example(tokenizer, RECORD, call, value)
    assert ex["input_ids"] == ex["treatment_input_ids"]
    assert ex["attention_mask"] == ex["treatment_attention_mask"] == [1] * len(ex["input_ids"])
    assert ex["completion_start"] == ex["treatment_completion_start"]
    control, treatment = ex["labels"]["control"], ex["labels"]["treatment"]
    start = ex["completion_start"]
    assert control[:start] == treatment[:start] == [IGNORE_INDEX] * start
    assert control[start:] == ex["input_ids"][start:]                  # every completion token
    assert tokenizer.decode(ex["input_ids"][start:]) == \
        ex["completion_text"] + tokenizer.eos_token
    supervised = [i for i, t in enumerate(treatment) if t != IGNORE_INDEX]
    decoded = tokenizer.decode([treatment[i] for i in supervised])
    assert supervised[-1] == len(ex["input_ids"]) - 1 and \
        treatment[-1] == tokenizer.eos_token_id, f"treatment supervised {decoded!r}"
    assert decoded.strip() == value + tokenizer.eos_token or \
        decoded.strip() == (value + tokenizer.eos_token).strip(), \
        f"treatment supervises {decoded!r}, expected the value {value!r} and EOS"
    assert supervised == list(range(supervised[0], len(ex["input_ids"]))), decoded
    assert control != treatment


def test_prompt_and_completion_tokenization_match_the_trainer_exactly(tokenizer):
    from engine.test_generation_prompt import format_chat_prompt
    dp = make_datapoint(RECORD, "f(3)", "42")
    ex = build_example(tokenizer, RECORD, "f(3)", "42")
    prompt_ids = tokenizer(format_chat_prompt(tokenizer, dp.prompt),
                           add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(dp.completion.strip() + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]
    assert ex["input_ids"] == list(prompt_ids) + list(completion_ids)


# --- dataset level: order, hashes, legacy compatibility ----------------------------------------

def _datapoints():
    return [make_datapoint(RECORD, call, value) for call, value in VALUES]


def test_arm_datasets_share_order_and_input_hashes_but_not_label_hashes(tokenizer):
    control, c_stats = prepare_arm(tokenizer, _datapoints(), "control")
    treatment, t_stats = prepare_arm(tokenizer, _datapoints(), "treatment")
    assert len(control) == len(treatment) == len(VALUES)
    for key in ("input_ids_sha256", "attention_mask_sha256"):
        assert c_stats[key] == t_stats[key]
    assert c_stats["labels_sha256"] != t_stats["labels_sha256"]
    assert c_stats["objective_mode"] == "full_completion"
    assert t_stats["objective_mode"] == "value_only"
    assert t_stats["supervised_tokens"] < c_stats["supervised_tokens"]
    for c, t in zip(control, treatment):
        assert c["input_ids"] == t["input_ids"] and c["completion_start"] == t["completion_start"]


def test_legacy_path_is_unchanged_and_explicit_full_completion_matches_it(tokenizer):
    dps = _datapoints()
    legacy_trainer = dataset_trainer(tokenizer, None)
    legacy = legacy_trainer.prepare_dataset(dps)
    assert "labels" not in legacy.column_names                       # legacy columns only
    assert "objective_mode" not in legacy_trainer.dataset_stats
    control, _ = prepare_arm(tokenizer, dps, "control")
    collator = CompletionOnlyDataCollator(tokenizer)
    rows = list(range(len(dps)))
    legacy_batch = collator([legacy[i] for i in rows])
    control_batch = collator([control[i] for i in rows])
    assert legacy_batch["input_ids"].tolist() == control_batch["input_ids"].tolist()
    assert legacy_batch["attention_mask"].tolist() == control_batch["attention_mask"].tolist()
    assert legacy_batch["labels"].tolist() == control_batch["labels"].tolist()


def test_the_production_collator_keeps_treatment_labels_through_padding(tokenizer):
    treatment, _ = prepare_arm(tokenizer, _datapoints(), "treatment")
    batch = CompletionOnlyDataCollator(tokenizer)([treatment[i] for i in range(len(VALUES))])
    for row in range(len(VALUES)):
        real = batch["attention_mask"][row] == 1
        assert batch["labels"][row][real].tolist() == list(treatment[row]["labels"])
        assert set(batch["labels"][row][~real].tolist()) <= {IGNORE_INDEX}


def test_a_mixed_batch_is_refused(tokenizer):
    control, _ = prepare_arm(tokenizer, _datapoints()[:1], "control")
    legacy = dataset_trainer(tokenizer, None).prepare_dataset(_datapoints()[:1])
    with pytest.raises(ValueError, match="mixes"):
        CompletionOnlyDataCollator(tokenizer)([control[0], legacy[0]])


def test_the_trainer_facing_batch_retains_the_masks(tokenizer, tmp_path):
    torch = pytest.importorskip("torch")
    trl = pytest.importorskip("trl")
    config = transformers.Qwen2Config(vocab_size=len(tokenizer), hidden_size=8,
                                      intermediate_size=16, num_hidden_layers=1,
                                      num_attention_heads=1, num_key_value_heads=1)
    model = transformers.Qwen2ForCausalLM(config)
    for arm in ("control", "treatment"):
        dataset, _ = prepare_arm(tokenizer, _datapoints(), arm)
        expected = {tuple(r["input_ids"]): list(r["labels"]) for r in dataset}
        args = trl.SFTConfig(output_dir=str(tmp_path / arm), per_device_train_batch_size=4,
                             report_to="none", max_seq_length=3072, use_cpu=True,
                             dataset_kwargs={"skip_prepare_dataset": True},
                             remove_unused_columns=False, bf16=False, fp16=False)
        trainer = trl.SFTTrainer(model=model, args=args, train_dataset=dataset,
                                 processing_class=tokenizer,
                                 data_collator=CompletionOnlyDataCollator(tokenizer))
        seen = 0
        for batch in trainer.get_train_dataloader():
            for row in range(batch["input_ids"].shape[0]):
                real = batch["attention_mask"][row] == 1
                key = tuple(batch["input_ids"][row][real].tolist())
                assert batch["labels"][row][real].tolist() == expected[key], arm
                seen += 1
        assert seen == len(VALUES)
        # the loss the Trainer optimises is computed from exactly these labels
        first = next(iter(trainer.get_train_dataloader()))
        with torch.no_grad():
            out = model(**{k: first[k] for k in ("input_ids", "attention_mask", "labels")})
        assert torch.isfinite(out.loss)


# --- refusals ---------------------------------------------------------------------------------

def _dp(completion, span, **over):
    base = dict(prompt=make_datapoint(RECORD, "f(3)", "42").prompt, completion=completion,
                function_id="toy::refuse", supervised_char_span=span)
    base.update(over)
    return SFTDataPoint(**base)


@pytest.mark.parametrize("dp_kwargs, reason", [
    (dict(completion="assert f(3) == 42", span=(16, 17)), "start at the value"),     # "2" only
    (dict(completion="assert f(3) == 'hello'", span=(15, 19)), "end at the value"),  # mid-token
    (dict(completion="assert f(3) == 42", span=(13, 17)), "start at the value"),     # "= 42"
    (dict(completion="assert f(3) == 42, 'msg'", span=(15, 17)), "contiguous end"),  # not last
    (dict(completion="assert f(3) == 42", span=None), "no supervised_char_span"),
    (dict(completion="assert f(3) == 42 ", span=(15, 17)), "outer whitespace"),
    (dict(completion="assert f(3) == 42", span=(15, 17),
          execution_mode="repository_pytest_fragment"), "function-assertion"),
])
def test_unalignable_or_unsupported_examples_are_refused(tokenizer, dp_kwargs, reason):
    dp = _dp(dp_kwargs.pop("completion"), dp_kwargs.pop("span"), **dp_kwargs)
    trainer = dataset_trainer(tokenizer, "value_only")
    with pytest.raises(ObjectiveAlignmentError, match=reason):
        trainer.prepare_dataset([dp])


def test_value_span_is_positional_not_a_substring_search(tokenizer):
    body, start, end = completion_text("g(42, '42')", "42")
    ids = tokenizer(body + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    labels = value_token_labels(tokenizer, body + tokenizer.eos_token, ids, (start, end))
    kept = [t for t in labels if t != IGNORE_INDEX]
    assert tokenizer.decode(kept).strip() == "42" + tokenizer.eos_token
    assert start == body.rindex("42")        # the final occurrence, located by position


def test_non_canonical_values_are_refused():
    for bad in (" 42", "42 ", ""):
        with pytest.raises(ObjectiveAlignmentError):
            make_datapoint(RECORD, "f(3)", bad)


# --- C4 manifest validation on toy rows (unchanged rules) ---------------------------------------

def _row(i, group="g1", **over):
    row = {"record_id": f"r{i}", "group_id": group, "call": f"f({i})", "value": str(i),
           "execution_mode": "function", "representation": None, "passes_reference": True,
           "discriminates": True, "dataset": "toy", "bug_family": "arith",
           "complexity_tier": "simple", "input_ids_sha256": "x",
           "control_input_ids_sha256": f"in{i}", "treatment_input_ids_sha256": f"in{i}",
           "control_labels_sha256": f"c{i}", "treatment_labels_sha256": f"t{i}",
           "order_index_control": i, "order_index_treatment": i,
           "tokens": {"input": 10, "control_supervised": 6, "treatment_supervised": 2}}
    row.update(over)
    return row


@pytest.mark.parametrize("bad, message", [
    (dict(passes_reference=False), "fails on the reference"),
    (dict(discriminates=False), "does not discriminate"),
    (dict(execution_mode="repository_pytest_fragment"), "non-function"),
    (dict(group_id="gate1"), "gate/confirmation"),
    (dict(treatment_input_ids_sha256="other"), "input sequences differ"),
    (dict(treatment_labels_sha256="c1"), "label masks are identical"),
    (dict(order_index_treatment=99), "order differs"),
])
def test_manifest_rules_refuse(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_manifest([_row(0), _row(1, **bad)], gate_groups={"gate1"},
                          confirmation_groups={"conf1"}, group_cap=5)


def test_manifest_refuses_repeats_and_group_cap_and_reports_composition():
    with pytest.raises(ValueError, match="repeated row"):
        validate_manifest([_row(0), _row(0)], gate_groups=(), confirmation_groups=(),
                          group_cap=5)
    with pytest.raises(ValueError, match="cap"):
        validate_manifest([_row(i) for i in range(3)], gate_groups=(), confirmation_groups=(),
                          group_cap=2)
    report = validate_manifest([_row(0), _row(1, group="g2")], gate_groups=(),
                               confirmation_groups=(), group_cap=2)
    assert report["rows"] == 2 and report["tokens"]["treatment_supervised"] == 4


def test_verify_pair_executes_on_toy_functions():
    ref = "def f(x):\n    return x + 1 if x > 2 else x\n"
    mut = "def f(x):\n    return x + 1 if x >= 2 else x\n"
    assert verify_pair(ref, mut, "f(2)", "2") == {"passes_reference": True, "discriminates": True}
    assert verify_pair(ref, mut, "f(5)", "6") == {"passes_reference": True, "discriminates": False}
    assert verify_pair(ref, mut, "f(2)", "3")["passes_reference"] is False
