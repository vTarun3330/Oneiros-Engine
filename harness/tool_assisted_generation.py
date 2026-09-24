"""Bounded execution-feedback repair (arm C) and its compute-matched control (arm B).

Both arms spend exactly the same generation budget per target and differ only
in whether demonstrably invalid candidates are shown structured feedback.

Round 1 (shared).  Eight samples from the canonical prompt, one call, seeded
per record.  B and C use the *same* round-1 generations, so the arms are
paired at the source.

Round 2 (eight single-sequence calls, one per slot, each seeded per slot).

* B: every slot is a fresh sample from the canonical prompt.
* C: slot ``i`` is a repair of round-1 candidate ``i`` if, and only if, that
  candidate is in a REPAIRABLE category; otherwise it is the identical
  canonical sample B receives.  So C differs from B only where feedback was
  given.

Final slots.  One frozen policy for both arms, blind to every assertion
outcome and to the fixed implementation: candidates are interleaved
(round-1 slot 1, round-2 slot 1, round-1 slot 2, ...), well-formed
non-duplicates come first in that order, and the first eight are returned.
"Well-formed" is decided on the code under test only and treats passing,
failing and raising in the target identically.

Information boundary.  The loop receives a *permitted view* of each record:
specification, the code under test and its permitted context.  The fixed
implementation and gold tests are removed before the loop sees the record and
are used only by the evaluator, after the eight final slots are frozen.

Everything model-facing is injected (sampler, parser, executor) so the whole
loop runs on the CPU under test with no model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from harness.execution_feedback import (
    FEEDBACK_SCHEMA_VERSION, REPAIRABLE, WELL_FORMED, build_feedback,
    classify_execution, classify_policy_failure, normalised_code,
)

LOOP_VERSION = "oneiros_tool_assisted_generation_v1"
ARMS = ("B", "C")
INITIAL_SAMPLES = 8
SECOND_ROUND_SLOTS = 8
FINAL_SLOTS = 8
SEQUENCES_PER_TARGET = INITIAL_SAMPLES + SECOND_ROUND_SLOTS
MAX_REPAIRS_PER_TARGET = SECOND_ROUND_SLOTS
FINAL_SLOT_POLICY = ("interleave round-1/round-2 by slot; well-formed non-duplicates "
                     "first in that order; first eight; blind to assertion outcomes and "
                     "to the fixed implementation")

#: Fields the loop may see.  Anything else in a record is dropped.
PERMITTED_FIELDS = (
    "id", "group_id", "task_type", "source", "entry_point", "execution_mode",
    "task_mode", "test_format", "target_symbols", "support_context",
    "prompt_code_under_test", "specification", "source_name", "dataset_name",
    "dataset_identity_policy", "project", "complexity_tier",
)
#: Record fields that are evaluator-only.  Never in the view, always scanned.
HIDDEN_FIELDS = ("golden_code", "reference_code", "test_cases", "tests")


class Sampler(Protocol):
    def __call__(self, view: Mapping[str, Any], additions: Sequence[str | None],
                 sequences_per_prompt: int, seed: int) -> list[list[dict[str, Any]]]:
        """One model call.  One prompt per addition (None = canonical prompt).

        Returns, per prompt, ``sequences_per_prompt`` dicts with ``text``,
        ``input_tokens``, ``output_tokens`` and ``prompt_sha256``.
        """


@dataclass(frozen=True)
class Budget:
    sequences_per_target: int = SEQUENCES_PER_TARGET
    max_repairs_per_target: int = MAX_REPAIRS_PER_TARGET
    max_new_tokens: int = 1024


class BudgetExceeded(RuntimeError):
    pass


class RepairPromptOverBudget(RuntimeError):
    """Raised by a sampler when a repair prompt cannot fit the prompt budget.

    The slot then receives the canonical sample B receives; the feedback is
    recorded as not delivered.  Budgets stay identical.
    """


#: The model's earlier attempt is echoed back at most this long (characters).
MAX_ECHO_CHARS = 1500


def permitted_view(record: Mapping[str, Any], protected_ids: Iterable[str] = ()) -> dict:
    """The record as the loop may see it: hidden fields removed, identity checked.

    ``code_under_test`` is the buggy implementation the evaluator also runs;
    executing against it is buggy-side behaviour the model is allowed to see.
    """
    protected = set(map(str, protected_ids))
    if str(record.get("id")) in protected or str(record.get("group_id")) in protected:
        raise PermissionError(f"record {record.get('id')!r} belongs to a protected split "
                              "or lineage")
    view = {key: record[key] for key in PERMITTED_FIELDS if key in record}
    view["code_under_test"] = str(record.get("mutant_code") or "")
    if not view["code_under_test"]:
        raise ValueError("record has no code under test")
    return view


def hidden_material(record: Mapping[str, Any]) -> list[str]:
    material: list[str] = []
    for key in HIDDEN_FIELDS:
        value = record.get(key)
        if isinstance(value, str):
            material.append(value)
        elif isinstance(value, (list, tuple)):
            material.extend(str(item.get("code") if isinstance(item, Mapping) else item)
                            for item in value)
    return material


def derived_seed(base_seed: int, record_id: str, *parts: object) -> int:
    material = "|".join([str(base_seed), str(record_id), *map(str, parts)])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


class Journal:
    """Append-only record of completed model calls, so a resume never repeats one."""

    def __init__(self, path: Path | None):
        self.path = path
        self.entries: dict[str, list[list[dict[str, Any]]]] = {}
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry["outputs"]

    def call(self, key: str, run: Callable[[], list[list[dict[str, Any]]]]):
        if key in self.entries:
            return self.entries[key], True
        outputs = run()
        self.entries[key] = outputs
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"key": key, "outputs": outputs},
                                        separators=(",", ":")) + "\n")
        return outputs, False


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def assess_batch(texts: Sequence[str], view: Mapping[str, Any], parse: Callable,
                 execute: Callable, seen: set[str], timeout: float,
                 cache: dict[str, dict[str, Any]], workers: int | None = None
                 ) -> list[dict[str, Any]]:
    """Parse, validate and run candidates on the code under test only.

    Parsing and duplicate detection are sequential (order decides which copy
    is the duplicate); executions run in parallel, each in its own sandboxed
    process.  Identical executable text is executed once per target: the
    result is a deterministic function of the text and the code under test.
    """
    from harness.parallel_execution import map_jobs
    verdicts: list[dict[str, Any]] = []
    pending: dict[str, None] = {}
    for text in texts:
        parsed = parse(text, str(view["id"]), str(view["entry_point"]))
        verdict: dict[str, Any] = {"parse_valid": bool(parsed["valid"]), "code": parsed["code"],
                                   "shape": parsed.get("shape"),
                                   "policy_reason": parsed["reason"], "execution": None,
                                   "_executable": None}
        if not parsed["valid"]:
            verdict["category"], verdict["detail"] = classify_policy_failure(parsed["reason"])
        else:
            key = normalised_code(parsed["code"])
            if key in seen:
                verdict["category"], verdict["detail"] = "duplicate_candidate", ""
            else:
                verdict["_executable"] = parsed["executable"]
                if parsed["executable"] not in cache:
                    pending[parsed["executable"]] = None
            seen.add(key)
        verdicts.append(verdict)
    jobs = list(pending)
    results = map_jobs(lambda code: execute(str(view["code_under_test"]), code, timeout),
                       jobs, workers=workers)
    cache.update(zip(jobs, results))
    for verdict in verdicts:
        executable = verdict.pop("_executable")
        if executable is not None:
            verdict["execution"] = dict(cache[executable])
            verdict["category"], verdict["detail"] = classify_execution(verdict["execution"])
    return verdicts


def make_frozen_parser(generator=None) -> Callable[[str, str, str], dict[str, Any]]:
    """The successor protocol's whole-output parser, exactly as generation uses it.

    ``generator`` is the loaded generator in a GPU run; on the CPU a generator
    object is constructed without loading weights, because parsing needs only
    its methods.  The shape and executable form come from the same candidate
    policy the evaluator applies.
    """
    from harness.candidate_policy import executable_candidate, validate_generated_test
    if generator is None:
        from engine.generator import Phi3Generator
        generator = Phi3Generator(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                  model_revision="2e1fd397ee46e1388853d2af2c993145b0f1098a")
    generator.parse_mode = "whole_output"

    def parse(text: str, function_id: str, entry_point: str) -> dict[str, Any]:
        parsed = generator._parse_output(text, function_id, entry_point)
        if not parsed.is_valid:
            return {"valid": False, "code": parsed.input_code or None, "shape": None,
                    "reason": str(parsed.parse_error or "unparsable"), "executable": None}
        shape = validate_generated_test(parsed.input_code, entry_point,
                                        allow_test_function=True).shape
        return {"valid": True, "code": parsed.input_code, "shape": shape, "reason": "",
                "executable": executable_candidate(parsed.input_code, shape)}
    return parse


def select_final(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The frozen, outcome-blind final-slot policy (identical for B and C)."""
    ordered = sorted(candidates, key=lambda c: (c["slot"], c["round"]))
    well = [c for c in ordered if c["category"] in WELL_FORMED]
    rest = [c for c in ordered if c["category"] not in WELL_FORMED]
    chosen = (well + rest)[:FINAL_SLOTS]
    return [dict(candidate, final_rank=rank, selection_reason=(
        "well_formed_in_interleaved_order" if candidate["category"] in WELL_FORMED
        else "filled_after_all_well_formed")) for rank, candidate in enumerate(chosen, 1)]


def run_target(record: Mapping[str, Any], *, sampler: Sampler, parse: Callable,
               execute: Callable, base_seed: int, journal: Journal,
               protected_ids: Iterable[str] = (), timeout: float = 5.0,
               budget: Budget = Budget(), provenance: Mapping[str, Any] | None = None,
               workers: int | None = None) -> dict[str, Any]:
    """Run both B and C for one target from one shared round 1."""
    view = permitted_view(record, protected_ids)
    hidden = hidden_material(record)
    record_id = str(view["id"])
    lineage = str(view.get("group_id") or record_id)
    started = time.perf_counter()
    calls = {"B": 0, "C": 0}
    sequences = {"B": 0, "C": 0}

    def timed(key: str, additions: Sequence[str | None], per_prompt: int, seed: int):
        t0 = time.perf_counter()
        outputs, reused = journal.call(key, lambda: sampler(view, additions, per_prompt, seed))
        elapsed = time.perf_counter() - t0
        if len(outputs) != len(additions) or any(len(o) != per_prompt for o in outputs):
            raise RuntimeError(f"sampler returned the wrong shape for {key}")
        for group in outputs:
            for item in group:
                if int(item["output_tokens"]) > budget.max_new_tokens:
                    raise BudgetExceeded("a generation exceeded max_new_tokens")
        return outputs, reused, elapsed

    seed1 = derived_seed(base_seed, record_id, "round1")
    round1, reused1, wall1 = timed(f"{record_id}|round1", [None], INITIAL_SAMPLES, seed1)
    for arm in ARMS:
        calls[arm] += 1
        sequences[arm] += INITIAL_SAMPLES

    candidates: dict[str, list[dict[str, Any]]] = {"B": [], "C": []}
    seen_r1: set[str] = set()
    cache: dict[str, dict[str, Any]] = {}
    round1_assessed = assess_batch([item["text"] for item in round1[0]], view, parse,
                                   execute, set(), timeout, cache, workers)
    for slot, (item, verdict) in enumerate(zip(round1[0], round1_assessed), 1):
        for arm in ARMS:
            candidates[arm].append(_candidate(arm, 1, slot, None, item, verdict, 1, seed1,
                                              wall1 / INITIAL_SAMPLES, reused1))

    repairs = 0
    undelivered: list[int] = []
    generated: dict[str, list[tuple]] = {"B": [], "C": []}
    for slot in range(1, SECOND_ROUND_SLOTS + 1):
        seed2 = derived_seed(base_seed, record_id, "round2", slot)
        fresh, reused_b, wall_b = timed(f"{record_id}|round2|{slot}|canonical", [None], 1, seed2)
        calls["B"] += 1
        sequences["B"] += 1
        generated["B"].append((slot, seed2, fresh[0][0], reused_b, wall_b, None))
        parent = round1_assessed[slot - 1]
        if parent["category"] in REPAIRABLE:
            repairs += 1
            if repairs > budget.max_repairs_per_target:
                raise BudgetExceeded("repair budget exceeded")
            feedback = build_feedback(parent["category"], parent["detail"], hidden)
            echo = parent["code"] if parent["code"] else round1[0][slot - 1]["text"]
            addition = _repair_addition(echo, feedback)
            try:
                output, reused_c, wall_c = timed(f"{record_id}|round2|{slot}|repair",
                                                 [addition], 1, seed2)
                generated["C"].append((slot, seed2, output[0][0], reused_c, wall_c,
                                       feedback))
            except RepairPromptOverBudget:
                undelivered.append(slot)
                generated["C"].append((slot, seed2, fresh[0][0], reused_b, wall_b, None))
        else:
            generated["C"].append((slot, seed2, fresh[0][0], reused_b, wall_b, None))
        calls["C"] += 1
        sequences["C"] += 1

    for arm in ARMS:
        seen = {normalised_code(c["code"]) for c in candidates[arm] if c["parse_valid"]}
        verdicts = assess_batch([entry[2]["text"] for entry in generated[arm]], view, parse,
                                execute, seen, timeout, cache, workers)
        for (slot, seed2, item, reused, wall, feedback), verdict in zip(generated[arm], verdicts):
            parent = round1_assessed[slot - 1]
            if (feedback is not None and verdict["code"] and parent["code"]
                    and normalised_code(verdict["code"]) == normalised_code(parent["code"])):
                verdict["category"], verdict["detail"] = "identical_to_parent", ""
            candidates[arm].append(_candidate(
                arm, 2, slot, f"{record_id}|r1|{slot}" if feedback is not None else None,
                item, verdict, 1 + slot, seed2, wall, reused, feedback))

    for arm in ARMS:
        if sequences[arm] != budget.sequences_per_target:
            raise BudgetExceeded(f"arm {arm} used {sequences[arm]} sequences")
    finals = {arm: select_final(candidates[arm]) for arm in ARMS}
    for arm in ARMS:
        if len(finals[arm]) != FINAL_SLOTS:
            raise RuntimeError(f"arm {arm} did not produce {FINAL_SLOTS} final slots")
    tokens = {arm: {"input": sum(c["input_tokens"] for c in candidates[arm]),
                    "output": sum(c["output_tokens"] for c in candidates[arm])}
              for arm in ARMS}
    return {
        "loop_version": LOOP_VERSION, "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        "record_id": record_id, "function_lineage": lineage,
        "provenance": dict(provenance or {}),
        "budget": {"sequences_per_target": budget.sequences_per_target,
                   "max_new_tokens": budget.max_new_tokens, "sequences": sequences,
                   "model_calls": calls, "repairs": repairs,
                   "repairs_undelivered_prompt_over_budget": undelivered, "tokens": tokens},
        "candidates": candidates,
        "final": {arm: [{"raw_output": c["raw_output"], "code": c["code"] if c["parse_valid"]
                         else None, "candidate_id": c["candidate_id"],
                         "selection_reason": c["selection_reason"]} for c in finals[arm]]
                  for arm in ARMS},
        "final_slot_policy": FINAL_SLOT_POLICY,
        "loop_wall_seconds": time.perf_counter() - started,
    }


def _repair_addition(candidate_text: str, feedback: Mapping[str, Any]) -> str:
    """Appended to the canonical prompt exactly as the generation adapter appends."""
    return "\n\n".join(["You previously answered:",
                        str(candidate_text).strip()[:MAX_ECHO_CHARS],
                        str(feedback["message"]),
                        "Write one corrected test. Output only the test."])


def _candidate(arm: str, round_: int, slot: int, parent: str | None, item: Mapping[str, Any],
               verdict: Mapping[str, Any], call_number: int, seed: int, wall: float,
               reused: bool, feedback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = str(item["text"])
    return {
        "candidate_id": f"{arm}|r{round_}|{slot}", "arm": arm, "round": round_, "slot": slot,
        "status": "repair" if feedback is not None else "initial" if round_ == 1 else "resample",
        "parent_candidate": parent, "prompt_sha256": item.get("prompt_sha256"),
        "raw_output": raw, "raw_output_sha256": _sha(raw), "code": verdict["code"],
        "parse_valid": verdict["parse_valid"], "candidate_shape": verdict.get("shape"),
        "policy_reason": verdict["policy_reason"], "execution": verdict["execution"],
        "category": verdict["category"], "detail": verdict["detail"],
        "feedback": dict(feedback) if feedback else None,
        "model_call_number": call_number, "input_tokens": int(item["input_tokens"]),
        "output_tokens": int(item["output_tokens"]), "wall_seconds": round(wall, 6),
        "generation_seed": seed, "replayed_from_journal": reused,
    }
