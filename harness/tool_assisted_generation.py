"""Bounded execution-feedback repair (arm C) against a sham-feedback control (arm B).

B is a call-, sequence-, cap- and rendered-input-token-matched control
("request-budget-matched").  Per target:

* model calls, sequence counts and max-new-token caps are equal by construction;
* on every slot where C receives a repair prompt, B receives a SHAM prompt:
  the identical canonical prompt, the identical echoed round-1 candidate, a
  neutral sentence with no execution result or category, the identical final
  request, and neutral padding so the rendered Qwen-chat input has exactly
  C's token count;
* actual output tokens and wall-clock time are measured outcomes and may
  differ; they are never described as equal compute.

Round 1 (shared).  Eight samples from the canonical prompt, one call, seeded
per record, used by both arms.

Round 2 (eight single-sequence calls per arm, each seeded per slot).  If the
round-1 candidate in slot ``i`` is demonstrably invalid, C gets a repair prompt
and B the matched sham; if an exact token match is infeasible, NEITHER arm is
repaired and both take the identical canonical sample.  Otherwise both take
the identical canonical sample.

Final slots.  One frozen policy for both arms that reads only candidate TEXT:
interleave (round-1 slot 1, round-2 slot 1, ...), keep parser/policy-valid,
non-duplicate candidates first in that order, take eight.  No execution
outcome, assertion outcome, kill status or fixed-side validity is read.

Information boundary.  The loop receives a permitted view of each record; the
fixed implementation and gold tests are removed and are used only by the
evaluator after the final slots are frozen.  Sampler, parser, executor and
token matcher are injected so the whole loop runs on the CPU under test.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from harness.execution_feedback import (  # noqa: F401 - MAX_ECHO_CHARS is re-exported
    FEEDBACK_SCHEMA_VERSION, MAX_ECHO_CHARS, REPAIRABLE, build_feedback,
    classify_execution, classify_policy_failure, normalised_code, repair_addition,
    sham_addition,
)

LOOP_VERSION = "oneiros_tool_assisted_generation_v2_sham_matched"
ARMS = ("B", "C")
INITIAL_SAMPLES = 8
SECOND_ROUND_SLOTS = 8
FINAL_SLOTS = 8
SEQUENCES_PER_TARGET = INITIAL_SAMPLES + SECOND_ROUND_SLOTS
MAX_REPAIRS_PER_TARGET = SECOND_ROUND_SLOTS
CONTROL_DESCRIPTION = ("call-, sequence-, cap- and rendered-input-token-matched sham-feedback "
                       "control (request-budget-matched); actual output tokens and wall-clock "
                       "time are measured outcomes and may differ")
FINAL_SLOT_POLICY = ("interleave round-1/round-2 by slot; parser/policy-valid, non-duplicate "
                     "candidates first in that order; first eight; reads candidate text only "
                     "- never execution, assertion, kill or fixed-side outcomes")

PERMITTED_FIELDS = (
    "id", "group_id", "task_type", "source", "entry_point", "execution_mode",
    "task_mode", "test_format", "target_symbols", "support_context",
    "prompt_code_under_test", "specification", "source_name", "dataset_name",
    "dataset_identity_policy", "project", "complexity_tier",
)
HIDDEN_FIELDS = ("golden_code", "reference_code", "test_cases", "tests")


class Sampler(Protocol):
    def __call__(self, view: Mapping[str, Any], additions: Sequence[str | None],
                 sequences_per_prompt: int, seed: int) -> list[list[dict[str, Any]]]:
        """One model call; returns ``text``, ``input_tokens``, ``output_tokens``,
        ``prompt_sha256`` per sequence."""


@dataclass(frozen=True)
class Budget:
    sequences_per_target: int = SEQUENCES_PER_TARGET
    max_repairs_per_target: int = MAX_REPAIRS_PER_TARGET
    max_new_tokens: int = 1024


@dataclass(frozen=True)
class MatchedPair:
    repair_addition: str
    sham_addition: str
    repair_tokens: int
    sham_tokens_unpadded: int
    sham_tokens: int
    pad_units: int


class BudgetExceeded(RuntimeError):
    pass


class MatchInfeasible(RuntimeError):
    """An exact rendered-token match cannot be built within the frozen budget."""


def permitted_view(record: Mapping[str, Any], protected_ids: Iterable[str] = ()) -> dict:
    """The record as the loop may see it: hidden fields removed, identity checked."""
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


def make_token_matcher(tokenizer, settings, build_prompt) -> Callable:
    """Exact rendered-input-token matching on the real Qwen chat template.

    Both prompts are rendered exactly as the generation adapter renders them
    (canonical prompt + blank line + addition, section-aware compaction, chat
    template).  Either prompt needing compaction, i.e. any truncation of the
    original prompt, echo or feedback, makes the slot infeasible.  The sham is
    padded one neutral unit at a time until its count equals the repair's.
    """
    from engine.prompt_budget import PromptBudgetError, compact_unified_user_prompt
    from engine.test_generation_prompt import format_chat_prompt

    limit = min(settings.prompt_token_limit,
                settings.max_sequence_tokens - settings.generation_completion_token_limit)

    def rendered_tokens(view: Mapping[str, Any], addition: str) -> int:
        prompt = f"{build_prompt(view)}\n\n{addition}"
        try:
            result = compact_unified_user_prompt(tokenizer, prompt, limit, format_chat_prompt)
        except (PromptBudgetError, ValueError) as exc:
            raise MatchInfeasible(f"prompt over budget: {exc}") from exc
        if result.compacted:
            raise MatchInfeasible("prompt would be compacted")
        return len(result.token_ids)

    def match(view: Mapping[str, Any], repair: str, echo: str) -> MatchedPair:
        target = rendered_tokens(view, repair)
        unpadded = rendered_tokens(view, sham_addition(echo, 0))
        if unpadded > target:
            raise MatchInfeasible("sham is longer than the repair before padding")
        for units in range(0, target - unpadded + 3):
            sham = sham_addition(echo, units)
            count = rendered_tokens(view, sham)
            if count == target:
                return MatchedPair(repair, sham, target, unpadded, count, units)
            if count > target:
                break
        raise MatchInfeasible("no exact padding length exists")
    return match


def make_frozen_parser(generator=None) -> Callable[[str, str, str], dict[str, Any]]:
    """The successor protocol's whole-output parser, exactly as generation uses it."""
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


def assess_batch(texts: Sequence[str], view: Mapping[str, Any], parse: Callable,
                 execute: Callable, seen: set[str], timeout: float,
                 cache: dict[str, dict[str, Any]], workers: int | None = None
                 ) -> list[dict[str, Any]]:
    """Parse, validate and run candidates on the code under test only.

    The resulting category decides only whether a round-1 candidate may be
    repaired.  It is never read by the final-slot policy.
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


def select_final(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The frozen final-slot policy (identical for B and C), text-only.

    Reads ``slot``, ``round``, ``parse_valid`` and ``code`` and nothing else.
    """
    ordered = sorted(candidates, key=lambda c: (c["slot"], c["round"]))
    seen: set[str] = set()
    preferred, rest = [], []
    for candidate in ordered:
        key = normalised_code(candidate["code"]) if candidate["parse_valid"] else None
        if key is not None and key not in seen:
            seen.add(key)
            preferred.append(candidate)
        else:
            rest.append(candidate)
    preferred_ids = {id(candidate) for candidate in preferred}
    chosen = (preferred + rest)[:FINAL_SLOTS]
    return [dict(candidate, final_rank=rank, selection_reason=(
        "policy_valid_first_occurrence" if id(candidate) in preferred_ids
        else "filled_after_all_policy_valid")) for rank, candidate in enumerate(chosen, 1)]


def run_target(record: Mapping[str, Any], *, sampler: Sampler, parse: Callable,
               execute: Callable, matcher: Callable, base_seed: int, journal: Journal,
               protected_ids: Iterable[str] = (), timeout: float = 0.5,
               budget: Budget = Budget(), provenance: Mapping[str, Any] | None = None,
               workers: int | None = None) -> dict[str, Any]:
    """Run B and C for one target from one shared round 1."""
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
    cache: dict[str, dict[str, Any]] = {}
    round1_assessed = assess_batch([item["text"] for item in round1[0]], view, parse,
                                   execute, set(), timeout, cache, workers)
    candidates: dict[str, list[dict[str, Any]]] = {"B": [], "C": []}
    for slot, (item, verdict) in enumerate(zip(round1[0], round1_assessed), 1):
        for arm in ARMS:
            candidates[arm].append(_candidate(arm, 1, slot, None, item, verdict, 1, seed1,
                                              wall1 / INITIAL_SAMPLES, reused1))

    attempted: list[int] = []
    delivered: list[int] = []
    skipped: dict[str, str] = {}
    generated: dict[str, list[dict[str, Any]]] = {"B": [], "C": []}
    for slot in range(1, SECOND_ROUND_SLOTS + 1):
        seed2 = derived_seed(base_seed, record_id, "round2", slot)
        parent = round1_assessed[slot - 1]
        pair = None
        feedback = None
        if parent["category"] in REPAIRABLE:
            attempted.append(slot)
            if len(attempted) > budget.max_repairs_per_target:
                raise BudgetExceeded("repair budget exceeded")
            feedback = build_feedback(parent["category"], parent["detail"], hidden)
            echo = parent["code"] if parent["code"] else round1[0][slot - 1]["text"]
            try:
                pair = matcher(view, repair_addition(echo, feedback, hidden), echo)
            except MatchInfeasible as exc:
                skipped[str(slot)] = str(exc)
        if pair is not None:
            repair, reused_c, wall_c = timed(f"{record_id}|round2|{slot}|repair",
                                             [pair.repair_addition], 1, seed2)
            sham, reused_b, wall_b = timed(f"{record_id}|round2|{slot}|sham",
                                           [pair.sham_addition], 1, seed2)
            if (int(repair[0][0]["input_tokens"]) != pair.repair_tokens
                    or int(sham[0][0]["input_tokens"]) != pair.sham_tokens):
                raise RuntimeError(f"rendered input tokens drifted from the match at {slot}")
            delivered.append(slot)
            parent_id = f"{record_id}|r1|{slot}"
            generated["C"].append({"slot": slot, "seed": seed2, "item": repair[0][0],
                                   "reused": reused_c, "wall": wall_c, "feedback": feedback,
                                   "status": "repair", "parent": parent_id,
                                   "addition": pair.repair_addition, "pair": pair})
            generated["B"].append({"slot": slot, "seed": seed2, "item": sham[0][0],
                                   "reused": reused_b, "wall": wall_b, "feedback": None,
                                   "status": "sham", "parent": parent_id,
                                   "addition": pair.sham_addition, "pair": pair})
        else:
            fresh, reused, wall = timed(f"{record_id}|round2|{slot}|canonical", [None], 1, seed2)
            for arm in ARMS:
                generated[arm].append({"slot": slot, "seed": seed2, "item": fresh[0][0],
                                       "reused": reused, "wall": wall, "feedback": None,
                                       "status": "resample", "parent": None,
                                       "addition": None, "pair": None})
        for arm in ARMS:
            calls[arm] += 1
            sequences[arm] += 1

    for arm in ARMS:
        seen = {normalised_code(c["code"]) for c in candidates[arm] if c["parse_valid"]}
        verdicts = assess_batch([entry["item"]["text"] for entry in generated[arm]], view,
                                parse, execute, seen, timeout, cache, workers)
        for entry, verdict in zip(generated[arm], verdicts):
            parent = round1_assessed[entry["slot"] - 1]
            if (entry["status"] in ("repair", "sham") and verdict["code"] and parent["code"]
                    and normalised_code(verdict["code"]) == normalised_code(parent["code"])):
                verdict["category"], verdict["detail"] = "identical_to_parent", ""
            candidates[arm].append(_candidate(
                arm, 2, entry["slot"], entry["parent"], entry["item"], verdict,
                1 + entry["slot"], entry["seed"], entry["wall"], entry["reused"],
                entry["feedback"], entry["status"], entry["addition"], entry["pair"]))

    for arm in ARMS:
        if sequences[arm] != budget.sequences_per_target:
            raise BudgetExceeded(f"arm {arm} used {sequences[arm]} sequences")
    finals = {arm: select_final(candidates[arm]) for arm in ARMS}
    tokens = {arm: {"input": sum(c["input_tokens"] for c in candidates[arm]),
                    "output": sum(c["output_tokens"] for c in candidates[arm])}
              for arm in ARMS}
    return {
        "loop_version": LOOP_VERSION, "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        "control": CONTROL_DESCRIPTION,
        "record_id": record_id, "function_lineage": lineage,
        "provenance": dict(provenance or {}),
        "budget": {"sequences_per_target": budget.sequences_per_target,
                   "max_new_tokens": budget.max_new_tokens, "sequences": sequences,
                   "model_calls": calls, "repairs_attempted": attempted,
                   "repairs_delivered": delivered,
                   "repairs_skipped_match_infeasible": skipped,
                   "tokens": tokens},
        "candidates": candidates,
        "final": {arm: [{"raw_output": c["raw_output"], "code": c["code"] if c["parse_valid"]
                         else None, "candidate_id": c["candidate_id"],
                         "selection_reason": c["selection_reason"]} for c in finals[arm]]
                  for arm in ARMS},
        "final_slot_policy": FINAL_SLOT_POLICY,
        "loop_wall_seconds": time.perf_counter() - started,
    }


def _candidate(arm: str, round_: int, slot: int, parent: str | None, item: Mapping[str, Any],
               verdict: Mapping[str, Any], call_number: int, seed: int, wall: float,
               reused: bool, feedback: Mapping[str, Any] | None = None,
               status: str | None = None, addition: str | None = None,
               pair: MatchedPair | None = None) -> dict[str, Any]:
    raw = str(item["text"])
    return {
        "candidate_id": f"{arm}|r{round_}|{slot}", "arm": arm, "round": round_, "slot": slot,
        "status": status or "initial", "parent_candidate": parent,
        "prompt_sha256": item.get("prompt_sha256"), "prompt_addition": addition,
        "matched_input_tokens": None if pair is None else {
            "repair": pair.repair_tokens, "sham_unpadded": pair.sham_tokens_unpadded,
            "sham": pair.sham_tokens, "pad_units": pair.pad_units},
        "raw_output": raw, "raw_output_sha256": _sha(raw), "code": verdict["code"],
        "parse_valid": verdict["parse_valid"], "candidate_shape": verdict.get("shape"),
        "policy_reason": verdict["policy_reason"], "execution": verdict["execution"],
        "category": verdict["category"], "detail": verdict["detail"],
        "feedback": dict(feedback) if feedback else None,
        "model_call_number": call_number, "input_tokens": int(item["input_tokens"]),
        "output_tokens": int(item["output_tokens"]), "wall_seconds": round(wall, 6),
        "generation_seed": seed, "replayed_from_journal": reused,
    }
