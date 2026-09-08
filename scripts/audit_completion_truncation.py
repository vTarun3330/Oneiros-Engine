"""Does the 128-token completion limit cut multi-assertion tests in half?

The successor protocol lets a model emit one test function carrying several
assertions. A completion budget that stops mid-function would make that
impossible regardless of what the model can do, and would do it silently: a
truncated function fails to parse, is counted as an invalid candidate, and
looks exactly like a model that writes bad code.

The budget must not be raised on suspicion. This measures whether truncation
actually happens, using the raw generations the pilot retained:

* completion length in TOKENS, with percentiles, using the same tokenizer the
  generator used
* how many completions reach the limit exactly, which is what a length stop
  looks like from the artifact
* how many are unparseable, and among those, how many parse once a trailing
  incomplete line is dropped - the signature of a cut-off generation rather
  than a malformed one
* assertions present, and assertions recoverable, so the cost is counted in
  the unit that matters

Raising the budget is justified only if the measured truncation is material.
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.candidate_policy import count_assertions
from harness.corpus import write_json
from scripts.analyze_parser_pilot import whole_output_of


def _drop_last_line(code: str) -> str:
    lines = code.rstrip().split("\n")
    return "\n".join(lines[:-1]) if len(lines) > 1 else ""


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def analyse(artifact: Path, limit_tokens: int, tokenizer_name: str) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    split = str(payload.get("evaluation_split") or "")
    if payload.get("final_test_measurement") or split == "test":
        raise SystemExit(f"{artifact} is sealed final-test data; refusing")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    lengths: list[int] = []
    at_limit = 0
    unparseable = 0
    parses_after_dropping_last_line = 0
    assertions_present = 0
    assertions_recovered = 0
    retained = 0

    for result in payload.get("function_results") or []:
        for outcome in result.get("candidate_outcomes") or []:
            raw = outcome.get("raw_output")
            if not isinstance(raw, str):
                continue
            retained += 1
            tokens = len(tokenizer.encode(raw, add_special_tokens=False))
            lengths.append(tokens)
            if tokens >= limit_tokens:
                at_limit += 1
            # Unwrap fenced blocks exactly as the successor parser does. Left
            # raw, a chat model's ```python fence makes every output
            # "unparseable" and the audit measures its own omission: the base
            # model scored 4207 of 4336 unparseable that way, against a real
            # kill@8 of 0.5959, which cannot both be true.
            code = whole_output_of(raw)
            assertions_present += count_assertions(code)
            if _parses(code):
                continue
            unparseable += 1
            trimmed = _drop_last_line(code)
            if trimmed and _parses(trimmed):
                parses_after_dropping_last_line += 1
                assertions_recovered += count_assertions(trimmed)

    percentiles: dict[str, int | None] = {}
    if lengths:
        ordered = sorted(lengths)
        for name, fraction in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90),
                               ("p99", 0.99)):
            index = min(len(ordered) - 1, int(fraction * len(ordered)))
            percentiles[name] = ordered[index]
        percentiles["max"] = ordered[-1]
        percentiles["mean"] = int(statistics.fmean(ordered))

    return {
        "schema_version": "oneiros_completion_truncation_audit_v1",
        "artifact": artifact.as_posix().split("results/", 1)[-1],
        "evaluation_split": split,
        "sealed_final_test_accessed": False,
        "completion_token_limit": limit_tokens,
        "tokenizer": tokenizer_name,
        "completions_with_retained_text": retained,
        "completion_token_percentiles": percentiles,
        "completions_at_or_over_the_limit": at_limit,
        "share_at_or_over_the_limit": round(at_limit / retained, 6) if retained else None,
        "unparseable_completions": unparseable,
        "unparseable_that_parse_after_dropping_the_last_line":
            parses_after_dropping_last_line,
        "assertions_present": assertions_present,
        "assertions_recoverable_from_truncated_completions": assertions_recovered,
        "verdict_rule": (
            "raise the budget only if completions at the limit are a material "
            "share AND dropping a trailing partial line rescues a meaningful "
            "number of assertions. A high unparseable count with few rescues "
            "is bad generation, not truncation, and a larger budget would not "
            "fix it."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--limit-tokens", type=int, default=128)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = analyse(arguments.artifact, arguments.limit_tokens, arguments.tokenizer)
    write_json(arguments.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
