"""Rescore both prompt arms from stored raw outputs, under one extraction rule.

Why this exists
---------------
Under the frozen ``whole_output`` parser the oracle_first arm lost 446 of 4,336
candidates to ``generation_invalid``, against 72 for the baseline. Inspection
shows those candidates are not malformed: the two-step instruction makes the
model emit several independent top-level assertions, and the candidate policy
admits one assertion or one test function, not a list of assertions. So the
loss is a policy interaction with the prompt, not a model failure, and the
published comparison charges it to the model.

Two readings, neither replacing the frozen result
-------------------------------------------------
Both apply ``whole_output`` first, exactly as the frozen protocol parses it,
and differ only in what they do with a candidate the frozen run discarded:

``--mode first_assertion``
    Keep the first ``assert `` line and drop the rest. This answers "did the
    output contain at least one immediately usable assertion". It is NOT a
    faithful reading of the candidate: discarding later assertions can hide one
    that would have killed the mutant, and equally one that would have failed
    the reference. Read it as a lower bound on usable content, not as a score.

``--mode conjunctive``
    Run every emitted statement as one test function, which is the multi-
    assertion semantics the frozen policy already defines: all assertions must
    hold on the reference for the candidate to be reference-valid, and any
    assertion failing on the mutant kills it. This is the closer reading of
    what the model actually wrote.

Both are reached only for candidates the frozen run already discarded, so
neither can alter a candidate that already scored. That is a mechanical
property of the replay relative to the old score - it does NOT by itself
establish that a recovered candidate is a valid test, which is precisely why
both readings are reported rather than one. The no-multi-assertion precondition
is re-checked at run time and refused if it ever stops holding.

Both modes were chosen after inspecting the failures, so every number here is
exploratory, not confirmatory.

No model is loaded and no weights are written. The stored ``raw_output`` of
every candidate is replayed through the same parser, candidate policy and
execution harness the original run used; only the extraction differs.

Development-only. Rescoring a development arm does not make it a selection or
generalization result.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.candidate_policy import count_assertions  # noqa: E402
from harness.evaluation_admission import scope_split  # noqa: E402

ARMS = {
    "baseline": "results/rehearsal_ablationdev_base_qwen_s42_v1/rehearsal_result.json",
    "oracle_first": "results/promptab_oracle_first_ablation_dev_s42/rehearsal_result.json",
}


def first_assertion(raw: str) -> str | None:
    """The legacy extractor: first top-level line starting with ``assert ``.

    This keeps one assertion and discards the rest of the candidate, so it
    answers only "did the output contain at least one immediately usable
    assertion". It can hide a later assertion that would have killed the
    mutant, and equally a later assertion that would have failed the
    reference. It is a diagnostic reading, not a faithful one.
    """
    for line in str(raw or "").replace("\r", "").split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert "):
            return stripped
    return None


def unfence(output: str) -> str:
    """Strip a chat model's code fence, exactly as the frozen parser does."""
    code = str(output or "").strip()
    if code.startswith("```"):
        body = code.split("```")
        if len(body) >= 2:
            candidate = body[1]
            first, _, rest = candidate.partition("\n")
            code = (rest if first.strip().isalpha() else candidate).strip()
    return code


def conjunctive_candidate(raw: str) -> str | None:
    """All emitted statements as ONE test function: the multi-assertion reading.

    Every assertion must pass on the reference for the candidate to be
    reference-valid, and any assertion failing on the mutant kills it - which is
    the semantics a multi-assertion test function already has under the frozen
    policy, so this uses the existing validated shape rather than a new one.

    Returns None when the output is not a straight-line block of statements
    (an import, a def or a class means the candidate was never of this shape).
    """
    code = unfence(raw)
    if not code.strip():
        return None
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    if (len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name.startswith("test")):
        return code                      # already the shape we would build
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            return None
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        return None
    body = "\n".join("    " + line if line.strip() else line
                     for line in code.split("\n"))
    return "def test_generated():\n" + body


def load_arm(path: Path):
    """Stored candidates per record, in rank order, with their frozen verdict."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored = {}
    for result in artifact["function_results"]:
        outcomes = sorted(result["candidate_outcomes"], key=lambda c: c.get("rank", 0))
        stored[result["record_id"]] = outcomes
    return artifact, stored


def check_additive(stored) -> int:
    """Refuse if any already-valid candidate holds more than one assertion.

    If one did, replacing it under stage 2 could change a result that the
    frozen run already scored, and the rescore would no longer be additive.
    """
    offenders = 0
    for outcomes in stored.values():
        for outcome in outcomes:
            if outcome.get("policy_valid") and count_assertions(outcome.get("code") or "") > 1:
                offenders += 1
    return offenders


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="ablation_dev")
    parser.add_argument("--mode", default="first_assertion",
                        choices=("first_assertion", "conjunctive"),
                        help="first_assertion: keep one assertion, discarding the "
                             "rest. conjunctive: run every emitted statement as one "
                             "test function, so all assertions must hold on the "
                             "reference and any may kill the mutant.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out_path = args.out or f"results/prompt_variant_rescore_{args.mode}.json"

    from harness.generation_adapter import successor_settings
    from harness.rehearsal_evaluator import run_rehearsal_evaluation
    from scripts.train_on_dataset import _record_to_pair

    settings = successor_settings()
    corpus = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
    splits = json.loads((corpus / "splits.json").read_text(encoding="utf-8"))
    records = json.loads((corpus / "records.json").read_text(encoding="utf-8"))
    scope = scope_split(splits, records, args.split, adapt=_record_to_pair)
    targets = scope.eligible
    print(f"targets in scope: {len(targets)}   scope sha256: {scope.scope_sha256()[:16]}...")

    summary = {}
    for arm, rel in ARMS.items():
        artifact, stored = load_arm(ROOT / rel)
        offenders = check_additive(stored)
        if offenders:
            print(f"REFUSED: {arm} has {offenders} valid multi-assertion candidates; "
                  f"stage 2 would no longer be purely additive")
            return 2
        recovered = [0]
        extract = first_assertion if args.mode == "first_assertion" else conjunctive_candidate

        def generate_batch(batch, _stored=stored, _recovered=recovered, _extract=extract):
            rows = []
            for record in batch:
                outcomes = _stored[str(record["id"])]
                slots = []
                for outcome in outcomes:
                    raw = outcome.get("raw_output", "")
                    if outcome.get("policy_valid"):
                        slots.append({"raw_output": raw, "code": outcome.get("code")})
                        continue
                    salvaged = _extract(raw)
                    if salvaged:
                        _recovered[0] += 1
                        slots.append({"raw_output": raw, "code": salvaged})
                    else:
                        slots.append({"raw_output": raw, "code": outcome.get("code")})
                rows.append(slots)
            return rows

        out_dir = ROOT / "results" / f"rescore_{arm}_{args.mode}_{args.split}"
        print(f"\n--- rescoring {arm} from stored raw outputs (no model) ---", flush=True)
        rescored = run_rehearsal_evaluation(
            load_records=lambda: targets,
            generate_batch=generate_batch,
            output_dir=out_dir,
            split_name=args.split,
            scope_summary=scope.to_dict(),
            frozen_settings=settings.to_dict(),
            candidates_per_target=settings.candidates_per_function,
            generation_batch_size=settings.generation_batch_size,
            allow_test_function=settings.allow_test_function_candidates,
            log=lambda m: None,
            seed_record={"replay": True, "no_generation_performed": True},
            generator_identity={"replay_of": rel, "model_loaded": False,
                                "extraction": "whole_output, then first_assertion fallback"},
        )
        taxonomy = rescored["failure_taxonomy"]
        total = sum(taxonomy.values())
        summary[arm] = {
            "frozen": {k: artifact["kill_at_k"][k]["rate"] for k in ("1", "2", "4", "8")},
            "rescored": {k: rescored["kill_at_k"][k]["rate"] for k in ("1", "2", "4", "8")},
            "frozen_taxonomy": artifact["failure_taxonomy"],
            "rescored_taxonomy": taxonomy,
            "candidates_recovered": recovered[0],
            "candidates": total,
        }
        print(f"  candidates recovered by stage 2: {recovered[0]}")
        for k in ("1", "2", "4", "8"):
            before = artifact["kill_at_k"][k]["rate"]
            after = rescored["kill_at_k"][k]["rate"]
            print(f"  Kill@{k:<2} {before:.6f} -> {after:.6f}  ({100 * (after - before):+.4f} pts)")

    print("\n" + "=" * 74)
    print(f"{'':10} {'frozen parser':>28}   {'same extraction both arms':>28}")
    for k in ("1", "2", "4", "8"):
        fb = summary["baseline"]["frozen"][k]
        fo = summary["oracle_first"]["frozen"][k]
        rb = summary["baseline"]["rescored"][k]
        ro = summary["oracle_first"]["rescored"][k]
        print(f"  Kill@{k:<2}  base {fb:.4f} oracle {fo:.4f} ({100*(fo-fb):+6.2f})   "
              f"base {rb:.4f} oracle {ro:.4f} ({100*(ro-rb):+6.2f})")
    print("=" * 74)

    Path(out_path).write_text(json.dumps({
        "schema_version": "oneiros_prompt_variant_rescore_v1",
        "label": "development-only rescore - not model selection, not a generalization result",
        "split": args.split,
        "evaluation_scope_sha256": scope.scope_sha256(),
        "extraction_mode": args.mode,
        "additive": "verified: no policy-valid candidate in either arm holds >1 assertion",
        "model_loaded": False, "weights_written": False,
        "arms": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
