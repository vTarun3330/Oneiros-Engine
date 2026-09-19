"""A/B one prompt variant against the frozen protocol, base model only.

Runs the full measured path - admission, mode-aware scoping, generation,
whole-output parsing, safe execution, raw-output retention, scoring - exactly as
the rehearsal does, changing only ``output_instruction_variant``.

The comparison target is the already-measured base figure on the same split at
the same seed under ``self_contained``: Kill@8 = 0.605166, evaluation-scope
digest d37f75d6..., recorded in results/v4_2_rehearsal_execution_receipt.json.
Same scope, same seed, same everything else, so the difference is the prompt.

Operational measurement on a permitted split. Not model selection, not a
generalization estimate, not a final-test result. No weights are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.evaluation_admission import refuse_refused_split, scope_split  # noqa: E402

BASELINE_KILL_AT_8 = 0.605166          # self_contained, same split/seed
BASELINE_SCOPE_SHA = "d37f75d6f44e7f25f8ec0b38ef1be46efd77c85c31cda115e81f7a77fb91f6ca"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="oracle_first")
    ap.add_argument("--split", default="ablation_dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    refuse_refused_split(args.split)

    from config.settings import immutable_revision_for
    from engine.generator import Phi3Generator
    from harness.generation_adapter import generate_candidate_slots, successor_settings
    from harness.generation_rng import seed_generation_rngs
    from harness.prompt_factory import prompt_factory
    from harness.rehearsal_evaluator import run_rehearsal_evaluation
    from scripts.train_on_dataset import _record_to_pair

    from research.experimental.prompt_variants import variant_builder

    # The variant is NOT a value of the frozen ``output_instruction_variant``.
    # Adding one to the prompt engine changes its hash and invalidates the
    # rehearsal receipt's prompt binding, so settings stay canonical and the
    # substitution happens in the experimental module instead.
    settings = successor_settings()
    bad = settings.problems()
    if bad:
        print(f"REFUSED: invalid settings: {bad}")
        return 1

    corpus = ROOT / "data" / "corpus" / "v4_1_research_hardened_candidate"
    splits = json.loads((corpus / "splits.json").read_text(encoding="utf-8"))
    records = json.loads((corpus / "records.json").read_text(encoding="utf-8"))
    scope = scope_split(splits, records, args.split, adapt=_record_to_pair)
    targets = scope.eligible[:args.limit] if args.limit else scope.eligible

    print(f"variant      : {args.variant}")
    print(f"split        : {args.split}")
    print(f"targets      : {len(targets)} (of {scope.target_count} in scope)")
    print(f"scope sha256 : {scope.scope_sha256()}")
    print(f"scope matches the baseline run: {scope.scope_sha256() == BASELINE_SCOPE_SHA}")
    print(f"baseline Kill@8 (self_contained, same split/seed): {BASELINE_KILL_AT_8}")

    name = settings.base_model_name
    rev = settings.base_model_revision or immutable_revision_for(name)
    print(f"\nloading {name} @ {rev} (base only, no adapter)...", flush=True)
    gen = Phi3Generator(model_name=name, model_revision=rev,
                        attention_implementation=settings.attention_implementation)
    gen.temperature = settings.temperature
    gen.top_p = settings.top_p
    gen.parse_mode = settings.candidate_parse_mode
    gen.load_model()

    build_prompt = variant_builder(prompt_factory(settings.prompt_settings()), args.variant)
    seed_record = seed_generation_rngs(settings.seed)
    seed_record["applied_immediately_before_generation"] = True

    def generate_batch(batch):
        rows = [dict(r) for r in batch]
        acc = generate_candidate_slots(gen, rows, settings, build_prompt,
                                       seed_before_generation=False)
        return [[{"raw_output": s.get("raw_output", ""), "code": s.get("code")}
                 for s in acc[i]["candidate_slots"]] for i in range(len(rows))]

    run_name = f"promptab_{args.variant}_{args.split}_s{settings.seed}"
    out_dir = ROOT / "results" / run_name
    started = time.time()
    artifact = run_rehearsal_evaluation(
        load_records=lambda: targets,
        generate_batch=generate_batch,
        output_dir=out_dir,
        split_name=args.split,
        scope_summary=scope.to_dict(),
        frozen_settings=settings.to_dict(),
        candidates_per_target=settings.candidates_per_function,
        generation_batch_size=settings.generation_batch_size,
        allow_test_function=settings.allow_test_function_candidates,
        log=lambda m: print(m, flush=True),
        seed_record=seed_record,
        generator_identity={"python_object_id": id(gen), "model_name": name,
                            "model_revision": rev, "adapter": None,
                            "prompt_output_instruction_variant": args.variant},
    )

    k8 = artifact["kill_at_k"]["8"]["rate"]
    print("\n" + "=" * 70)
    print(f"variant   {args.variant:22} Kill@8 = {k8:.6f}")
    print(f"baseline  {'self_contained':22} Kill@8 = {BASELINE_KILL_AT_8:.6f}")
    print(f"delta                            {100*(k8-BASELINE_KILL_AT_8):+.4f} points")
    print("=" * 70)
    for k in (1, 2, 4, 8):
        e = artifact["kill_at_k"][str(k)]
        print(f"  Kill@{k:<2} {e['rate']:.6f}  ({e['functions']} functions)")
    tax = artifact["failure_taxonomy"]
    tot = sum(tax.values())
    ref_fail = sum(v for k_, v in tax.items() if k_.startswith("reference"))
    print(f"\n  fails the reference : {ref_fail}/{tot} = {ref_fail/tot:.1%}"
          f"   (baseline run: 41.8%)")
    print(f"  survived (no discrim): {tax.get('survived',0)}/{tot} = {tax.get('survived',0)/tot:.1%}"
          f"   (baseline run: 31.0%)")
    print(f"  wall time: {round(time.time()-started,1)}s")
    print("\nOperational measurement. Not model selection, not a generalization result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
