"""Freeze a diagnostic receipt for the oracle_first prompt experiment.

This is deliberately NOT a rehearsal or final receipt. It records what a
development experiment was run with and what it produced, so the numbers can be
checked later without re-running a GPU job, and so nobody mistakes them for a
selection or generalization result.

What it binds:

  * the exact variant instruction text and its hash, now living in
    ``research/experimental/prompt_variants.py`` rather than in the frozen
    prompt engine;
  * the canonical prompt-engine hash, which must equal the value the rehearsal
    receipt recorded - the experiment is only comparable to the baseline if the
    baseline's prompt sources are unchanged;
  * both run artifacts (frozen-parser and rescored) by hash;
  * the evaluation scope digest shared by every arm.

CPU only. Writes one JSON file and nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECEIPT_VERSION = "oneiros_prompt_variant_diagnostic_v1"

ARTIFACTS = {
    "baseline_frozen_parser": "results/rehearsal_ablationdev_base_qwen_s42_v1/rehearsal_result.json",
    "oracle_first_frozen_parser": "results/promptab_oracle_first_ablation_dev_s42/rehearsal_result.json",
    "baseline_first_assertion": "results/rescore_baseline_ablation_dev/rehearsal_result.json",
    "oracle_first_first_assertion": "results/rescore_oracle_first_ablation_dev/rehearsal_result.json",
    "baseline_conjunctive": "results/rescore_baseline_conjunctive_ablation_dev/rehearsal_result.json",
    "oracle_first_conjunctive": "results/rescore_oracle_first_conjunctive_ablation_dev/rehearsal_result.json",
    "rescore_summary_first_assertion": "results/prompt_variant_rescore.json",
    "rescore_summary_conjunctive": "results/prompt_variant_rescore_conjunctive.json",
    "tap_pilot_split": "results/tap_pilot_split.json",
}

#: The prompt engine hash the rehearsal receipt froze. The variant must not
#: change it; if it has, the baseline is no longer a valid comparison target.
EXPECTED_PROMPT_ENGINE_CANONICAL = "b522d35222f3599e"


def file_hashes(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/prompt_variant_diagnostic_receipt.json")
    args = parser.parse_args(argv)

    from harness.source_identity import canonical_sha256, raw_sha256
    from research.experimental.prompt_variants import CANONICAL_SELF_CONTAINED, ORACLE_FIRST

    engine = ROOT / "engine/test_generation_prompt.py"
    engine_canonical = canonical_sha256(engine)
    if not engine_canonical.startswith(EXPECTED_PROMPT_ENGINE_CANONICAL):
        print(f"REFUSED: prompt engine canonical hash is {engine_canonical[:16]}..., "
              f"expected {EXPECTED_PROMPT_ENGINE_CANONICAL}... - the frozen "
              f"baseline is no longer a valid comparison target")
        return 2

    missing = [rel for rel in ARTIFACTS.values() if not (ROOT / rel).exists()]
    if missing:
        print(f"REFUSED: missing artifacts: {missing}")
        return 2

    variant_module = ROOT / "research/experimental/prompt_variants.py"
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "label": "development diagnostic - NOT a rehearsal, selection, "
                 "generalization or final-test result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "ablation_dev",
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
        "adapter": None,
        "seed": 42,
        "variant": "oracle_first",
        "variant_instruction_sha256": hashlib.sha256(
            ORACLE_FIRST.encode("utf-8")).hexdigest(),
        "replaced_instruction_sha256": hashlib.sha256(
            CANONICAL_SELF_CONTAINED.encode("utf-8")).hexdigest(),
        "variant_module": {
            "path": "research/experimental/prompt_variants.py",
            "raw_sha256": raw_sha256(variant_module),
            "canonical_sha256": canonical_sha256(variant_module),
        },
        "prompt_engine_unchanged": {
            "path": "engine/test_generation_prompt.py",
            "raw_sha256": raw_sha256(engine),
            "canonical_sha256": engine_canonical,
            "note": "equals the value bound by the rehearsal receipt; the "
                    "variant lives outside the frozen prompt engine",
        },
        "extraction_readings": {
            "first_assertion": "keep the first assert line, discard the rest; a "
                               "lower bound on usable content, not a faithful "
                               "reading of the candidate",
            "conjunctive": "run every emitted statement as one test function; all "
                           "assertions must hold on the reference, any may kill "
                           "the mutant",
            "applied": "identically to both arms; reached only for candidates the "
                       "frozen run already discarded",
        },
        "artifacts": {name: file_hashes(ROOT / rel) for name, rel in ARTIFACTS.items()},
    }

    first = json.loads((ROOT / "results/prompt_variant_rescore.json").read_text(encoding="utf-8"))
    conj = json.loads((ROOT / "results/prompt_variant_rescore_conjunctive.json").read_text(encoding="utf-8"))
    receipt["evaluation_scope_sha256"] = first["evaluation_scope_sha256"]
    receipt["readings"] = {
        "frozen_parser": {k: {"baseline": first["arms"]["baseline"]["frozen"][k],
                              "oracle_first": first["arms"]["oracle_first"]["frozen"][k]}
                          for k in ("1", "2", "4", "8")},
        "first_assertion": {k: {"baseline": first["arms"]["baseline"]["rescored"][k],
                                "oracle_first": first["arms"]["oracle_first"]["rescored"][k]}
                            for k in ("1", "2", "4", "8")},
        "conjunctive": {k: {"baseline": conj["arms"]["baseline"]["rescored"][k],
                            "oracle_first": conj["arms"]["oracle_first"]["rescored"][k]}
                        for k in ("1", "2", "4", "8")},
    }
    receipt["statistical_status"] = (
        "Exploratory, not confirmatory: both extraction readings were chosen "
        "after inspecting the failures, and four nested Kill@k comparisons were "
        "examined. Under Bonferroni x4, Kill@1 survives p<0.01 under every "
        "reading and Kill@8 survives under none.")

    out = ROOT / args.out
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"prompt engine canonical : {engine_canonical[:16]}... (unchanged)")
    print(f"variant instruction     : {receipt['variant_instruction_sha256'][:16]}...")
    print(f"evaluation scope        : {receipt['evaluation_scope_sha256'][:16]}...")
    for name, meta in receipt["artifacts"].items():
        print(f"  {name:32} {meta['sha256'][:16]}...  {meta['bytes']:>9,d} bytes")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
