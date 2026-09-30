"""Read-only launch gate CLI (amendment v2.2 section D). Writes nothing; launches nothing.

    python scripts/native_generated_tests_launch_gate.py --preflight FILE --authorization FILE
        --job FILE --arm {base,sft} --out DIR [--condition primary_whole_module]

Prints pipeline_ready, gpu_authorized and launch_ready as distinct states. Exit status 0 only
when launch_ready.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from harness.native_launch_gate import evaluate
    from scripts.native_generated_tests_generate import (ARMS, adapter_sha256, model_identity,
                                                         source_tree_identity)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition", default="primary_whole_module")
    parser.add_argument("--backend", default="hf", choices=("hf", "mock"))
    args = parser.parse_args(argv)
    result = evaluate(ROOT, ROOT / args.preflight, ROOT / args.authorization,
                      job_path=ROOT / args.job, condition=args.condition, arm=args.arm,
                      out_dir=ROOT / args.out, model_identity=model_identity,
                      adapter_sha256=adapter_sha256, backend=args.backend,
                      generation_source=source_tree_identity)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["launch_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
