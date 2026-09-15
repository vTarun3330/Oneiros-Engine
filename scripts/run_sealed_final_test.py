"""The authorized entrypoint for the one-time sealed final measurement.

Separate from ``scripts/train_on_dataset.py`` on purpose. The development
entrypoint can reach every split except this one, and refuses the sealed split
outright; this file is the only path that can open it, and it cannot do so
without a token that was issued in a separate, explicit act against one frozen
bundle hash.

It refuses by default. Running it with no arguments does nothing but explain
what would be required, because the failure mode worth designing against is not
someone maliciously opening the split - it is someone opening it casually, to
"just check", before the thing being checked was frozen.

Nothing in this file may be run without explicit human authorization. It is
committed so that the sealed procedure is reviewable in advance, not so that it
is convenient to execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.sealed_final import (  # noqa: E402
    SEALED_SPLIT, FinalBundle, SealedAccessError, SealedFinalGuard,
)

READINESS_RECEIPT = "results/v4_2_sealed_final_readiness_receipt.json"
STATE_PATH = "results/sealed_final_state.json"
AUDIT_LOG_PATH = "results/sealed_final_audit.log"
FINAL_RUN_NAME = "sealed_final_base_qwen_s42"


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_bundle(receipt_path: Path, expected_sha256: str) -> FinalBundle:
    """Rebuild the frozen bundle from the readiness receipt, or refuse."""
    if not receipt_path.is_file():
        raise SealedAccessError(f"readiness receipt not found: {receipt_path}")
    actual = sha256_file(receipt_path)
    if actual != expected_sha256:
        raise SealedAccessError(
            "readiness receipt hash mismatch; refusing to open the sealed split.\n"
            f"  expected: {expected_sha256}\n  found   : {actual}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("ready_for_authorization") is not True:
        raise SealedAccessError("readiness receipt is not marked ready_for_authorization")
    if receipt.get("sealed_split_accessed") is not False:
        raise SealedAccessError("readiness receipt claims the sealed split was already accessed")
    bundle = FinalBundle(fields=receipt["frozen_bundle"]["fields"])
    if bundle.sha256() != receipt["bundle_sha256"]:
        raise SealedAccessError(
            "the frozen bundle does not hash to the value recorded in the receipt; "
            "something was edited after the freeze")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-receipt", default=READINESS_RECEIPT)
    parser.add_argument("--expected-receipt-sha256", default=None)
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument(
        "--i-understand-this-is-one-time-and-irreversible", action="store_true",
        dest="acknowledged",
        help="Required. The sealed split cannot be reopened and nothing may be "
             "tuned afterwards.")
    parser.add_argument("--run-name", default=FINAL_RUN_NAME)
    args = parser.parse_args(argv)

    missing = [name for name, value in (
        ("--expected-receipt-sha256", args.expected_receipt_sha256),
        ("--authorization-token", args.authorization_token),
    ) if not value]
    if missing or not args.acknowledged:
        print("REFUSED: the sealed final test was not run.\n")
        print("This entrypoint opens the one measurement in this project that cannot")
        print("be repeated. It requires, all together:\n")
        print("  --readiness-receipt                              (default: %s)" % READINESS_RECEIPT)
        print("  --expected-receipt-sha256 <sha256>               the frozen receipt's hash")
        print("  --authorization-token <token>                    issued once, separately")
        print("  --i-understand-this-is-one-time-and-irreversible")
        if missing:
            print("\nmissing: %s" % ", ".join(missing))
        if not args.acknowledged:
            print("missing: --i-understand-this-is-one-time-and-irreversible")
        print("\nAfter execution these are prohibited, without exception:")
        for item in ("retraining of any kind", "prompt changes", "threshold changes",
                     "adding, reweighting or re-selecting data",
                     "selecting a different checkpoint or model",
                     "re-running the sealed final test",
                     "re-scoring the result under a different parser or policy"):
            print(f"  - {item}")
        return 2

    try:
        bundle = load_frozen_bundle(ROOT / args.readiness_receipt,
                                    args.expected_receipt_sha256)
        guard = SealedFinalGuard(ROOT / STATE_PATH, ROOT / AUDIT_LOG_PATH)
        guard.open_sealed_split(
            SEALED_SPLIT, bundle, args.authorization_token,
            caller=f"{Path(__file__).name}:{args.run_name}")
    except SealedAccessError as exc:
        print(f"REFUSED: {exc}")
        return 1

    # Reaching here means the guard granted the single authorized read and has
    # already spent the token. The measurement itself is intentionally not
    # implemented in this commit: it must not become runnable before the
    # authorization it depends on has actually been granted.
    print("Authorization accepted and spent. The sealed measurement is not "
          "implemented in this commit; implement it only under the approved "
          "bundle, and never re-run this entrypoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
