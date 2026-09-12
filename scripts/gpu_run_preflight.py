"""Refuse a GPU run that is not actually the run you meant to launch.

This exists because of a specific, expensive failure. A metamorphic-prompt
evaluation was launched without ``--base-model-name``,
``--attention-implementation`` and ``--sft-prompt-token-limit``. Those flags
have defaults, so nothing errored: the run silently used the canonical Phi-3
backend at a 512-token prompt budget instead of Qwen at 1024, every one of 542
prompts overflowed, and it reported a kill rate of 0.0 that measured the
misconfiguration. Thirteen minutes of GPU and a wholly invalid artifact,
because three defaults were wrong and nothing said so.

The defaults cannot simply be changed - they are the canonical configuration
that older runs were measured under, and moving them would silently re-label
history. So instead: state the intended configuration up front, and refuse to
launch if the command does not express it.

Checks performed:

* CUDA is actually available and a device is visible - a "GPU run" that
  quietly falls back to CPU takes twenty times as long and looks identical in
  the log until the wall time arrives;
* the base model, attention backend and prompt budget are named explicitly;
* the split is one this run is allowed to touch, and the sealed test is
  refused outright;
* when a comparison baseline is named, its recorded configuration matches, so
  the new artifact will be comparable to it rather than merely adjacent.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Flags whose defaults are canonical-but-wrong for every Qwen run on this
#: machine. Each must be stated explicitly rather than inherited.
REQUIRED_FLAGS = (
    "--base-model-name",
    "--attention-implementation",
    "--sft-prompt-token-limit",
)

SEALED_SPLIT = "test"

EXPECTED = {
    "--base-model-name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "--attention-implementation": "sdpa",
    "--sft-prompt-token-limit": "1024",
}


def parse_command(command: str) -> dict[str, str]:
    tokens = shlex.split(command)
    flags: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if token.startswith("--"):
            if "=" in token:
                name, _, value = token.partition("=")
                flags[name] = value
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                flags[token] = tokens[index + 1]
            else:
                flags[token] = "true"
    return flags


def check_cuda() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is False"}
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "count": torch.cuda.device_count(),
    }


def check(command: str, baseline: Path | None = None,
          require_cuda: bool = True) -> dict[str, Any]:
    flags = parse_command(command)
    problems: list[str] = []

    for flag in REQUIRED_FLAGS:
        if flag not in flags:
            problems.append(
                f"{flag} is not stated. Its default is the canonical Phi-3 "
                f"configuration, not this project's Qwen one; leaving it "
                f"implicit is what produced an invalid 0.0 artifact before."
            )
        elif flag in EXPECTED and flags[flag] != EXPECTED[flag]:
            problems.append(
                f"{flag} is {flags[flag]!r}, expected {EXPECTED[flag]!r}")

    split = flags.get("--evaluation-split", "val")
    if split == SEALED_SPLIT or flags.get("--confirm-final-test"):
        problems.append(
            "this command targets the sealed final test, which stays closed "
            "until every model-selection decision is frozen")

    cuda = check_cuda()
    if require_cuda and not cuda["available"]:
        problems.append(
            "CUDA is not available, so this would run on CPU roughly twenty "
            f"times slower and look identical in the log ({cuda.get('reason')})")

    comparable: dict[str, Any] | None = None
    if baseline is not None:
        if not baseline.exists():
            problems.append(f"baseline artifact {baseline} does not exist")
        else:
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            runtime = payload.get("model_runtime_profile") or {}
            fingerprint = str(payload.get("dataset_fingerprint") or "")
            comparable = {
                "baseline_model": runtime.get("model_name"),
                "baseline_attention": runtime.get("attention_implementation"),
                "baseline_prompt_token_limit": (
                    "1024" if "prompt_token_limit=1024" in fingerprint
                    else "unknown"),
                "baseline_split": payload.get("evaluation_split"),
            }
            if comparable["baseline_model"] != flags.get("--base-model-name"):
                problems.append(
                    f"baseline was measured on {comparable['baseline_model']} "
                    f"but this run requests {flags.get('--base-model-name')}; "
                    "the artifacts would not be comparable")
            if comparable["baseline_split"] != split:
                problems.append(
                    f"baseline is split {comparable['baseline_split']} but this "
                    f"run is split {split}")

    return {
        "schema_version": "oneiros_gpu_run_preflight_v1",
        "command": command,
        "flags": flags,
        "cuda": cuda,
        "baseline": comparable,
        "problems": problems,
        "ok": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", help="the full command line to be launched")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="artifact this run will be compared against")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="permit a run with no CUDA device (for tests)")
    arguments = parser.parse_args()

    report = check(arguments.command, arguments.baseline,
                   require_cuda=not arguments.allow_cpu)
    print(json.dumps({k: v for k, v in report.items() if k != "flags"}, indent=2))
    if not report["ok"]:
        print("\nPREFLIGHT FAILED - not launching.")
        return 1
    print("\npreflight ok: " + str(report["cuda"].get("device")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
