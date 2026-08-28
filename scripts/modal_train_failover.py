"""Credit-aware two-profile launcher for resumable Oneiros Modal training.

This wrapper never changes the globally active Modal profile. It selects a
configured profile through MODAL_PROFILE, launches the normal training entry
point, and retries on the next profile only when the failure is attributable
to exhausted credits or a workspace budget limit.

Example:
    py -3.12 scripts/modal_train_failover.py --profiles primary backup --estimated-cost 8 -- \
        --corpus-version v3_final_candidate --phase sft \
        --run-name v3_full_sft --sft-epochs 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


# Resolve Oneiros modules from this workspace even when the desktop process has
# inherited a PYTHONPATH containing another project's top-level ``config.py``.
# This is a launcher-only path fix; it does not alter the corpus or checkpoints.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root_text = str(PROJECT_ROOT)
if sys.path[:1] != [project_root_text]:
    if project_root_text in sys.path:
        sys.path.remove(project_root_text)
    sys.path.insert(0, project_root_text)


DEFAULT_PROFILES = tuple(
    profile.strip()
    for profile in os.environ.get("ONEIROS_MODAL_PROFILES", "").split(",")
    if profile.strip()
)
DEFAULT_CREDIT_LIMIT = 30.0
CREDIT_FAILURE_PATTERN = re.compile(
    r"(?:insufficient|out of|exhausted).{0,30}credits?"
    r"|credits?.{0,30}(?:exhausted|depleted|limit)"
    r"|workspace.{0,40}budget"
    r"|budget.{0,40}(?:exceeded|exhausted|limit)"
    r"|spending limit"
    r"|payment required",
    re.IGNORECASE | re.DOTALL,
)


def _profile_environment(profile: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MODAL_PROFILE"] = profile
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def monthly_usage(profile: str) -> float:
    """Return this month's billed usage for one configured Modal profile."""
    report = subprocess.run(
        [sys.executable, "-m", "modal", "billing", "report", "--for", "this month", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_profile_environment(profile),
        check=False,
    )
    if report.returncode:
        raise RuntimeError(
            f"Could not query Modal billing for profile {profile!r}: "
            f"{report.stderr.strip() or report.stdout.strip()}"
        )
    rows = json.loads(report.stdout or "[]")
    # Modal CLI 1.5 emits lowercase ``cost``/``object_id`` while older
    # clients emitted title-cased columns. Accept both so a client upgrade
    # cannot incorrectly report $0 usage and select an exhausted profile.
    return sum(float(row.get("cost", row.get("Cost", 0.0))) for row in rows)


def _option_value(arguments: list[str], option: str) -> str | None:
    try:
        index = arguments.index(option)
    except ValueError:
        return None
    if index + 1 >= len(arguments):
        raise ValueError(f"{option} requires a value")
    return arguments[index + 1]


def bounded_sft_optimizer_upper_bound(
    max_pairs: int,
    epochs: int,
    batch_size: int,
    max_real_repeats: int,
    gradient_accumulation_steps: int = 16,
) -> int:
    """Conservative local upper bound before any Modal GPU call is submitted."""
    if min(max_pairs, epochs, batch_size, max_real_repeats) <= 0:
        raise ValueError("Bounded SFT capacity inputs must be positive")
    # Each canonical record contributes at most three winners. Treat every
    # record as repeatable real data to make this an upper, never lower, bound.
    maximum_examples = max_pairs * 3 * max_real_repeats
    return (
        math.ceil(maximum_examples / (batch_size * gradient_accumulation_steps))
        * epochs
    )


def validate_bounded_sft_monitor_capacity(
    max_pairs: int,
    epochs: int,
    batch_size: int,
    max_real_repeats: int,
    checkpoint_steps: int = 50,
    minimum_checkpoints: int = 2,
) -> int:
    """Reject a smoke that cannot reach its declared monitor checkpoints."""
    upper_bound = bounded_sft_optimizer_upper_bound(
        max_pairs, epochs, batch_size, max_real_repeats
    )
    effective_checkpoint_steps = min(checkpoint_steps, upper_bound)
    required_steps = effective_checkpoint_steps * minimum_checkpoints
    if upper_bound < required_steps:
        raise ValueError(
            "Bounded SFT smoke is underpowered even at its theoretical maximum: "
            f"optimizer_steps<={upper_bound}, required={required_steps}. Increase "
            "--max-pairs before submitting Modal work."
        )
    return upper_bound


def _failure_log_path(run_name: str) -> Path:
    return Path(f"pipeline_oneiros_training_{run_name}.log")


def _read_failure_log(run_name: str, start_offset: int = 0) -> str:
    path = _failure_log_path(run_name)
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(start_offset, size - 256_000, 0))
        return handle.read().decode("utf-8", errors="replace")


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def is_credit_failure(message: str, remaining: float, reserve: float) -> bool:
    """Classify only explicit credit/budget failures or an exhausted balance."""
    return bool(CREDIT_FAILURE_PATTERN.search(message)) or remaining <= reserve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--credit-limit", type=float, default=DEFAULT_CREDIT_LIMIT)
    parser.add_argument(
        "--reserve",
        type=float,
        default=0.25,
        help="Do not start new GPU work when estimated remaining credit is below this amount.",
    )
    parser.add_argument(
        "--estimated-cost",
        type=float,
        default=0.0,
        help="Optional expected run cost used to choose a profile before launch.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    namespace = parser.parse_args()
    if namespace.training_args[:1] == ["--"]:
        namespace.training_args = namespace.training_args[1:]
    return namespace


def main() -> int:
    args = parse_args()
    training_args = list(args.training_args)
    profiles = _deduplicate(args.profiles)

    if not training_args:
        raise ValueError("Pass modal_train.py arguments after --")
    if "--fresh" in training_args:
        raise ValueError(
            "Failover launches forbid --fresh. Use a unique --run-name so checkpoints "
            "can be resumed without deleting or resetting artifacts."
        )
    phase = _option_value(training_args, "--phase") or "sft"
    monitor_enabled = "--no-sft-monitor-kill-rate" not in training_args
    max_pairs_value = _option_value(training_args, "--max-pairs")
    if phase == "sft" and monitor_enabled and max_pairs_value:
        from config import training_config
        validate_bounded_sft_monitor_capacity(
            int(max_pairs_value),
            int(_option_value(training_args, "--sft-epochs") or training_config.sft_epochs),
            int(_option_value(training_args, "--sft-batch-size") or training_config.sft_batch_size),
            int(_option_value(training_args, "--sft-max-real-repeats") or 8),
            checkpoint_steps=training_config.sft_checkpoint_steps,
            minimum_checkpoints=int(
                _option_value(training_args, "--sft-min-monitor-checkpoints")
                or training_config.sft_min_monitor_checkpoints
            ),
        )
    run_name = _option_value(training_args, "--run-name")
    if not run_name or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("A safe explicit --run-name is required for failover training")
    if args.credit_limit <= 0 or args.reserve < 0 or args.estimated_cost < 0:
        raise ValueError("Credit limit, reserve, and estimated cost must be non-negative")
    if not profiles:
        raise ValueError(
            "Pass one or more Modal profile names with --profiles, or set "
            "ONEIROS_MODAL_PROFILES to a comma-separated list."
        )

    profile_state: list[dict[str, float | str]] = []
    for profile in profiles:
        usage = monthly_usage(profile)
        profile_state.append(
            {
                "profile": profile,
                "usage": usage,
                "remaining": args.credit_limit - usage,
            }
        )

    required_credit = args.estimated_cost + args.reserve
    eligible = [state for state in profile_state if state["remaining"] >= required_credit]
    if not eligible:
        balances = ", ".join(
            f"{state['profile']}=${state['remaining']:.2f}" for state in profile_state
        )
        raise RuntimeError(
            f"No Modal profile has the requested ${required_credit:.2f} allowance; {balances}"
        )

    initial_profile = eligible[0]
    fallback_profiles = [
        state for state in profile_state
        if state is not initial_profile and state["remaining"] >= args.reserve
    ]
    attempt_order = [initial_profile, *fallback_profiles]

    print("Modal failover preflight:")
    for state in profile_state:
        print(
            f"  {state['profile']}: used ${state['usage']:.2f}; "
            f"estimated remaining ${state['remaining']:.2f}"
        )
    print(f"  selected first profile: {initial_profile['profile']}")
    if args.dry_run:
        print("Dry run only; no Modal app was launched.")
        return 0

    for state in attempt_order:
        profile = str(state["profile"])
        print(f"\nLaunching {run_name} with Modal profile {profile}...")
        log_path = _failure_log_path(run_name)
        log_start = log_path.stat().st_size if log_path.exists() else 0
        completed = subprocess.run(
            [sys.executable, "scripts/modal_train.py", *training_args],
            env=_profile_environment(profile),
            check=False,
        )
        if completed.returncode == 0:
            print(f"Training completed successfully on profile {profile}.")
            return 0

        failure_log = _read_failure_log(run_name, log_start)
        updated_usage = monthly_usage(profile)
        updated_remaining = args.credit_limit - updated_usage
        credit_failure = is_credit_failure(failure_log, updated_remaining, args.reserve)
        if not credit_failure:
            raise RuntimeError(
                f"Training failed on {profile}, but the failure is not a confirmed "
                "credit/budget exhaustion. Refusing to spend another account's credits."
            )

        checkpoint_dir = Path("checkpoints") / run_name
        if checkpoint_dir.exists():
            print(f"Recovered persisted checkpoint locally at {checkpoint_dir}.")
        else:
            print("No checkpoint had been committed yet; the next profile will start this run from zero.")
        print(f"Profile {profile} exhausted its usable credit; preparing the next profile.")

    raise RuntimeError(f"All eligible Modal profiles were exhausted for run {run_name}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
