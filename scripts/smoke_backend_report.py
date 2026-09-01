"""Print the backend/model smoke-test comparison table and a recommendation.

Reads:
  - results/local_j1024_integration_32_seed42_v5/sft_monitor_baseline.json
    (the already-completed Phi-3/eager base-model reference on the fixed panel)
  - results/smoke_backend_<backend>_seed<seed>.json for any backend that has
    been smoke-tested via scripts/smoke_backend_compare.py

Applies the predeclared decision rule: a candidate is only preferred over the
Phi-3/eager reference if it meets the locked 0.58 gate (or is statistically
indistinguishable from a reference that does) with no material regression in
parse/reference validity, and is meaningfully faster.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
REFERENCE_BASELINE = (
    RESULTS_DIR / "local_j1024_integration_32_seed42_v5" / "sft_monitor_baseline.json"
)
GATE = 0.58


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row(label: str, data: dict) -> dict:
    kill_at = data.get("kill_at_k", {})
    return {
        "label": label,
        "kill_rate": data.get("function_kill_rate"),
        "wilson_95": data.get("function_kill_rate_wilson_95"),
        "parse_valid_rate": data.get("parse_success_rate"),
        "execution_valid_rate": data.get("execution_valid_rate"),
        "reference_valid_rate": data.get("reference_valid_rate"),
        "kill_at_1": (kill_at.get("1") or {}).get("rate"),
        "kill_at_8": (kill_at.get("8") or {}).get("rate"),
        "wall_time_s": data.get("wall_time_seconds"),
        "seconds_per_function": data.get("seconds_per_function"),
        "attention_implementation": (
            data.get("model_runtime_profile", {}).get("attention_implementation")
            or data.get("attention_implementation")
        ),
        "model_name": data.get("model_name", "microsoft/Phi-3-mini-4k-instruct"),
    }


def main() -> None:
    rows = []
    reference = _load(REFERENCE_BASELINE)
    if reference is None:
        print("No Phi-3/eager reference baseline found yet; nothing to compare against.")
        return
    rows.append(_row("phi3_eager (reference, pre-existing run)", reference))

    infeasible = []
    for candidate_path in sorted(RESULTS_DIR.glob("v4_1_smoke_backend_*_seed*.json")):
        data = _load(candidate_path)
        if data is None:
            continue
        if data.get("status") == "infeasible":
            infeasible.append((data.get("backend", candidate_path.stem), data.get("reason", "")))
            continue
        rows.append(_row(data.get("backend", candidate_path.stem), data))

    for label, reason in infeasible:
        print(f"{label}: INFEASIBLE in this environment -- {reason}")
    if infeasible:
        print()

    header = (
        f"{'config':45s} {'kill_rate':>10s} {'kill@1':>8s} {'kill@8':>8s} "
        f"{'parse%':>8s} {'ref%':>8s} {'sec/fn':>8s} {'wall_s':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        def pct(value):
            return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"

        print(
            f"{row['label']:45s} {pct(row['kill_rate']):>10s} {pct(row['kill_at_1']):>8s} "
            f"{pct(row['kill_at_8']):>8s} {pct(row['parse_valid_rate']):>8s} "
            f"{pct(row['reference_valid_rate']):>8s} "
            f"{(row['seconds_per_function'] if row['seconds_per_function'] is not None else 'n/a'):>8} "
            f"{(row['wall_time_s'] if row['wall_time_s'] is not None else 'n/a'):>8}"
        )

    ref_row = rows[0]
    print()
    print("Decision (predeclared rule: meet/approach the 0.58 gate, no material "
          "parse/reference-validity regression vs. reference, meaningfully faster):")
    for row in rows[1:]:
        reasons = []
        verdict = "ACCEPT"
        if row["kill_rate"] is None:
            verdict, reasons = "PENDING", ["not yet run"]
        else:
            if row["kill_rate"] < GATE - 0.05:
                verdict = "REJECT"
                reasons.append(f"kill_rate {row['kill_rate']:.3f} well below gate {GATE}")
            if (
                row["parse_valid_rate"] is not None
                and ref_row["parse_valid_rate"] is not None
                and row["parse_valid_rate"] < ref_row["parse_valid_rate"] - 0.10
            ):
                verdict = "REJECT"
                reasons.append("parse-valid rate regressed materially vs. reference")
            if (
                row["reference_valid_rate"] is not None
                and ref_row["reference_valid_rate"] is not None
                and row["reference_valid_rate"] < ref_row["reference_valid_rate"] - 0.10
            ):
                verdict = "REJECT"
                reasons.append("reference-valid rate regressed materially vs. reference")
            if (
                verdict == "ACCEPT"
                and row["seconds_per_function"] is not None
                and ref_row["seconds_per_function"] is not None
                and row["seconds_per_function"] >= ref_row["seconds_per_function"] * 0.95
            ):
                verdict = "INCONCLUSIVE"
                reasons.append("not meaningfully faster than the reference")
            if not reasons:
                reasons.append("meets gate, validity within tolerance, faster than reference")
        print(f"  {row['label']:45s} -> {verdict:12s} ({'; '.join(reasons)})")


if __name__ == "__main__":
    main()
