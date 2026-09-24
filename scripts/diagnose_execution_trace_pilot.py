"""Post-gate, train-only diagnosis of the execution-order (trace) pilot.

Downstream of the frozen gate, like ``diagnose_execution_supervision_pilot``:
it cannot change the gate, open confirmation identifiers, or read validation,
ablation_dev, test or sealed-final data. It refuses to run unless the frozen
analysis recorded a failed gate, and every artifact it reads is hash-checked
against that analysis.

Three arms are compared pairwise on the same 97-item train-derived panel:
the earlier prompt-only control, the unordered-event arm and the ordered-trace
arm. Beyond transition matrices and format classes it asks one mechanistic
question the frozen gate does not: on items where the shown (mutated) code and
the intended behaviour disagree, does a model's answer track the value the
shown code actually computes, the intended value, or neither?

Everything below the gate is exploratory. Two interval methods are added as a
post-hoc sensitivity check on the frozen paired Wald interval, which collapses
to a point when a comparison has no discordant pairs:

* Newcombe's hybrid score interval for paired proportions (method 10);
* an exact one-sided Clopper-Pearson bound on the gain rate alone. A paired
  difference can never exceed the rate of items that switched to correct, so
  this bound is conservative whatever the losses were.

Neither replaces the frozen verdict.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_execution_supervision_pilot import _format_class, _index, _sha  # noqa: E402

CONDITIONS = ("intended_output", "shown_actual_output")
ARMS = ("control", "unordered_events", "ordered_trace")
PAIRS = (("control", "unordered_events"), ("control", "ordered_trace"),
         ("unordered_events", "ordered_trace"))
Z90 = 1.644853627
#: Mirrors the frozen analysis constant; used only to phrase conclusions.
MIN_GAIN_PP = 5.0

ANSWER_CLASSES = {
    "correct": "correct",
    "wrong_value": "wrong_answer",
    "wrong_type": "wrong_answer",
}


def answer_class(verdict: str) -> str:
    """Collapse scorer verdicts to correct / wrong answer / no usable answer."""
    return ANSWER_CLASSES.get(str(verdict), "no_usable_answer")


def _value(literal: Any) -> tuple[bool, Any]:
    """Parse a stored literal; (False, text) when it is not a Python literal."""
    text = literal.get("literal") if isinstance(literal, dict) else literal
    if text is None:
        return False, None
    try:
        return True, ast.literal_eval(str(text))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, str(text).strip()


def same_value(predicted: Any, reference: Any) -> bool:
    """Value equality with the type check the scorer applies."""
    ok_p, left = _value(predicted)
    ok_r, right = _value(reference)
    if left is None or right is None:
        return False
    if ok_p != ok_r:
        return False
    try:
        return type(left) is type(right) and left == right
    except Exception:  # noqa: BLE001 - exotic __eq__ on a parsed literal
        return False


def track(predicted: Any, intended: Any, actual: Any) -> str:
    """Which behaviour an answer reproduces, on an item where the two differ."""
    if predicted is None:
        return "no_usable_answer"
    hit_intended = same_value(predicted, intended)
    hit_actual = same_value(predicted, actual)
    if hit_intended and not hit_actual:
        return "intended"
    if hit_actual and not hit_intended:
        return "shown_code_actual"
    if hit_actual and hit_intended:  # cannot happen when the two differ
        return "both"
    return "neither"


def wilson(successes: int, n: int, z: float = Z90) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def newcombe_paired(pairs: list[tuple[bool, bool]], z: float = Z90) -> tuple[float, float]:
    """Newcombe (1998) method 10 for the paired difference b - a, in pp."""
    n = len(pairs)
    both = sum(1 for a, b in pairs if a and b)
    only_a = sum(1 for a, b in pairs if a and not b)
    only_b = sum(1 for a, b in pairs if b and not a)
    neither = n - both - only_a - only_b
    p1, p2 = (both + only_a) / n, (both + only_b) / n
    l1, u1 = wilson(both + only_a, n, z)
    l2, u2 = wilson(both + only_b, n, z)
    product = (both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither)
    phi = 0.0 if product == 0 else (both * neither - only_a * only_b) / math.sqrt(product)
    theta = p2 - p1
    lower = theta - math.sqrt(max(
        (p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1), 0.0))
    upper = theta + math.sqrt(max(
        (u2 - p2) ** 2 + (p1 - l1) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1), 0.0))
    return 100 * lower, 100 * upper


def exact_gain_upper(gains: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound on the gain rate, in pp."""
    if gains >= n:
        return 100.0
    from scipy.stats import beta
    return 100 * float(beta.ppf(1 - alpha, gains + 1, n - gains))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--unordered-events", type=Path, required=True)
    parser.add_argument("--ordered-trace", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True,
                        help="pilot_development.execution.json the panel was drawn from")
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis.get("mechanism_gate_passed") is not False:
        raise SystemExit("REFUSED: this diagnosis is only for a failed frozen gate")
    paths = {"control": args.control, "unordered_events": args.unordered_events,
             "ordered_trace": args.ordered_trace}
    for arm, path in paths.items():
        if analysis["inputs"][arm]["sha256"] != _sha(path):
            raise SystemExit(f"REFUSED: {arm} artifact is not the one the frozen analysis read")
    data = {arm: json.loads(path.read_text(encoding="utf-8")) for arm, path in paths.items()}
    pilot_hash = _sha(args.pilot)
    for arm, artifact in data.items():
        if artifact.get("pilot_development_sha256") != pilot_hash:
            raise SystemExit(f"REFUSED: {arm} is not bound to the supplied pilot")
    pilot = _index(json.loads(args.pilot.read_text(encoding="utf-8")))
    if any(item.get("evaluation_split") != "train" for item in pilot.values()):
        raise SystemExit("REFUSED: mechanism panel is not train-only")
    rows = {arm: {condition: _index(data[arm]["detail"][condition])
                  for condition in CONDITIONS} for arm in ARMS}
    ids = sorted(pilot)
    for arm in ARMS:
        for condition in CONDITIONS:
            if sorted(rows[arm][condition]) != ids:
                raise SystemExit(f"REFUSED: panel mismatch for {arm}/{condition}")
    n = len(ids)
    differing = [i for i in ids if pilot[i]["execution_evidence"]["differs"]]

    # --- per-arm answer and format profile ------------------------------------
    profile: dict[str, Any] = {}
    for arm in ARMS:
        profile[arm] = {}
        for condition in CONDITIONS:
            table = rows[arm][condition]
            profile[arm][condition] = {
                "lenient_answer_classes": dict(sorted(Counter(
                    answer_class(table[i]["lenient"]["verdict"]) for i in ids).items())),
                "lenient_verdicts": dict(sorted(Counter(
                    str(table[i]["lenient"]["verdict"]) for i in ids).items())),
                "strict_verdicts": dict(sorted(Counter(
                    str(table[i]["strict"]["verdict"]) for i in ids).items())),
                "format_classes": dict(sorted(Counter(
                    _format_class(str(table[i]["raw"])) for i in ids).items())),
                "completion_limit_hits": sum(
                    bool(table[i]["hit_completion_limit"]) for i in ids),
                "median_raw_chars": sorted(table[i]["raw_chars"] for i in ids)[n // 2],
            }

    # --- which behaviour the answers reproduce, where the two differ ----------
    tracking: dict[str, Any] = {"items_where_shown_differs_from_intended": len(differing)}
    for arm in ARMS:
        tracking[arm] = {}
        for condition in CONDITIONS:
            table = rows[arm][condition]
            counts = Counter()
            for record_id in differing:
                row = table[record_id]
                predicted = (row["lenient"].get("predicted_literal")
                             if answer_class(row["lenient"]["verdict"]) != "no_usable_answer"
                             else None)
                evidence = pilot[record_id]["execution_evidence"]
                counts[track(predicted, evidence["intended"], evidence["actual"])] += 1
            tracking[arm][condition] = dict(sorted(counts.items()))

    # --- pairwise transitions, identity and interval sensitivity --------------
    pairwise: dict[str, Any] = {}
    for left, right in PAIRS:
        key = f"{left} -> {right}"
        pairwise[key] = {}
        for condition in CONDITIONS:
            a, b = rows[left][condition], rows[right][condition]
            strict = Counter()
            lenient = Counter()
            identical = 0
            gained, lost = [], []
            for record_id in ids:
                la, lb = a[record_id], b[record_id]
                strict[(str(la["strict"]["verdict"]), str(lb["strict"]["verdict"]))] += 1
                lenient[(str(la["lenient"]["verdict"]), str(lb["lenient"]["verdict"]))] += 1
                identical += int(la["raw_sha256"] == lb["raw_sha256"])
                was = la["lenient"]["verdict"] == "correct"
                now = lb["lenient"]["verdict"] == "correct"
                if now and not was:
                    gained.append(record_id)
                elif was and not now:
                    lost.append(record_id)
            pairs = [(a[i]["lenient"]["verdict"] == "correct",
                      b[i]["lenient"]["verdict"] == "correct") for i in ids]
            difference = 100 * (len(gained) - len(lost)) / n
            wald_half = 100 * Z90 * math.sqrt(max(
                (len(gained) + len(lost) - (len(gained) - len(lost)) ** 2 / n) / (n * n), 0.0))
            newcombe = newcombe_paired(pairs)
            gain_bound = exact_gain_upper(len(gained), n)
            pairwise[key][condition] = {
                "raw_output_identical": identical,
                "raw_output_identical_rate": identical / n,
                "lenient_gained_ids": gained,
                "lenient_lost_ids": lost,
                "discordant_pairs": len(gained) + len(lost),
                "difference_pp": difference,
                "frozen_wald_ci90_pp": [difference - wald_half, difference + wald_half],
                "wald_interval_degenerate": len(gained) + len(lost) == 0,
                "posthoc_newcombe_ci90_pp": list(newcombe),
                "posthoc_exact_gain_rate_upper95_one_sided_pp": gain_bound,
                "min_gain_excluded_by": {
                    "frozen_wald": difference + wald_half < MIN_GAIN_PP,
                    "newcombe": newcombe[1] < MIN_GAIN_PP,
                    "exact_gain_rate_bound": gain_bound < MIN_GAIN_PP,
                },
                "strict_transitions": {f"{x} -> {y}": c for (x, y), c in sorted(strict.items())},
                "lenient_transitions": {f"{x} -> {y}": c for (x, y), c in sorted(lenient.items())},
            }

    # --- descriptive strata (exploratory; pre-existing panel metadata only) ---
    strata: dict[str, Any] = defaultdict(lambda: {"n": 0, **{arm: 0 for arm in ARMS}})
    for record_id in ids:
        item = pilot[record_id]
        for dimension, value in (
            ("source", item["source_dataset"]),
            ("complexity", item["complexity_tier"]),
            ("bug_family", item["bug_family"]),
            ("shown_differs_from_intended", str(bool(item["execution_evidence"]["differs"])).lower()),
        ):
            bucket = strata[f"{dimension}::{value}"]
            bucket["n"] += 1
            for arm in ARMS:
                bucket[arm] += int(
                    rows[arm]["intended_output"][record_id]["lenient"]["verdict"] == "correct")

    manifest = json.loads(args.trace_manifest.read_text(encoding="utf-8"))
    excluded = {key: value["intended_output"]["min_gain_excluded_by"]
                for key, value in pairwise.items()}
    shown_excluded = {key: value["shown_actual_output"]["min_gain_excluded_by"]
                      for key, value in pairwise.items()}
    report = {
        "schema_version": "oneiros_execution_trace_failure_diagnosis_v1",
        "label": ("post-gate train-only diagnosis; exploratory below the frozen gate; "
                  "no confirmation/validation/ablation_dev/test/sealed access"),
        "frozen_gate_passed": False,
        "frozen_gate_unchanged": True,
        "inputs": {
            "analysis_sha256": _sha(args.analysis),
            **{f"{arm}_sha256": _sha(path) for arm, path in paths.items()},
            "pilot_development_sha256": pilot_hash,
            "trace_manifest_sha256": _sha(args.trace_manifest),
        },
        "panel": {"items": n, "train_only": True,
                  "source_counts": dict(sorted(Counter(
                      pilot[i]["source_dataset"] for i in ids).items())),
                  "complexity_counts": dict(sorted(Counter(
                      pilot[i]["complexity_tier"] for i in ids).items()))},
        "training_intervention": {
            "examples_per_arm": manifest.get("examples_per_arm"),
            "execution_examples_per_arm": manifest.get("replacement_examples"),
            "shared_canonical_examples": manifest.get("shared_canonical_examples"),
            "execution_share": (manifest.get("replacement_examples") or 0)
            / max(manifest.get("examples_per_arm") or 1, 1),
            "replacement_source_counts": manifest.get("replacement_source_counts"),
            "token_report": manifest.get("token_report"),
            "arms_differ_only_in": "temporal order of an identical execution-event multiset",
        },
        "arm_profiles": profile,
        "answer_tracking_where_shown_differs": tracking,
        "pairwise": pairwise,
        "strata_intended_lenient_correct_exploratory": dict(sorted(strata.items())),
        "gate_mechanics_findings": [
            ("The frozen paired Wald interval collapses to [0, 0] when a comparison has "
             "no discordant pairs, so a '90% lower bound >= 0' check passes vacuously. "
             "unordered_events/intended_output passed that single check this way; it is "
             "not evidence of a direction."),
            ("Future frozen analyses should report a non-degenerate interval (Newcombe "
             "or exact) alongside Wald; this recommendation does not alter the present gate."),
        ],
        "min_gain_exclusion_intended_output": excluded,
        "min_gain_exclusion_shown_actual_output": shown_excluded,
        "sealed_final_test_accessed": False,
        "validation_accessed": False,
        "ablation_dev_accessed": False,
        "confirmation_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, not text mode: on Windows text mode writes CRLF while
    # .gitattributes stores *.json as LF, so a hash taken here would stop
    # matching the committed file after a fresh checkout.
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))

    print(f"panel n={n}; shown differs from intended on {len(differing)}")
    for key, value in pairwise.items():
        for condition in CONDITIONS:
            v = value[condition]
            print(f"  {key:36} {condition:20} identical {v['raw_output_identical_rate']:5.1%}"
                  f"  +{len(v['lenient_gained_ids'])}/-{len(v['lenient_lost_ids'])}"
                  f"  Wald90 [{v['frozen_wald_ci90_pp'][0]:+.2f},{v['frozen_wald_ci90_pp'][1]:+.2f}]"
                  f"  Newcombe90 [{v['posthoc_newcombe_ci90_pp'][0]:+.2f},{v['posthoc_newcombe_ci90_pp'][1]:+.2f}]"
                  f"  exact-gain<= {v['posthoc_exact_gain_rate_upper95_one_sided_pp']:.2f}")
    print("\nanswer tracking where shown code differs from intended:")
    for arm in ARMS:
        for condition in CONDITIONS:
            print(f"  {arm:17} {condition:20} {tracking[arm][condition]}")
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
