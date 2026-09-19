"""Gate 1 analysis: did ordinary SFT improve, preserve or damage output prediction?

Reporting rules this script enforces, because the first pass got them wrong:

* **Per-requested accuracy is primary.** Every one of the primary items counts,
  including those an arm failed to answer. An arm that declines the hard items
  and answers the easy ones must not be rewarded for it.
* **Parseable-subset accuracy is diagnostic only.** It is computed on a
  self-selected subset whose size differs by arm, so it is printed in a clearly
  separate block and never used for the headline claim.
* **Unanswered items get worst-case bounds.** Accuracy is reported as an
  interval: every non-answer wrong (lower) to every non-answer right (upper).
* **Equivalence is tested, not inferred.** A non-significant difference is not
  evidence of equivalence. The paired difference gets a confidence interval and
  a TOST against a predeclared margin; equivalence is claimed only if the whole
  interval sits inside that margin.
* **Code sensitivity only where both conditions answered.** Two non-answers are
  not an "identical prediction", and counting them as one inflates the apparent
  insensitivity of whichever arm declined most.

Predeclared before the numbers were seen: MARGIN = 5 percentage points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

#: Practical-equivalence margin, in percentage points. Declared in this file
#: rather than chosen after looking at the interval.
MARGIN_PP = 5.0
Z95 = 1.959963985
Z90 = 1.644853627  # TOST at alpha=.05 uses a two-sided 90% CI.

ANSWERED = ("correct", "wrong_value")


def paired_difference(pairs, *, z=Z95):
    """Difference in paired proportions with a normal-approximation CI.

    ``pairs`` is a sequence of (a_correct, b_correct) booleans. Returns the
    difference b - a in percentage points, its CI, and the discordant counts.
    """
    n = len(pairs)
    n10 = sum(1 for a, b in pairs if b and not a)
    n01 = sum(1 for a, b in pairs if a and not b)
    diff = (n10 - n01) / n
    var = (n10 + n01 - (n10 - n01) ** 2 / n) / (n * n)
    half = z * math.sqrt(max(var, 0.0))
    return (100 * diff, 100 * (diff - half), 100 * (diff + half), n10, n01)


def tost(low, high, margin=MARGIN_PP):
    """TOST equivalence verdict from a 90% CI and predeclared margin."""
    if low > -margin and high < margin:
        return f"EQUIVALENT at alpha=.05 within +/-{margin:g}pp"
    return f"NOT EQUIVALENT at +/-{margin:g}pp"


def mcnemar_exact(n10: int, n01: int) -> float:
    """Two-sided exact McNemar p-value from discordant paired outcomes."""
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(n10, n01) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def validate_artifact(data: dict, split: dict) -> dict[str, dict[str, dict]]:
    """Refuse incomplete, clipped, mismatched, or differently-scoped evidence."""
    detail = data.get("detail", {})
    expected_keys = {
        f"{arm}::{condition}"
        for arm in ("base", "a431", "a431_sysprompt")
        for condition in ("TAP-ref", "TAP-mut")
    }
    if set(detail) != expected_keys:
        raise ValueError(
            f"artifact conditions mismatch: missing={sorted(expected_keys-set(detail))}, "
            f"extra={sorted(set(detail)-expected_keys)}"
        )
    indexed = {}
    canonical_ids = None
    for key, rows in detail.items():
        by_id = {}
        for row in rows:
            item_id = row.get("id")
            if item_id in by_id:
                raise ValueError(f"duplicate item ID in {key}: {item_id}")
            raw = row.get("raw")
            if not isinstance(raw, str):
                raise ValueError(f"missing full raw output in {key}/{item_id}")
            import hashlib
            if row.get("raw_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest():
                raise ValueError(f"raw hash mismatch in {key}/{item_id}")
            if row.get("raw_chars") != len(raw):
                raise ValueError(f"raw length mismatch in {key}/{item_id}")
            by_id[item_id] = row
        if canonical_ids is None:
            canonical_ids = set(by_id)
        elif set(by_id) != canonical_ids:
            raise ValueError(f"item IDs differ across conditions at {key}")
        indexed[key] = by_id
    pilot = set(split.get("pilot_ids", []))
    if not pilot <= (canonical_ids or set()):
        raise ValueError("pilot IDs are not a subset of the artifact IDs")
    if len(pilot) != split.get("pilot_n"):
        raise ValueError("pilot count does not match the split receipt")
    contract = data.get("run_contract", {})
    if contract.get("items_file_sha256") != split.get("items_file_sha256"):
        raise ValueError("artifact and pilot split bind different TAP item files")
    if data.get("status") != "complete":
        raise ValueError("TAP artifact is not marked complete")
    return indexed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="results/tap_adapter_compare.json")
    parser.add_argument("--split", default="results/tap_pilot_split.json")
    parser.add_argument("--out", default="results/tap_capacity_gate_analysis.json")
    args = parser.parse_args(argv)

    artifact_path = Path(args.artifact)
    split_path = Path(args.split)
    artifact_bytes = artifact_path.read_bytes()
    split_bytes = split_path.read_bytes()
    data = json.loads(artifact_bytes)
    split = json.loads(split_bytes)
    pilot = set(split["pilot_ids"])
    det = validate_artifact(data, split)
    primary = [i for i in det["base::TAP-ref"] if i not in pilot]
    arms = sorted({k.split("::")[0] for k in det})
    report = {
        "primary_n": len(primary),
        "pilot_n": len(pilot),
        "margin_pp": MARGIN_PP,
        "tost_alpha": 0.05,
        "inputs": {
            "artifact": str(artifact_path),
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "pilot_split": str(split_path),
            "pilot_split_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "analysis_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
        "arms": {},
    }

    def verdict(arm, cond, i):
        return det[f"{arm}::{cond}"][i]["verdict"]

    print(f"primary n={len(primary)}  (pilot {len(pilot)} excluded: design data)\n")
    print("=" * 78)
    print("PRIMARY: per-requested accuracy over every primary item")
    print("=" * 78)
    print(f"{'arm':18} {'cond':8} {'accuracy':>9} {'worst-case bounds':>22} {'unanswered':>11}")
    for arm in arms:
        for cond in ("TAP-ref", "TAP-mut"):
            key = f"{arm}::{cond}"
            if key not in det:
                continue
            ok = sum(1 for i in primary if verdict(arm, cond, i) == "correct")
            un = sum(1 for i in primary if verdict(arm, cond, i) not in ANSWERED)
            n = len(primary)
            lo, hi = ok / n, (ok + un) / n
            print(f"{arm:18} {cond:8} {ok/n:>8.1%} {f'[{lo:.1%}, {hi:.1%}]':>22} {un:>7} ({un/n:.0%})")
            report["arms"][key] = {"n": n, "correct": ok, "unanswered": un,
                                  "per_requested_accuracy": ok / n,
                                  "worst_case_lower": lo, "worst_case_upper": hi}

    print("\n" + "=" * 78)
    print("DESCRIPTIVE STRATA: per-requested accuracy (no cross-stratum selection)")
    print("=" * 78)
    report["strata"] = {}
    for benchmark in ("humaneval", "mbpp"):
        ids = [i for i in primary
               if det["base::TAP-ref"][i].get("benchmark") == benchmark]
        report["strata"][benchmark] = {"n": len(ids), "arms": {}}
        print(f"  {benchmark} n={len(ids)}")
        for arm in arms:
            for cond in ("TAP-ref", "TAP-mut"):
                key = f"{arm}::{cond}"
                ok = sum(1 for i in ids if verdict(arm, cond, i) == "correct")
                un = sum(1 for i in ids if verdict(arm, cond, i) not in ANSWERED)
                rate = ok / len(ids) if ids else None
                shown = f"{rate:6.1%}" if rate is not None else "   n/a"
                print(f"    {key:32} {shown}  unanswered={un}")
                report["strata"][benchmark]["arms"][key] = {
                    "correct": ok,
                    "unanswered": un,
                    "per_requested_accuracy": rate,
                }

    print("\n" + "=" * 78)
    print(f"EQUIVALENCE vs base, per-requested, all {len(primary)} primary items")
    print(f"predeclared margin: +/-{MARGIN_PP:g} percentage points")
    print("=" * 78)
    report["equivalence"] = {}
    for arm in arms:
        if arm == "base":
            continue
        for cond in ("TAP-ref", "TAP-mut"):
            if f"{arm}::{cond}" not in det:
                continue
            pairs = [(verdict("base", cond, i) == "correct",
                      verdict(arm, cond, i) == "correct") for i in primary]
            diff, lo95, hi95, n10, n01 = paired_difference(pairs, z=Z95)
            _, lo90, hi90, _, _ = paired_difference(pairs, z=Z90)
            call = tost(lo90, hi90)
            exact_p = mcnemar_exact(n10, n01)
            print(f"  {arm:18} {cond:8} {diff:+6.2f}pp  "
                  f"95% CI [{lo95:+6.2f}, {hi95:+6.2f}]  "
                  f"90% TOST CI [{lo90:+6.2f}, {hi90:+6.2f}]  "
                  f"(+{n10}/-{n01}) p={exact_p:.6g}  {call}")
            report["equivalence"][f"{arm}::{cond}"] = {
                "difference_pp": diff,
                "ci95_low_pp": lo95, "ci95_high_pp": hi95,
                "tost_ci90_low_pp": lo90, "tost_ci90_high_pp": hi90,
                "gained": n10, "lost": n01,
                "mcnemar_exact_p": exact_p, "verdict": call}

    print("\n" + "=" * 78)
    print("CODE SENSITIVITY: only items where BOTH conditions were answered")
    print("=" * 78)
    report["code_sensitivity"] = {}
    for arm in arms:
        if f"{arm}::TAP-mut" not in det:
            print(f"  {arm:18} NOT MEASURABLE - no TAP-mut arm")
            continue
        both = [i for i in primary
                if verdict(arm, "TAP-ref", i) in ANSWERED
                and verdict(arm, "TAP-mut", i) in ANSWERED]
        if not both:
            print(f"  {arm:18} no items answered in both conditions")
            continue
        same = sum(1 for i in both
                   if det[f"{arm}::TAP-ref"][i]["predicted"]
                   == det[f"{arm}::TAP-mut"][i]["predicted"])
        cr = sum(1 for i in both if verdict(arm, "TAP-ref", i) == "correct")
        cm = sum(1 for i in both if verdict(arm, "TAP-mut", i) == "correct")
        print(f"  {arm:18} n={len(both):3d}  identical {same/len(both):6.1%}"
              f"   ref-correct {cr/len(both):6.1%}  mut-correct {cm/len(both):6.1%}"
              f"   gap {100*(cr-cm)/len(both):+5.1f}pp")
        report["code_sensitivity"][arm] = {
            "n_both_answered": len(both), "identical_rate": same / len(both),
            "ref_correct": cr / len(both), "mut_correct": cm / len(both),
            "coverage_of_primary": len(both) / len(primary)}

    print("\n" + "=" * 78)
    print("DIAGNOSTIC ONLY: accuracy among parseable answers (self-selected subsets)")
    print("=" * 78)
    for arm in arms:
        for cond in ("TAP-ref", "TAP-mut"):
            key = f"{arm}::{cond}"
            if key not in det:
                continue
            answered = [i for i in primary if verdict(arm, cond, i) in ANSWERED]
            ok = sum(1 for i in answered if verdict(arm, cond, i) == "correct")
            share = len(answered) / len(primary)
            conditional = ok / len(answered) if answered else None
            shown = f"{conditional:6.1%}" if conditional is not None else "   n/a"
            print(f"  {arm:18} {cond:8} {shown} of {len(answered):3d} answered"
                  f"  ({share:.0%} of primary) - NOT comparable across arms")
            report["arms"][key]["parseable_subset_accuracy"] = conditional
            report["arms"][key]["parseable_subset_n"] = len(answered)

    report["label"] = ("diagnostic only - not model selection, not a "
                       "generalization or final-test result")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
