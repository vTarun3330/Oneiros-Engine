"""Build Test-Assert Pairs in two conditions, to locate the oracle bottleneck.

The question this exists to settle
----------------------------------
On the locked panel, 65.5% of the tuned arm's candidates fail against the
CORRECT implementation, while only 9.1% fail by probing an input where the
reference and the mutant agree. The model finds the bug and then misstates what
should happen there.

Published assertion-generation work (ATLAS, and the large-scale fine-tuning
study built on it) reports fine-tuning working well on a task that looks
similar. The difference we suspect is not capability but *information*: their
training pair is

    (CORRECT focal method, test prefix) -> assertion

where the answer is derivable from the input, and the correct code is present
at inference time too. Ours is effectively

    (MUTANT, thin specification, input) -> correct value

where the value lives only in a reference the model never sees.

So we build the same item in two conditions and compare:

    TAP-ref   correct code  + call  -> assertion        (their setting)
    TAP-mut   mutant + spec + call  -> assertion        (our setting)

If TAP-ref scores well and TAP-mut does not, the bottleneck is the information
asymmetry of mutation testing, not model capability, corpus quality or the
training recipe. If TAP-mut also fails, that explanation is wrong and the
problem is capability - which is equally worth knowing.

The ground truth is the reference implementation's actual return value on the
chosen call, computed here by execution. For the joinable subset it can also be
checked against the EvalPlus third-party reference; see
papers/common/data/evalplus_reference_agreement.json, which found 100%
agreement on HumanEval and 86.68% on MBPP.

CPU only. Permitted splits only; the consumed split is refused by name.
Generates nothing, loads no model, trains nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.evaluation_admission import (  # noqa: E402
    execution_mode_of, is_function_mode, refuse_refused_split,
)

CORPUS_VERSION = "v4_1_research_hardened_candidate"
PERMITTED_SPLITS = ("train", "ablation_dev", "val")

#: Refused by name. The final split was consumed on 2026-09-16; see
#: docs/SEALED_FINAL_INCIDENT.md. A TAP set is not a reason to reopen it.
REFUSED_SPLITS = {"test"}

#: Executed in a subprocess so a pathological reference cannot hang the build.
EVAL_HARNESS = '''
import json, sys
ns = {}
try:
    exec(SRC, ns)
except Exception as e:
    print(json.dumps({"status": "load_error", "err": type(e).__name__})); sys.exit()
if EP not in ns:
    print(json.dumps({"status": "missing_entrypoint"})); sys.exit()
try:
    value = eval(CALL, dict(ns))
except Exception as e:
    print(json.dumps({"status": "raised", "err": type(e).__name__})); sys.exit()
try:
    rendered = repr(value)
    round_trips = eval(rendered, {"__builtins__": __builtins__}) == value
except Exception:
    print(json.dumps({"status": "unrepresentable"})); sys.exit()
if not round_trips:
    print(json.dumps({"status": "no_round_trip"})); sys.exit()
print(json.dumps({"status": "ok", "value": rendered[:400]}))
'''


def extract_calls(record, entry_point, limit=4):
    """Balanced `entry_point(...)` expressions taken from the record's own tests."""
    calls = []
    for test in (record.get("tests") or []):
        code = test.get("code", "")
        for m in re.finditer(rf"\b{re.escape(entry_point)}\s*\(", code):
            start, depth = m.start(), 0
            for j in range(m.end() - 1, min(len(code), start + 800)):
                if code[j] == "(":
                    depth += 1
                elif code[j] == ")":
                    depth -= 1
                    if depth == 0:
                        expr = code[start:j + 1]
                        if expr not in calls:
                            calls.append(expr)
                        break
            if len(calls) >= limit:
                return calls
    return calls


def evaluate(source, entry_point, call, timeout=15):
    src = ("SRC=%r\nEP=%r\nCALL=%r\n" % (source, entry_point, call)) + EVAL_HARNESS
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=timeout)
        out = (proc.stdout or "").strip().splitlines()
        return json.loads(out[-1]) if out else {"status": "no_output"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception:                                          # noqa: BLE001
        return {"status": "harness_error"}
    finally:
        os.unlink(path)


PROMPT_REF = """### ASSERTION GENERATION TASK

You are given a Python function and one call to it. Write the single `assert`
statement that states what this call must evaluate to.

Return only the assert statement. No explanation, no test function, no imports.

### Function
{code}

### Call
{call}

### Assertion
"""

PROMPT_MUT = """### ASSERTION GENERATION TASK

You are given a Python function, a description of what it is intended to do,
and one call to it. The function shown may be defective. Write the single
`assert` statement that states what this call must evaluate to for a CORRECT
implementation.

Return only the assert statement. No explanation, no test function, no imports.

### Intended behaviour
{spec}

### Function (may be defective)
{code}

### Call
{call}

### Assertion
"""


def build(split, limit, calls_per_record):
    corpus = ROOT / "data" / "corpus" / CORPUS_VERSION
    splits = json.loads((corpus / "splits.json").read_text(encoding="utf-8"))
    records = json.loads((corpus / "records.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}

    items, stats = [], {"records_seen": 0, "no_entrypoint": 0, "no_calls": 0,
                        "ref_unusable": 0, "mutant_agrees": 0, "emitted": 0}

    seen_problem = set()
    for rid in splits[split]:
        if limit and stats["emitted"] >= limit:
            break
        rec = by_id.get(rid)
        if rec is None:
            continue
        # Mode comes from the shared admission helper, never from a local
        # re-reading of the field. Function-mode records leave execution_mode
        # ABSENT and default to function_assertion; an earlier version of this
        # loop compared the raw field to the literal and silently emitted
        # nothing. That is the same shape as the defect that consumed the final
        # split - treating an absent-by-default field as an exclusion.
        if not is_function_mode(execution_mode_of(rec)):
            continue
        # one record per underlying problem, so the set is not dominated by
        # many mutants of the same function.
        problem = re.sub(r"_mut_\d+$", "", rid)
        if problem in seen_problem:
            continue
        stats["records_seen"] += 1
        entry_point = rec.get("entry_point")
        if not entry_point:
            stats["no_entrypoint"] += 1
            continue
        calls = extract_calls(rec, entry_point, calls_per_record)
        if not calls:
            stats["no_calls"] += 1
            continue

        emitted_for_record = 0
        for call in calls:
            ref = evaluate(rec["reference_code"], entry_point, call)
            if ref.get("status") != "ok":
                stats["ref_unusable"] += 1
                continue
            mut = evaluate(rec["code_under_test"], entry_point, call)
            # A call the mutant answers identically cannot distinguish anything,
            # so it tells us nothing about the oracle under test.
            distinguishing = not (mut.get("status") == "ok"
                                  and mut.get("value") == ref["value"])
            if not distinguishing:
                stats["mutant_agrees"] += 1
                continue

            spec = (rec.get("specification") or "").strip() or "(no specification available)"
            items.append({
                "id": f"{rid}::{len(items)}",
                "record_id": rid,
                "split": split,
                "benchmark": "humaneval" if "humaneval" in rid else
                             ("mbpp" if "mbpp" in rid else "other"),
                "entry_point": entry_point,
                "call": call,
                # ground truth: what the reference actually returns
                "expected_repr": ref["value"],
                "mutant_status": mut.get("status"),
                "mutant_repr": mut.get("value"),
                "spec_chars": len(spec),
                "prompt_ref": PROMPT_REF.format(code=rec["reference_code"].strip(), call=call),
                "prompt_mut": PROMPT_MUT.format(spec=spec,
                                                code=rec["code_under_test"].strip(),
                                                call=call),
            })
            stats["emitted"] += 1
            emitted_for_record += 1
            if limit and stats["emitted"] >= limit:
                break
        if emitted_for_record:
            seen_problem.add(problem)

    return items, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="ablation_dev")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--calls-per-record", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.split in REFUSED_SPLITS:
        print(f"REFUSED: {args.split!r} was consumed on 2026-09-16 and may never "
              f"be read. See docs/SEALED_FINAL_INCIDENT.md")
        return 2
    if args.split not in PERMITTED_SPLITS:
        print(f"REFUSED: unknown split {args.split!r}; permitted: {PERMITTED_SPLITS}")
        return 2

    print(f"building TAP items from split {args.split!r} (CPU only, no model)...")
    items, stats = build(args.split, args.limit, args.calls_per_record)

    out = Path(args.out) if args.out else (
        ROOT / "results" / f"tap_{args.split}_items.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

    import collections
    bench = collections.Counter(i["benchmark"] for i in items)
    spec_by_bench = collections.defaultdict(list)
    for i in items:
        spec_by_bench[i["benchmark"]].append(i["spec_chars"])

    print(f"\nemitted {len(items)} TAP items -> {out.relative_to(ROOT)}")
    print(f"  by benchmark: {dict(bench)}")
    for b, lens in sorted(spec_by_bench.items()):
        lens.sort()
        print(f"    {b:10} median specification {lens[len(lens)//2]:5d} chars")
    print(f"  build stats: {stats}")
    print("\nEach item carries two prompts for the SAME call and the SAME ground truth:")
    print("  prompt_ref -> correct code + call   (the published setting)")
    print("  prompt_mut -> mutant + spec + call  (the Oneiros setting)")
    print("Scoring: does the asserted value equal expected_repr?")
    print("\nNo model was loaded. Running the two conditions requires GPU inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
