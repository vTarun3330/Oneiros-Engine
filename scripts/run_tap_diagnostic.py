"""Run the two-condition TAP diagnostic. Base model only, no training.

Asks the SAME question twice for every item:

    TAP-ref   correct code  + call  -> assertion   (the published ATLAS setting)
    TAP-mut   mutant + spec + call  -> assertion   (the Oneiros setting)

Ground truth is the reference implementation's actual return value, computed at
build time by execution. Scoring compares the asserted right-hand side to that
value using ``ast.literal_eval`` - no generated text is ever executed.

If TAP-ref scores well and TAP-mut does not, the bottleneck is the information
asymmetry of mutation testing rather than model capability. If both fail, that
explanation is wrong.

Greedy decoding, so the result is the model's single best answer and is
reproducible. Permitted splits only; writes no weights.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSERT_RE = re.compile(r"assert\s+(.+)", re.S)


def strip_markdown(text: str) -> str:
    """Remove the chat model's markdown wrapping before any parsing.

    The model emits ``**assert f(x) == 'json'**`` and fenced blocks. Left in,
    the trailing ``**`` rides along on the right-hand side and every literal
    fails to parse - which reads as a model failure and is not one. The first
    smoke run scored 23 of 40 as "non-literal" for exactly this reason.
    """
    text = re.sub(r"```[a-zA-Z]*", " ", text).replace("```", " ")
    return text.replace("**", " ").replace("`", " ")


def asserted_rhs(text: str):
    """The right-hand side of the first `assert ... == <rhs>` in the output."""
    m = ASSERT_RE.search(strip_markdown(text))
    if not m:
        return None
    body = m.group(1).splitlines()[0].strip().rstrip(",;")
    # split on the LAST top-level '==' so tuples/dicts on either side survive
    depth, split_at = 0, None
    i = 0
    while i < len(body) - 1:
        c = body[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and body[i:i + 2] == "==" and body[i + 2:i + 3] != "=":
            split_at = i
        i += 1
    if split_at is None:
        return None
    rhs = body[split_at + 2:].strip()
    # trailing punctuation the chat format leaves behind
    return rhs.strip().strip('*').strip().rstrip(',;').strip()


def values_match(rhs: str, expected_repr: str):
    """Compare by value where both sides are literals; never execute the model."""
    try:
        want = ast.literal_eval(expected_repr)
    except Exception:                                          # noqa: BLE001
        return "expected_not_literal"
    try:
        got = ast.literal_eval(rhs)
    except Exception:                                          # noqa: BLE001
        return "prediction_not_literal"
    try:
        if got == want:
            return "correct"
        if isinstance(got, float) and isinstance(want, float) and abs(got - want) < 1e-6:
            return "correct"
    except Exception:                                          # noqa: BLE001
        return "uncomparable"
    return "wrong_value"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="results/tap_train_items.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", default="results/tap_diagnostic_result.json")
    args = parser.parse_args(argv)

    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[:args.limit]
    print(f"TAP items: {len(items)}  ({Counter(i['benchmark'] for i in items)})")

    import torch
    from engine.generator import Phi3Generator
    from config import model_config
    from config.settings import immutable_revision_for

    name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    rev = immutable_revision_for(name)
    print(f"loading {name} @ {rev} (base only, no adapter)...", flush=True)
    gen = Phi3Generator(model_name=name, model_revision=rev,
                        attention_implementation="sdpa")
    gen.load_model()
    tok, model = gen.tokenizer, gen.model
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def complete(prompts, batch=8):
        out = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
                     for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=3072).to(model.device)
            with torch.inference_mode():
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                out.append(tok.decode(ids[j][enc.input_ids.shape[1]:],
                                      skip_special_tokens=True))
            if (i // batch) % 10 == 0:
                print(f"    {min(i+batch, len(prompts))}/{len(prompts)}", flush=True)
        return out

    results = {}
    started = time.time()
    for cond, key in (("TAP-ref", "prompt_ref"), ("TAP-mut", "prompt_mut")):
        print(f"\n--- {cond} ---", flush=True)
        outs = complete([it[key] for it in items])
        per_bench = defaultdict(Counter)
        detail = []
        for it, text in zip(items, outs):
            rhs = asserted_rhs(text)
            verdict = "no_assertion" if rhs is None else values_match(rhs, it["expected_repr"])
            per_bench[it["benchmark"]][verdict] += 1
            per_bench["ALL"][verdict] += 1
            detail.append({"id": it["id"], "benchmark": it["benchmark"],
                           "verdict": verdict, "expected": it["expected_repr"][:60],
                           "predicted": (rhs or "")[:60]})
        results[cond] = {"per_benchmark": {k: dict(v) for k, v in per_bench.items()},
                         "detail": detail}
        for bench in ("ALL", "humaneval", "mbpp"):
            c = per_bench.get(bench)
            if not c:
                continue
            n = sum(c.values())
            print(f"  {bench:10} n={n:4d}  correct={c['correct']:4d} "
                  f"({c['correct']/n:6.1%})  wrong={c['wrong_value']:4d}  "
                  f"no_assert={c['no_assertion']:3d}  non_literal={c['prediction_not_literal']:3d}")

    elapsed = round(time.time() - started, 1)
    summary = {
        "schema_version": "oneiros_tap_diagnostic_v1",
        "model": name, "model_revision": rev, "adapter": None,
        "decoding": "greedy", "max_new_tokens": args.max_new_tokens,
        "items_file": args.items, "items": len(items),
        "wall_time_seconds": elapsed,
        "weights_written": False, "training_performed": False,
        "conditions": {c: r["per_benchmark"] for c, r in results.items()},
        "label": "diagnostic only - not a model-performance or selection result",
    }
    Path(args.out).write_text(json.dumps(
        {**summary, "detail": {c: r["detail"] for c, r in results.items()}},
        indent=2) + "\n", encoding="utf-8")

    print(f"\n=== VERDICT ({elapsed}s) ===")
    for bench in ("ALL", "humaneval", "mbpp"):
        a = results["TAP-ref"]["per_benchmark"].get(bench)
        b = results["TAP-mut"]["per_benchmark"].get(bench)
        if not a or not b:
            continue
        na, nb = sum(a.values()), sum(b.values())
        print(f"  {bench:10} TAP-ref {a['correct']/na:6.1%}   "
              f"TAP-mut {b['correct']/nb:6.1%}   "
              f"gap {(a['correct']/na - b['correct']/nb):+.1%}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
