"""v2.5 SFT data path: exact (prompt, completion) pairs for arm C (protocol v2.5 B, addendum 1
section 5). Builds DATA ONLY: no model weights are touched.

- prompt: the frozen v2.5 model-visible prompt (harness.native_generated_test_prompt_v25),
  built from the permitted buggy-side view; synthetic targets use their stable train-only
  interface module ``oneiros_target``.
- completion: the complete verified ``pytest_module_v1`` module, byte for byte.
- admission: only rows whose verdict is a verified positive in BOTH repeatability runs (kill,
  fixed valid, rerun agrees, policy pass, conversion accepted) and that the frozen selection
  chose; pending, failed, duplicate or non-policy rows refuse with a recorded reason.
- tokens: chat-templated prompt, completion + EOS and combined counts with the frozen base
  tokenizer; a row over the prompt, completion or 3,072 sequence budget refuses (never
  truncated or compacted).
- mixture (deterministic, outcome-free): unique canonical test once; <= 3 per function and per
  lineage, <= 40 per repository; synthetic replay <= 1 per repository example and <= 60% of
  the corpus; no bug family above 35%; never repetition to reach a share.

    python scripts/v25_sft_data.py dry-run --repository-sources FILE [FILE...]
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "oneiros_v25_sft_pairs_v1"
KILLS = ("semantic_kill", "crash_kill")
SYNTHETIC_GROUP = "synthetic_mutation_functions"
INTERFACE_FILE = "oneiros_target.py"
CAPS = {"per_function": 3, "per_lineage": 3, "per_repository": 40, "family_share": 0.35,
        "synthetic_share": 0.60, "synthetic_per_repository_example": 1.0}
LIMITS = {"prompt_tokens": 2048, "completion_tokens": 1024,     # addendum 2 section 3
          "sequence_tokens": 3072}
GATE = {"repository_tests": 150, "repositories": 8, "lineages": 60}
R2 = "results/sft_root_cause/v25_corpus_stage1_r2"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# --- admission ------------------------------------------------------------------------------

def verdict_problems(runs: Sequence[Optional[Mapping[str, Any]]]) -> List[str]:
    """A verified positive in EVERY repeatability run, with identical verdicts."""
    if len(runs) < 2:
        return ["fewer than two repeatability runs"]
    if any(r is None for r in runs):
        return ["missing from a repeatability run"]
    problems = []
    first = runs[0]
    if any(dict(r) != dict(first) for r in runs[1:]):
        problems.append("verdicts differ between runs")
    if first.get("status") != "executed":
        problems.append(f"not executed ({first.get('status')})")
    if first.get("class") not in KILLS:
        problems.append(f"not a kill ({first.get('class')})")
    if first.get("fixed_valid") is not True:
        problems.append("fixed revision not valid")
    if first.get("rerun_agrees") is not True:
        problems.append("rerun does not agree")
    if first.get("accepted") is not True:
        problems.append("verifier did not accept")
    return problems


def admit(candidate: Mapping[str, Any], runs: Sequence[Optional[Mapping[str, Any]]],
          selected: bool) -> List[str]:
    problems = []
    conversion = candidate.get("conversion") or {}
    if conversion.get("accepted") is not True:
        problems.append("conversion rejected / not policy-valid")
    if not selected:
        problems.append("not chosen by the frozen selection")
    problems += verdict_problems(runs)
    for r in runs:
        if r is not None and r.get("module_sha256") != conversion.get("module_sha256"):
            problems.append("verdict is for a different module")
            break
    return problems


# --- prompts --------------------------------------------------------------------------------

def synthetic_dto(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {"target_key": candidate["id"], "repository": f"synthetic/{candidate['dataset']}",
            "buggy_commit": "synthetic", "target_file": INTERFACE_FILE,
            "qualname": candidate["entry_point"]}


def make_pair(*, pair_id: str, dto: Mapping[str, Any], buggy_source: str, completion: str,
              identity: Mapping[str, Any]) -> Dict[str, Any]:
    from harness.native_generated_test_prompt_v25 import build_prompt
    built = build_prompt(dto, buggy_source)
    return {"pair_id": pair_id, "prompt": built["prompt"], "prompt_sha256": built["prompt_sha256"],
            "builder_version": built["builder_version"], "view_sha256": built["view_sha256"],
            "completion": completion, "completion_sha256": sha_text(completion),
            **dict(identity)}


def token_counter(base_model: str, revision: str):
    """(tokenizer, count) with the frozen base tokenizer; CPU, no model loaded."""
    import transformers
    from engine.test_generation_prompt import format_chat_prompt
    tok = transformers.AutoTokenizer.from_pretrained(base_model, revision=revision,
                                                     local_files_only=True)

    def count(pair: Mapping[str, Any]) -> Dict[str, int]:
        p = len(tok(format_chat_prompt(tok, pair["prompt"]), add_special_tokens=False)
                ["input_ids"])
        c = len(tok(pair["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"])
        return {"prompt": p, "completion": c, "combined": p + c}
    return tok, count


def fit_problems(tokens: Mapping[str, int]) -> List[str]:
    out = []
    if tokens["prompt"] > LIMITS["prompt_tokens"]:
        out.append(f"prompt {tokens['prompt']} > {LIMITS['prompt_tokens']}")
    if tokens["completion"] > LIMITS["completion_tokens"]:
        out.append(f"completion {tokens['completion']} > {LIMITS['completion_tokens']}")
    if tokens["combined"] > LIMITS["sequence_tokens"]:
        out.append(f"sequence {tokens['combined']} > {LIMITS['sequence_tokens']}")
    return out


# --- mixture --------------------------------------------------------------------------------

def _order(p: Mapping[str, Any]) -> Tuple:
    return (p.get("rank", 0), p["canonical_test"])


def select_mixture(repository: Sequence[Mapping[str, Any]],
                   synthetic: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Deterministic and outcome-free: depends only on identities and the frozen caps."""
    chosen, dropped = [], []
    seen, per_fn, per_lin, per_repo = set(), Counter(), Counter(), Counter()
    for p in sorted(repository, key=_order):
        why = ("duplicate canonical test" if p["canonical_test"] in seen else
               "function cap" if per_fn[p["function"]] >= CAPS["per_function"] else
               "lineage cap" if per_lin[p["lineage"]] >= CAPS["per_lineage"] else
               "repository cap" if per_repo[p["repository"]] >= CAPS["per_repository"] else None)
        if why:
            dropped.append({"pair_id": p["pair_id"], "reason": why})
            continue
        seen.add(p["canonical_test"])
        per_fn[p["function"]] += 1
        per_lin[p["lineage"]] += 1
        per_repo[p["repository"]] += 1
        chosen.append(p)
    n_repo = len(chosen)
    budget = min(int(n_repo * CAPS["synthetic_per_repository_example"]),
                 int(CAPS["synthetic_share"] / (1 - CAPS["synthetic_share"]) * n_repo))
    syn = []
    for p in sorted(synthetic, key=_order):
        why = ("duplicate canonical test" if p["canonical_test"] in seen else
               "function cap" if per_fn[p["function"]] >= CAPS["per_function"] else
               "synthetic replay budget" if len(syn) >= budget else None)
        if why:
            dropped.append({"pair_id": p["pair_id"], "reason": why})
            continue
        seen.add(p["canonical_test"])
        per_fn[p["function"]] += 1
        syn.append(p)
    mixture = chosen + syn
    # family cap: drop the latest-ordered synthetic rows of an over-cap family, then repository
    while mixture:
        share = family_share(mixture)
        over = [f for f, v in share.items() if v > CAPS["family_share"]]
        if not over:
            break
        victims = [p for p in reversed(mixture) if over[0] in p["families"]]
        victim = next((p for p in victims if p["source_group"] == SYNTHETIC_GROUP), victims[0])
        mixture.remove(victim)
        dropped.append({"pair_id": victim["pair_id"], "reason": f"family cap ({over[0]})"})
    # removals may have lowered the repository count: re-apply the replay budget
    n_repo = sum(p["source_group"] != SYNTHETIC_GROUP for p in mixture)
    budget = min(int(n_repo * CAPS["synthetic_per_repository_example"]),
                 int(CAPS["synthetic_share"] / (1 - CAPS["synthetic_share"]) * n_repo))
    for victim in [p for p in mixture if p["source_group"] == SYNTHETIC_GROUP][budget:]:
        mixture.remove(victim)
        dropped.append({"pair_id": victim["pair_id"], "reason": "synthetic replay budget"})
    return {"pairs": mixture, "dropped": dropped}


def family_share(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    weight = Counter()
    for p in pairs:
        for f in p["families"]:
            weight[f] += 1 / len(p["families"])
    return {f: round(w / len(pairs), 4) for f, w in weight.most_common()} if pairs else {}


def mixture_report(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    repo = [p for p in pairs if p["source_group"] != SYNTHETIC_GROUP]
    syn = [p for p in pairs if p["source_group"] == SYNTHETIC_GROUP]
    canon = [p["canonical_test"] for p in pairs]
    report = {
        "pairs": len(pairs), "repository_pairs": len(repo), "synthetic_pairs": len(syn),
        "repositories": len({p["repository"] for p in repo}),
        "repository_lineages": len({p["lineage"] for p in repo}),
        "max_per_function": max(Counter(p["function"] for p in pairs).values(), default=0),
        "max_per_lineage": max(Counter(p["lineage"] for p in repo).values(), default=0),
        "max_per_repository": max(Counter(p["repository"] for p in repo).values(), default=0),
        "duplicate_canonical_tests": len(canon) - len(set(canon)),
        "family_share": family_share(pairs),
        "synthetic_share": round(len(syn) / len(pairs), 4) if pairs else 0.0,
        "by_complexity": dict(Counter(p["complexity"] for p in pairs)),
        "by_repository": dict(Counter(p["repository"] for p in repo))}
    gate = {"repository_tests": len(repo) >= GATE["repository_tests"],
            "repositories": report["repositories"] >= GATE["repositories"],
            "lineages": report["repository_lineages"] >= GATE["lineages"]}
    caps = {"per_function": report["max_per_function"] <= CAPS["per_function"],
            "per_lineage": report["max_per_lineage"] <= CAPS["per_lineage"],
            "per_repository": report["max_per_repository"] <= CAPS["per_repository"],
            "family_share": all(v <= CAPS["family_share"] for v in
                                report["family_share"].values()),
            "synthetic_share": report["synthetic_share"] <= CAPS["synthetic_share"],
            "replay_ratio": len(syn) <= len(repo) * CAPS["synthetic_per_repository_example"],
            "no_duplicates": report["duplicate_canonical_tests"] == 0}
    return {**report, "training_gate": gate, "training_gate_passed": all(gate.values()),
            "caps": caps, "caps_passed": all(caps.values())}


# --- sources --------------------------------------------------------------------------------

def synthetic_pairs(directory: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Selected verified synthetic rows -> pairs; every other row refused with a reason."""
    cands = {c["index"]: c for c in jsonl(directory / "converted_candidates.jsonl")}
    runs = [{v["index"]: v for v in jsonl(p)}
            for p in sorted(directory.glob("verification_run[0-9].jsonl"))]
    selection = jsonl(directory / "selection_view_synthetic.jsonl")
    chosen = {}
    for s in selection:
        # one representative row per selected canonical test: the lowest-index row among the
        # mutants the selection says it KILLS (a sibling row of the same text may time out)
        killed = set(s["mutants_killed"])
        rows = sorted((c for c in cands.values()
                       if c["group_id"] == s["function"] and c["id"] in killed
                       and (c["conversion"].get("identities") or {}).get("canonical_ast")
                       == s["canonical_ast"]), key=lambda c: c["index"])
        if rows:
            chosen[rows[0]["index"]] = s
    pairs, refused = [], []
    for index, c in sorted(cands.items()):
        if c["execution_mode"] != "function_assertion":
            continue
        verdicts = [r.get(index) for r in runs]
        problems = admit(c, verdicts, index in chosen)
        if problems:
            if index in chosen or not any("selection" in p for p in problems):
                refused.append({"index": index, "id": c["id"], "reasons": problems})
            continue
        s = chosen[index]
        pairs.append(make_pair(
            pair_id=f"syn:{index}", dto=synthetic_dto(c), buggy_source=c["code_under_test"],
            completion=c["conversion"]["module"],
            identity={"source_group": SYNTHETIC_GROUP, "dataset": c["dataset"],
                      "repository": f"synthetic/{c['dataset']}", "lineage": c["group_id"],
                      "function": c["group_id"], "canonical_test": s["canonical_ast"],
                      "rank": s["rank"], "families": s["families"],
                      "complexity": c["complexity"], "candidate_index": index,
                      "candidate_id": c["id"],
                      "module_sha256": c["conversion"]["module_sha256"]}))
    return pairs, refused


def repository_pairs(sources: Iterable[Mapping[str, Any]],
                     runs: Sequence[Mapping[int, Mapping[str, Any]]],
                     cands: Mapping[int, Mapping[str, Any]]) -> Tuple[List, List]:
    """``sources``: rows from the WSL prompt-source export (dto + buggy source per verified
    repository fragment). Verdicts must be positives in every run."""
    pairs, refused = [], []
    for s in sources:
        c = cands[s["index"]]
        problems = admit(c, [r.get(s["index"]) for r in runs], True)
        if problems:
            refused.append({"index": s["index"], "id": c["id"], "reasons": problems})
            continue
        dto = s["dto"]
        pairs.append(make_pair(
            pair_id=f"repo:{s['index']}", dto=dto, buggy_source=s["buggy_source"],
            completion=c["conversion"]["module"],
            identity={"source_group": c.get("dataset") or "repository",
                      "dataset": c.get("dataset"), "repository": s["project"],
                      "lineage": s["lineage"],
                      "function": f"{s['project']}:{dto['target_file']}:{dto['qualname']}",
                      "canonical_test": c["conversion"]["identities"]["canonical_ast"],
                      "rank": 0, "families": [c.get("bug_family") or "unknown"],
                      "complexity": c.get("complexity") or "unknown",
                      "candidate_index": s["index"], "candidate_id": c["id"],
                      "module_sha256": c["conversion"]["module_sha256"]}))
    return pairs, refused


def to_datapoints(pairs: Sequence[Mapping[str, Any]]):
    from engine.sft_trainer import SFTDataPoint
    from engine.sft_trainer_v25 import PYTEST_MODULE_TASK_KIND
    return [SFTDataPoint(prompt=p["prompt"], completion=p["completion"],
                         function_id=p["pair_id"], project=p["repository"],
                         bug_family=(p["families"] or ["unknown"])[0],
                         semantic_group=p["function"],
                         execution_mode=("function_assertion"
                                         if p["source_group"] == SYNTHETIC_GROUP
                                         else "repository_pytest_module"),
                         dataset=str(p.get("dataset")),
                         dataset_family=f"{p['source_group']}::{p.get('dataset')}",
                         task_kind=PYTEST_MODULE_TASK_KIND) for p in pairs]


def dry_run(repository_source_files: Sequence[str], verify_dirs: Sequence[str],
            out_dir: str) -> Dict[str, Any]:
    from engine.sft_trainer_v25 import plan_v25_exposure
    from scripts.native_generated_tests_generate import CONTRACT
    from scripts.v25_converted_corpus import install_audit
    install_audit()
    directory = ROOT / R2
    syn, syn_refused = synthetic_pairs(directory)
    cands = {c["index"]: c for c in jsonl(directory / "converted_candidates.jsonl")}
    repo_runs = []
    for d in verify_dirs:
        files = sorted((ROOT / d).glob("repository_verification_run[0-9].jsonl"))
        repo_runs.append([{v["index"]: v for v in jsonl(f)} for f in files])
    # each directory verifies only the projects whose environments it was given; a row it
    # lists as "no_environment_row" belongs to another directory and never overrides a verdict
    merged = [{k: v for runs in repo_runs for k, v in runs[i].items()
               if v.get("status") != "no_environment_row"} for i in range(2)] \
        if repo_runs else [{}, {}]
    sources = [r for f in repository_source_files for r in jsonl(ROOT / f)]
    repo, repo_refused = repository_pairs(sources, merged, cands)
    _, count = token_counter(CONTRACT["base_model"], CONTRACT["base_revision"])
    fit_refused = []
    for p in syn + repo:
        p["tokens"] = count(p)
    syn_ok = [p for p in syn if not fit_problems(p["tokens"])]
    repo_ok = [p for p in repo if not fit_problems(p["tokens"])]
    fit_refused = [{"pair_id": p["pair_id"], "reasons": fit_problems(p["tokens"])}
                   for p in syn + repo if fit_problems(p["tokens"])]
    mix = select_mixture(repo_ok, syn_ok)
    report = mixture_report(mix["pairs"])
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs_dry_run.jsonl"
    pairs_path.write_bytes(("\n".join(json.dumps(p, sort_keys=True) for p in mix["pairs"])
                            + "\n").encode("utf-8"))
    tokens = [p["tokens"] for p in mix["pairs"]]
    return {
        "schema_version": SCHEMA, "status": "DRY RUN: data only, no model weights touched",
        "pairs_file": {"path": f"{out_dir}/pairs_dry_run.jsonl", "sha256": sha_file(pairs_path)},
        "candidate_pool": {"synthetic_admitted": len(syn), "synthetic_refused": len(syn_refused),
                           "repository_admitted": len(repo),
                           "repository_refused": repo_refused,
                           "synthetic_refusal_reasons": Counter(
                               r for x in syn_refused for r in x["reasons"]),
                           "token_fit_refused": fit_refused},
        "mixture": report, "dropped_by_mixture": Counter(d["reason"] for d in mix["dropped"]),
        "tokens": {k: {"sum": sum(t[k] for t in tokens), "max": max((t[k] for t in tokens),
                                                                     default=0)}
                   for k in ("prompt", "completion", "combined")},
        "supervised_target_tokens": sum(t["completion"] for t in tokens),
        "exposure_plan_addendum2": (plan_v25_exposure(len(tokens),
                                                      sum(t["completion"] for t in tokens))
                                    if tokens else None),
        "limits": LIMITS, "caps": CAPS, "gate": GATE,
        "train_ready": report["training_gate_passed"] and report["caps_passed"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--repository-sources", nargs="*", default=[])
    d.add_argument("--verify-dirs", nargs="*", default=[])
    d.add_argument("--out-dir", required=True)
    d.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    from scripts.native_rehearsal_rebuild_v22 import publish_once
    result = dry_run(args.repository_sources, args.verify_dirs, args.out_dir)
    status = publish_once(args.receipt, result)
    print(json.dumps({"status": status, "mixture": {k: result["mixture"][k] for k in (
        "pairs", "repository_pairs", "synthetic_pairs", "repositories", "repository_lineages",
        "training_gate_passed", "caps_passed")}, "tokens": result["tokens"],
        "train_ready": result["train_ready"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
