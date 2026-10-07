"""v2.7 Phase 6: compact, redacted, COMMITTABLE evidence bundle for the frozen panel's
model-independent execution evidence, so the common-subset freeze reproduces from a clean clone.

Per panel target (all 34): native target reach (official difference-exposing tests entered the
target on both revisions; panel admission evidence), the model-free sandbox control (both
revisions, two repetitions), and the Atheris eligibility v2 probe (adapter plan, stage flags,
fuzz-input consumption, determinism, failure class/reason and every child's exit evidence).
Bulky raw artifacts stay git-ignored; each is bound here by SHA-256. Paths are scrubbed.

v3 (supersedes v2, which is preserved): every invocation child's STRUCTURED per-seed stage record
is kept (receiver_built, arguments_built, consumed_bytes, target_invoked, outcome kind, a
bounded outcome detail - e.g. the SystemExit code - and the full outcome's SHA-256), for each
revision and repetition, so eligibility can be recomputed from committed evidence alone.

    python scripts/v27_phase6_evidence_bundle.py --out results/<bundle>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import receipt_sanitize  # noqa: E402

D = "results/sft_root_cause/v27_confirmation"
PANEL = "results/sft_root_cause_v27_confirmation_panel_r4.json"
REACH = f"{D}/r4_native_reach.jsonl"
CONTROL = f"{D}/phase6/native_control.jsonl"
ELIG = f"{D}/phase6/atheris_eligibility_v2"
TAIL = 200
DETAIL = 80
SCHEMA = "oneiros_v27_phase6_evidence_bundle_v3"
CHILD_FIELDS = ("returncode", "signal", "timed_out", "stdout_sha256", "stderr_sha256")


def sha(rel) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def jsonl(rel) -> dict:
    rows = {}
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row.get("key") or row.get("target_id")] = row
    return rows


def child(evidence: dict | None) -> dict | None:
    if not evidence:
        return None
    proc = evidence["process"]
    return {**{k: proc.get(k) for k in CHILD_FIELDS},
            "normal_exit": evidence.get("normal_exit"),
            "stderr_tail": receipt_sanitize.scrub(proc.get("stderr_tail") or "")[-TAIL:]}


def seed_record(s: dict) -> dict:
    outcome = s.get("outcome") or ["none"]
    detail = outcome[1] if len(outcome) > 1 else None
    return {"receiver_built": s.get("receiver_built"), "arguments_built": s.get("arguments_built"),
            "consumed_bytes": s.get("consumed_bytes"), "target_invoked": s.get("target_invoked"),
            "outcome_kind": outcome[0],
            "outcome_detail": receipt_sanitize.scrub(json.dumps(detail))[:DETAIL],
            "outcome_sha256": hashlib.sha256(json.dumps(outcome, sort_keys=True)
                                             .encode()).hexdigest()}


def structured_runs(raw: dict | None) -> dict | None:
    """Per revision, per repetition: normal_exit and the per-seed structured stage records."""
    if not raw or not raw.get("invocations"):
        return None
    out = {}
    for label, runs in raw["invocations"].items():
        out[label] = [{"normal_exit": r.get("normal_exit"),
                       "returncode": (r.get("process") or {}).get("returncode"),
                       "seeds": [seed_record(s) for s in ((r.get("result") or {}).get("seeds")
                                                          or [])]
                       if r.get("normal_exit") else None} for r in runs]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists (bundles are never overwritten)")
    panel = json.loads((ROOT / PANEL).read_text(encoding="utf-8"))
    keys = sorted(t["target_id"] for t in panel["targets"])
    reach, control = jsonl(REACH), jsonl(CONTROL)
    contract = json.loads((ROOT / ELIG / "contract.json").read_text(encoding="utf-8"))
    if contract["results"]["sha256"] != sha(contract["results"]["path"]):
        raise SystemExit("REFUSED: eligibility v2 results changed since the probe")
    elig = jsonl(contract["results"]["path"])
    for name, rows in (("reach", reach), ("control", control), ("eligibility", elig)):
        missing = [k for k in keys if k not in rows]
        if missing:
            raise SystemExit(f"REFUSED: {name} evidence missing for {len(missing)} panel targets")
    sources = {PANEL: sha(PANEL), REACH: sha(REACH), CONTROL: sha(CONTROL),
               f"{ELIG}/contract.json": sha(f"{ELIG}/contract.json"),
               contract["results"]["path"]: contract["results"]["sha256"]}
    targets = []
    for key in keys:
        r, c, e = reach[key], control[key], elig[key]
        raw = None
        if e.get("evidence"):
            if sha(e["evidence"]["path"]) != e["evidence"]["sha256"]:
                raise SystemExit(f"REFUSED: per-target evidence changed for {key}")
            raw = json.loads((ROOT / e["evidence"]["path"]).read_text(encoding="utf-8"))
            sources[e["evidence"]["path"]] = e["evidence"]["sha256"]
        children = None
        if raw:
            children = {"probe": {l: child(raw.get("probe", {}).get(l)) for l in ("buggy", "fixed")}
                        if raw.get("probe") else None,
                        "instrumentation": child(raw.get("instrumentation")),
                        "invocations": {l: [child(x) for x in runs] for l, runs in
                                        (raw.get("invocations") or {}).items()} or None}
        targets.append({
            "target_id": key,
            "native_reach": {"reach": r["reach"], "entered": r.get("entered"),
                             "pytest_exit": r.get("exit")},
            "sandbox_control": {"passed": c["control_passed"], "deterministic": c["deterministic"],
                                "repetitions": c["repetitions"], "uids": c["uids"],
                                "evidence_sha256": c["evidence_sha256"]},
            "atheris_v2": {k: e.get(k) for k in (
                "probed", "adapter_covered", "eligible_for_fuzzing", "reason", "failure_class",
                "plan", "structural_fuzz_inputs", "receiver_built", "arguments_built",
                "target_invoked", "consumes_fuzz_input", "deterministic", "v5_terminating",
                "per_revision")},
            "atheris_v2_children": children,
            "atheris_v2_structured_runs": structured_runs(raw),
            "atheris_v2_evidence_sha256": (e.get("evidence") or {}).get("sha256"),
        })
    bundle = receipt_sanitize.scrub_json({
        "schema_version": SCHEMA,
        "panel": {"path": PANEL, "sha256": sha(PANEL), "targets": len(keys)},
        "eligibility_contract": {k: contract[k] for k in (
            "schema_version", "design_version", "harness_script_sha256", "probe_script_sha256",
            "inner_sha256", "seeds_sha256", "fuzzing", "model", "counts")},
        "raw_sources_sha256": dict(sorted(sources.items())),
        "redaction": "host paths scrubbed; stderr tails truncated to 200 characters; raw "
                     "artifacts remain git-ignored and are bound by SHA-256",
        "targets": targets})
    out.write_bytes((json.dumps(bundle, indent=1, sort_keys=True) + "\n").encode("utf-8"))  # LF
    problems = receipt_sanitize.check_file(out)
    if problems:
        out.unlink()
        raise SystemExit(f"REFUSED: unredacted material in bundle: {problems[:3]}")
    print(json.dumps({"targets": len(targets), "sha256": sha(args.out),
                      "bytes": out.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
