"""v2.7 Phase 4 gate: validate a finished confirmation acquisition WITHOUT modifying it.

    python scripts/v27_phase4_gate.py --config docs/<acquisition config>.json --run-id <durable run>
        [--launch-commit SHA] --out results/<gate receipt>.json --ledger results/<ledger>.json

Read-only over the store, the durable run directory and the published report. Every condition is
reported; the gate PASSES only if all hold. Nothing here is outcome-dependent: it never looks at
which candidates were admitted except to count them for the funnel.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import v27_roles as roles  # noqa: E402
from harness.acquisition_receipt import validate_receipt  # noqa: E402
from scripts.run_repository_native_acquisition_pilot import load_list, status_of  # noqa: E402

TOKEN = re.compile(rb"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                   rb"[Aa]uthorization:\s*(token|[Bb]earer)\s+\S{8,})")
SOURCES = ("scripts/run_repository_native_acquisition_pilot.py", "scripts/v27_acquire.py",
           "scripts/v27_runner.py", "harness/github_acquisition.py",
           "harness/acquisition_receipt.py", "harness/repository_isolation.py",
           "harness/v27_roles.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def token_scan(paths) -> dict:
    hits, scanned = [], 0
    for p in paths:
        if p.is_file():
            scanned += 1
            if TOKEN.search(p.read_bytes()):
                hits.append(p.relative_to(ROOT).as_posix())
    return {"files_scanned": scanned, "files_with_credential_shaped_material": hits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--launch-commit", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", required=True)
    args = ap.parse_args(argv)
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    store = ROOT / cfg["store"]
    journal_path = store / "journal.jsonl"
    raw_lines = journal_path.read_text(encoding="utf-8").splitlines()
    entries, bad_lines = [], 0
    for line in raw_lines:
        try:
            entries.append(json.loads(line))
        except ValueError:
            bad_lines += 1
    checks, failed = {}, []

    def check(name, ok, detail):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            failed.append(name)

    # -- frozen inputs and source binding --------------------------------------------------
    listed = load_list(ROOT / cfg["repositories_file"])
    check("repository_list_hash_matches_config",
          sha(ROOT / cfg["repositories_file"]) == cfg["expected_repository_list_sha256"],
          {"list": cfg["repositories_file"], "repositories": len(listed)})
    launch = args.launch_commit
    src = {}
    for rel in SOURCES + (args.config,):
        now = sha(ROOT / rel)
        at_launch = None
        if launch:
            blob = subprocess.run(["git", "show", f"{launch}:{rel}"], cwd=ROOT,
                                  capture_output=True).stdout
            at_launch = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
        now_norm = hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        src[rel] = {"sha256": now, "lf_normalised_sha256": now_norm,
                    "matches_launch_commit": None if launch is None else now_norm == at_launch}
    check("source_and_config_match_launch_contract",
          launch is not None and all(v["matches_launch_commit"] for v in src.values()),
          {"launch_commit": launch, "files": src})

    # -- journal integrity ----------------------------------------------------------------
    keys = [e.get("key") for e in entries]
    dup = sorted(k for k, n in Counter(keys).items() if n > 1)
    check("journal_parses_and_is_append_only_unique",
          bad_lines == 0 and not dup and all(keys),
          {"lines": len(raw_lines), "unparseable": bad_lines, "duplicate_keys": dup[:20],
           "sha256": sha(journal_path)})

    # -- audit sessions -------------------------------------------------------------------
    starts = {k.split(":", 1)[1] for k in keys if k.startswith("audit_start:")}
    ends = {k.split(":", 1)[1]: e for e, k in zip(entries, keys) if k.startswith("audit_end:")}
    check("every_audit_session_closed", starts and starts == set(ends),
          {"started": sorted(starts), "closed": sorted(ends),
           "incomplete": sorted(starts - set(ends))})
    work = [e for e in entries if not e["key"].startswith(("audit_start:", "audit_end:"))]
    unsnap = [e["key"] for e in work if not e.get("audit_snapshot")]
    sessions_seen = {(e.get("audit_snapshot") or {}).get("scope_start_utc") for e in work}
    session_starts = {s.rsplit(":", 1)[0] for s in starts}
    check("no_unaudited_interval",
          not unsnap and sessions_seen <= session_starts,
          {"records_without_snapshot": unsnap[:20],
           "snapshot_scopes": sorted(x for x in sessions_seen if x),
           "session_scopes": sorted(session_starts)})
    accesses = sum(len((e.get("audit_snapshot") or {}).get("protected_accesses") or [])
                   for e in work)
    accesses += sum(len(e["evidence"].get("protected_accesses") or []) for e in ends.values())
    check("zero_protected_access", accesses == 0,
          {"protected_accesses": accesses, "snapshots": len(work) - len(unsnap),
           "opens_checked_final": [e["evidence"].get("opens_checked") for e in ends.values()]})

    # -- published report ------------------------------------------------------------------
    report_path = ROOT / cfg["report"]
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    problems = validate_receipt(report) if report else ["report not published"]
    check("report_published_and_valid", report and not problems,
          {"report": cfg["report"], "sha256": sha(report_path) if report else None,
           "problems": problems})
    check("report_binds_this_journal",
          (report.get("identity") or {}).get("journal_sha256") == sha(journal_path),
          {"report_journal_sha256": (report.get("identity") or {}).get("journal_sha256")})
    check("report_gate_no_protected_data_access",
          (report.get("gate") or {}).get("no_protected_data_access") is True,
          {"gate": report.get("gate")})

    # -- all 70 accounted; ledger ----------------------------------------------------------
    by_key = {roles.repository_key(r): r for r in listed}
    repo_entries = {roles.repository_key(e["repository"]) if e.get("repository") else
                    roles.repository_key(e["key"].split(":", 1)[1]): e
                    for e in work if e["key"].startswith("repo:")}
    scans = {roles.repository_key(e["key"].split(":", 1)[1]): e
             for e in work if e["key"].startswith("scan:")}
    cands = defaultdict(list)
    for e in work:
        if e["key"].startswith("cand:"):
            cands[roles.repository_key(e["repository"])].append(e)
    ledger, unaccounted, unreasoned = [], [], []
    funnel = Counter()
    for k, name in sorted(by_key.items()):
        r = repo_entries.get(k)
        row = {"repository": name}
        if r is None:
            unaccounted.append(name)
            row["stage"] = "unaccounted"
        elif r.get("failure") or r.get("status") == "not_found":
            row.update(stage="repository_unavailable", reason=r.get("failure") or "not_found")
        elif r.get("screen_problems"):
            row.update(stage="screen_excluded", reason=r["screen_problems"])
        else:
            s = scans.get(k)
            if s is None:
                row.update(stage="no_scan_record")
                unaccounted.append(name)
            else:
                row.update(stage="scanned", commits_scanned=s.get("commits_scanned"),
                           selected=len(s.get("selected") or []), scan_failure=s.get("failure"))
                statuses = Counter()
                for c in cands.get(k, []):
                    st = status_of(c)
                    if not st or st == "?":
                        unreasoned.append(c["key"])
                    statuses[st] += 1
                row["candidates"] = dict(statuses)
                row["admitted"] = statuses.get("ADMITTED", 0)
                funnel["candidates_evaluated"] += sum(statuses.values())
                funnel["admitted"] += row["admitted"]
                funnel["repositories_with_admitted"] += bool(row["admitted"])
        funnel["stage:" + row["stage"]] += 1
        ledger.append(row)
    stray = sorted(set(repo_entries) - set(by_key))
    check("all_frozen_repositories_accounted",
          len(ledger) == len(listed) == 70 and not unaccounted and not stray,
          {"frozen": len(listed), "accounted": len(ledger) - len(unaccounted),
           "unaccounted": unaccounted, "not_in_frozen_list": stray})
    check("every_exclusion_has_a_machine_readable_reason", not unreasoned,
          {"candidates_without_reason": unreasoned[:20]})

    # -- credentials, durable run, protected splits ----------------------------------------
    run_dir = ROOT / "runs" / args.run_id
    scan_paths = [p for p in list(store.glob("*")) + list(run_dir.rglob("*"))
                  if p.is_file() and p.suffix in ("", ".log", ".jsonl", ".json", ".txt", ".tmp")]
    scan_paths.append(report_path)
    ts = token_scan(scan_paths)
    check("no_credential_material_in_outputs", not ts["files_with_credential_shaped_material"],
          ts)
    run_status = {}
    for name in ("status.json", "manifest.json"):
        p = run_dir / name
        if p.is_file():
            run_status[name] = json.loads(p.read_text(encoding="utf-8"))
    argv_text = json.dumps(run_status)
    check("no_credential_in_command_arguments", not TOKEN.search(argv_text.encode()),
          {"checked": sorted(run_status)})
    check("validation_and_sealed_final_untouched", accesses == 0,
          {"basis": "protected-access monitor (oneiros_protected_locations_v2) covers locked "
                    "validation and sealed-final locations; zero recorded accesses"})

    result = {
        "schema_version": "oneiros_v27_phase4_gate_v1",
        "config": {"path": args.config, "sha256": sha(cfg_path)},
        "run_id": args.run_id,
        "status": "PASS" if not failed else "FAIL",
        "failed_conditions": failed,
        "checks": checks,
        "funnel": dict(funnel),
        "ledger": {"path": args.ledger},
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
    }
    ledger_doc = {"schema_version": "oneiros_v27_acquisition_exclusion_ledger_v1",
                  "config": args.config, "journal_sha256": sha(journal_path),
                  "repositories": ledger}
    Path(ROOT / args.ledger).write_text(json.dumps(ledger_doc, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")
    result["ledger"]["sha256"] = sha(ROOT / args.ledger)
    Path(ROOT / args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed": failed, "funnel": result["funnel"]},
                     indent=1))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
