"""v2.7 credential-safe acquisition wrapper (trusted parent process).

    python scripts/v27_acquire.py --config docs/<acquisition config>.json [--log FILE]

1. resolves the absolute GitHub CLI executable (PATH, then the standard install location);
2. checks ``gh auth status`` (its output is never printed: only the account name is parsed);
3. captures ``gh auth token`` INTERNALLY (never echoed, logged, written or put in arguments);
4. passes it ONLY as ``GITHUB_TOKEN`` in the environment of the trusted acquisition child
   (scripts/run_repository_native_acquisition_pilot.py, read-only GitHub REST + public git);
5. redacts the token and any token-shaped string from every line the child prints, in the
   terminal and in the log;
6. drops the token from memory when the child exits; records only
   ``authenticated_requests: true`` and the account / host.
``untrusted_env`` builds the environment for repository code, official tests, native
environments and Atheris: every GitHub / GH / SSH / git-credential / token / secret variable is
removed. A token that ever appears unredacted in an output stops the run (exit 3).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_GH = Path(r"C:\Program Files\GitHub CLI\gh.exe")
RUNNER = "scripts/v27_runner.py"            # unchanged runner + line-split cache
TOKEN_SHAPES = re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
CREDENTIAL_PREFIXES = ("GITHUB_", "GH_", "SSH_", "GIT_ASKPASS", "GIT_CREDENTIAL",
                       "GCM_", "GIT_TERMINAL_PROMPT")
CREDENTIAL_WORDS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "API_KEY",
                    "AUTH")


class TokenExposure(RuntimeError):
    """A credential appeared unredacted in an output."""


def redact(text: str, secrets=()) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[REDACTED]")
    return TOKEN_SHAPES.sub("[REDACTED]", text)


def untrusted_env(base=None) -> dict:
    """Environment for untrusted repository code: no credential of any kind."""
    base = dict(os.environ if base is None else base)
    return {k: v for k, v in base.items()
            if not k.upper().startswith(CREDENTIAL_PREFIXES)
            and not any(w in k.upper() for w in CREDENTIAL_WORDS)}


def resolve_gh(explicit=None) -> list:
    if explicit:
        return list(explicit)
    found = shutil.which("gh")
    if found:
        return [found]
    if DEFAULT_GH.is_file():
        return [str(DEFAULT_GH)]
    raise SystemExit("REFUSED: GitHub CLI not found (not reinstalled automatically)")


def auth_account(gh: list) -> str:
    done = subprocess.run([*gh, "auth", "status", "--hostname", "github.com"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=untrusted_env_for_gh())
    text = done.stdout + done.stderr
    m = re.search(r"Logged in to github\.com account (\S+)", text)
    if done.returncode != 0 or not m:
        raise SystemExit("REFUSED: gh is not authenticated for github.com")
    return m.group(1)


def untrusted_env_for_gh() -> dict:
    """gh reads its own credential store; it needs no credential variable from us."""
    return untrusted_env()


def stalled_candidate(heartbeat: Path, timeout: float, now: float,
                      since: float = 0.0) -> dict | None:
    """The candidate the heartbeat has reported for longer than ``timeout`` seconds. A heartbeat
    written before ``since`` (the current child's start) belongs to an earlier process and is
    never a stall of this one."""
    try:
        beat = json.loads(heartbeat.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if beat.get("phase") != "candidate" or not beat.get("commit"):
        return None
    import calendar
    started = calendar.timegm(time.strptime(beat["utc"], "%Y-%m-%dT%H:%M:%SZ"))
    if started < since:
        return None
    return beat if now - started > timeout else None


def run_child(cmd: list, token: str, log_path: Path, heartbeat: Path | None = None,
              timeout: float | None = None) -> tuple:
    """-> (exit code, stalled heartbeat or None). A stall stops the child."""
    import threading
    env = untrusted_env()
    env["GITHUB_TOKEN"] = token
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"exposed": False}
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")

        def pump():
            for line in proc.stdout:
                clean = redact(line, [token])
                state["exposed"] |= clean != line and token in line
                sys.stdout.write(clean)
                log.write(clean)
                log.flush()
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        stalled = None
        child_start = time.time() - 1
        while proc.poll() is None:
            time.sleep(5)
            if heartbeat is not None and timeout is not None:
                stalled = stalled_candidate(heartbeat, timeout, time.time(), child_start)
                if stalled is not None:
                    proc.kill()
                    break
        code = proc.wait()
        reader.join(timeout=30)
    env.clear()
    if state["exposed"]:
        sys.stdout.write("[credential-safety] the child printed the token; it was redacted, "
                         "the run is stopped: revoke with gh auth logout and log in again\n")
        return 3, None
    return code, stalled


def record_timeout(journal_path: Path, beat: dict, timeout: float) -> str:
    """Journal a stalled candidate as the INFRASTRUCTURE exclusion ``evaluation_timeout``."""
    from harness.github_acquisition import Journal
    key = f"cand:{beat['repository']}@{beat['commit']}"
    journal = Journal(journal_path)
    if not journal.done(key):
        journal.record(key, {"repository": beat["repository"], "fixed_commit": beat["commit"],
                             "failure": "evaluation_timeout",
                             "detail": f"candidate evaluation exceeded {int(timeout)} s "
                                       "(v2.7 wrapper watchdog; infrastructure exclusion)"})
    return key


def main(argv=None, *, gh=None, child=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--candidate-timeout", type=float, default=1800.0,
                        help="seconds one candidate may stay in evaluation (frozen: 1800)")
    parser.add_argument("--max-restarts", type=int, default=25)
    args = parser.parse_args(argv)
    gh_cmd = resolve_gh(gh)
    account = auth_account(gh_cmd)
    got = subprocess.run([*gh_cmd, "auth", "token", "--hostname", "github.com"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         env=untrusted_env_for_gh())
    token = got.stdout.strip()
    if got.returncode != 0 or not token:
        raise SystemExit("REFUSED: no token from gh (value never printed)")
    cmd = child or [sys.executable, str(ROOT / RUNNER), "--config", args.config]
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    log = Path(args.log) if args.log else ROOT / config["store"] / "wrapper.log"
    print(json.dumps({"authenticated_requests": True, "account": account,
                      "host": "github.com", "config": args.config}))
    store = ROOT / config["store"]
    timeouts = []
    try:
        for _ in range(args.max_restarts + 1):
            code, stalled = run_child(cmd, token, log, store / "heartbeat.json",
                                      args.candidate_timeout)
            if stalled is None:
                break
            timeouts.append(record_timeout(store / "journal.jsonl", stalled,
                                           args.candidate_timeout))
            print(json.dumps({"watchdog": "evaluation_timeout", "candidate": timeouts[-1],
                              "restarting": True}), flush=True)
        else:
            code = 4
    finally:
        token = None                                    # noqa: F841 - drop the reference
    print(json.dumps({"child_exit": code, "evaluation_timeouts": timeouts}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
