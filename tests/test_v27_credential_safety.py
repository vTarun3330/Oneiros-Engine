"""v2.7 credential safety with a FAKE token: never printed, logged, persisted or passed to
untrusted code; token exposure stops the run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts import v27_acquire as wa

FAKE = "gho_FAKEfaketokenFAKE0123456789abcdefABCD"


def fake_gh(tmp_path: Path) -> list:
    script = tmp_path / "fake_gh.py"
    script.write_text(
        "import sys\n"
        "a = sys.argv[1:]\n"
        "if a[:2] == ['auth', 'status']:\n"
        "    print('github.com\\n  Logged in to github.com account tester (store)')\n"
        f"    print('  - Token: {FAKE[:4]}***')\n"
        "elif a[:2] == ['auth', 'token']:\n"
        f"    print('{FAKE}')\n", encoding="utf-8")
    return [sys.executable, str(script)]


def child(tmp_path: Path, body: str) -> list:
    script = tmp_path / "child.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def config(tmp_path: Path) -> str:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"store": str(tmp_path / "store")}))
    return str(cfg)


def test_token_reaches_only_the_trusted_child_and_never_any_output(tmp_path, capsys,
                                                                   monkeypatch):
    monkeypatch.setattr(wa, "ROOT", tmp_path)
    marker = tmp_path / "child_saw.txt"
    want = hashlib.sha256(FAKE.encode()).hexdigest()       # the child never embeds the value
    kid = child(tmp_path, "import hashlib, os, pathlib\n"
                f"pathlib.Path(r'{marker}').write_text(str(hashlib.sha256(os.environ.get("
                f"'GITHUB_TOKEN', '').encode()).hexdigest() == '{want}'))\n"
                "print('journal entry ok')\n")
    log = tmp_path / "w.log"
    code = wa.main(["--config", config(tmp_path), "--log", str(log)], gh=fake_gh(tmp_path),
                   child=kid)
    out = capsys.readouterr()
    assert code == 0 and marker.read_text() == "True"
    for text in (out.out, out.err, log.read_text(encoding="utf-8")):
        assert FAKE not in text
    assert '"authenticated_requests": true' in out.out and '"account": "tester"' in out.out
    for p in tmp_path.rglob("*"):
        if p.is_file() and p.name != "fake_gh.py":
            assert FAKE not in p.read_text(encoding="utf-8", errors="ignore"), p


def test_a_child_that_prints_the_token_is_redacted_and_stopped(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(wa, "ROOT", tmp_path)
    kid = child(tmp_path, "import os\nprint('leak', os.environ['GITHUB_TOKEN'])\n"
                "raise SystemExit('failure with ' + os.environ['GITHUB_TOKEN'])\n")
    log = tmp_path / "w.log"
    code = wa.main(["--config", config(tmp_path), "--log", str(log)], gh=fake_gh(tmp_path),
                   child=kid)
    out = capsys.readouterr()
    assert code == 3
    assert FAKE not in out.out + out.err + log.read_text(encoding="utf-8")
    assert "[REDACTED]" in log.read_text(encoding="utf-8")


def test_untrusted_environment_has_no_credentials():
    base = {"PATH": "/bin", "GITHUB_TOKEN": FAKE, "GH_TOKEN": FAKE, "SSH_AUTH_SOCK": "/s",
            "GIT_ASKPASS": "x", "MY_API_KEY": "k", "AWS_SECRET_ACCESS_KEY": "s",
            "HOME": "/h", "GCM_INTERACTIVE": "1"}
    env = wa.untrusted_env(base)
    assert env == {"PATH": "/bin", "HOME": "/h"}
    assert FAKE not in json.dumps(env)


@pytest.mark.parametrize("text", [f"error {FAKE} here", "x github_pat_" + "A1b2" * 10,
                                  "ghs_" + "Z" * 36])
def test_token_shaped_strings_are_redacted(text):
    assert "[REDACTED]" in wa.redact(text) and "gh" not in wa.redact(text).split()[-1][:3]


def test_unauthenticated_gh_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(wa, "ROOT", tmp_path)
    bad = child(tmp_path, "import sys\nprint('not logged in'); sys.exit(1)\n")
    with pytest.raises(SystemExit, match="not authenticated"):
        wa.main(["--config", config(tmp_path)], gh=bad, child=bad)
