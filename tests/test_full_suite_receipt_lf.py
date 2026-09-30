"""The full-suite receipt is written as platform-independent LF bytes (amendment v2.4 J)."""
from __future__ import annotations

import json
import subprocess

from scripts import run_full_suite_receipt as suite

RECEIPT = {"schema_version": "oneiros_native_full_suite_receipt_v2", "exit": 0, "passed": 3,
           "failed": 0, "tail": "line one\nline two\r\nC:\\Users\\someone\\x.py"}


def test_written_bytes_are_valid_json_lf_only_with_one_final_newline(tmp_path):
    out = tmp_path / "nested" / "full_suite.json"
    written = suite.write_receipt(out, dict(RECEIPT))
    data = out.read_bytes()
    assert b"\r" not in data.replace(b"\\r", b"")             # no CR byte (escaped \r is text)
    assert data.endswith(b"}\n") and not data.endswith(b"\n\n")
    assert json.loads(data.decode("utf-8")) == written
    assert "someone" not in data.decode("utf-8")              # scrubbed before writing
    assert data == suite.receipt_bytes(written)               # reproducible from content


def test_checkout_reproduces_the_generated_bytes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                    cwd=repo, capture_output=True, text=True, check=True)
    run("init", "-q")
    run("config", "core.autocrlf", "true")                    # the Windows default here
    (repo / ".gitattributes").write_text("*.json text eol=lf\n")
    out = repo / "results" / "full_suite.json"
    suite.write_receipt(out, dict(RECEIPT))
    generated = out.read_bytes()
    run("add", "-A")
    run("commit", "-q", "-m", "receipt")
    out.unlink()
    run("checkout", "--", "results/full_suite.json")
    assert out.read_bytes() == generated
    blob = subprocess.run(["git", "show", "HEAD:results/full_suite.json"], cwd=repo,
                          capture_output=True, check=True).stdout
    assert blob == generated
