from __future__ import annotations

import unittest

from harness.corpus import official_evidence_verifies_pair
from harness.swebench_verified import build_repository_record
from scripts.modal_ingest_swebench_verified import (
    _read_files, _sandbox_write_text, _select_instance_ids,
)


class FakeFilesystem:
    def __init__(self) -> None:
        self.files = {"/testbed/pkg/module.py": "def f():\n    return 1\n"}
        self.writes: list[tuple[str, str]] = []

    def read_text(self, remote_path: str) -> str:
        return self.files[remote_path]

    def write_text(self, content: str, remote_path: str) -> None:
        self.writes.append((content, remote_path))


class FakeSandbox:
    def __init__(self) -> None:
        self.filesystem = FakeFilesystem()


class FakeRunner:
    def __init__(self) -> None:
        self.sandbox = FakeSandbox()


class SandboxFilesystemAdapterTests(unittest.TestCase):
    def test_explicit_swebench_resume_reuses_checkpoint_selection(self) -> None:
        ledger = {"selection": {"instance_ids": ["django__django-1", "sympy__sympy-1"]}}
        selected = _select_instance_ids(
            "resume", ledger, "", ["pallets__flask-1"],
            {"django__django-1", "sympy__sympy-1", "pallets__flask-1"},
        )
        self.assertEqual(selected, ["django__django-1", "sympy__sympy-1"])

    def test_structured_swebench_evidence_is_a_verified_pair(self) -> None:
        evidence = {
            "buggy_fail_verified": True,
            "fixed_pass_verified": True,
            "base_counts": {
                "f2p_success": 0, "f2p_failure": 1, "p2p_failure": 0,
            },
            "fixed_counts": {
                "f2p_success": 1, "f2p_failure": 0, "p2p_failure": 0,
            },
        }
        self.assertTrue(official_evidence_verifies_pair(evidence))

    def test_record_uses_canonical_repository_project(self) -> None:
        instance = {
            "instance_id": "matplotlib__matplotlib-1",
            "repo": "matplotlib/matplotlib",
            "base_commit": "abc123",
            "FAIL_TO_PASS": ["tests/test_bug.py::test_bug"],
            "PASS_TO_PASS": [],
            "patch": "",
            "test_patch": "",
        }
        verification = {"fixed_returncode": 0, "buggy_returncode": 1}
        tests = {"tests/test_bug.py": "def test_bug():\n    assert True\n"}
        record = build_repository_record(
            instance,
            {"module.py": "def value():\n    return 0\n"},
            {"module.py": "def value():\n    return 1\n"},
            tests,
            tests,
            verification,
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["provenance"]["project"], "matplotlib")

    def test_write_uses_new_filesystem_argument_order(self) -> None:
        sandbox = FakeSandbox()
        written = _sandbox_write_text(sandbox, "/tmp/patch.diff", "patch-data")
        self.assertTrue(written)
        self.assertEqual(
            sandbox.filesystem.writes,
            [("patch-data", "/tmp/patch.diff")],
        )

    def test_read_files_uses_filesystem_namespace(self) -> None:
        runner = FakeRunner()
        self.assertEqual(
            _read_files(runner, ["pkg/module.py", "missing.py"]),
            {"pkg/module.py": "def f():\n    return 1\n"},
        )


if __name__ == "__main__":
    unittest.main()
