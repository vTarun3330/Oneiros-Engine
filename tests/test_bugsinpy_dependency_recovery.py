from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from harness import bugsinpy_v2
from scripts.modal_ingest_bugsinpy_v3 import _select_task_ids


def completed(returncode: int, output: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode=returncode, stdout=output)


class DurableSelectionTests(unittest.TestCase):
    def test_explicit_resume_reuses_checkpoint_selection(self) -> None:
        ledger = {
            "selection": {
                "explicit_task_ids": ["bugsinpy::black::10", "bugsinpy::keras::1"],
            }
        }
        selected = _select_task_ids(
            "resume",
            ledger,
            ["bugsinpy::black::10", "bugsinpy::keras::1", "bugsinpy::pandas::1"],
            "",
            {"bugsinpy::black::10", "bugsinpy::keras::1", "bugsinpy::pandas::1"},
        )
        self.assertEqual(
            selected, ["bugsinpy::black::10", "bugsinpy::keras::1"],
        )


class RequirementAliasTests(unittest.TestCase):
    def test_historical_import_names_resolve_to_declared_distributions(self) -> None:
        requirements = [
            "attrs==19.3.0",
            "pyOpenSSL==19.1.0",
            "PyDispatcher==2.0.5",
            "requests-async==0.5.0",
        ]
        self.assertEqual(
            bugsinpy_v2._requirement_for_module("attr", requirements),
            "attrs==19.3.0",
        )
        self.assertEqual(
            bugsinpy_v2._requirement_for_module("OpenSSL", requirements),
            "pyOpenSSL==19.1.0",
        )
        self.assertEqual(
            bugsinpy_v2._requirement_for_module("pydispatch", requirements),
            "PyDispatcher==2.0.5",
        )
        self.assertEqual(
            bugsinpy_v2._requirement_for_module("requests_async", requirements),
            "requests-async==0.5.0",
        )
        self.assertEqual(
            bugsinpy_v2._requirement_for_module("past", requirements),
            "future",
        )

    def test_inherited_unittest_target_is_compacted_with_its_subclass(self) -> None:
        source = """
class BaseCase:
    def setUp(self):
        self.value = self.make_value()

    def make_value(self):
        return 1

    def _inherited_test(self):
        assert self.value == 2

class SelectedCase(BaseCase):
    setting = 2

    def make_value(self):
        return self.setting
"""
        fragment = bugsinpy_v2._test_fragment(
            source,
            ["SelectedCase", "_inherited_test"],
        )
        self.assertIsNotNone(fragment)
        self.assertIn("class BaseCase", fragment)
        self.assertIn("class SelectedCase(BaseCase)", fragment)
        self.assertIn("def _inherited_test", fragment)
        self.assertIn("def make_value", fragment)
        compile(fragment or "", "<test-fragment>", "exec")


class FailureRecoveryTests(unittest.TestCase):
    def run_recovery(
        self, failure: str, requirements: list[str], expected_requirement: str,
    ) -> None:
        evidence = {"installed_requirements": [], "failed_requirements": []}
        with patch.object(
            bugsinpy_v2,
            "_run",
            side_effect=[completed(4, failure), completed(0), completed(0)],
        ) as run:
            result = bugsinpy_v2._run_with_missing_dependency_install(
                ["python", "-m", "pytest"],
                Path("checkout"),
                {},
                Path("python"),
                requirements,
                evidence,
                60,
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn(expected_requirement, evidence["installed_requirements"])
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["python", "-m", "pip", "install", expected_requirement],
        )

    def test_recovers_declared_pytest_xdist(self) -> None:
        self.run_recovery(
            "error: unrecognized arguments: -n tests/test_model.py::test_model",
            ["pytest-xdist==1.32.0"],
            "pytest-xdist==1.32.0",
        )

    def test_recovers_declared_protobuf_pin(self) -> None:
        self.run_recovery(
            "TypeError: Descriptors cannot not be created directly.",
            ["protobuf==3.12.2"],
            "protobuf==3.12.2",
        )

    def test_recovers_mocker_fixture(self) -> None:
        self.run_recovery(
            "fixture 'mocker' not found",
            ["pytest-mock==3.1.0"],
            "pytest-mock==3.1.0",
        )

    def test_pins_pytest_for_removed_get_marker_api(self) -> None:
        self.run_recovery(
            "AttributeError: 'Function' object has no attribute 'get_marker'",
            ["pytest==5.4.2"],
            "pytest==3.10.1",
        )

    def test_repins_cryptography_for_historical_pyopenssl(self) -> None:
        self.run_recovery(
            "AttributeError: module 'lib' has no attribute X509_V_FLAG_NOTIFY_POLICY",
            ["cryptography==2.9.2", "pyOpenSSL==19.1.0"],
            "cryptography==2.9.2",
        )

    def test_scipy_release_candidate_has_stable_fallback(self) -> None:
        self.assertEqual(
            bugsinpy_v2._installation_candidates("scipy==1.5.0rc1"),
            ["scipy==1.5.0rc1", "scipy==1.5.0"],
        )

    def test_ft2font_import_failure_triggers_native_build(self) -> None:
        result = completed(
            4,
            "ImportError: cannot import name 'ft2font' from partially initialized module 'matplotlib'",
        )
        self.assertTrue(bugsinpy_v2._needs_local_extension_build(result))


if __name__ == "__main__":
    unittest.main()
