"""View rule v2: a shipped runtime ``<pkg>/testing`` package is kept; every test file, test
directory, conftest and official test stays out of the view."""
from __future__ import annotations

from pathlib import Path

from scripts import native_generated_tests_execute_wsl as ex
from scripts import v25_view_rule as vr


def _tree(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


BASE = {"pkg/__init__.py": "", "pkg/core.py": "X = 1\n",
        "pkg/testing/__init__.py": "", "pkg/testing/pytest.py": "def raises(): pass\n",
        "pkg/testing/tests/__init__.py": "", "pkg/testing/tests/test_pytest.py": "def test(): 0\n",
        "pkg/testing/test_runner.py": "def test(): 0\n", "pkg/testing/conftest.py": "",
        "pkg/tests/__init__.py": "", "pkg/tests/test_core.py": "def test(): 0\n",
        "pkg/sub/__init__.py": "", "pkg/sub/core_test.py": "def test(): 0\n",
        "testing/__init__.py": "", "conftest.py": ""}


def test_a_literally_shipped_testing_package_is_kept_without_its_tests(tmp_path):
    co = _tree(tmp_path / "co", {**BASE, "setup.py": "packages=['pkg', 'pkg.testing']\n"})
    out = vr.build_view_v2(co, "", tmp_path / "view")
    view = tmp_path / "view"
    assert out["kept_testing_packages"] == {"pkg.testing": {"evidence": "literal",
                                                            "metadata": ["setup.py"]}}
    assert (view / "pkg/testing/pytest.py").is_file()
    for rel in ("pkg/testing/tests/test_pytest.py", "pkg/testing/test_runner.py",
                "pkg/testing/conftest.py", "pkg/tests/test_core.py", "pkg/sub/core_test.py",
                "testing/__init__.py", "conftest.py"):
        assert not (view / rel).exists(), rel


def test_find_packages_ships_it_unless_excluded(tmp_path):
    co = _tree(tmp_path / "a", {**BASE, "setup.py": "packages=find_packages()\n"})
    assert "pkg.testing" in vr.shipped_testing_packages(co, "")
    co = _tree(tmp_path / "b", {**BASE, "setup.py":
                                "packages=find_packages(exclude=['pkg.testing'])\n"})
    assert vr.shipped_testing_packages(co, "") == {}


def test_without_shipping_evidence_the_view_equals_the_frozen_rule(tmp_path):
    co = _tree(tmp_path / "co", {**BASE, "setup.py": "packages=['pkg']\n"})
    new = vr.build_view_v2(co, "", tmp_path / "v2")
    old = ex.build_view(co, "", tmp_path / "v1")
    assert new["kept_testing_packages"] == {}
    assert new["manifest_sha256"] == old["manifest_sha256"]
    assert not (tmp_path / "v2/pkg/testing").exists()


def test_official_test_files_are_proven_absent(tmp_path):
    co = _tree(tmp_path / "co", {**BASE, "setup.py": "packages=['pkg', 'pkg.testing']\n"})
    vr.build_view_v2(co, "", tmp_path / "view")
    assert vr.official_files_absent(tmp_path / "view", "", [
        "pkg/testing/tests/test_pytest.py::test", "pkg/tests/test_core.py"])["ok"]
    bad = vr.official_files_absent(tmp_path / "view", "", ["pkg/testing/pytest.py"])
    assert not bad["ok"] and bad["present"] == ["pkg/testing/pytest.py"]


def test_v3_keeps_a_shipped_test_free_test_package_but_never_a_test_suite(tmp_path):
    files = {**BASE, "setup.py": "packages=find_packages()\n",
             "pkg/test/__init__.py": "", "pkg/test/client.py": "class Client: pass\n",
             "pkg/test/utils.py": "def override(): pass\n",
             "other/__init__.py": "", "other/test/__init__.py": "",
             "other/test/util.py": "X = 1\n", "other/test/gen_test.py": "def test(): 0\n"}
    co = _tree(tmp_path / "co", files)
    out = vr.build_view_v3(co, "", tmp_path / "view")
    view = tmp_path / "view"
    assert out["view_rule"] == vr.VERSION_V3
    assert "pkg.test" in out["kept_testing_packages"]
    assert "other.test" not in out["kept_testing_packages"]          # a project test suite
    assert (view / "pkg/test/client.py").is_file()
    assert not (view / "other/test").exists()
    assert not (view / "pkg/tests/test_core.py").exists()
    v2 = vr.build_view_v2(co, "", tmp_path / "v2")
    assert v2["view_rule"] == vr.VERSION and "pkg.test" not in v2["kept_testing_packages"]
