"""v2.7 native reach v3: generic measurement fixes (test-id resolution, unmeasured runs,
shutdown-safe tracer, build-generated files)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("v27_reach", ROOT / "scripts/v27_native_reach_wsl.py")
reach = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reach)


def test_class_based_ids_resolve_to_the_module_file(tmp_path):
    (tmp_path / "tests/runners").mkdir(parents=True)
    (tmp_path / "tests/runners/test_harness.py").write_text("")
    (tmp_path / "tests/test_utils.py").write_text("")
    assert reach.pytest_ids(["tests.runners.test_harness.TestA::test_b[x.y-1]",
                             "tests.test_utils::test_c[a.b]"], tmp_path) == [
        "tests/runners/test_harness.py::TestA::test_b[x.y-1]",
        "tests/test_utils.py::test_c[a.b]"]


def test_only_missing_generated_files_are_copied(tmp_path):
    view, checkout = tmp_path / "view", tmp_path / "co"
    (view / "pkg").mkdir(parents=True)
    (checkout / "pkg").mkdir(parents=True)
    (view / "pkg/_version.py").write_text("v = '1'")
    (view / "pkg/core.py").write_text("VIEW")
    (checkout / "pkg/core.py").write_text("CHECKOUT")
    assert reach.copy_generated(view, checkout) == ["pkg/_version.py"]
    assert (checkout / "pkg/core.py").read_text() == "CHECKOUT"


def _run(tmp_path, body, name="target", env_file="mod.py"):
    (tmp_path / "plugin").mkdir()
    (tmp_path / "plugin/oneiros_reach_plugin.py").write_text(reach.PLUGIN)
    (tmp_path / "mod.py").write_text("def target():\n    return 1\n")
    (tmp_path / "test_x.py").write_text(body)
    out = tmp_path / "hit.json"
    env = {"PYTHONPATH": f"{tmp_path / 'plugin'}{__import__('os').pathsep}{tmp_path}",
           "ONEIROS_REACH_NAME": name, "ONEIROS_REACH_FILE": env_file,
           "ONEIROS_REACH_OUT": str(out), "SYSTEMROOT": __import__("os").environ.get(
               "SYSTEMROOT", "")}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-p", "oneiros_reach_plugin", "test_x.py"], cwd=tmp_path, env=env,
                          capture_output=True, text=True)
    return done.returncode, json.loads(out.read_text()) if out.exists() else None


def test_entry_is_recorded_even_if_a_project_plugin_crashes_at_teardown(tmp_path):
    (tmp_path / "conftest.py").write_text(
        "import pytest\n"
        "@pytest.hookimpl(tryfirst=True)\n"
        "def pytest_unconfigure(config):\n    raise RuntimeError('project teardown crash')\n")
    code, hit = _run(tmp_path, "import mod\ndef test_a():\n    assert mod.target() == 1\n")
    assert hit == {"entered": True}


def test_target_not_called_is_measured_false(tmp_path):
    code, hit = _run(tmp_path, "def test_a():\n    assert True\n")
    assert code == 0 and hit == {"entered": False}
