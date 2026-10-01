"""v2.5 Django verification layer (WSL, stdlib only).

Django test code needs configured settings before ``django.test`` can be used; the frozen pytest
sandbox provides none. This layer is a SEPARATE copy of a qualified environment (the qualified
environment is never modified) with two added files in its site-packages:
- ``oneiros_django_settings.py``: the settings of Django's own ``tests/test_sqlite.py``
  (sqlite databases, the same secret key, MD5 hasher, the contrib apps the test runner
  installs) - no project test package, no official test;
- ``oneiros_django.pth``: at interpreter start, sets ``DJANGO_SETTINGS_MODULE`` to it and calls
  ``django.setup()`` (Django itself is imported from the view on PYTHONPATH).
The candidate module, the static policy and the sandbox are unchanged. The layered environment
is locked (``env_lock``) and a canary proves Django is configured from the view.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

VERSION = "oneiros_v25_django_layer_v1"
SETTINGS = '''"""Oneiros v2.5 Django verification settings (mirror of django tests/test_sqlite.py)."""
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3"},
             "other": {"ENGINE": "django.db.backends.sqlite3"}}
SECRET_KEY = "django_tests_secret_key"
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"
USE_TZ = False
INSTALLED_APPS = ["django.contrib.contenttypes", "django.contrib.auth",
                  "django.contrib.sessions", "django.contrib.messages",
                  "django.contrib.admin.apps.SimpleAdminConfig", "django.contrib.staticfiles"]
'''
SETUP = '''"""Configure Django at interpreter start (Oneiros v2.5 Django verification layer)."""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "oneiros_django_settings")
try:
    import django
    django.setup()
except Exception as exc:                                  # surfaced by the layer canary
    os.environ["ONEIROS_DJANGO_SETUP_ERROR"] = type(exc).__name__
'''
PTH = "import oneiros_django_setup\n"


def layered_env(env_dir: str) -> Path:
    """Create (once) ``<env_dir>_django`` = copy of the qualified env + the layer files."""
    src = Path(env_dir)
    dest = src.parent / (src.name + "_django")
    if not dest.exists():
        tmp = src.parent / (src.name + "_django.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp, symlinks=True)
        site = next(tmp.glob("lib/python*/site-packages"))
        (site / "oneiros_django_settings.py").write_text(SETTINGS, encoding="utf-8")
        (site / "oneiros_django_setup.py").write_text(SETUP, encoding="utf-8")
        (site / "oneiros_django.pth").write_text(PTH, encoding="utf-8")
        tmp.rename(dest)
    return dest


def canary(python: str, view: str) -> dict:
    """Django configured from the view, in the layered environment."""
    code = ("import os, django; from django.conf import settings; "
            "print(settings.configured, os.path.realpath(django.__file__), "
            "os.environ.get('ONEIROS_DJANGO_SETUP_ERROR'))")
    done = subprocess.run([python, "-B", "-c", code], capture_output=True, text=True, cwd="/",
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": view,
                               "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    parts = done.stdout.split()
    ok = done.returncode == 0 and len(parts) == 3 and parts[0] == "True" and \
        parts[1].startswith(view.rstrip("/") + "/") and parts[2] == "None"
    return {"ok": ok, "error": None if ok else (done.stdout + done.stderr)[-300:]}


def layer_files_sha256() -> str:
    return hashlib.sha256((SETTINGS + SETUP + PTH).encode("utf-8")).hexdigest()
