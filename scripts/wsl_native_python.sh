#!/usr/bin/env bash
# Run a repository script under WSL uv CPython 3.13 (stdlib-only harness scripts).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this checkout, wherever it lives
PY="$(uv python find 3.13)"
SCRIPT="$1"; shift
"$PY" "$REPO/$SCRIPT" "$@"
