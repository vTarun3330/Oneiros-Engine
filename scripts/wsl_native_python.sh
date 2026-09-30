#!/usr/bin/env bash
# Run a repository script under WSL uv CPython 3.13 (stdlib-only harness scripts).
REPO=/mnt/c/Users/Student2/Desktop/Capstone/oneiros
PY="$(uv python find 3.13)"
SCRIPT="$1"; shift
"$PY" "$REPO/$SCRIPT" "$@"
