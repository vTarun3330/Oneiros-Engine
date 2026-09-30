#!/usr/bin/env bash
REPO=/mnt/c/Users/Student2/Desktop/Capstone/oneiros
PY="$(uv python find 3.13)"
/usr/bin/python3.11 --version
"$PY" "$REPO/scripts/native_generated_tests_execute_wsl.py" canaries "$REPO/results/sft_root_cause/native_v21_canaries"
