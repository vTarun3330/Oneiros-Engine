#!/usr/bin/env bash
# Copy the formal v2.1 preparation outputs (records and the separated buggy/verifier views)
# into ignored local storage in the repository. Environments and views stay in WSL.
set -euo pipefail
SRC=/root/oneiros_native_v21_prep
DST=/mnt/c/Users/Student2/Desktop/Capstone/oneiros/results/sft_root_cause/native_v21_rehearsal
rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC/exports/buggy_view" "$SRC/exports/verifier" "$DST/"
cp "$SRC/records.jsonl" "$SRC/contract.json" "$SRC/run_summary.json" "$SRC/console.log" "$DST/"
ls "$DST"
