@echo off
REM Oneiros Phase 3 Review 1 - panel demonstration preflight for cmd.exe
REM Run from the oneiros root:  scripts\demo_preflight.bat

setlocal
if "%PY%"=="" set PY=C:\Users\saira\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
if not exist "%PY%" set PY=py

echo Interpreter
"%PY%" --version
echo.

echo === 0. Required artifacts ===
"%PY%" scripts\demo_check_artifacts.py
echo.

echo === 1. Arnav - canonical corpus verification ===
"%PY%" -c "from pathlib import Path; from harness.corpus import verify_corpus; m=verify_corpus(Path('data/corpus/v3_final_candidate')); print('Corpus:',m['corpus_id']); print('Records:',m['training_records']); print('Splits:',{k:v['record_count'] for k,v in m['splits'].items()}); print('All quality gates:',all(m['quality_gate'].values()))"
echo.

echo === 2. Saaj - execution oracle replay ===
"%PY%" scripts\demo_oracle_replay.py
echo.

echo === 3. Saaj - focused regression suite (expect 42 passed) ===
"%PY%" -m pytest -q tests\test_model_runtime_alignment.py tests\test_sft_best_checkpoint.py tests\test_sft_monitor_comparison.py tests\test_preflight_sft_run.py tests\test_training_smoke_selection.py
echo.

echo === 4. Venkat - adapter and checkpoint metadata ===
"%PY%" scripts\demo_adapter_info.py
echo.

echo === 5. Anushka - result table from saved JSON ===
"%PY%" scripts\demo_results_table.py
echo.

echo Preflight complete. Compare every value above against the runbook.
endlocal
