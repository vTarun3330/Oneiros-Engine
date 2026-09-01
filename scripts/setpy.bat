@echo off
REM Sets PY for the current cmd session. Run as:  scripts\setpy.bat
REM No setlocal on purpose - the variable must survive into the calling shell.
set PY=C:\Users\saira\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
if not exist "%PY%" (
  echo Verified interpreter not found - falling back to the py launcher.
  set PY=py
)
echo PY is set. Interpreter:
"%PY%" --version
