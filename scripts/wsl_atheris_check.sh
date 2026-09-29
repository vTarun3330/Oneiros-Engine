#!/usr/bin/env bash
# Print the atheris version and Python version of the WSL Atheris interpreter, or "missing".
P=/opt/atheris311/bin/python
if [ -x "$P" ]; then "$P" -c "import atheris, sys, importlib.metadata as m; print(m.version(\"atheris\"), sys.version.split()[0])"; else echo missing; fi
