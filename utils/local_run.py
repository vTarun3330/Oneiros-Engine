"""Output isolation for native GPU runs; no dataset or cloud access."""
from pathlib import Path
import re


def local_run_paths(project_root: Path, run_name: str, *, fresh: bool = False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', run_name):
        raise ValueError('Run name must be 1-100 letters, digits, underscores or hyphens')
    if run_name.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('Run name is reserved on Windows')
    root = project_root.resolve()
    paths = tuple(root / directory / run_name for directory in ('checkpoints', 'results'))
    for path in paths:
        if not path.resolve().is_relative_to(root / path.parent.name):
            raise ValueError('Run output resolves outside its project output directory')
        if path.exists() and not path.is_dir():
            raise ValueError(f'Run output is not a directory: {path}')
        if fresh and path.exists():
            raise ValueError('Refusing --fresh against existing outputs; choose a new run name')
    return paths
