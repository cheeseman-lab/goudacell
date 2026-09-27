"""Checks that goudacell runs in a working goudacell environment.

Stdlib only, and importable on older Pythons, so a notebook kernel or CLI started in the
wrong conda env (e.g. the CellProfiler env) gets a clear error instead of an import error
deep inside a dependency.
"""

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

# pyproject.toml's requires-python and numpy floor
MIN_PYTHON = (3, 10)
MIN_NUMPY = 2
# The conda env scripts/setup_cellprofiler_env.sh creates; goudacell only calls it as a subprocess
CELLPROFILER_ENV = "goudacell_cp"
WRONG_KERNEL = (
    "this notebook runs in the `goudacell` env; the CellProfiler env is only called as a "
    "subprocess — switch the kernel to goudacell"
)


def check_environment(require_cellpose: bool = False) -> None:
    """Raise if this interpreter is not a working goudacell environment.

    CellProfiler being importable is fine on its own; it only marks the interpreter as the
    CellProfiler env in the error when goudacell's requirements aren't met.

    Args:
        require_cellpose: Also require Cellpose (the notebook and segmentation need it).

    Raises:
        RuntimeError: If this is the ``goudacell_cp`` env, Python is older than goudacell
            requires, numpy is missing or older than 2, or Cellpose is required and missing.
    """
    where = f"this interpreter is {sys.executable} (Python {sys.version.split()[0]})"
    if Path(sys.prefix).name == CELLPROFILER_ENV:
        raise RuntimeError(f"{WRONG_KERNEL} ({where}, the CellProfiler env).")
    problem = None
    if tuple(sys.version_info[:2]) < MIN_PYTHON:
        problem = f"goudacell needs Python >= {'.'.join(map(str, MIN_PYTHON))} but {where}"
    else:
        try:
            numpy_version = importlib.metadata.version("numpy")
        except importlib.metadata.PackageNotFoundError:
            problem = f"numpy is not installed: {where}"
        else:
            if int(numpy_version.split(".")[0]) < MIN_NUMPY:
                problem = (
                    f"goudacell needs numpy >= {MIN_NUMPY} but {where} has numpy "
                    f"{numpy_version}"
                )
    if problem and importlib.util.find_spec("cellprofiler"):
        raise RuntimeError(f"{WRONG_KERNEL} ({problem}; it has CellProfiler).")
    if problem:
        raise RuntimeError(f"{problem}; switch the kernel to (or activate) the goudacell env.")
    if require_cellpose and importlib.util.find_spec("cellpose") is None:
        raise RuntimeError(
            f"Cellpose is not installed: {where}. Switch to the goudacell env, or install it "
            "there with uv pip install -e '.[cellpose3]' (or '.[cellpose4]')."
        )
