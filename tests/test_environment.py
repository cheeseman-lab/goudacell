"""The environment checks the notebook and CLI run before any work."""

import importlib.metadata
import importlib.util
import sys

import pytest

from goudacell import environment
from goudacell.environment import WRONG_KERNEL, check_environment


def test_goudacell_env_passes():
    check_environment()


def test_cellprofiler_env_is_the_wrong_kernel(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "/conda/envs/goudacell_cp")
    with pytest.raises(RuntimeError, match="switch the kernel to goudacell") as err:
        check_environment()
    assert WRONG_KERNEL in str(err.value)


def test_old_python_or_numpy(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 9, 19))
    with pytest.raises(RuntimeError, match="needs Python >= 3.10"):
        check_environment()
    monkeypatch.undo()

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "1.26.4" if name == "numpy" else real_version(name),
    )
    with pytest.raises(RuntimeError, match="needs numpy >= 2 .* has numpy 1.26.4"):
        check_environment()


def test_missing_cellpose(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "cellpose" else real_find_spec(name, *args),
    )
    check_environment()
    with pytest.raises(RuntimeError, match="Cellpose is not installed"):
        check_environment(require_cellpose=True)


def test_notebook_refuses_the_wrong_kernel(monkeypatch):
    from goudacell.notebook import ParameterUI

    monkeypatch.setattr(sys, "prefix", "/conda/envs/" + environment.CELLPROFILER_ENV)
    with pytest.raises(RuntimeError, match="switch the kernel to goudacell"):
        ParameterUI()
