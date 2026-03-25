"""Shared test fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def example_config(project_root):
    """Return path to example config."""
    return project_root / "data" / "segmentation_config.yaml"
