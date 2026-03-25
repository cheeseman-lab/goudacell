"""Smoke tests: does the package install and import correctly?"""

import subprocess


def test_import():
    """Package imports without error."""
    import goudacell  # noqa: F401


def test_version():
    """Version string is set and follows semver."""
    from goudacell import __version__

    assert __version__
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cli_entry_point():
    """CLI entry point is registered and callable."""
    result = subprocess.run(["goudacell", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "segment" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_core_dependencies_importable():
    """All core dependencies are importable."""
    import yaml  # noqa: F401
    import typer  # noqa: F401
    import rich  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import skimage  # noqa: F401
    import tifffile  # noqa: F401
    import mahotas  # noqa: F401


def test_public_api():
    """Public API functions are accessible."""
    from goudacell import (
        load_image,
        save_mask,
        segment,
        segment_nuclei_and_cells,
        get_cellpose_version,
        make_mask_cmap,
        extract_features,
    )

    assert callable(load_image)
    assert callable(save_mask)
    assert callable(segment)
    assert callable(segment_nuclei_and_cells)
    assert callable(get_cellpose_version)
    assert callable(make_mask_cmap)
    assert callable(extract_features)
