"""Cytoplasm is only built from reconciled nuclei and cell masks."""

import numpy as np
import pytest

from goudacell.features import extract_features
from goudacell.segment import identify_cytoplasm, masks_reconciled


def _masks():
    """Two cells, each around the nucleus of the same label."""
    nuclei = np.zeros((40, 40), dtype=int)
    cells = np.zeros((40, 40), dtype=int)
    cells[2:18, 2:18], nuclei[6:14, 6:14] = 1, 1
    cells[22:38, 22:38], nuclei[26:34, 26:34] = 2, 2
    return nuclei, cells


def test_reconciled_masks_give_cytoplasm():
    nuclei, cells = _masks()
    assert masks_reconciled(nuclei, cells)
    cytoplasm = identify_cytoplasm(nuclei, cells)
    assert set(np.unique(cytoplasm)) == {0, 1, 2}
    assert not np.any(cytoplasm[nuclei > 0])


def test_equal_counts_with_unpaired_labels_skip_cytoplasm():
    nuclei, cells = _masks()
    swapped = np.where(nuclei == 1, 2, np.where(nuclei == 2, 1, 0))
    assert len(np.unique(swapped)) == len(np.unique(cells))
    assert not masks_reconciled(swapped, cells)
    with pytest.warns(UserWarning, match="not reconciled"):
        assert identify_cytoplasm(swapped, cells) is None


def test_unequal_counts_skip_cytoplasm():
    nuclei, cells = _masks()
    nuclei[30:32, 2:4] = 3
    assert not masks_reconciled(nuclei, cells)
    with pytest.warns(UserWarning, match="3 nuclei, 2 cells"):
        assert identify_cytoplasm(nuclei, cells) is None


def test_unreconciled_features_have_no_cytoplasm_columns():
    nuclei, cells = _masks()
    swapped = np.where(nuclei == 1, 2, np.where(nuclei == 2, 1, 0))
    image = np.random.default_rng(0).integers(0, 1000, (2, 40, 40)).astype(np.uint16)
    with pytest.warns(UserWarning, match="not reconciled"):
        df = extract_features(
            image, swapped, cells, include_texture=False, include_correlation=False
        )
    assert not any(c.startswith("cytoplasm_") for c in df.columns)
    assert any(c.startswith("cell_") for c in df.columns)
