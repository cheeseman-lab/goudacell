"""Integration tests: segmentation + feature extraction with all backends.

Tests use two fixture sets:
- Synthetic: float32 blobs (always available, fast)
- Real crops: uint16 single-cell crops from actual microscopy data (tests/data/)

Segmentation tests require cellpose (marked with pytest.mark.gpu).
"""

import numpy as np
import pandas as pd
import pytest
import tifffile
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures: real crop data (uint16, single cell from microscopy images)
# ---------------------------------------------------------------------------

CROP_DIR = Path(__file__).parent / "data"


@pytest.fixture
def crop_image():
    """4-channel uint16 single-cell crop from MAX-projected TIFF."""
    path = CROP_DIR / "crop_image.tif"
    if not path.exists():
        pytest.skip("Test crop data not found (run crop generation script)")
    return tifffile.imread(path)


@pytest.fixture
def crop_masks():
    """Nuclei + cell masks for the single-cell crop."""
    nuc_path = CROP_DIR / "crop_nuclei.tif"
    cell_path = CROP_DIR / "crop_cells.tif"
    if not nuc_path.exists() or not cell_path.exists():
        pytest.skip("Test crop masks not found")
    nuclei = tifffile.imread(nuc_path)
    cells = tifffile.imread(cell_path)
    return nuclei, cells


@pytest.fixture
def nd2_crop_image():
    """4-channel uint16 crop from ND2 source (saved as TIFF)."""
    path = CROP_DIR / "crop_nd2_as_tiff.tif"
    if not path.exists():
        pytest.skip("ND2 crop data not found")
    return tifffile.imread(path)


@pytest.fixture
def dv_crop_image():
    """4-channel uint16 crop from DeltaVision source (saved as TIFF)."""
    path = CROP_DIR / "crop_dv_as_tiff.tif"
    if not path.exists():
        pytest.skip("DV crop data not found")
    return tifffile.imread(path)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data (float32, always available)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_image():
    """3-channel 128x128 float32 image with two bright blobs."""
    rng = np.random.default_rng(42)
    img = rng.random((3, 128, 128)).astype(np.float32) * 0.1

    yy, xx = np.ogrid[:128, :128]
    mask1 = ((yy - 40) ** 2 + (xx - 40) ** 2) < 15**2
    mask2 = ((yy - 90) ** 2 + (xx - 90) ** 2) < 12**2

    for ch in range(3):
        img[ch][mask1] += 0.5 + 0.2 * ch
        img[ch][mask2] += 0.3 + 0.1 * ch

    return img


@pytest.fixture
def synthetic_masks():
    """Labeled nuclei and cell masks matching synthetic_image blobs."""
    yy, xx = np.ogrid[:128, :128]

    nuclei = np.zeros((128, 128), dtype=np.int32)
    nuclei[((yy - 40) ** 2 + (xx - 40) ** 2) < 10**2] = 1
    nuclei[((yy - 90) ** 2 + (xx - 90) ** 2) < 8**2] = 2

    cells = np.zeros((128, 128), dtype=np.int32)
    cells[((yy - 40) ** 2 + (xx - 40) ** 2) < 15**2] = 1
    cells[((yy - 90) ** 2 + (xx - 90) ** 2) < 12**2] = 2

    return nuclei, cells


# ---------------------------------------------------------------------------
# Fixtures: optional dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def has_cp_measure():
    """Skip if cp_measure is not installed."""
    pytest.importorskip("cp_measure", reason="cp_measure not installed")


@pytest.fixture
def has_cellprofiler():
    """Skip if cellprofiler CLI is not available."""
    import shutil

    if shutil.which("cellprofiler") is None:
        pytest.skip("cellprofiler CLI not found in PATH")


# ---------------------------------------------------------------------------
# Segmentation (requires cellpose)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_segment_cellpose3(synthetic_image):
    """Cellpose 3 segments synthetic blobs."""
    from goudacell.segment import segment, get_cellpose_version

    version = get_cellpose_version()
    assert version[0] == 3, f"Expected cellpose 3, got {version}"

    single_ch = synthetic_image[0]
    masks = segment(single_ch, diameter=25, model="cyto3", gpu=False)

    assert masks.shape == (128, 128)
    assert masks.max() >= 1, "Should find at least one object"


@pytest.mark.gpu
def test_segment_nuclei_and_cells(synthetic_image):
    """Dual segmentation finds nuclei and cells."""
    from goudacell.segment import segment_nuclei_and_cells

    nuclei, cells = segment_nuclei_and_cells(
        synthetic_image,
        nuclei_channel=0,
        cyto_channel=1,
        nuclei_diameter=18,
        cell_diameter=28,
        gpu=False,
    )

    assert nuclei.shape == (128, 128)
    assert cells.shape == (128, 128)
    assert nuclei.max() >= 1
    assert cells.max() >= 1


# ===========================================================================
# Feature extraction on REAL crop data (uint16)
# ===========================================================================


class TestCpEmulatorRealData:
    """cp_emulator backend on real uint16 microscopy crops."""

    def test_basic_features(self, crop_image, crop_masks):
        """Intensity + shape features on real crop."""
        from goudacell.features import extract_features

        nuclei, cells = crop_masks
        df = extract_features(
            crop_image,
            nuclei_masks=nuclei,
            cell_masks=cells,
            channel_names=["DAPI", "Vimentin", "PML", "EdU"],
            include_texture=False,
            include_correlation=False,
            include_neighbors=True,
            method="cp_emulator",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1, "Should find at least one cell"
        assert "label" in df.columns
        nucleus_cols = [c for c in df.columns if c.startswith("nucleus_")]
        assert len(nucleus_cols) > 0

    def test_texture_features(self, crop_image, crop_masks):
        """Haralick + PFTAS texture on real uint16 data (mahotas needs integers)."""
        from goudacell.features import extract_features

        nuclei, cells = crop_masks
        df = extract_features(
            crop_image,
            nuclei_masks=nuclei,
            cell_masks=cells,
            include_texture=True,
            include_correlation=True,
            method="cp_emulator",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert len(df.columns) > 50, f"Expected many features with texture, got {len(df.columns)}"

    def test_all_crop_sources(self, crop_image, nd2_crop_image, dv_crop_image, crop_masks):
        """All image sources (TIFF, ND2, DV) produce compatible feature tables."""
        from goudacell.features import extract_features

        nuclei, cells = crop_masks
        results = {}
        for name, img in [
            ("tiff", crop_image),
            ("nd2", nd2_crop_image),
            ("dv", dv_crop_image),
        ]:
            df = extract_features(
                img,
                nuclei_masks=nuclei,
                cell_masks=cells,
                include_texture=False,
                include_correlation=False,
                include_neighbors=False,
                method="cp_emulator",
            )
            results[name] = df
            assert len(df) >= 1, f"{name}: should extract features"

        # All sources should produce same column set
        assert set(results["tiff"].columns) == set(results["nd2"].columns)
        assert set(results["tiff"].columns) == set(results["dv"].columns)


class TestCpMeasureRealData:
    """cp_measure backend on real uint16 microscopy crops."""

    def test_basic_features(self, has_cp_measure, crop_image, crop_masks):
        """cp_measure extracts features from real crop."""
        from goudacell.features import extract_features

        nuclei, cells = crop_masks
        df = extract_features(
            crop_image,
            nuclei_masks=nuclei,
            cell_masks=cells,
            channel_names=["DAPI", "Vimentin", "PML", "EdU"],
            include_correlation=True,
            include_neighbors=True,
            method="cp_measure",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 10

    def test_nuclei_only(self, has_cp_measure, crop_image, crop_masks):
        """cp_measure works with nuclei only (no cell masks)."""
        from goudacell.features import extract_features

        nuclei, _ = crop_masks
        df = extract_features(
            crop_image,
            nuclei_masks=nuclei,
            cell_masks=None,
            method="cp_measure",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestCellProfilerRealData:
    """CellProfiler headless CLI backend tests."""

    def test_requires_pipeline_file(self, crop_image, crop_masks):
        """CellProfiler method raises ValueError without pipeline_file."""
        from goudacell.features import extract_features

        nuclei, cells = crop_masks
        with pytest.raises(ValueError, match="pipeline_file is required"):
            extract_features(
                crop_image,
                nuclei_masks=nuclei,
                method="cellprofiler",
            )

    def test_missing_pipeline_file(self, crop_image, crop_masks):
        """CellProfiler method raises FileNotFoundError for bad path."""
        from goudacell.features import extract_features

        nuclei, cells = crop_masks
        with pytest.raises(FileNotFoundError):
            extract_features(
                crop_image,
                nuclei_masks=nuclei,
                method="cellprofiler",
                pipeline_file="/nonexistent/pipeline.cppipe",
            )

    def test_basic_features(self, has_cellprofiler, crop_image, crop_masks):
        """CellProfiler headless extracts features from real crop (needs CP + pipeline)."""
        pipeline = CROP_DIR / "test_pipeline.cppipe"
        if not pipeline.exists():
            pytest.skip("No test pipeline file (tests/data/test_pipeline.cppipe)")

        from goudacell.features import extract_features

        nuclei, cells = crop_masks
        df = extract_features(
            crop_image,
            nuclei_masks=nuclei,
            cell_masks=cells,
            channel_names=["DAPI", "Vimentin", "PML", "EdU"],
            method="cellprofiler",
            pipeline_file=str(pipeline),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ===========================================================================
# Feature extraction on SYNTHETIC data (float32)
# ===========================================================================


def test_extract_cp_emulator_synthetic(synthetic_image, synthetic_masks):
    """cp_emulator on synthetic float32 data (no texture — mahotas needs int)."""
    from goudacell.features import extract_features

    nuclei, cells = synthetic_masks
    df = extract_features(
        synthetic_image,
        nuclei_masks=nuclei,
        cell_masks=cells,
        channel_names=["DAPI", "GFP", "RFP"],
        include_texture=False,
        include_correlation=False,
        include_neighbors=True,
        method="cp_emulator",
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "label" in df.columns
    nucleus_cols = [c for c in df.columns if c.startswith("nucleus_")]
    assert len(nucleus_cols) > 0
    cell_cols = [c for c in df.columns if c.startswith("cell_")]
    assert len(cell_cols) > 0


def test_extract_cp_measure_synthetic(has_cp_measure, synthetic_image, synthetic_masks):
    """cp_measure on synthetic float32 data."""
    from goudacell.features import extract_features

    nuclei, cells = synthetic_masks
    df = extract_features(
        synthetic_image,
        nuclei_masks=nuclei,
        cell_masks=cells,
        channel_names=["DAPI", "GFP", "RFP"],
        include_correlation=True,
        include_neighbors=True,
        method="cp_measure",
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(df.columns) > 10


def test_extract_cellprofiler_requires_pipeline(synthetic_image, synthetic_masks):
    """CellProfiler method requires a pipeline file."""
    from goudacell.features import extract_features

    nuclei, cells = synthetic_masks
    with pytest.raises(ValueError, match="pipeline_file is required"):
        extract_features(
            synthetic_image,
            nuclei_masks=nuclei,
            method="cellprofiler",
        )


# ===========================================================================
# Zarr I/O
# ===========================================================================


def test_zarr_roundtrip_image(tmp_path, crop_image):
    """Round-trip: save real uint16 image as OME-Zarr and reload."""
    from goudacell.io import load_image, save_image

    zarr_path = tmp_path / "test.zarr"
    save_image(crop_image, zarr_path, channel_names=["DAPI", "Vim", "PML", "EdU"])

    loaded = load_image(zarr_path)
    assert loaded.shape == crop_image.shape
    assert loaded.dtype == crop_image.dtype
    np.testing.assert_array_equal(loaded, crop_image)


def test_zarr_roundtrip_mask(tmp_path, crop_masks):
    """Round-trip: save real mask as OME-Zarr and reload."""
    from goudacell.io import load_image, save_mask

    nuclei, _ = crop_masks
    zarr_path = tmp_path / "nuclei.zarr"
    save_mask(nuclei, zarr_path)

    loaded = load_image(zarr_path)
    assert loaded.shape == nuclei.shape
    np.testing.assert_array_equal(loaded, nuclei)


def test_tiff_roundtrip(tmp_path, crop_masks):
    """TIFF save/load still works after zarr additions."""
    from goudacell.io import load_image, save_mask

    nuclei, _ = crop_masks
    tiff_path = tmp_path / "nuclei.tif"
    save_mask(nuclei, tiff_path)

    loaded = load_image(tiff_path)
    np.testing.assert_array_equal(loaded, nuclei)


def test_zarr_roundtrip_synthetic(tmp_path, synthetic_image):
    """OME-Zarr round-trip with float32 synthetic data."""
    from goudacell.io import load_image, save_image

    zarr_path = tmp_path / "synth.zarr"
    save_image(synthetic_image, zarr_path)

    loaded = load_image(zarr_path)
    assert loaded.shape == synthetic_image.shape
    np.testing.assert_allclose(loaded, synthetic_image, atol=1e-5)


# ===========================================================================
# Dispatch & config validation
# ===========================================================================


def test_invalid_method_raises(synthetic_image, synthetic_masks):
    """Unknown extraction method raises ValueError."""
    from goudacell.features import extract_features

    nuclei, _ = synthetic_masks
    with pytest.raises(ValueError, match="Unknown extraction method"):
        extract_features(synthetic_image, nuclei_masks=nuclei, method="nonexistent")


def test_config_method_field():
    """FeatureExtractionParams supports method field."""
    from goudacell.config import FeatureExtractionParams

    fe = FeatureExtractionParams(method="cp_measure")
    assert fe.method == "cp_measure"

    fe2 = FeatureExtractionParams()
    assert fe2.method == "cp_emulator"


def test_config_method_roundtrip(tmp_path):
    """Method field survives YAML roundtrip."""
    from goudacell.config import FeatureExtractionParams, SegmentationConfig

    cfg = SegmentationConfig(
        input_dir=".",
        output_dir="./out",
        feature_extraction=FeatureExtractionParams(
            enabled=True, method="cp_measure"
        ),
    )
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    cfg2 = SegmentationConfig.from_yaml(path)
    assert cfg2.feature_extraction.method == "cp_measure"
