"""Config loading and validation tests."""

from goudacell.config import (
    DualSegmentationParams,
    FeatureExtractionParams,
    SegmentationConfig,
)


def test_load_example_config(example_config):
    """Example config loads without error."""
    cfg = SegmentationConfig.from_yaml(str(example_config))
    assert cfg.input_dir is not None
    assert cfg.output_dir is not None


def test_config_roundtrip(tmp_path):
    """Config can be saved and reloaded."""
    cfg = SegmentationConfig(input_dir=".", output_dir="./out")
    path = tmp_path / "test_config.yaml"
    cfg.to_yaml(str(path))
    cfg2 = SegmentationConfig.from_yaml(str(path))
    assert cfg2.input_dir == cfg.input_dir
    assert cfg2.output_dir == cfg.output_dir
    assert cfg2.model == cfg.model
    assert cfg2.gpu == cfg.gpu


def test_config_defaults():
    """Default config has sensible values."""
    cfg = SegmentationConfig(input_dir=".", output_dir="./masks")
    assert cfg.file_pattern == "*.tif"
    assert cfg.model == "cyto3"
    assert cfg.diameter == 30.0
    assert cfg.flow_threshold == 0.4
    assert cfg.mode == "cells"
    assert cfg.gpu is True


def test_dual_config_roundtrip(tmp_path):
    """Dual-mode config with nested params survives roundtrip."""
    dual = DualSegmentationParams(nuclei_channel=1, cyto_channel=0, nuclei_diameter=15.0)
    cfg = SegmentationConfig(input_dir=".", output_dir="./out", mode="dual", dual=dual)
    path = tmp_path / "dual_config.yaml"
    cfg.to_yaml(str(path))
    cfg2 = SegmentationConfig.from_yaml(str(path))
    assert cfg2.mode == "dual"
    assert cfg2.dual is not None
    assert cfg2.dual.nuclei_channel == 1
    assert cfg2.dual.nuclei_diameter == 15.0


def test_feature_extraction_params():
    """FeatureExtractionParams defaults are sensible."""
    fe = FeatureExtractionParams()
    assert fe.enabled is False
    assert fe.include_texture is True
    assert fe.output_path == "features.csv"


def test_segmentation_modes():
    """All three modes produce valid output paths."""
    from pathlib import Path

    input_file = Path("test_image.tif")

    for mode in ["nuclei", "cells"]:
        cfg = SegmentationConfig(input_dir=".", output_dir="./masks", mode=mode)
        out = cfg.get_output_path(input_file)
        assert out.suffix == ".tif"
        if mode == "nuclei":
            assert "nuclei_mask" in out.name
        else:
            assert out.name == "test_image_mask.tif"

    cfg = SegmentationConfig(
        input_dir=".",
        output_dir="./masks",
        mode="dual",
        dual=DualSegmentationParams(),
    )
    nuc, cell = cfg.get_dual_output_paths(input_file)
    assert "nuclei_mask" in nuc.name
    assert "cell_mask" in cell.name
