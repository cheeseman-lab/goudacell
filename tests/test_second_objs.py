import numpy as np
import pytest

from goudacell.config import SecondaryObjectParams, SegmentationConfig


def test_flat_config_unchanged():
    p = SecondaryObjectParams(second_obj_detection=True, second_obj_channel_index=2)
    out = p.to_phenotype_config()
    assert out == p.to_dict()
    assert "threshold_k" not in out and "second_obj_mode" not in out
    assert p.needs_1_5_2() == []


def test_named_count_and_mad():
    p = SecondaryObjectParams(
        second_obj_detection=True,
        second_obj_channel_index=2,
        second_obj_name="granule",
        threshold_method="cell_median_mad",
        threshold_k=5,
        second_obj_mode="count",
    )
    out = p.to_phenotype_config()
    spec = out["second_objs"]["granule"]
    assert out["second_obj_detection"] is True and "second_obj_detection" not in spec
    assert spec["threshold_k"] == 5 and spec["second_obj_mode"] == "count"
    assert spec["foci_radius"] == 3 and spec["foci_threshold"] == 10
    assert len(p.needs_1_5_2()) == 3


@pytest.mark.parametrize("name", ["Granule", "cell_body", "nucleus", "second_obj_x", "a__b", "1x"])
def test_bad_names(name):
    with pytest.raises(ValueError):
        SecondaryObjectParams(second_obj_name=name)


def test_yaml_roundtrip(tmp_path):
    so = SecondaryObjectParams(
        second_obj_detection=True,
        second_obj_channel_index=1,
        second_obj_name="granule",
        threshold_method="cell_median_mad",
    )
    cfg = SegmentationConfig(input_dir="in", output_dir="out", mode="dual", secondary_objects=so)
    from goudacell.config import DualSegmentationParams

    cfg.dual = DualSegmentationParams()
    cfg.to_yaml(tmp_path / "c.yaml")
    back = SegmentationConfig.from_yaml(tmp_path / "c.yaml").secondary_objects
    assert back.second_obj_name == "granule" and back.threshold_method == "cell_median_mad"


def test_run_gated_until_resync():
    from goudacell.segment import _needs_newer_brieflow, segment_second_objects

    p = SecondaryObjectParams(
        second_obj_detection=True, second_obj_channel_index=0, threshold_method="cell_median_mad"
    )
    if not _needs_newer_brieflow(p):
        pytest.skip("vendored brieflow already supports cell_median_mad")
    img = np.zeros((1, 32, 32), np.uint16)
    cells = np.ones((32, 32), np.int32)
    with pytest.raises(NotImplementedError, match="1.5.2"):
        segment_second_objects(img, cells, cells, p, gpu=False)
    assert _needs_newer_brieflow(SecondaryObjectParams()) == []
