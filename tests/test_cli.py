"""``goudacell segment`` exit status: 1 only when every file fails."""

import importlib

import numpy as np
import pytest
import tifffile

CliRunner = pytest.importorskip("typer.testing").CliRunner


def _run(tmp_path, monkeypatch, good: int, bad: int):
    from goudacell.cli import app
    from goudacell.config import SegmentationConfig

    def fake_segment(image, **kwargs):
        masks = np.zeros(image.shape[-2:], dtype=np.uint16)
        masks[2:6, 2:6] = 1
        return masks

    monkeypatch.setattr(importlib.import_module("goudacell.segment"), "segment", fake_segment)
    (tmp_path / "in").mkdir()
    for i in range(good):
        tifffile.imwrite(tmp_path / "in" / f"good{i}.tif", np.ones((16, 16), dtype=np.uint16))
    for i in range(bad):
        (tmp_path / "in" / f"bad{i}.tif").write_bytes(b"not a tiff")
    config = SegmentationConfig(
        input_dir=str(tmp_path / "in"), output_dir=str(tmp_path / "out"), mode="cells", gpu=False
    )
    config.to_yaml(tmp_path / "config.yaml")
    return CliRunner().invoke(app, ["segment", str(tmp_path / "config.yaml")])


def test_all_succeed_exits_0(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, good=2, bad=0)
    assert result.exit_code == 0, result.output
    assert "All 2 files succeeded" in result.output


def test_partial_failure_exits_0_with_count(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, good=1, bad=2)
    assert result.exit_code == 0, result.output
    assert "2 of 3 files failed" in result.output
    assert (tmp_path / "out" / "good0_mask.tif").is_file()


def test_all_fail_exits_1(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, good=0, bad=2)
    assert result.exit_code == 1, result.output
    assert "2 of 2 files failed" in result.output
