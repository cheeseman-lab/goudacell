"""Fine-tuning from a labeler export: run-time inputs, scoring, and a one-epoch cyto3 run."""

import io
import json
import zipfile

import numpy as np
import pytest
import tifffile

from goudacell import labels as L
from goudacell import train as T


def _spec(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 50, (2, 96, 96)).astype(np.uint16)
    cells = np.zeros((96, 96), np.int32)
    for i, (y, x) in enumerate([(6, 6), (6, 50), (50, 10), (52, 56), (30, 30)], 1):
        yy, xx = np.ogrid[:96, :96]
        disk = (yy - y - 10) ** 2 + (xx - x - 10) ** 2 < 100
        cells[disk] = i
        img[1][disk] += 800
        img[0][(yy - y - 10) ** 2 + (xx - x - 10) ** 2 < 16] += 900
    np.save(tmp_path / "img.npy", img)
    np.savez(tmp_path / "pre.npz", cells=cells)
    L.write_meta(tmp_path / "pre.npz", packages=["numpy"], model="synthetic")
    field = dict(
        image=str(tmp_path / "img.npy"),
        um_per_px=None,
        channels={"cyto": 1, "dapi": 0},
        tile_px=64,
        n_tiles=1,
        prefills={"a": [str(tmp_path / "pre.npz"), "cells"]},
    )
    spec = dict(
        pack_id="t",
        name="t",
        channels=[
            dict(key="cyto", name="C", color="#0f0"),
            dict(key="dapi", name="D", color="#00f"),
        ],
        prefill_models=[dict(key="a", name="A")],
        min_area=5,
        fields=[
            dict(field, id="f0", split="train"),
            dict(field, id="f1", split="test", cell_diameter_px=20),
        ],
    )
    return spec, cells


def _export(path, pack, cells):
    rows = []
    with zipfile.ZipFile(path, "w") as z:
        for t in pack["tiles"]:
            s = t["source"]
            m = cells[s["y0"] : s["y0"] + t["h"], s["x0"] : s["x0"] + t["w"]]
            b = io.BytesIO()
            tifffile.imwrite(b, m.astype(np.uint16))
            z.writestr(f"masks/{t['id']}.tif", b.getvalue())
            rows.append(
                dict(
                    tile_id=t["id"],
                    file=f"masks/{t['id']}.tif",
                    split=t["split"],
                    base="a",
                    done=True,
                    edited=False,
                    tags={},
                    source=s,
                )
            )
        z.writestr("labels.json", json.dumps(dict(pack_id=pack["pack_id"], tiles=rows)))


def test_training_tiles_use_runtime_input(tmp_path):
    from goudacell.brieflow.shared.segment_cellpose import prepare_cellpose

    spec, cells = _spec(tmp_path)
    _export(tmp_path / "e.zip", L.build_pack(spec), cells)
    tiles = T.training_tiles(spec, tmp_path / "e.zip")
    assert [t["split"] for t in tiles] == ["train", "test"]
    full = prepare_cellpose(np.load(tmp_path / "img.npy"), 0, 1)
    t = tiles[0]
    assert t["rgb"].dtype == np.uint8 and t["rgb"].shape == (3, *t["mask"].shape)
    assert np.array_equal(t["rgb"], full[:, t["y0"] : t["y0"] + 64, t["x0"] : t["x0"] + 64])
    assert 15 < t["diameter"] < 23 and tiles[1]["diameter"] == 20
    assert [t["split"] for t in T.training_tiles(spec, tmp_path / "e.zip", {"test"})] == ["test"]
    with pytest.raises(ValueError, match="nucleus channel"):
        T.training_tiles(spec, tmp_path / "e.zip", nuclei="pcnt")


def test_match():
    gt = np.zeros((20, 20), int)
    gt[:5, :5], gt[10:15, 10:15] = 1, 2
    pred = np.zeros_like(gt)
    pred[:5, :4], pred[16:, 16:] = 7, 8
    assert T.match(gt, pred) == (1, 1, 1)


def test_finetune_cyto3_one_epoch_and_score(tmp_path):
    spec, cells = _spec(tmp_path)
    _export(tmp_path / "e.zip", L.build_pack(spec), cells)
    path = T.finetune(spec, tmp_path / "e.zip", tmp_path / "ft", model="cyto3", epochs=1, gpu=False)
    assert path.exists() and path.parent == tmp_path / "ft" / "models"
    meta = L.read_meta(str(path))
    assert meta["tiles"] == ["f0_t0"] and meta["base_model"] == "cyto3"
    res = T.evaluate(spec, tmp_path / "e.zip", ["cyto3", str(path)], gpu=False)
    assert set(res["cyto3"]["tiles"]) == {"f1_t0"} and res[str(path)]["tp"] >= 0
    assert 0 <= res["cyto3"]["mean_iou"] <= 1 and "f1_75" in res["cyto3"]


def test_outline_stats():
    gt = np.zeros((40, 40), int)
    gt[5:25, 5:25] = 1
    pred = np.zeros_like(gt)
    pred[7:25, 5:25] = 3
    st = T.outline_stats(gt, pred)
    assert (st["tp"], st["tp75"]) == (1, 1) and abs(st["ious"][0] - 0.9) < 1e-9
    assert 0 < st["dists"][0] < 2
    assert T.boundary_distance(gt == 1, gt == 1) == 0
