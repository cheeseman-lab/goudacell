"""Annotation packs: encoding, tile choice, HTML embedding, export round trip and the next round."""

import io
import json
import zipfile

import numpy as np
import pytest
import tifffile

from goudacell import labels as L


def _field(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 200, (2, 128, 160)).astype(np.uint16)
    cells = np.zeros((128, 160), np.int32)
    for i, (y, x) in enumerate([(10, 10), (10, 60), (60, 20), (70, 90), (100, 130)], 1):
        cells[y : y + 20, x : x + 20] = i
        img[1, y : y + 20, x : x + 20] += 500
    np.save(tmp_path / "img.npy", img)
    np.savez(tmp_path / "pre.npz", a=cells, b=(cells > 0).astype(np.int32))
    L.write_meta(tmp_path / "pre.npz", packages=["numpy"], model="synthetic", diameter=20)
    return dict(
        pack_id="t",
        name="Test pack",
        channels=[
            dict(key="cyto", name="Cell body", color="#4fd17a"),
            dict(key="dapi", name="DAPI", color="#5a8cff"),
        ],
        prefill_models=[dict(key="a", name="A"), dict(key="b", name="B")],
        min_area=5,
        fields=[
            dict(
                id="f0",
                image=str(tmp_path / "img.npy"),
                channels={"cyto": 1, "dapi": 0},
                um_per_px=0.2,
                tile_px=64,
                n_tiles=2,
                split="train",
                prefills={
                    "a": [str(tmp_path / "pre.npz"), "a"],
                    "b": [str(tmp_path / "pre.npz"), "b"],
                },
            )
        ],
    )


def test_rle_round_trip():
    m = np.zeros((7, 9), np.uint16)
    m[1:3, 2:6] = 4
    m[5, :] = 1
    assert np.array_equal(L.unrle(L.rle(m), m.shape), m)
    assert L.rle(np.zeros((3, 3), np.uint16)) == []


def test_clean_mask_keeps_largest_piece_and_drops_small():
    m = np.zeros((10, 10), np.int32)
    m[0:4, 0:4] = 3
    m[8, 8] = 3
    m[6, 0] = 7
    out = L.clean_mask(m, min_area=2)
    assert out.max() == 1 and out[8, 8] == 0 and out[6, 0] == 0 and (out == 1).sum() == 16


def test_pick_windows_do_not_overlap():
    cells = np.zeros((200, 200), np.int32)
    for i in range(16):
        y, x = divmod(i, 4)
        cells[y * 50 + 5 : y * 50 + 20, x * 50 + 5 : x * 50 + 20] = i + 1
    wins = L.pick_windows(cells, (64, 64), 3)
    assert len(wins) == 3
    for i, (y, x) in enumerate(wins):
        for y2, x2 in wins[i + 1 :]:
            assert abs(y - y2) >= 64 or abs(x - x2) >= 64


def test_build_pack_and_html(tmp_path):
    pack = L.build_pack(_field(tmp_path))
    assert pack["format"] == L.FORMAT and len(pack["tiles"]) == 2
    t = pack["tiles"][0]
    assert set(t["prefills"]) == {"a", "b"} and t["prefill"] == t["prefills"]["a"]
    assert set(t["images"]) == {"cyto", "dapi"} and "thumb" in pack["fields"]["f0"]
    page = L.build_html(pack)
    assert page.startswith("<!doctype html>") and 'id="embedded-pack"' in page
    assert "cdn.jsdelivr.net/npm/fflate" not in page and "<title>Test pack</title>" in page
    blob = page.split('id="embedded-pack">', 1)[1].split("</script>", 1)[0]
    assert json.loads(blob.replace("<\\/", "</"))["pack_id"] == "t"
    assert not L.build_html(pack, artifact=True).startswith("<!doctype")


def _export(path, pack, masks, done=True):
    rows = []
    with zipfile.ZipFile(path, "w") as z:
        for tid, m in masks.items():
            b = io.BytesIO()
            tifffile.imwrite(b, m.astype(np.uint16))
            z.writestr(f"masks/{tid}.tif", b.getvalue())
            rows.append(dict(tile_id=tid, file=f"masks/{tid}.tif", done=done, edited=True, tags={}))
        z.writestr(
            "labels.json",
            json.dumps(dict(format="goudacell-labels", pack_id=pack["pack_id"], tiles=rows)),
        )


def test_export_round_trip_and_next_round(tmp_path):
    pack = L.build_pack(_field(tmp_path))
    t = pack["tiles"][0]
    m = L.unrle(t["prefills"]["a"], (t["h"], t["w"])).copy()
    m[m == m.max()] = 0
    m[0, 0] = 9
    _export(tmp_path / "e.zip", pack, {t["id"]: m})
    man, masks, problems = L.read_export(tmp_path / "e.zip")
    assert np.array_equal(masks[t["id"]], m) and (t["id"], 9, "1 px") in problems
    nxt = L.next_round(pack, tmp_path / "e.zip")
    assert nxt["pack_id"] == "t-r2" and nxt["prefill_models"][0]["key"] == "labels"
    t2 = next(x for x in nxt["tiles"] if x["id"] == t["id"])
    assert np.array_equal(L.unrle(t2["prefill"], m.shape), m)
    assert "labels" not in nxt["tiles"][1]["prefills"]
    paths = L.write_masks(tmp_path / "e.zip", tmp_path / "out")
    assert paths and (tmp_path / "out" / "labels.json").exists()


def test_next_round_rejects_other_pack(tmp_path):
    pack = L.build_pack(_field(tmp_path))
    _export(tmp_path / "e.zip", dict(pack_id="other"), {})
    with pytest.raises(ValueError):
        L.next_round(pack, tmp_path / "e.zip")


def test_build_pack_requires_and_embeds_segmentation_record(tmp_path):
    spec = _field(tmp_path)
    seg = L.build_pack(spec)["fields"]["f0"]["segmentation"]
    assert seg["a"]["meta"]["model"] == "synthetic" and seg["a"]["meta"]["packages"]["numpy"]
    assert len(seg["a"]["sha256"]) == 64
    (tmp_path / "pre.npz.meta.json").unlink()
    with pytest.raises(ValueError, match="meta.json"):
        L.build_pack(spec)
    assert (
        L.build_pack(spec, require_meta=False)["fields"]["f0"]["segmentation"]["a"]["meta"] is None
    )


def _dual_run(tmp_path, second_objs=True):
    from goudacell.cli import _record_mask
    from goudacell.config import DualSegmentationParams, SecondaryObjectParams, SegmentationConfig
    from goudacell.io import save_mask

    (tmp_path / "in").mkdir()
    cfg = SegmentationConfig(
        input_dir="in",
        output_dir="masks",
        mode="dual",
        dual=DualSegmentationParams(nuclei_channel=0, cyto_channel=1),
        secondary_objects=SecondaryObjectParams(
            second_obj_detection=second_objs, second_obj_channel_index=1
        ),
    )
    cfg.to_yaml(tmp_path / "run.yaml")
    cfg.output_dir = str(tmp_path / "masks")
    rng = np.random.default_rng(1)
    for k in range(3):
        img = rng.integers(0, 200, (2, 96, 96)).astype(np.uint16)
        cells = np.zeros((96, 96), np.int32)
        cells[10:40, 10:40], cells[50:90, 40:80] = 1, 2
        spots = np.zeros_like(cells)
        spots[20:23, 20:23], spots[60:63, 60:63] = 1, 2
        f = tmp_path / "in" / f"field{k}.tif"
        tifffile.imwrite(f, img)
        nuc_p, cell_p = cfg.get_dual_output_paths(f)
        for m, p, kind in [(cells, nuc_p, "nuclei"), (cells, cell_p, "cells")] + (
            [(spots, cfg.get_second_obj_output_path(f), "second_objs")] if second_objs else []
        ):
            save_mask(m, p)
            _record_mask(p, cfg, tmp_path / "run.yaml", f, kind)
    return tmp_path / "run.yaml"


def test_pack_from_run_cells_and_second_objs(tmp_path):
    config = _dual_run(tmp_path)
    spec = L.spec_from_run(config, "cells", n_tiles=1, tile_px=64, max_fields=2)
    assert len(spec["fields"]) == 2 and spec["reference_name"] == "nuclei"
    pack = L.build_pack(spec)
    seg = pack["fields"]["field0"]["segmentation"]["run"]["meta"]
    assert seg["params"]["cyto_channel"] == 1 and seg["config"].endswith("run.yaml")
    spec = L.spec_from_run(config, "second_objs", n_tiles=1, tile_px=64)
    assert spec["min_area"] == 3 and spec["channels"][0]["key"] == "obj"
    assert len(L.build_pack(spec)["tiles"]) == 3


def test_from_run_cli_writes_labeler(tmp_path):
    from typer.testing import CliRunner

    from goudacell.cli import app

    config = _dual_run(tmp_path, second_objs=False)
    out = tmp_path / "lab"
    res = CliRunner().invoke(app, ["labels", "from-run", str(config), str(out), "--tile-px", "64"])
    assert res.exit_code == 0, res.output
    assert (out / "run-cells_labeler.html").stat().st_size > 1000
    res = CliRunner().invoke(
        app, ["labels", "from-run", str(config), str(out), "--objects", "second_objs"]
    )
    assert res.exit_code == 1 and "secondary objects" in res.output
