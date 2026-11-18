"""Fine-tune a Cellpose cell model on labels returned from the labeler, and score models on them.

Training and scoring feed the model exactly what ``goudacell segment`` (brieflow's
``prepare_cellpose``) feeds it at run time: the whole field as an 8-bit [helper, cell body, DAPI]
image, then the tile cut from it. Tiles are rescaled so cells are ~30 px across, the size both
Cellpose versions train at, so a tuned model runs with the same cell diameter as the base model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from goudacell.labels import _load_image, read_export, write_meta

PathLike = Union[str, Path]
TRAIN_DIAMETER = 30.0
# Cellpose authors' fine-tuning settings for each version
CP4_TRAIN = dict(learning_rate=1e-5, weight_decay=0.1, batch_size=1)
CP3_TRAIN = dict(learning_rate=0.1, weight_decay=1e-4, SGD=True, batch_size=8)
NUCLEI_KEYS = ("nuc", "dapi", "nuclei")


def _channels(field: Dict[str, Any], cyto: str, nuclei: Optional[str], helper: Optional[str]):
    ch = field["channels"]
    nuclei = nuclei or next((k for k in NUCLEI_KEYS if k in ch), None)
    if cyto not in ch or nuclei not in ch:
        raise ValueError(
            f"field {field['id']!r} needs a cell-body and a nucleus channel; it has {list(ch)}. "
            "Name them with cyto= and nuclei=."
        )
    return ch[nuclei], ch[cyto], (ch[helper] if helper else None)


def field_input(
    field: Dict[str, Any], cyto: str = "cyto", nuclei: Optional[str] = None, helper=None
) -> np.ndarray:
    """The whole field as the 8-bit (3, Y, X) image goudacell's dual mode gives Cellpose."""
    from goudacell.brieflow.shared.segment_cellpose import prepare_cellpose

    nuc, cyt, hlp = _channels(field, cyto, nuclei, helper)
    return prepare_cellpose(_load_image(field["image"]), nuc, cyt, helper_index=hlp)


def median_diameter(mask: np.ndarray) -> float:
    """Median equivalent diameter (px) of the objects in a label image."""
    _, area = np.unique(mask[mask > 0], return_counts=True)
    return float(np.median(2 * np.sqrt(area / np.pi))) if len(area) else TRAIN_DIAMETER


def training_tiles(
    spec: Dict[str, Any],
    export: PathLike,
    splits: Optional[Iterable[str]] = None,
    cyto: str = "cyto",
    nuclei: Optional[str] = None,
    helper: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Accepted tiles of an export, each with its run-time input crop, mask and cell diameter.

    Args:
        spec: The pack spec (``build_pack`` input); its fields give the raw images.
        export: The labeler's export zip.
        splits: Keep only tiles in these splits (e.g. ``{"train"}``); None keeps all.
        cyto: Spec channel key of the cell body.
        nuclei: Spec channel key of the nuclei; default the first of "nuc", "dapi", "nuclei".
        helper: Spec channel key of the optional helper (red) channel.

    Returns:
        Dicts with ``tile_id``, ``split``, ``field``, ``y0``, ``x0``, ``rgb`` (3, h, w) uint8,
        ``mask`` and ``diameter`` (the field's ``cell_diameter_px``, else the labels' median).
    """
    manifest, masks, _ = read_export(export, min_area=1)
    fields = {f["id"]: f for f in spec["fields"]}
    keep = set(splits) if splits is not None else None
    out, cache = [], {}
    for t in manifest["tiles"]:
        if not t.get("done") or (keep is not None and t.get("split") not in keep):
            continue
        s, m = t["source"], masks[t["tile_id"]]
        f = fields[s["field"]]
        if s["field"] not in cache:
            cache[s["field"]] = field_input(f, cyto, nuclei, helper)
        y, x = s["y0"], s["x0"]
        rgb = cache[s["field"]][:, y : y + m.shape[0], x : x + m.shape[1]]
        d = f.get("cell_diameter_px") or median_diameter(m)
        out.append(
            dict(
                tile_id=t["tile_id"],
                split=t.get("split", ""),
                field=s["field"],
                y0=y,
                x0=x,
                rgb=rgb,
                mask=m,
                diameter=float(d),
            )
        )
    if not out:
        raise ValueError(
            f"no accepted tiles in {export}" + (f" with split in {keep}" if keep else "")
        )
    return out


def _rescale(tile: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    from skimage.transform import resize

    f = TRAIN_DIAMETER / tile["diameter"]
    hw = tuple(max(1, round(n * f)) for n in tile["mask"].shape)
    img = resize(tile["rgb"].astype(np.float32), (3, *hw), order=1, preserve_range=True)
    mask = resize(tile["mask"], hw, order=0, preserve_range=True, anti_aliasing=False)
    return img.astype(np.float32), mask.astype(np.int32)


def _sha256(path: PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finetune(
    spec: Dict[str, Any],
    export: PathLike,
    out_dir: PathLike,
    model: str = "cpsam",
    epochs: int = 200,
    holdout: bool = True,
    name: Optional[str] = None,
    gpu: bool = True,
    **channels: Optional[str],
) -> Path:
    """Fine-tune a Cellpose cell model on an export's accepted tiles.

    Args:
        spec: The pack spec the labeler was built from.
        export: The labeler's export zip.
        out_dir: Cellpose writes the weights to ``out_dir/models/<name>``.
        model: Base model: "cpsam" (Cellpose 4), "cyto3" (Cellpose 3.1.0) or a weights path.
        epochs: Training epochs.
        holdout: Leave out tiles in the "test" split, to score the model on them after.
        name: Weights file name; default ``<pack_id>_<model>_ft``.
        gpu: Train on the GPU.
        **channels: ``cyto``, ``nuclei`` and ``helper`` channel keys (see ``training_tiles``).

    Returns:
        Path to the weights. Use it as the cell model, with the same cell diameter as before.
    """
    from goudacell.brieflow.shared.segment_cellpose import cellpose_version_for_model
    from goudacell.segment import create_cellpose_model

    tiles = training_tiles(spec, export, None, **channels)
    if holdout:
        tiles = [t for t in tiles if t["split"] != "test"]
    if not tiles:
        raise ValueError("every accepted tile is in the test split; train with holdout=False")
    data = [_rescale(t) for t in tiles]
    version = cellpose_version_for_model(model)
    base = Path(model).stem if Path(model).exists() else model
    name = name or f"{spec['pack_id']}_{base}_ft"
    net = create_cellpose_model(model, gpu=gpu).net
    Path(out_dir, "models").mkdir(parents=True, exist_ok=True)
    if version == 4:
        from cellpose import train

        kw = dict(CP4_TRAIN, channel_axis=0)
    else:
        from cellpose3 import train

        kw = dict(CP3_TRAIN, channels=[2, 3], channel_axis=0)
    res = train.train_seg(
        net,
        train_data=[d[0] for d in data],
        train_labels=[d[1] for d in data],
        n_epochs=epochs,
        min_train_masks=1,
        save_path=str(out_dir),
        model_name=name,
        **kw,
    )
    path = Path(res[0] if isinstance(res, tuple) else res)
    write_meta(
        path,
        packages=["cellpose", "cellpose3", "torch", "numpy"],
        base_model=model,
        cellpose_version=version,
        export=str(Path(export).resolve()),
        export_sha256=_sha256(export),
        pack_id=spec["pack_id"],
        tiles=[t["tile_id"] for t in tiles],
        n_objects=int(sum(len(np.unique(t["mask"])) - 1 for t in tiles)),
        held_out="test split" if holdout else "none",
        epochs=epochs,
        train_settings={k: v for k, v in kw.items() if k != "channel_axis"},
        input="prepare_cellpose [helper, cell body, DAPI] uint8, whole field, then the tile",
        rescale=f"cells to {TRAIN_DIAMETER:g} px; run with the field's cell diameter",
    )
    return path


def match(gt: np.ndarray, pred: np.ndarray, iou: float = 0.5) -> Tuple[int, int, int]:
    """``(TP, FP, FN)`` object counts of ``pred`` against ``gt`` at an IoU threshold >= 0.5."""
    both = (gt > 0) & (pred > 0)
    inter = Counter(zip(gt[both].tolist(), pred[both].tolist()))
    ga = dict(zip(*np.unique(gt[gt > 0], return_counts=True)))
    pa = dict(zip(*np.unique(pred[pred > 0], return_counts=True)))
    tp = sum(n / (ga[a] + pa[b] - n) >= iou for (a, b), n in inter.items())
    return tp, len(pa) - tp, len(ga) - tp


def segment_cells(model, rgb: np.ndarray, diameter: float, version: int) -> np.ndarray:
    """Cell masks for a prepared image, with the eval call brieflow's dual mode makes."""
    ch = [1, 2, 3] if version == 4 else [2, 3]
    return model.eval(rgb, diameter=diameter, channels=ch, batch_size=8)[0].astype(np.int32)


def evaluate(
    spec: Dict[str, Any],
    export: PathLike,
    models: Sequence[str],
    splits: Optional[Iterable[str]] = ("test",),
    gpu: bool = True,
    **channels: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Score cell models against an export: each runs on the whole field, then is cut to the tile.

    Returns:
        ``{model: {tp, fp, fn, precision, recall, f1, tiles: {tile_id: (tp, fp, fn)}}}``.
    """
    from goudacell.brieflow.shared.segment_cellpose import cellpose_version_for_model
    from goudacell.segment import create_cellpose_model

    tiles = training_tiles(spec, export, splits, **channels)
    fields = {f["id"]: f for f in spec["fields"]}
    loaded = {m: (create_cellpose_model(m, gpu=gpu), cellpose_version_for_model(m)) for m in models}
    res = {m: dict(tiles={}) for m in models}
    for fid in dict.fromkeys(t["field"] for t in tiles):
        rgb = field_input(fields[fid], **channels)
        here = [t for t in tiles if t["field"] == fid]
        for m, (net, v) in loaded.items():
            cells = segment_cells(net, rgb, here[0]["diameter"], v)
            for t in here:
                h, w = t["mask"].shape
                res[m]["tiles"][t["tile_id"]] = match(
                    t["mask"], cells[t["y0"] : t["y0"] + h, t["x0"] : t["x0"] + w]
                )
    for r in res.values():
        tp, fp, fn = (int(sum(c[i] for c in r["tiles"].values())) for i in range(3))
        r.update(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / max(tp + fp, 1),
            recall=tp / max(tp + fn, 1),
            f1=2 * tp / max(2 * tp + fp + fn, 1),
        )
    return res


def read_spec(path: PathLike) -> Dict[str, Any]:
    """Load a pack spec JSON."""
    return json.loads(Path(path).read_text())
