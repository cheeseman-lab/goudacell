"""Annotation packs for correcting segmentations in a browser, and reading the labels back.

A pack holds native-resolution tiles cut from whole fields, each with one or more model
segmentations ("prefills") to correct. ``build_html`` embeds a pack in a single self-contained HTML
labeler that opens offline from disk; the annotator exports a zip of uint16 TIFF masks plus a
``labels.json`` manifest, which ``read_export`` reads and ``next_round`` turns into the next pack.
"""

from __future__ import annotations

import base64
import html
import io
import json
import zipfile
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PathLike = Union[str, Path]
FORMAT = "goudacell-labelpack"


def rle(mask: np.ndarray) -> List[int]:
    """Run-length encode a label image as flat ``[start, length, value]`` triples (nonzero runs)."""
    f = np.ascontiguousarray(mask).ravel()
    ch = np.flatnonzero(np.diff(f)) + 1
    starts, ends = np.r_[0, ch], np.r_[ch, f.size]
    keep = f[starts] > 0
    return np.stack([starts[keep], (ends - starts)[keep], f[starts][keep]], 1).ravel().tolist()


def unrle(runs: Sequence[int], shape: Tuple[int, int]) -> np.ndarray:
    """Decode ``rle`` output back to a uint16 label image of ``shape``."""
    m = np.zeros(int(np.prod(shape)), np.uint16)
    for s, n, v in np.asarray(runs, int).reshape(-1, 3):
        m[s : s + n] = v
    return m.reshape(shape)


def _sqrt8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # sqrt keeps more 8-bit levels for dim signal; the page squares it back (channel enc='sqrt')
    v = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-9), 0, 1)
    return np.round(np.sqrt(v) * 255).astype(np.uint8)


def _jpeg_uri(a8: np.ndarray, quality: int = 88) -> str:
    b = io.BytesIO()
    Image.fromarray(a8).save(b, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _load(src: Union[str, Sequence[str]]) -> np.ndarray:
    """Load an array from ``.npy``, ``.tif`` or ``[path.npz, key]``."""
    if not isinstance(src, str):
        return np.load(src[0])[src[1]]
    if src.endswith(".npy"):
        return np.load(src, mmap_mode="r")
    import tifffile

    return tifffile.imread(src)


def _load_image(src: str) -> np.ndarray:
    """Load a field as C×Y×X: ``.npy`` directly, anything else through ``goudacell.io``."""
    if src.endswith(".npy"):
        img = np.load(src, mmap_mode="r")
    else:
        from goudacell.io import load_image

        img = load_image(src, channel=None, z_project=True)
    return img[None] if img.ndim == 2 else img


META_SUFFIX = ".meta.json"


def _file(src: Union[str, Sequence[str]]) -> Path:
    return Path(src if isinstance(src, str) else src[0])


def sha256(path: PathLike, chunk: int = 1 << 22) -> str:
    """SHA-256 of a file, read in chunks."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_meta(path: PathLike, packages: Sequence[str] = (), **info: Any) -> Path:
    """Record how a mask file was made, next to it as ``<path>.meta.json``.

    Call this from the script that writes the masks. Captures the time, host, Slurm job, command
    line, the calling script's SHA-256 and the versions of ``packages``; ``info`` holds the rest
    (model, inputs, channels, diameters, thresholds). Uses only the standard library, so it can be
    copied into scripts that run in environments without goudacell.
    """
    import datetime
    import os
    import platform
    import sys
    from importlib import metadata

    script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] not in ("", "-c") else None
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    meta = dict(
        file=Path(path).name,
        created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        host=platform.node(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        argv=sys.argv,
        script=str(script) if script else None,
        script_sha256=sha256(script) if script and script.is_file() else None,
        python=platform.python_version(),
        packages=versions,
        **info,
    )
    out = Path(str(path) + META_SUFFIX)
    out.write_text(json.dumps(meta, indent=1, default=str))
    return out


def read_meta(src: Union[str, Sequence[str]]) -> Optional[Dict[str, Any]]:
    """The ``.meta.json`` record of a mask source, or None if it has none."""
    p = Path(str(_file(src)) + META_SUFFIX)
    return json.loads(p.read_text()) if p.exists() else None


def largest_parts(mask: np.ndarray) -> np.ndarray:
    """Keep each object's largest connected piece (cropping can cut an object in two)."""
    out = np.zeros_like(mask)
    for i, sl in enumerate(ndi.find_objects(mask), 1):
        if sl is None:
            continue
        lab, n = ndi.label(mask[sl] == i)
        if n:
            k = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            out[sl][lab == k] = i
    return out


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Largest piece per object, objects below ``min_area`` px removed, labels made sequential."""
    from skimage.segmentation import relabel_sequential

    m = largest_parts(np.asarray(mask, np.int32))
    area = np.bincount(m.ravel())
    m[area[m] < min_area] = 0
    return relabel_sequential(m)[0]


def pick_windows(
    labels: np.ndarray, size: Tuple[int, int], n: int, percentile: float = 70
) -> List[Tuple[int, int]]:
    """Top-left corners of ``n`` non-overlapping windows with a typical object count.

    Windows step by a quarter tile; the count is whole objects inside the window, and windows
    nearest the ``percentile`` of nonzero counts win, so tiles are neither empty nor only the
    densest clump.
    """
    (H, W), (th, tw) = labels.shape, size
    ys = np.unique(np.r_[np.arange(0, max(1, H - th + 1), max(1, th // 4)), max(0, H - th)])
    xs = np.unique(np.r_[np.arange(0, max(1, W - tw + 1), max(1, tw // 4)), max(0, W - tw)])
    boxes = np.array(
        [(s[0].start, s[1].start, s[0].stop, s[1].stop) for s in ndi.find_objects(labels) if s]
    ).reshape(-1, 4)
    cand = []
    for y in ys:
        for x in xs:
            inside = (boxes[:, 0] >= y) & (boxes[:, 1] >= x)
            inside &= (boxes[:, 2] <= y + th) & (boxes[:, 3] <= x + tw)
            cand.append((int(inside.sum()), int(y), int(x)))
    counts = np.array([c for c, _, _ in cand])
    target = np.percentile(counts[counts > 0], percentile) if (counts > 0).any() else 0
    out: List[Tuple[int, int]] = []
    for _, y, x in sorted(cand, key=lambda t: (abs(t[0] - target), t[1], t[2])):
        if all(abs(y - y2) >= th or abs(x - x2) >= tw for y2, x2 in out):
            out.append((y, x))
        if len(out) == n:
            break
    return out


def _thumb(
    img: np.ndarray, chans: Dict[str, int], keys: List[str], lims: Dict, px: int = 240
) -> str:
    sc = px / max(img.shape[1:])
    size = (max(1, round(img.shape[2] * sc)), max(1, round(img.shape[1] * sc)))

    def small(k):
        a8 = _sqrt8(img[chans[k]], *lims[k])
        return np.asarray(Image.fromarray(a8).resize(size, Image.BILINEAR))

    g, b = small(keys[0]), small(keys[min(1, len(keys) - 1)])
    return _jpeg_uri(np.stack([np.zeros_like(g), g, np.maximum(b, g // 3)], -1), 80)


def build_pack(spec: Dict[str, Any], require_meta: bool = True) -> Dict[str, Any]:
    """Cut tiles from whole fields and attach every prefill, following ``spec``.

    Args:
        spec: Dict with ``pack_id``, ``name``, ``channels`` ([{key, name, color, gain?}], cell
            body first), ``prefill_models`` ([{key, name}], the first is the default start) and
            ``fields``, each with ``id``, ``image`` (C×Y×X ``.npy``/``.tif``), ``channels``
            ({key: index}), ``um_per_px``, ``tile_px``, ``n_tiles``, ``split``, ``note``,
            ``prefills`` ({key: source}) and optional ``reference`` (e.g. nuclei). A source is a
            path or ``[path.npz, key]``. Optional: ``object``, ``tags``, ``instructions``,
            ``reference_name``, ``min_area`` (30), ``typical_percentile`` (70, also per field),
            ``provenance`` (free-form: how the prefills were made; carried into exports),
            ``return_to`` (``host:/dir/`` the labeler says to copy the export to), and
            ``params`` on each prefill model.

        require_meta: Refuse prefill or reference files without a ``.meta.json`` record (see
            ``write_meta``), so every pack says how its outlines were made.

    Returns:
        The pack as a JSON-serialisable dict. Each field carries ``segmentation``: per source, its
        ``.meta.json`` record plus the SHA-256 of the mask file, and the image file's SHA-256.

    Raises:
        ValueError: If ``require_meta`` and any mask source has no record.
    """
    if require_meta:
        missing = sorted(
            {
                str(_file(v))
                for f in spec["fields"]
                for v in [
                    *f["prefills"].values(),
                    *([f["reference"]] if f.get("reference") else []),
                ]
                if read_meta(v) is None
            }
        )
        if missing:
            raise ValueError(
                f"{len(missing)} mask file(s) have no {META_SUFFIX} record (write one with "
                f"goudacell.labels.write_meta when segmenting): {missing[:3]}"
            )
    keys = [c["key"] for c in spec["channels"]]
    models = [m["key"] for m in spec["prefill_models"]]
    min_area = spec.get("min_area", 30)
    fields, tiles = {}, []
    for f in spec["fields"]:
        img = _load_image(f["image"])
        th, tw = min(f["tile_px"], img.shape[1]), min(f["tile_px"], img.shape[2])
        pre = {k: _load(f["prefills"][k]) for k in models if k in f["prefills"]}
        ref = _load(f["reference"]) if f.get("reference") else None
        ck = [k for k in keys if k in f["channels"]]
        lims = {k: np.percentile(img[f["channels"][k]], [0.5, 99.9]) for k in ck}
        fields[f["id"]] = dict(
            w=int(img.shape[2]),
            h=int(img.shape[1]),
            um_per_px=f["um_per_px"],
            note=f.get("note", ""),
            thumb=_thumb(img, f["channels"], ck, lims),
            image=str(f["image"]) if isinstance(f["image"], str) else list(f["image"]),
            channels=f["channels"],
            prefills={k: v if isinstance(v, str) else list(v) for k, v in f["prefills"].items()},
            image_sha256=sha256(_file(f["image"])),
            segmentation={
                k: dict(
                    source=v if isinstance(v, str) else list(v),
                    sha256=sha256(_file(v)),
                    meta=read_meta(v),
                )
                for k, v in [
                    *f["prefills"].items(),
                    *([("reference", f["reference"])] if f.get("reference") else []),
                ]
            },
        )
        pct = f.get("typical_percentile", spec.get("typical_percentile", 70))
        corners = pick_windows(pre[models[0]], (th, tw), f.get("n_tiles", 1), pct)
        for i, (y0, x0) in enumerate(corners):
            box = (slice(y0, y0 + th), slice(x0, x0 + tw))
            P = {k: clean_mask(m[box], min_area) for k, m in pre.items()}
            images = {k: _jpeg_uri(_sqrt8(img[f["channels"][k]][box], *lims[k])) for k in ck}
            tile = dict(
                id=f"{f['id']}_t{i}",
                h=int(th),
                w=int(tw),
                split=f.get("split", ""),
                note=f.get("note", ""),
                source=dict(field=f["id"], y0=int(y0), x0=int(x0), um_per_px=f["um_per_px"]),
                images=images,
                prefill=rle(P[models[0]]),
                prefills={k: rle(v) for k, v in P.items()},
                reference=rle(clean_mask(ref[box], 1)) if ref is not None else None,
            )
            tiles.append(tile)
    channels = [dict(c, enc="sqrt") for c in spec["channels"]]
    return dict(
        format=FORMAT,
        version=2,
        pack_id=spec["pack_id"],
        name=spec["name"],
        object=spec.get("object", "cell"),
        channels=channels,
        tags=spec.get("tags", []),
        instructions=spec.get("instructions", ""),
        reference_name=spec.get("reference_name", "Reference"),
        min_area=min_area,
        prefill_model=models[0],
        prefill_models=spec["prefill_models"],
        provenance=spec.get("provenance", {}),
        return_to=spec.get("return_to", ""),
        fields=fields,
        tiles=tiles,
    )


COLORS = ("#4fd17a", "#5a8cff", "#ff7ad9", "#ffc04f")


def _resolve(path: str, base: Path) -> Path:
    # relative run paths: next to the config if there, else where the batch job ran (cwd)
    p = Path(path)
    return p if p.is_absolute() or not (base / p).exists() else base / p


def spec_from_run(
    config: PathLike,
    objects: str = "cells",
    n_tiles: int = 3,
    tile_px: int = 512,
    um_per_px: Optional[float] = None,
    max_fields: Optional[int] = None,
) -> Dict[str, Any]:
    """Pack spec for a finished ``goudacell segment run``: its images with its masks as prefill.

    Args:
        config: The run's YAML config.
        objects: "cells" (the run's cells, or nuclei in nuclei mode) or "second_objs" (the
            secondary objects of a dual run, with the cells as reference outlines).
        n_tiles: Tiles cut per field.
        tile_px: Tile size in pixels.
        um_per_px: Pixel size, shown on the page; None shows sizes in pixels.
        max_fields: Use at most this many fields, evenly spaced through the run.

    Returns:
        A spec for ``build_pack``. Fields whose masks are missing are skipped.

    Raises:
        ValueError: If ``objects`` doesn't match the run, or no field has its masks.
    """
    from goudacell.config import SegmentationConfig

    config = Path(config).resolve()
    cfg = SegmentationConfig.from_yaml(config)
    base = config.parent
    cfg.input_dir = str(_resolve(cfg.input_dir, base))
    cfg.output_dir = str(_resolve(cfg.output_dir, base))
    so = cfg.secondary_objects
    dual = cfg.mode == "dual" and cfg.dual is not None
    names = (cfg.feature_extraction.channel_names if cfg.feature_extraction else None) or []

    def chan(key: str, idx: int, default: str) -> Dict[str, Any]:
        name = names[idx] if idx < len(names) else default
        return dict(key=key, name=name, index=idx)

    if objects == "second_objs":
        if not (dual and so and so.second_obj_detection):
            raise ValueError("objects='second_objs' needs a dual run with secondary objects on")
        chans = [
            chan("obj", so.second_obj_channel_index, "Objects"),
            chan("nuc", cfg.dual.nuclei_channel, "Nuclei"),
        ]
        obj, model, ref_name = "object", f"{so.second_obj_method} (goudacell run)", "cells"
    elif objects == "cells":
        if dual:
            chans = [
                chan("cyto", cfg.dual.cyto_channel, "Cells"),
                chan("nuc", cfg.dual.nuclei_channel, "Nuclei"),
            ]
            model = f"{cfg.dual.cell_model} (goudacell run)"
        else:
            chans = [chan("cyto", cfg.channel_to_segment or 0, "Signal")]
            model = f"{cfg.model} (goudacell run)"
        obj = "nucleus" if cfg.mode == "nuclei" else "cell"
        ref_name = "nuclei" if dual else "Reference"
    else:
        raise ValueError(f"objects must be 'cells' or 'second_objs', not {objects!r}")

    files = cfg.get_input_files()
    if max_fields and len(files) > max_fields:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_fields).round().astype(int)]
    fields = []
    for f in files:
        if objects == "second_objs":
            prefill, ref = cfg.get_second_obj_output_path(f), cfg.get_dual_output_paths(f)[1]
        elif dual:
            ref, prefill = cfg.get_dual_output_paths(f)
        else:
            prefill, ref = cfg.get_output_path(f), None
        if not prefill.exists():
            continue
        field = dict(
            id=f.stem,
            image=str(f),
            channels={c["key"]: c["index"] for c in chans},
            um_per_px=um_per_px,
            tile_px=tile_px,
            n_tiles=n_tiles,
            split="",
            note=f.name,
            prefills={"run": str(prefill)},
        )
        if ref is not None and ref.exists():
            field["reference"] = str(ref)
        fields.append(field)
    if not fields:
        raise ValueError(f"no masks found in {cfg.output_dir} for {len(files)} input file(s)")
    small = objects == "second_objs"
    return dict(
        pack_id=f"{config.stem}-{objects.replace('_', '-')}",
        name=f"{config.stem}: {obj} outlines",
        object=obj,
        channels=[
            dict(key=c["key"], name=c["name"], color=COLORS[i % len(COLORS)])
            for i, c in enumerate(chans)
        ],
        prefill_models=[dict(key="run", name=model)],
        reference_name=ref_name,
        min_area=3 if small else 30,
        fields=fields,
        tags=[],
        instructions=f"One {obj} = one outline. Fix merges, splits, misses and false calls.",
        provenance=dict(config=str(config), objects=objects),
    )


def build_html(pack: Dict[str, Any], artifact: bool = False) -> str:
    """Return one self-contained labeler page with ``pack`` embedded.

    The page needs no network or account: labels autosave in the browser and "Export labels"
    downloads a zip. With ``artifact=True`` the HTML skeleton is left out (for a claude.ai
    Artifact).
    """
    data = resources.files("goudacell") / "data"
    page = (data / "labeler.html").read_text()
    lib = (data / "fflate-0.8.2.umd.js").read_text().replace("</script", "<\\/script")
    tag = '<script src="https://cdn.jsdelivr.net/npm/fflate@0.8.2/umd/index.js"></script>'
    page = page.replace(tag, f"<script>{lib}</script>")
    page = "\n".join(line for line in page.splitlines() if "fonts.googleapis.com" not in line)
    title = html.escape(pack.get("name", "Labeler"))
    page = page.replace("<title>Goudacell Labeler</title>", f"<title>{title}</title>", 1)
    blob = json.dumps(pack, separators=(",", ":")).replace("</", "<\\/")
    i = page.index("<script>\nconst $ =")
    embed = f'<script type="application/json" id="embedded-pack">{blob}</script>\n'
    page = page[:i] + embed + page[i:]
    if artifact:
        return page
    head = (
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    )
    return f'<!doctype html><html lang="en"><head>{head}</head><body>{page}</body></html>'


def read_export(
    path: PathLike, min_area: int = 10
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], List[Tuple[str, int, str]]]:
    """Read a labeler export zip.

    Returns:
        ``(manifest, {tile_id: mask}, problems)``; problems lists objects in several pieces or below
        ``min_area`` px as ``(tile_id, label, note)``.
    """
    import tifffile

    masks, problems = {}, []
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("labels.json"))
        for t in manifest["tiles"]:
            m = tifffile.imread(io.BytesIO(z.read(t["file"])))
            for i, sl in enumerate(ndi.find_objects(m), 1):
                if sl is None:
                    continue
                n, area = ndi.label(m[sl] == i)[1], int((m[sl] == i).sum())
                if n > 1:
                    problems.append((t["tile_id"], i, f"{n} disconnected pieces"))
                if area < min_area:
                    problems.append((t["tile_id"], i, f"{area} px"))
            masks[t["tile_id"]] = m
    return manifest, masks, problems


def next_round(
    pack: Dict[str, Any],
    export: PathLike,
    pack_id: Optional[str] = None,
    name: str = "Round 1 labels",
) -> Dict[str, Any]:
    """Make the next round's pack: tiles the annotator accepted or edited start from their labels.

    The returned labels become the first prefill model (key ``labels``), so they open by default;
    the original models stay available. Tiles not in the export are unchanged.
    """
    manifest, masks, _ = read_export(export, min_area=1)
    if manifest.get("pack_id") != pack["pack_id"]:
        raise ValueError(f"export is for pack {manifest.get('pack_id')!r}, not {pack['pack_id']!r}")
    out = json.loads(json.dumps(pack))
    out["pack_id"] = pack_id or f"{pack['pack_id']}-r2"
    out["prefill_models"] = [dict(key="labels", name=name)] + [
        m for m in pack["prefill_models"] if m["key"] != "labels"
    ]
    out["prefill_model"] = "labels"
    rows = {t["tile_id"]: t for t in manifest["tiles"]}
    for t in out["tiles"]:
        if t["id"] in masks:
            t["prefills"]["labels"] = rle(masks[t["id"]])
            t["prefill"] = t["prefills"]["labels"]
            t["note"] = (
                t.get("note", "") + " · reviewed"
                if rows[t["id"]].get("done")
                else t.get("note", "")
            )
    return out


def write_masks(export: PathLike, out_dir: PathLike) -> List[Path]:
    """Write each tile's mask from an export as ``<tile_id>.tif``, plus the manifest."""
    import tifffile

    manifest, masks, _ = read_export(export)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for tid, m in masks.items():
        p = out / f"{tid}.tif"
        tifffile.imwrite(p, m.astype(np.uint16))
        paths.append(p)
    (out / "labels.json").write_text(json.dumps(manifest, indent=1))
    return paths
