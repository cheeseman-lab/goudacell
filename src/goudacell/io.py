"""Image I/O module for loading and saving microscopy images.

Supports ND2, TIFF, DeltaVision (DV), and OME-Zarr file formats.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile

# Zarr v3 / OME-NGFF v0.5 format
ZARR_FORMAT = 3

PathLike = Union[str, Path]


def load_image(
    file_path: Union[str, Path],
    channel: Optional[int] = None,
    z_project: bool = True,
) -> np.ndarray:
    """Load a microscopy image from various formats.

    Supports ND2, TIFF, DeltaVision (DV), and OME-Zarr files.

    Args:
        file_path: Path to the image file.
        channel: Specific channel to load (0-indexed). If None, loads all channels.
        z_project: Whether to max-project Z-stacks. Default True.

    Returns:
        Image array with shape (C, Y, X) or (Y, X) if single channel.

    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    # Detect zarr: .zarr suffix, zarr.json sentinel, or directory within .zarr
    if _is_zarr_path(file_path):
        return _load_zarr(file_path, channel=channel, z_project=z_project)
    elif suffix == ".nd2":
        return _load_nd2(file_path, channel=channel, z_project=z_project)
    elif suffix in (".tif", ".tiff"):
        return _load_tiff(file_path, channel=channel, z_project=z_project)
    elif suffix == ".dv":
        return _load_dv(file_path, channel=channel, z_project=z_project)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported: .nd2, .tif, .tiff, .dv, .zarr"
        )


def _load_nd2(
    file_path: Path,
    channel: Optional[int] = None,
    z_project: bool = True,
) -> np.ndarray:
    """Load ND2 file using nd2 library."""
    import nd2

    with nd2.ND2File(file_path) as f:
        data = f.asarray()

    # Handle different dimension orders
    # ND2 files typically have shape (T, Z, C, Y, X) or subsets
    data = _normalize_dimensions(data, f.sizes if hasattr(f, "sizes") else None)

    if z_project and data.ndim > 3:
        # Assume Z is axis 0 if we have 4D after normalization
        data = np.max(data, axis=0)

    if channel is not None:
        if data.ndim >= 3:
            data = data[channel]

    return data


def _load_tiff(
    file_path: Path,
    channel: Optional[int] = None,
    z_project: bool = True,
) -> np.ndarray:
    """Load TIFF file using tifffile."""
    data = tifffile.imread(file_path)

    # Handle Z-stacks (assume shape is Z, C, Y, X or Z, Y, X or C, Y, X)
    if z_project and data.ndim == 4:
        # Assume (Z, C, Y, X)
        data = np.max(data, axis=0)
    elif z_project and data.ndim == 3:
        # Could be (Z, Y, X) or (C, Y, X) - check metadata or assume C, Y, X
        pass

    if channel is not None:
        if data.ndim >= 3:
            data = data[channel]

    return data


def _load_dv(
    file_path: Path,
    channel: Optional[int] = None,
    z_project: bool = True,
) -> np.ndarray:
    """Load DeltaVision (DV) file using mrc package.

    The mrc library typically returns DV files as (C, Z, Y, X).
    This function handles z-projection and channel selection to return
    data in (C, Y, X) format.
    """
    import mrc

    data = mrc.imread(file_path)

    # mrc.imread returns DV files as (C, Z, Y, X) for 4D data
    if data.ndim == 4:
        # Shape is (C, Z, Y, X)
        if z_project:
            # Max project along Z (axis 1)
            data = np.max(data, axis=1)  # Now (C, Y, X)

        if channel is not None:
            data = data[channel]  # Now (Y, X)

    elif data.ndim == 3:
        # Could be (Z, Y, X) single channel or (C, Y, X)
        # Heuristic: if first dim is much smaller than others, likely C or Z
        if z_project and data.shape[0] > 10:
            # Assume (Z, Y, X), project along Z
            data = np.max(data, axis=0)
        # If channel specified and still 3D, take that channel
        if channel is not None and data.ndim == 3:
            data = data[channel]

    return data


def _normalize_dimensions(data: np.ndarray, sizes: Optional[dict] = None) -> np.ndarray:
    """Normalize image dimensions to (C, Y, X) or (Z, C, Y, X)."""
    # Simple heuristic: Y and X should be the largest dimensions
    # This is a basic implementation - may need refinement based on actual data
    return data


def save_mask(
    mask: np.ndarray,
    output_path: Union[str, Path],
    compress: bool = True,
    pixel_size: Optional[Union[float, Tuple[float, ...]]] = None,
) -> None:
    """Save segmentation mask as TIFF or OME-Zarr.

    Output format is determined by the file extension of output_path.
    Use .zarr for OME-Zarr, .tif/.tiff for TIFF.

    Args:
        mask: Segmentation mask array (integer labels).
        output_path: Path to save the mask.
        compress: Whether to use compression (TIFF only). Default True.
        pixel_size: Pixel size in microns (OME-Zarr only).
    """
    output_path = Path(output_path)

    # Ensure mask is integer type
    if mask.dtype not in (np.uint8, np.uint16, np.uint32, np.int32):
        if mask.max() <= 255:
            mask = mask.astype(np.uint8)
        elif mask.max() <= 65535:
            mask = mask.astype(np.uint16)
        else:
            mask = mask.astype(np.uint32)

    if _is_zarr_path(output_path):
        save_image(mask, output_path, is_label=True, pixel_size=pixel_size)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compression = "zlib" if compress else None
        tifffile.imwrite(output_path, mask, compression=compression)


# ---------------------------------------------------------------------------
# OME-Zarr high-level API
# ---------------------------------------------------------------------------


def save_image(
    image: np.ndarray,
    output_path: PathLike,
    *,
    pixel_size: Optional[Union[float, Tuple[float, ...]]] = None,
    channel_names: Optional[Sequence[str]] = None,
    coarsening_factor: int = 2,
    max_levels: int = 5,
    is_label: bool = False,
) -> None:
    """Save an image to TIFF or OME-Zarr depending on the output path suffix.

    Args:
        image: Image array. Shape (Y, X), (C, Y, X), or (C, Z, Y, X).
        output_path: Path to save. Use .zarr for OME-Zarr, .tif for TIFF.
        pixel_size: Pixel size in microns (float for isotropic, tuple for (y, x)).
        channel_names: Names for each channel.
        coarsening_factor: Downscale factor per pyramid level.
        max_levels: Number of pyramid levels.
        is_label: Whether image is a label/mask image.
    """
    out = Path(output_path)
    suffix = out.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out), image)
        return

    # Handle zarr.json sentinel path
    if out.name == "zarr.json":
        out = out.parent

    if _is_zarr_path(out) or suffix == "":
        # Normalize to 5D TCZYX
        data = image
        if image.ndim == 2:
            data = image[np.newaxis, np.newaxis, np.newaxis, ...]
        elif image.ndim == 3:
            data = image[np.newaxis, :, np.newaxis, ...]
        elif image.ndim == 4:
            data = image[np.newaxis, ...]
        else:
            raise ValueError(f"Unsupported image.ndim={image.ndim} for OME-Zarr export")

        axes = "tczyx"
        ch_names = list(channel_names) if channel_names is not None else None
        if ch_names is None:
            c_len = int(data.shape[axes.index("c")])
            ch_names = [f"c{i}" for i in range(c_len)]

        _write_image_omezarr(
            image_data=data,
            out_path=str(out),
            channel_names=ch_names,
            axes=axes,
            pixel_size_um=pixel_size,
            coarsening_factor=coarsening_factor,
            max_levels=max_levels,
            is_label=is_label,
        )
        return

    raise ValueError(f"Unsupported output path: {out}")


# ---------------------------------------------------------------------------
# OME-Zarr writing (adapted from brieflow zarr3 branch)
# ---------------------------------------------------------------------------


def _write_image_omezarr(
    image_data: np.ndarray,
    out_path: str,
    channel_names: Optional[List[str]] = None,
    axes: str = "tczyx",
    pixel_size_um: Optional[Union[float, Tuple[float, ...], Dict[str, float]]] = None,
    coarsening_factor: int = 2,
    max_levels: int = 4,
    is_label: bool = False,
    chunk_size: Optional[Tuple[int, ...]] = None,
) -> None:
    """Write an image array to OME-Zarr format with pyramids.

    Args:
        image_data: Array containing image data (should be 5D TCZYX).
        out_path: Path to the output .zarr directory.
        channel_names: List of channel names.
        axes: String describing axes, e.g., "tczyx".
        pixel_size_um: Pixel size in microns.
        coarsening_factor: Downscale factor per level.
        max_levels: Number of pyramid levels.
        is_label: Whether the image is a label image.
        chunk_size: Tuple for chunking.
    """
    import dask.array as da
    import zarr
    from ome_zarr.scale import Scaler
    from ome_zarr.writer import write_image

    os.makedirs(out_path, exist_ok=True)
    root = zarr.open_group(out_path, mode="w", zarr_format=ZARR_FORMAT)

    if not isinstance(image_data, da.Array):
        if len(axes) != len(image_data.shape):
            raise ValueError(
                f"Axes '{axes}' (len={len(axes)}) does not match "
                f"image_data.ndim={len(image_data.shape)} with shape={image_data.shape}."
            )
        if chunk_size is None:
            shape = image_data.shape
            chunks = list(shape)
            if "y" in axes and "x" in axes:
                y_idx = axes.find("y")
                x_idx = axes.find("x")
                chunks[y_idx] = min(shape[y_idx], 1024)
                chunks[x_idx] = min(shape[x_idx], 1024)
            chunk_size = tuple(chunks)
        image_data = da.from_array(image_data, chunks=chunk_size)

    ps = _parse_pixel_sizes(pixel_size_um)

    coordinate_transformations = []
    for i in range(max_levels):
        scale_transform = [1.0] * len(axes)
        if "z" in axes and ps.get("z") is not None:
            scale_transform[axes.find("z")] = ps["z"]
        if "y" in axes and ps.get("y") is not None:
            scale_transform[axes.find("y")] = ps["y"] * (coarsening_factor**i)
        if "x" in axes and ps.get("x") is not None:
            scale_transform[axes.find("x")] = ps["x"] * (coarsening_factor**i)
        coordinate_transformations.append([{"scale": scale_transform, "type": "scale"}])

    metadata: Dict[str, Any] = {}
    omero: Dict[str, Any] = {}
    if channel_names:
        if is_label:
            omero["channels"] = [{"label": name, "active": True} for name in channel_names]
        else:
            omero["channels"] = [
                {"label": name, "active": True, "color": "FFFFFF"} for name in channel_names
            ]
    if any(v is not None for v in ps.values()):
        omero["pixel_size"] = {k: v for k, v in ps.items() if v is not None}
    if omero:
        metadata["omero"] = omero
    if is_label:
        metadata["image-label"] = {"version": "0.5"}

    write_image(
        image=image_data,
        group=root,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        scaler=Scaler(
            method="nearest" if is_label else "gaussian",
            downscale=coarsening_factor,
            max_layer=max_levels - 1,
            labeled=is_label,
        ),
        **metadata,
    )

    # Merge metadata into OME namespace for v0.5 compliance
    if metadata:
        ome_attrs = dict(root.attrs.get("ome", {}))
        for k, v in metadata.items():
            ome_attrs[k] = v
        root.attrs["ome"] = ome_attrs


# ---------------------------------------------------------------------------
# Zarr helpers
# ---------------------------------------------------------------------------


def _is_zarr_path(path: Path) -> bool:
    """Check if a path points to a zarr store."""
    p = Path(path)
    if p.name == "zarr.json":
        return True
    if p.suffix.lower() == ".zarr":
        return True
    if any(part.endswith(".zarr") for part in p.parts):
        return True
    return False


def _load_zarr(
    file_path: Path,
    channel: Optional[int] = None,
    z_project: bool = True,
) -> np.ndarray:
    """Load image from OME-Zarr store.

    Returns highest-resolution (level 0) array, squeezed to (C, Y, X) or (Y, X).
    """
    import zarr

    p = Path(file_path)
    if p.name == "zarr.json":
        p = p.parent

    root = zarr.open_group(str(p), mode="r")

    # Find the dataset path from multiscales metadata
    ds_path: Optional[str] = None
    ms = root.attrs.get("multiscales")
    if isinstance(ms, list) and ms:
        datasets = ms[0].get("datasets", [])
        if datasets and isinstance(datasets, list):
            ds_path = datasets[0].get("path")
    # Check ome namespace (zarr v3)
    if ds_path is None:
        ome = root.attrs.get("ome", {})
        ms = ome.get("multiscales")
        if isinstance(ms, list) and ms:
            datasets = ms[0].get("datasets", [])
            if datasets and isinstance(datasets, list):
                ds_path = datasets[0].get("path")
    if ds_path is None and "0" in root:
        ds_path = "0"
    if ds_path is None:
        raise ValueError(f"Could not find image data in OME-Zarr at {p}")

    arr = root[ds_path][:]

    # Squeeze 5D TCZYX to CYX
    if arr.ndim == 5:
        arr = arr[0]  # squeeze T → CZYX
    if arr.ndim == 4:
        if z_project and arr.shape[1] > 1:
            arr = np.max(arr, axis=1)  # max-project Z → CYX
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :, :]  # squeeze singleton Z → CYX
    # Squeeze singleton channel
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if channel is not None and arr.ndim >= 3:
        arr = arr[channel]

    return arr


def _parse_pixel_sizes(
    ps: Optional[Union[float, Tuple[float, ...], Dict[str, float]]],
) -> Dict[str, Optional[float]]:
    """Parse pixel size specification into a dict of {x, y, z} floats."""
    if ps is None:
        return {"x": None, "y": None, "z": None}
    if isinstance(ps, (int, float)):
        v = float(ps)
        return {"x": v, "y": v, "z": None}
    if isinstance(ps, dict):
        return {
            "x": float(ps["x"]) if ps.get("x") is not None else None,
            "y": float(ps["y"]) if ps.get("y") is not None else None,
            "z": float(ps["z"]) if ps.get("z") is not None else None,
        }
    if isinstance(ps, tuple):
        if len(ps) == 2:
            y, x = ps
            return {
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "z": None,
            }
        if len(ps) == 3:
            z, y, x = ps
            return {
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "z": float(z) if z is not None else None,
            }
    raise ValueError(
        f"Unsupported pixel_size_um type: {type(ps)}. Expected float, tuple, or dict."
    )
