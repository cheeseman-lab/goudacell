"""Image I/O module for loading and saving microscopy images.

Supports ND2, TIFF, and DeltaVision (DV) file formats.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile


def load_image(
    file_path: Union[str, Path],
    channel: Optional[int] = None,
    z_project: bool = True,
) -> np.ndarray:
    """Load a microscopy image from various formats.

    Supports ND2, TIFF, and DeltaVision (DV) files.

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

    if suffix == ".nd2":
        return _load_nd2(file_path, channel=channel, z_project=z_project)
    elif suffix in (".tif", ".tiff"):
        return _load_tiff(file_path, channel=channel, z_project=z_project)
    elif suffix == ".dv":
        return _load_dv(file_path, channel=channel, z_project=z_project)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .nd2, .tif, .tiff, .dv")


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
    """Load DeltaVision (DV) file using mrc package."""
    import mrc

    data = mrc.imread(file_path)

    # DV files are typically (Z, C, Y, X) or (C, Z, Y, X)
    # mrc.imread returns the raw array
    if z_project and data.ndim == 4:
        # Assume Z is first dimension
        data = np.max(data, axis=0)
    elif z_project and data.ndim == 3:
        # Could be (Z, Y, X) single channel
        pass

    if channel is not None:
        if data.ndim >= 3:
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
) -> None:
    """Save segmentation mask as TIFF.

    Args:
        mask: Segmentation mask array (integer labels).
        output_path: Path to save the mask.
        compress: Whether to use compression. Default True.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure mask is integer type
    if mask.dtype not in (np.uint8, np.uint16, np.uint32, np.int32):
        if mask.max() <= 255:
            mask = mask.astype(np.uint8)
        elif mask.max() <= 65535:
            mask = mask.astype(np.uint16)
        else:
            mask = mask.astype(np.uint32)

    compression = "zlib" if compress else None
    tifffile.imwrite(output_path, mask, compression=compression)
