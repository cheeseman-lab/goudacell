"""Cellpose-based cell segmentation module.

This module provides GPU-compatible cell segmentation using Cellpose.
Supports both Cellpose 3.x and 4.x with automatic version detection.

Cellpose 3.x (3.1.0):
    - Models: cyto3, nuclei, cyto2, cyto
    - Good for rounded cells
    - Supports automatic diameter estimation

Cellpose 4.x (4.0.4+):
    - Models: cpsam (Cellpose-SAM)
    - Better for complex cell shapes
    - Requires explicit diameter specification
    - Requires GPU
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte


def get_cellpose_version() -> Tuple[int, int]:
    """Get the installed Cellpose version.

    Returns:
        Tuple of (major, minor) version numbers.

    Raises:
        ImportError: If Cellpose is not installed.
    """
    try:
        import cellpose

        version_str = cellpose.version if hasattr(cellpose, "version") else cellpose.__version__
        parts = version_str.split(".")[:2]
        return (int(parts[0]), int(parts[1]))
    except ImportError:
        raise ImportError(
            "Cellpose is not installed. Install with either:\n"
            "  uv pip install -e '.[cellpose3]'  # For cyto3, nuclei models\n"
            "  uv pip install -e '.[cellpose4]'  # For cpsam model"
        )


def _is_cellpose_4x() -> bool:
    """Check if Cellpose 4.x is installed."""
    version = get_cellpose_version()
    return version[0] >= 4


def _validate_model(model: str) -> None:
    """Validate model compatibility with installed Cellpose version."""
    is_4x = _is_cellpose_4x()
    version = get_cellpose_version()

    if is_4x and model != "cpsam":
        raise ValueError(
            f"Model '{model}' requires Cellpose 3.x. "
            f"Cellpose 4.x only supports the 'cpsam' model. "
            f"Either use model='cpsam', or install Cellpose 3.x: "
            f"uv pip install cellpose==3.1.0"
        )
    if not is_4x and model == "cpsam":
        raise ValueError(
            f"CPSAM model requires Cellpose 4.x. "
            f"You have Cellpose {version[0]}.{version[1]}. "
            f"Upgrade with: uv pip install cellpose==4.0.4 torch==2.7.0 torchvision==0.22.0"
        )


def segment(
    image: np.ndarray,
    diameter: float,
    model: str = "cyto3",
    channels: Optional[list] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu: bool = True,
    remove_edge_cells: bool = True,
) -> np.ndarray:
    """Segment cells using Cellpose.

    Args:
        image: Input image array. Can be:
            - 2D (Y, X): Single channel grayscale
            - 3D (C, Y, X): Multi-channel
        diameter: Estimated cell diameter in pixels.
        model: Cellpose model to use:
            - Cellpose 3.x: 'cyto3' (default), 'nuclei', 'cyto2', 'cyto'
            - Cellpose 4.x: 'cpsam' only
        channels: Channel specification for Cellpose [cytoplasm, nucleus].
            For grayscale: [0, 0]
            For RGB with cytoplasm in green, nuclei in blue: [2, 3]
            If None, auto-detected based on image shape.
        flow_threshold: Flow error threshold. Lower = fewer cells. Default 0.4.
        cellprob_threshold: Cell probability threshold. Higher = fewer cells. Default 0.0.
        gpu: Whether to use GPU. Default True.
        remove_edge_cells: Remove cells touching image border. Default True.

    Returns:
        Segmentation mask with integer labels (0 = background).

    Raises:
        ImportError: If Cellpose is not installed.
        ValueError: If model is incompatible with Cellpose version.
    """
    _validate_model(model)

    from cellpose.models import CellposeModel

    # Determine channels if not specified
    if channels is None:
        if image.ndim == 2:
            channels = [0, 0]  # Grayscale
        elif image.ndim == 3 and image.shape[0] <= 4:
            # Assume (C, Y, X), use first channel as cytoplasm
            channels = [1, 0] if image.shape[0] == 1 else [2, 3]
        else:
            channels = [0, 0]

    # Create model based on Cellpose version
    if _is_cellpose_4x():
        cellpose_model = CellposeModel(pretrained_model=model, gpu=gpu)
    else:
        cellpose_model = CellposeModel(model_type=model, gpu=gpu)

    # Run segmentation
    masks, flows, styles = cellpose_model.eval(
        image,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    # Remove cells touching borders
    if remove_edge_cells:
        masks = clear_border(masks)

    return masks


def segment_nuclei_and_cells(
    image: np.ndarray,
    nuclei_channel: int,
    cyto_channel: int,
    nuclei_diameter: float,
    cell_diameter: float,
    cell_model: str = "cyto3",
    nuclei_model: str = "nuclei",
    nuclei_flow_threshold: float = 0.4,
    nuclei_cellprob_threshold: float = 0.0,
    cell_flow_threshold: float = 0.4,
    cell_cellprob_threshold: float = 0.0,
    gpu: bool = True,
    remove_edge_cells: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment both nuclei and cells from a multi-channel image.

    Args:
        image: Input image array with shape (C, Y, X).
        nuclei_channel: Index of the nuclear channel (e.g., DAPI).
        cyto_channel: Index of the cytoplasmic channel.
        nuclei_diameter: Estimated nuclear diameter in pixels.
        cell_diameter: Estimated cell diameter in pixels.
        cell_model: Cellpose model for cell segmentation (e.g., "cyto3", "cpsam").
        nuclei_model: Cellpose model for nuclei segmentation (e.g., "nuclei", "cpsam").
        nuclei_flow_threshold: Flow threshold for nuclei segmentation.
        nuclei_cellprob_threshold: Cell prob threshold for nuclei segmentation.
        cell_flow_threshold: Flow threshold for cell segmentation.
        cell_cellprob_threshold: Cell prob threshold for cell segmentation.
        gpu: Whether to use GPU.
        remove_edge_cells: Remove cells touching image border.

    Returns:
        Tuple of (nuclei_mask, cell_mask).
    """
    _validate_model(cell_model)
    _validate_model(nuclei_model)

    from cellpose.models import CellposeModel

    # Prepare RGB image for cellpose
    rgb = _prepare_rgb(image, nuclei_channel, cyto_channel)

    # Create models
    if _is_cellpose_4x():
        nuclei_cp_model = CellposeModel(pretrained_model=nuclei_model, gpu=gpu)
        cell_cp_model = CellposeModel(pretrained_model=cell_model, gpu=gpu)
    else:
        nuclei_cp_model = CellposeModel(model_type=nuclei_model, gpu=gpu)
        cell_cp_model = CellposeModel(model_type=cell_model, gpu=gpu)

    # Segment nuclei (using blue channel = DAPI)
    nuclei_masks, _, _ = nuclei_cp_model.eval(
        rgb[2],  # Blue channel (DAPI)
        diameter=nuclei_diameter,
        flow_threshold=nuclei_flow_threshold,
        cellprob_threshold=nuclei_cellprob_threshold,
    )

    # Segment cells
    if _is_cellpose_4x():
        # Cellpose 4.x auto-detects channels from image shape (no channels param)
        cell_masks, _, _ = cell_cp_model.eval(
            rgb,
            diameter=cell_diameter,
            flow_threshold=cell_flow_threshold,
            cellprob_threshold=cell_cellprob_threshold,
        )
    else:
        # Cellpose 3.x: use [cytoplasm, nuclei] channel spec
        cell_masks, _, _ = cell_cp_model.eval(
            rgb,
            diameter=cell_diameter,
            channels=[2, 3],  # Green=cyto, Blue=nuclei
            flow_threshold=cell_flow_threshold,
            cellprob_threshold=cell_cellprob_threshold,
        )

    # Remove edge cells
    if remove_edge_cells:
        nuclei_masks = clear_border(nuclei_masks)
        cell_masks = clear_border(cell_masks)

    return nuclei_masks, cell_masks


def _prepare_rgb(
    image: np.ndarray,
    nuclei_channel: int,
    cyto_channel: int,
    logscale: bool = True,
) -> np.ndarray:
    """Prepare a 3-channel RGB image for Cellpose.

    Cellpose expects: [Red, Green, Blue] where typically:
    - Red: helper channel (zeros if not used)
    - Green: cytoplasm channel
    - Blue: nuclei channel (DAPI)

    Args:
        image: Input image with shape (C, Y, X).
        nuclei_channel: Index of nuclear channel.
        cyto_channel: Index of cytoplasmic channel.
        logscale: Whether to apply log scaling to cytoplasm.

    Returns:
        RGB image with shape (3, Y, X) as uint8.
    """
    dapi = image[nuclei_channel].astype(np.float32)
    cyto = image[cyto_channel].astype(np.float32)
    helper = np.zeros_like(cyto)

    # Log scale cytoplasm channel
    if logscale:
        cyto = np.log1p(cyto)
        if cyto.max() > 0:
            cyto = cyto / cyto.max()

    # Normalize DAPI
    dapi_upper = np.percentile(dapi, 99.5)
    if dapi_upper > 0:
        dapi = dapi / dapi_upper
    dapi = np.clip(dapi, 0, 1)

    # Convert to uint8
    red = img_as_ubyte(helper)
    green = img_as_ubyte(cyto)
    blue = img_as_ubyte(dapi)

    return np.array([red, green, blue])


# Parameters that can be swept, mapped to the attribute suffix they override.
SWEEP_PARAM_SUFFIX = {
    "diameter": "diameter",
    "flow": "flow_threshold",
    "cellprob": "cellprob_threshold",
}

# DualSegmentationParams field names, splatted into segment_nuclei_and_cells.
_DUAL_FIELDS = (
    "nuclei_channel",
    "cyto_channel",
    "nuclei_diameter",
    "cell_diameter",
    "cell_model",
    "nuclei_model",
    "nuclei_flow_threshold",
    "nuclei_cellprob_threshold",
    "cell_flow_threshold",
    "cell_cellprob_threshold",
)


@dataclass
class SweepResult:
    """One point of a parameter sweep.

    Attributes:
        value: The swept parameter value used for this point.
        masks: Labelled mask for the swept target ("nuclei" or "cells").
        count: Number of objects found (labels excluding background).
    """

    value: float
    masks: np.ndarray
    count: int


@dataclass
class GridCell:
    """One cell of a 2D (or 1D) parameter-grid sweep.

    Attributes:
        ix: Column index (x axis).
        iy: Row index (y axis); 0 for a 1D sweep.
        x: x-axis parameter value.
        y: y-axis parameter value, or None for a 1D sweep.
        masks: Labelled mask for the swept target at this combination.
        count: Number of objects found.
    """

    ix: int
    iy: int
    x: float
    y: Optional[float]
    masks: np.ndarray
    count: int


def _count_objects(masks: np.ndarray) -> int:
    """Count labelled objects in a mask, excluding background (label 0)."""
    return int(len(set(np.unique(masks)) - {0}))


def _resolve_target(mode: str, target: Optional[str]) -> str:
    """Validate mode/target and return the effective compartment to vary."""
    if mode not in ("nuclei", "cells", "dual"):
        raise ValueError(f"Unknown mode '{mode}'. Choose from 'nuclei', 'cells', 'dual'.")
    target = (target or "cells") if mode == "dual" else mode
    if target not in ("nuclei", "cells"):
        raise ValueError(f"Unknown target '{target}'. Choose 'nuclei' or 'cells'.")
    return target


def _param_attr(target: str, param: str) -> str:
    """Map a (target, param) pair to its DualSegmentationParams attribute name."""
    if param not in SWEEP_PARAM_SUFFIX:
        raise ValueError(
            f"Unknown sweep parameter '{param}'. Choose from {list(SWEEP_PARAM_SUFFIX)}."
        )
    prefix = "nuclei" if target == "nuclei" else "cell"
    return f"{prefix}_{SWEEP_PARAM_SUFFIX[param]}"


def _segment_with_overrides(
    image, params, overrides, mode, target, gpu, remove_edge_cells
) -> np.ndarray:
    """Segment once with baseline ``params`` plus per-attribute ``overrides``.

    ``overrides`` keys are DualSegmentationParams attribute names (e.g.
    ``cell_flow_threshold``). Returns the mask for ``target``.
    """
    if mode == "dual":
        kwargs = {field: getattr(params, field) for field in _DUAL_FIELDS}
        kwargs.update(overrides)
        nuclei_masks, cell_masks = segment_nuclei_and_cells(
            image, gpu=gpu, remove_edge_cells=remove_edge_cells, **kwargs
        )
        return nuclei_masks if target == "nuclei" else cell_masks

    prefix = "nuclei" if target == "nuclei" else "cell"
    channel = params.nuclei_channel if target == "nuclei" else params.cyto_channel
    seg_image = image[channel] if image.ndim == 3 else image
    baseline = {
        "model": getattr(params, f"{prefix}_model"),
        "diameter": getattr(params, f"{prefix}_diameter"),
        "flow_threshold": getattr(params, f"{prefix}_flow_threshold"),
        "cellprob_threshold": getattr(params, f"{prefix}_cellprob_threshold"),
    }
    for attr, value in overrides.items():
        baseline[attr.split("_", 1)[1]] = value  # e.g. cell_flow_threshold -> flow_threshold
    return segment(
        seg_image,
        diameter=baseline["diameter"],
        model=baseline["model"],
        flow_threshold=baseline["flow_threshold"],
        cellprob_threshold=baseline["cellprob_threshold"],
        gpu=gpu,
        remove_edge_cells=remove_edge_cells,
    )


def parameter_sweep(
    image: np.ndarray,
    params: object,
    param: str,
    values: List[float],
    *,
    mode: str = "dual",
    target: Optional[str] = None,
    gpu: bool = True,
    remove_edge_cells: bool = False,
) -> List[SweepResult]:
    """Sweep one segmentation parameter, returning a mask + count per value.

    A single helper for all three modes. Everything except the swept parameter
    is held at the baseline in ``params``.

    Args:
        image: Multi-channel image (C, Y, X), or 2D for a single channel.
        params: Baseline parameters with DualSegmentationParams-style attributes
            (e.g. ``nuclei_diameter``, ``cell_flow_threshold``, ``cyto_channel``).
        param: Parameter to vary — one of "diameter", "flow", "cellprob".
        values: Values to test for ``param``.
        mode: Segmentation mode — "nuclei", "cells", or "dual".
        target: Which compartment to vary and return. Required for "dual";
            defaults to "cells". Ignored for single modes (set to the mode).
        gpu: Whether to use the GPU.
        remove_edge_cells: Whether to drop objects touching the border.

    Returns:
        One :class:`SweepResult` per value, in order.

    Raises:
        ValueError: If ``param``, ``mode``, or ``target`` is invalid.
    """
    target = _resolve_target(mode, target)
    attr = _param_attr(target, param)

    results: List[SweepResult] = []
    for value in values:
        masks = _segment_with_overrides(
            image, params, {attr: value}, mode, target, gpu, remove_edge_cells
        )
        results.append(SweepResult(value=value, masks=masks, count=_count_objects(masks)))
    return results


def sweep_grid_counts(
    image: np.ndarray,
    params: object,
    axis_x: tuple,
    axis_y: Optional[tuple] = None,
    *,
    mode: str = "dual",
    target: Optional[str] = None,
    gpu: bool = True,
    remove_edge_cells: bool = False,
) -> np.ndarray:
    """Sweep one or two parameters and return only object counts (no masks).

    Memory-light grid sweep for finding a reproducible operating range. For a
    2D sweep the result is indexed ``counts[y, x]``.

    Args:
        image: Multi-channel image (C, Y, X), or 2D for a single channel.
        params: Baseline DualSegmentationParams-style parameters.
        axis_x: ``(param, values)`` for the x axis.
        axis_y: Optional ``(param, values)`` for the y axis (None -> 1D sweep).
        mode: Segmentation mode — "nuclei", "cells", or "dual".
        target: Compartment to vary (required for "dual"; default "cells").
        gpu: Whether to use the GPU.
        remove_edge_cells: Whether to drop objects touching the border.

    Returns:
        1D array of counts (no axis_y) or 2D array ``counts[y, x]``.
    """
    target = _resolve_target(mode, target)
    param_x, values_x = axis_x
    attr_x = _param_attr(target, param_x)

    if axis_y is None:
        counts = np.zeros(len(values_x), dtype=int)
        for i, vx in enumerate(values_x):
            masks = _segment_with_overrides(
                image, params, {attr_x: vx}, mode, target, gpu, remove_edge_cells
            )
            counts[i] = _count_objects(masks)
        return counts

    param_y, values_y = axis_y
    attr_y = _param_attr(target, param_y)
    counts = np.zeros((len(values_y), len(values_x)), dtype=int)
    for iy, vy in enumerate(values_y):
        for ix, vx in enumerate(values_x):
            masks = _segment_with_overrides(
                image, params, {attr_x: vx, attr_y: vy}, mode, target, gpu, remove_edge_cells
            )
            counts[iy, ix] = _count_objects(masks)
    return counts


def sweep_grid(
    image: np.ndarray,
    params: object,
    axis_x: tuple,
    axis_y: Optional[tuple] = None,
    *,
    mode: str = "dual",
    target: Optional[str] = None,
    gpu: bool = True,
    remove_edge_cells: bool = False,
) -> List[GridCell]:
    """Sweep one or two parameters, returning the mask + count for each cell.

    Like :func:`sweep_grid_counts` but keeps the masks so the results can be
    shown as overlays. Use for modest grids (it holds every mask in memory).

    Args:
        image: Multi-channel image (C, Y, X), or 2D for a single channel.
        params: Baseline DualSegmentationParams-style parameters.
        axis_x: ``(param, values)`` for the x axis.
        axis_y: Optional ``(param, values)`` for the y axis (None -> 1D sweep).
        mode: Segmentation mode — "nuclei", "cells", or "dual".
        target: Compartment to vary (required for "dual"; default "cells").
        gpu: Whether to use the GPU.
        remove_edge_cells: Whether to drop objects touching the border.

    Returns:
        A flat list of :class:`GridCell` in row-major order.
    """
    target = _resolve_target(mode, target)
    param_x, values_x = axis_x
    attr_x = _param_attr(target, param_x)

    cells: List[GridCell] = []
    if axis_y is None:
        for ix, vx in enumerate(values_x):
            masks = _segment_with_overrides(
                image, params, {attr_x: vx}, mode, target, gpu, remove_edge_cells
            )
            cells.append(GridCell(ix=ix, iy=0, x=vx, y=None, masks=masks,
                                  count=_count_objects(masks)))
        return cells

    param_y, values_y = axis_y
    attr_y = _param_attr(target, param_y)
    for iy, vy in enumerate(values_y):
        for ix, vx in enumerate(values_x):
            masks = _segment_with_overrides(
                image, params, {attr_x: vx, attr_y: vy}, mode, target, gpu, remove_edge_cells
            )
            cells.append(GridCell(ix=ix, iy=iy, x=vx, y=vy, masks=masks,
                                  count=_count_objects(masks)))
    return cells


def reproducible_range_1d(values: List[float], counts, tol: float = 0.15) -> Optional[dict]:
    """Find the widest contiguous value range where the object count is stable.

    "Stable" means the count varies by at most ``tol`` (relative to the local
    mean) across a 3-point window — i.e. the result is insensitive to the exact
    parameter value, the hallmark of a reproducible setting.

    Args:
        values: Swept parameter values (ascending).
        counts: Object count at each value.
        tol: Maximum allowed relative count spread within the local window.

    Returns:
        Dict with ``lo``, ``hi`` (range bounds), ``setpoint`` (suggested value),
        and ``i_lo``/``i_hi`` (index bounds); or None if nothing is stable.
    """
    counts = np.asarray(counts, dtype=float)
    n = len(counts)
    if n < 2:
        return None

    # A transition between adjacent values is "stable" if the count barely moves.
    stable = np.zeros(n - 1, dtype=bool)
    for i in range(n - 1):
        pair = counts[i : i + 2]
        local_mean = max(pair.mean(), 1.0)
        stable[i] = (pair.max() - pair.min()) / local_mean <= tol

    # Longest run of consecutive stable transitions -> widest reproducible range.
    best_start, best_end, cur_start = 0, -1, None
    for i in range(n - 1):
        if stable[i]:
            cur_start = i if cur_start is None else cur_start
            if i - cur_start > best_end - best_start:
                best_start, best_end = cur_start, i
        else:
            cur_start = None

    if best_end < best_start:
        return None
    i_lo, i_hi = best_start, best_end + 1  # transitions span one extra point
    mid = (i_lo + i_hi) // 2
    return {
        "lo": values[i_lo],
        "hi": values[i_hi],
        "setpoint": values[mid],
        "i_lo": i_lo,
        "i_hi": i_hi,
    }


def reproducible_plateau_2d(counts, tol: float = 0.15) -> Optional[dict]:
    """Find the largest connected stable region in a 2D count grid.

    Each cell is "stable" if the object count varies by at most ``tol``
    (relative to the local mean) across its 3x3 neighbourhood. The largest
    connected block of stable cells is the reproducible plateau.

    Args:
        counts: 2D array of object counts, indexed ``counts[y, x]``.
        tol: Maximum allowed relative count spread within the local window.

    Returns:
        Dict with boolean ``mask`` (the plateau) and ``iy``/``ix`` (a suggested
        setpoint cell inside it); or None if nothing is stable.
    """
    from scipy import ndimage

    counts = np.asarray(counts, dtype=float)
    ny, nx = counts.shape
    stable = np.zeros_like(counts, dtype=bool)
    for y in range(ny):
        for x in range(nx):
            window = counts[max(0, y - 1) : min(ny, y + 2), max(0, x - 1) : min(nx, x + 2)]
            local_mean = max(window.mean(), 1.0)
            stable[y, x] = (window.max() - window.min()) / local_mean <= tol

    if not stable.any():
        return None

    labels, n_labels = ndimage.label(stable)
    sizes = [int((labels == k).sum()) for k in range(1, n_labels + 1)]
    best = int(np.argmax(sizes)) + 1
    region = labels == best

    ys, xs = np.where(region)
    iy, ix = int(round(ys.mean())), int(round(xs.mean()))
    if not region[iy, ix]:
        nearest = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
        iy, ix = int(ys[nearest]), int(xs[nearest])
    return {"mask": region, "iy": iy, "ix": ix}


def estimate_diameter(
    image: np.ndarray,
    model: str = "cyto3",
    channels: Optional[list] = None,
    gpu: bool = True,
) -> float:
    """Estimate optimal cell diameter using Cellpose SizeModel.

    Note: Only available with Cellpose 3.x. Cellpose 4.x does not support
    automatic diameter estimation.

    Args:
        image: Input image array.
        model: Cellpose model type.
        channels: Channel specification.
        gpu: Whether to use GPU.

    Returns:
        Estimated diameter in pixels.

    Raises:
        NotImplementedError: If using Cellpose 4.x.
    """
    if _is_cellpose_4x():
        raise NotImplementedError(
            "Automatic diameter estimation is not supported with Cellpose 4.x. "
            "Please specify diameter explicitly, or use Cellpose 3.x: "
            "uv pip install cellpose==3.1.0"
        )

    from cellpose import models as cellpose_models
    from cellpose.models import CellposeModel, SizeModel

    if channels is None:
        channels = [0, 0] if image.ndim == 2 else [2, 3]

    cp_model = CellposeModel(model_type=model, gpu=gpu)
    size_model = SizeModel(
        cp_model=cp_model,
        pretrained_size=cellpose_models.size_model_path(model),
    )

    diameter, _ = size_model.eval(image, channels=channels)
    diameter = max(5.0, float(diameter))

    return diameter
