"""Cellpose segmentation: thin adapters over brieflow's vendored phenotype code.

Dual (nuclei + cells) and nuclei-only segmentation call brieflow's ``segment_cellpose``
(``goudacell.brieflow.shared.segment_cellpose``) exactly as brieflow-analysis's phenotype
notebook does; diameters come from brieflow's ``estimate_diameters``; reconciliation and
cytoplasm identification are brieflow's. Cells-only segmentation, sweeps and the
reproducibility helpers are goudacell extras. Supports Cellpose 3.x and 4.x.

Cellpose 3.x (3.1.0):
    - Models: cyto3, nuclei, cyto2, cyto
    - Supports automatic diameter estimation

Cellpose 4.x (4.0.4+):
    - Models: cpsam (Cellpose-SAM)
    - Diameters are measured from a diameter-free run instead of estimated

Either version accepts a path to a custom trained model in place of a model name.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from skimage.measure import regionprops

from goudacell.brieflow.phenotype.identify_cytoplasm_cellpose import (
    identify_cytoplasm_cellpose,
)
from goudacell.brieflow.shared.segmentation_utils import (  # noqa: F401
    count_nuclei_per_cell,
    reconcile_nuclei_cells,
)


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
    return get_cellpose_version()[0] >= 4


def _is_custom_model(model: Optional[str]) -> bool:
    """Whether ``model`` is a path to a custom trained model rather than a built-in name."""
    return model is not None and ("/" in model or "\\" in model)


def brieflow_nuclei_model() -> str:
    """The model brieflow segments nuclei with next to cells: nuclei (3.x) or cpsam (4.x)."""
    return "cpsam" if _is_cellpose_4x() else "nuclei"


def create_cellpose_model(model: str, gpu: bool = False):
    """Create a CellposeModel via brieflow's version-aware ``create_cellpose_model``.

    Args:
        model: Built-in model name (e.g. "cyto3", "nuclei", "cpsam") or a custom model path.
        gpu: Whether to use the GPU.

    Returns:
        An initialized ``cellpose.models.CellposeModel``.

    Raises:
        ValueError: If the model is incompatible with the installed Cellpose version.
    """
    from goudacell.brieflow.shared import segment_cellpose as bf

    return bf.create_cellpose_model(model, gpu=gpu)


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
    """Segment cells in one image with Cellpose (goudacell's cells-only mode).

    brieflow has no cells-only mode; this runs one Cellpose model on the image as given.

    Args:
        image: Input image array. Can be:
            - 2D (Y, X): Single channel grayscale
            - 3D (C, Y, X): Multi-channel
        diameter: Estimated cell diameter in pixels.
        model: Cellpose model to use:
            - Cellpose 3.x: 'cyto3' (default), 'nuclei', 'cyto2', 'cyto'
            - Cellpose 4.x: 'cpsam' only
            - Either version: a path to a custom trained model
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
    from skimage.segmentation import clear_border

    # Determine channels if not specified
    if channels is None:
        if image.ndim == 2:
            channels = [0, 0]  # Grayscale
        elif image.ndim == 3 and image.shape[0] <= 4:
            # Assume (C, Y, X), use first channel as cytoplasm
            channels = [1, 0] if image.shape[0] == 1 else [2, 3]
        else:
            channels = [0, 0]

    cellpose_model = create_cellpose_model(model, gpu=gpu)

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
    nuclei_diameter: Optional[float],
    cell_diameter: Optional[float],
    cell_model: str = "cyto3",
    nuclei_model: Optional[str] = None,
    nuclei_flow_threshold: float = 0.4,
    nuclei_cellprob_threshold: float = 0.0,
    cell_flow_threshold: float = 0.4,
    cell_cellprob_threshold: float = 0.0,
    gpu: bool = True,
    remove_edge_cells: bool = True,
    reconcile: Optional[str] = "consensus",
    helper_channel: Optional[int] = None,
    return_nuclei_per_cell: bool = False,
) -> tuple:
    """Segment nuclei and cells with brieflow's ``segment_cellpose``.

    The phenotype notebook's segmentation cell: the channels are merged into a
    (helper, cyto, DAPI) RGB image, nuclei are segmented on the DAPI plane with brieflow's
    nuclei model and cells on the RGB image with ``cell_model``, edge objects are
    cleared, then the masks are reconciled.

    Args:
        image: Input image array with shape (C, Y, X).
        nuclei_channel: Index of the nuclear channel (brieflow ``DAPI_INDEX``).
        cyto_channel: Index of the cytoplasmic channel (brieflow ``CYTO_INDEX``).
        nuclei_diameter: Nuclear diameter in pixels (None lets Cellpose pick).
        cell_diameter: Cell diameter in pixels (None lets Cellpose pick).
        cell_model: Cellpose model for cells ("cyto3", "cpsam", or a custom model path).
        nuclei_model: Accepted for backward compatibility; brieflow always segments nuclei
            with :func:`brieflow_nuclei_model`, so any other value is ignored with a warning.
        nuclei_flow_threshold: Flow threshold for nuclei segmentation.
        nuclei_cellprob_threshold: Cell prob threshold for nuclei segmentation.
        cell_flow_threshold: Flow threshold for cell segmentation.
        cell_cellprob_threshold: Cell prob threshold for cell segmentation.
        gpu: Whether to use GPU.
        remove_edge_cells: Remove objects touching the image border (brieflow always does).
        reconcile: Reconciliation method ("consensus" or "contained_in_cells"); None skips
            it and returns the raw, independently labelled masks.
        helper_channel: Optional channel for the red plane of the Cellpose input
            (brieflow ``HELPER_INDEX``); None leaves it blank.
        return_nuclei_per_cell: Also return ``{cell_label: n_nuclei}``.

    Returns:
        Tuple of (nuclei_mask, cell_mask), plus the nuclei-per-cell dict when
        ``return_nuclei_per_cell`` is set.
    """
    from goudacell.brieflow.shared import segment_cellpose as bf

    if nuclei_model is not None and nuclei_model != brieflow_nuclei_model():
        warnings.warn(
            f"nuclei_model={nuclei_model!r} is ignored: brieflow segments nuclei with "
            f"{brieflow_nuclei_model()!r} next to cells",
            stacklevel=2,
        )

    thresholds = dict(
        nuclei_flow_threshold=nuclei_flow_threshold,
        nuclei_cellprob_threshold=nuclei_cellprob_threshold,
        cell_flow_threshold=cell_flow_threshold,
        cell_cellprob_threshold=cell_cellprob_threshold,
    )
    if remove_edge_cells:
        nuclei, cells, _, nuclei_per_cell = bf.segment_cellpose(
            image,
            dapi_index=nuclei_channel,
            cyto_index=cyto_channel,
            nuclei_diameter=nuclei_diameter,
            cell_diameter=cell_diameter,
            cellpose_kwargs=thresholds,
            cellpose_model=cell_model,
            helper_index=helper_channel,
            gpu=gpu,
            reconcile=reconcile,
            cells=True,
            return_counts=True,
        )
    else:
        # segment_cellpose always clears edges; this is its body with remove_edges=False
        rgb = bf.prepare_cellpose(image, nuclei_channel, cyto_channel, helper_index=helper_channel)
        nuclei, cells, _, nuclei_per_cell = bf.segment_cellpose_rgb(
            rgb,
            nuclei_diameter,
            cell_diameter,
            cellpose_model=cell_model,
            reconcile=reconcile,
            remove_edges=False,
            return_counts=True,
            gpu=gpu,
            nuclei_kwargs=dict(
                flow_threshold=nuclei_flow_threshold, cellprob_threshold=nuclei_cellprob_threshold
            ),
            cell_kwargs=dict(
                flow_threshold=cell_flow_threshold, cellprob_threshold=cell_cellprob_threshold
            ),
        )

    if return_nuclei_per_cell:
        return nuclei, cells, nuclei_per_cell
    return nuclei, cells


def segment_nuclei(
    image: np.ndarray,
    nuclei_channel: int,
    nuclei_diameter: Optional[float],
    model: Optional[str] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu: bool = True,
    remove_edge_cells: bool = True,
) -> np.ndarray:
    """Segment nuclei only with brieflow's ``segment_cellpose(cells=False)``.

    Args:
        image: Input image, (C, Y, X) or a single 2D channel.
        nuclei_channel: Index of the nuclear channel (ignored for a 2D image).
        nuclei_diameter: Nuclear diameter in pixels (None lets Cellpose pick).
        model: Cellpose model (brieflow ``CELLPOSE_MODEL``; e.g. "nuclei", "cyto3", "cpsam"
            or a custom model path). None uses :func:`brieflow_nuclei_model`.
        flow_threshold: Flow error threshold.
        cellprob_threshold: Cell probability threshold.
        gpu: Whether to use GPU.
        remove_edge_cells: Remove nuclei touching the image border (brieflow always does).

    Returns:
        Labeled nuclei mask.
    """
    from goudacell.brieflow.shared import segment_cellpose as bf

    if image.ndim == 2:
        image, nuclei_channel = image[np.newaxis], 0
    model = model or brieflow_nuclei_model()

    if remove_edge_cells:
        return bf.segment_cellpose(
            image,
            dapi_index=nuclei_channel,
            cyto_index=nuclei_channel,
            nuclei_diameter=nuclei_diameter,
            cell_diameter=None,
            cellpose_kwargs=dict(
                nuclei_flow_threshold=flow_threshold, nuclei_cellprob_threshold=cellprob_threshold
            ),
            cellpose_model=model,
            gpu=gpu,
            cells=False,
        )
    # segment_cellpose always clears edges; this is its body with remove_edges=False
    rgb = bf.prepare_cellpose(image, nuclei_channel, nuclei_channel)
    return bf.segment_cellpose_nuclei_rgb(
        rgb,
        nuclei_diameter,
        cellpose_model=model,
        gpu=gpu,
        remove_edges=False,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )


def identify_cytoplasm(nuclei: np.ndarray, cells: np.ndarray) -> Optional[np.ndarray]:
    """Cytoplasm masks from brieflow's ``identify_cytoplasm_cellpose``.

    Args:
        nuclei: Labeled nuclei mask (labels shared with ``cells``).
        cells: Labeled cell mask.

    Returns:
        Labeled cytoplasm mask, or None (with a warning) when the masks are not reconciled
        (:func:`masks_reconciled`; e.g. goudacell's ``reconcile: null``), where brieflow
        would raise or pair unrelated objects.
    """
    if not masks_reconciled(nuclei, cells):
        n_nuclei, n_cells = len(np.unique(nuclei[nuclei > 0])), len(np.unique(cells[cells > 0]))
        warnings.warn(
            f"Skipping cytoplasm: the nuclei and cell masks are not reconciled ({n_nuclei} "
            f"nuclei, {n_cells} cells, labels not paired). Cytoplasm needs reconciled masks; "
            "set `reconcile` (e.g. 'contained_in_cells' or 'consensus') in the config.",
            stacklevel=2,
        )
        return None
    return identify_cytoplasm_cellpose(nuclei, cells)


def masks_reconciled(nuclei: np.ndarray, cells: np.ndarray) -> bool:
    """Whether nuclei and cells share labels as brieflow's ``reconcile_nuclei_cells`` leaves them.

    Reconciled masks have the same label set, and each nucleus overlaps the cell of its own
    label (brieflow pairs a nucleus with the cell under its centre pixel).

    Args:
        nuclei: Labeled nuclei mask.
        cells: Labeled cell mask.

    Returns:
        True if every nucleus label is a cell label that it overlaps, and vice versa.
    """
    nuclei_labels = np.unique(nuclei[nuclei > 0])
    if not np.array_equal(nuclei_labels, np.unique(cells[cells > 0])):
        return False
    paired = np.unique(nuclei[(nuclei > 0) & (nuclei == cells)])
    return np.array_equal(paired, nuclei_labels)


def segment_second_objects(image, nuclei_masks, cell_masks, params, gpu):
    """Detect secondary objects with brieflow's ``segment_second_objs_from_config``.

    Nucleus centroids for the cell-nucleus distances come from the nuclei mask, as in
    brieflow's phenotype notebook.

    Args:
        image: Multichannel image (C, H, W).
        nuclei_masks: Reconciled nuclei mask (centroids for nucleus distances).
        cell_masks: Reconciled cell mask.
        params: SecondaryObjectParams.
        gpu: Whether the ML methods may use the GPU.

    Returns:
        Tuple of (second_obj_masks, cell_second_obj_table, updated_cytoplasm_masks).
    """
    from goudacell.brieflow.phenotype.segment_secondary_object import (
        segment_second_objs_from_config,
    )

    centroids = {r.label: r.centroid for r in regionprops(nuclei_masks)}
    return segment_second_objs_from_config(
        image=image,
        cell_masks=cell_masks,
        cytoplasm_masks=identify_cytoplasm(nuclei_masks, cell_masks),
        second_obj_params=params.to_brieflow_params(gpu),
        nuclei_centroids=centroids,
    )


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
    "helper_channel",
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
    if mode == "nuclei":
        baseline = {
            "model": params.nuclei_model,
            "diameter": params.nuclei_diameter,
            "flow_threshold": params.nuclei_flow_threshold,
            "cellprob_threshold": params.nuclei_cellprob_threshold,
        }
        for attr, value in overrides.items():
            baseline[attr.split("_", 1)[1]] = value
        return segment_nuclei(
            image,
            nuclei_channel=channel,
            nuclei_diameter=baseline["diameter"],
            model=baseline["model"],
            flow_threshold=baseline["flow_threshold"],
            cellprob_threshold=baseline["cellprob_threshold"],
            gpu=gpu,
            remove_edge_cells=remove_edge_cells,
        )

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




def estimate_diameters(
    image: np.ndarray,
    nuclei_channel: int,
    cyto_channel: int,
    cell_model: str = "cyto3",
    helper_channel: Optional[int] = None,
    gpu: bool = True,
) -> Tuple[float, float]:
    """Estimate nuclei and cell diameters with brieflow's ``estimate_diameters``.

    Only Cellpose 3.x built-in models have a size model. For Cellpose 4.x, cpsam and
    custom models, segment with ``diameter=None`` and use :func:`derive_diameters`, as
    brieflow's phenotype notebook does.

    Args:
        image: Input image with shape (C, Y, X).
        nuclei_channel: Index of the nuclear channel.
        cyto_channel: Index of the cytoplasmic channel.
        cell_model: Built-in Cellpose 3.x model for the cell estimate.
        helper_channel: Optional helper channel for the red plane.
        gpu: Whether to use GPU.

    Returns:
        Tuple of (nuclei_diameter, cell_diameter) in pixels.

    Raises:
        NotImplementedError: With Cellpose 4.x, or for cpsam / custom model paths.
    """
    from goudacell.brieflow.shared import segment_cellpose as bf

    if bf.CELLPOSE_4X or cell_model == "cpsam" or _is_custom_model(cell_model):
        raise NotImplementedError(
            "Automatic diameter estimation needs Cellpose 3.x and a built-in model. "
            "Segment with diameter=None and use derive_diameters(), or set diameters "
            "explicitly."
        )
    return bf.estimate_diameters(
        image,
        dapi_index=nuclei_channel,
        cyto_index=cyto_channel,
        helper_index=helper_channel,
        cellpose_model=cell_model,
        gpu=gpu,
    )


def derive_diameters(
    nuclei: np.ndarray, cells: Optional[np.ndarray] = None
) -> Tuple[float, Optional[float]]:
    """Mean equivalent diameter of segmented nuclei and cells.

    How brieflow's phenotype notebook sets the configured diameters for cpsam, which
    has no size model: segment once with ``diameter=None``, then measure the objects.

    Args:
        nuclei: Labeled nuclei mask.
        cells: Optional labeled cell mask.

    Returns:
        Tuple of (nuclei_diameter, cell_diameter); cell_diameter is None without cells.
    """
    nuclei_diameter = float(np.mean([r.equivalent_diameter for r in regionprops(nuclei)]))
    cell_diameter = None
    if cells is not None:
        cell_diameter = float(np.mean([r.equivalent_diameter for r in regionprops(cells)]))
    return nuclei_diameter, cell_diameter
