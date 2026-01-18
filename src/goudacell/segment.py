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

from typing import Optional, Tuple

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
