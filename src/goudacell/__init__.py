"""GoudaCell: HPC-compatible cell segmentation using Cellpose."""

__version__ = "0.3.0"

from goudacell.features import extract_features
from goudacell.gpu import GPUStatus, detect_gpu, resolve_gpu
from goudacell.io import load_image, save_image, save_mask
from goudacell.segment import (
    GridCell,
    SweepResult,
    get_cellpose_version,
    parameter_sweep,
    reproducible_plateau_2d,
    reproducible_range_1d,
    segment,
    segment_nuclei_and_cells,
    sweep_grid,
    sweep_grid_counts,
)
from goudacell.viz import (
    make_mask_cmap,
    plot_sweep,
    plot_sweep_1d,
    plot_sweep_grid,
    plot_sweep_montage,
)

__all__ = [
    "load_image",
    "save_image",
    "save_mask",
    "segment",
    "segment_nuclei_and_cells",
    "parameter_sweep",
    "sweep_grid_counts",
    "sweep_grid",
    "reproducible_range_1d",
    "reproducible_plateau_2d",
    "SweepResult",
    "GridCell",
    "get_cellpose_version",
    "make_mask_cmap",
    "plot_sweep",
    "plot_sweep_1d",
    "plot_sweep_grid",
    "plot_sweep_montage",
    "extract_features",
    "detect_gpu",
    "resolve_gpu",
    "GPUStatus",
]
