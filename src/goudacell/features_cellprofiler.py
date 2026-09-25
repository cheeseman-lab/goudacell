"""Feature extraction backend using CellProfiler headless CLI.

Runs CellProfiler in headless mode via subprocess:
    cellprofiler -c -r -p pipeline.cppipe -i input_dir -o output_dir

The user builds a pipeline in the CellProfiler GUI, exports as .cppipe,
and provides the path. GoudaCell handles the I/O plumbing: writes image
channels and masks as individual TIFFs, runs CP, and reads the output CSV.

Requires: cellprofiler installed in PATH (can be a separate conda env).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tifffile

# Pipeline object name -> (staged mask file stem, goudacell column prefix)
STAGED_OBJECTS = {
    "Nuclei": ("nuclei_mask", "nucleus"),
    "Cells": ("cell_mask", "cell"),
    "Cytoplasm": ("cytoplasm_mask", "cytoplasm"),
}


def extract_features_cellprofiler(
    image: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
    pipeline_file: Optional[Union[str, Path]] = None,
    cellprofiler_cmd: str = "cellprofiler",
    output_dir: Optional[Union[str, Path]] = None,
    include_texture: bool = True,
    include_correlation: bool = True,
    include_neighbors: bool = True,
    cytoplasm_masks: Optional[np.ndarray] = None,
    timeout: int = 600,
) -> pd.DataFrame:
    """Extract features by running CellProfiler headlessly via CLI.

    Writes each channel as ``<channel name>.tif`` and the masks as ``nuclei_mask.tif``,
    ``cell_mask.tif`` and ``cytoplasm_mask.tif`` (uint32 labels) into one input folder,
    runs the pipeline on it, and reads the per-object CSVs its ExportToSpreadsheet writes.
    The pipeline's NamesAndTypes must pick those files by name, channels as grayscale
    images and the masks as objects named ``Nuclei``, ``Cells`` and ``Cytoplasm``. Those
    three object tables become ``nucleus_``, ``cell_`` and ``cytoplasm_`` columns joined on
    ``label`` (the mask label); tables of other objects the pipeline creates are not joined.

    Args:
        image: Multichannel image (C, H, W).
        nuclei_masks: Labeled nuclear mask (H, W).
        cell_masks: Optional labeled cell mask (H, W).
        channel_names: Names for each channel. Used as filenames.
        pipeline_file: Path to .cppipe pipeline file. Required.
        cellprofiler_cmd: CellProfiler executable (default: "cellprofiler").
            Can be a full path to a CP install in another conda env.
        output_dir: Directory for the staged input and CP output, kept afterwards. If
            None, a fresh directory in the current working directory is used and removed
            (never /tmp on shared HPC).
        include_texture: Unused (pipeline controls this). Kept for API compat.
        include_correlation: Unused (pipeline controls this). Kept for API compat.
        include_neighbors: Unused (pipeline controls this). Kept for API compat.
        cytoplasm_masks: Optional labeled cytoplasm mask (H, W).
        timeout: Seconds before the CellProfiler run is killed.

    Returns:
        DataFrame with a ``label`` column (the mask label) and the CellProfiler
        measurements of every exported object table.

    Raises:
        FileNotFoundError: If pipeline_file doesn't exist.
        RuntimeError: If the executable is not found, CellProfiler fails, or it
            exports no object table.
    """
    if pipeline_file is None:
        raise ValueError(
            "pipeline_file is required for the 'cellprofiler' extraction method. "
            "Build a pipeline in the CellProfiler GUI and export as .cppipe."
        )

    pipeline_file = Path(pipeline_file).resolve()
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")

    if shutil.which(cellprofiler_cmd) is None:
        raise RuntimeError(
            f"CellProfiler executable not found: '{cellprofiler_cmd}'. "
            "Install CellProfiler or provide the full path via cellprofiler_cmd."
        )

    if image.ndim == 2:
        image = image[np.newaxis, ...]
    if image.ndim != 3:
        raise ValueError(f"Image must be (C, H, W), got shape {image.shape}")

    n_channels = image.shape[0]
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    # Not a dot-directory: CellProfiler's default Images filter skips hidden folders
    if output_dir is None:
        staging_dir = Path(tempfile.mkdtemp(prefix="goudacell_cp_", dir="."))
    else:
        staging_dir = Path(output_dir)
    input_dir = staging_dir / "input"
    cp_output_dir = staging_dir / "output"
    cp_temp_dir = staging_dir / "tmp"
    for folder in (input_dir, cp_output_dir, cp_temp_dir):
        folder.mkdir(parents=True, exist_ok=True)

    try:
        for ch_idx, ch_name in enumerate(channel_names):
            tifffile.imwrite(input_dir / f"{ch_name}.tif", image[ch_idx])
        # Masks are staged as 1..n so CellProfiler's ObjectNumber maps back to the label
        masks = {"Nuclei": nuclei_masks, "Cells": cell_masks, "Cytoplasm": cytoplasm_masks}
        labels = {}
        for obj, mask in masks.items():
            if mask is not None:
                labels[obj] = np.unique(mask[mask > 0])
                staged = np.zeros(mask.shape, np.uint32)
                staged[mask > 0] = np.searchsorted(labels[obj], mask[mask > 0]) + 1
                tifffile.imwrite(input_dir / f"{STAGED_OBJECTS[obj][0]}.tif", staged)

        # -t keeps CellProfiler's temporary files out of /tmp
        cmd = [
            cellprofiler_cmd,
            "-c",
            "-r",
            "-p", str(pipeline_file),
            "-i", str(input_dir.resolve()),
            "-o", str(cp_output_dir.resolve()),
            "-t", str(cp_temp_dir.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            staged = sorted(f.name for f in input_dir.iterdir())
            raise RuntimeError(
                f"CellProfiler failed (exit code {result.returncode}); its NamesAndTypes must "
                f"match the staged files {staged}:\n{result.stderr[-2000:]}"
            )

        # ExportToSpreadsheet names files <prefix><object>.csv (default prefix "MyExpt_")
        tables = []
        for csv_file in sorted(cp_output_dir.glob("*.csv")):
            obj = csv_file.stem.split("_")[-1]
            if obj in labels:
                tables.append(_label_object_table(pd.read_csv(csv_file), obj, labels[obj]))
        if not tables:
            raise RuntimeError(
                f"CellProfiler exported no Nuclei/Cells/Cytoplasm table to {cp_output_dir}; "
                "the pipeline needs those objects from the staged masks and "
                f"ExportToSpreadsheet (CSV).\n{result.stderr[-2000:]}"
            )
        result_df = tables[0]
        for df in tables[1:]:
            result_df = result_df.merge(df, on="label", how="outer")
        return result_df.sort_values("label").reset_index(drop=True)

    finally:
        if output_dir is None:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _label_object_table(df: pd.DataFrame, obj: str, labels: np.ndarray) -> pd.DataFrame:
    """Prefix an exported object table's columns by compartment and key it by mask label."""
    prefix = STAGED_OBJECTS[obj][1]
    features = df.drop(columns=["ImageNumber", "ObjectNumber"], errors="ignore")
    features = features.rename(columns=lambda c: f"{prefix}_{c}")
    label = pd.Series(labels[df["ObjectNumber"].to_numpy() - 1], index=df.index, name="label")
    return pd.concat([label, features], axis=1)


def run_cellprofiler_batch(
    pipeline_file: Union[str, Path],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    cellprofiler_cmd: str = "cellprofiler",
    first_image: Optional[int] = None,
    last_image: Optional[int] = None,
    group: Optional[dict] = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess:
    """Run CellProfiler headlessly on a directory of images.

    Lower-level function for batch processing. For Slurm job arrays,
    use first_image/last_image to split work across jobs.

    Args:
        pipeline_file: Path to .cppipe pipeline file.
        input_dir: Directory containing input images.
        output_dir: Directory for CP output (CSVs, etc.).
        cellprofiler_cmd: CellProfiler executable path.
        first_image: First image set number to process (1-indexed).
        last_image: Last image set number to process (1-indexed).
        group: Grouping variables dict (e.g., {"Well": "A01"}).
        timeout: Timeout in seconds (default 1 hour).

    Returns:
        subprocess.CompletedProcess with stdout/stderr.
    """
    pipeline_file = Path(pipeline_file)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    (output_dir / "tmp").mkdir(parents=True, exist_ok=True)

    cmd = [
        cellprofiler_cmd,
        "-c",  # headless
        "-r",  # run
        "-p", str(pipeline_file),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-t", str(output_dir / "tmp"),  # CellProfiler's temp files, not /tmp
    ]

    if first_image is not None:
        cmd.extend(["-f", str(first_image)])
    if last_image is not None:
        cmd.extend(["-l", str(last_image)])
    if group:
        group_str = ",".join(f"{k}={v}" for k, v in group.items())
        cmd.extend(["-g", group_str])

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
