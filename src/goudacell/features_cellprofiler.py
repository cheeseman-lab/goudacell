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
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tifffile


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
) -> pd.DataFrame:
    """Extract features by running CellProfiler headlessly via CLI.

    Writes image channels and masks as individual TIFFs to a staging directory,
    runs the CellProfiler pipeline, and reads the resulting measurement CSV.

    Args:
        image: Multichannel image (C, H, W).
        nuclei_masks: Labeled nuclear mask (H, W).
        cell_masks: Optional labeled cell mask (H, W).
        channel_names: Names for each channel. Used as filenames.
        pipeline_file: Path to .cppipe pipeline file. Required.
        cellprofiler_cmd: CellProfiler executable (default: "cellprofiler").
            Can be a full path to a CP install in another conda env.
        output_dir: Directory for CP output. If None, uses a temp directory
            in the current working directory (never /tmp on shared HPC).
        include_texture: Unused (pipeline controls this). Kept for API compat.
        include_correlation: Unused (pipeline controls this). Kept for API compat.
        include_neighbors: Unused (pipeline controls this). Kept for API compat.

    Returns:
        DataFrame with CellProfiler measurements, or empty DataFrame on failure.

    Raises:
        FileNotFoundError: If pipeline_file doesn't exist.
        RuntimeError: If cellprofiler executable is not found.
    """
    if pipeline_file is None:
        raise ValueError(
            "pipeline_file is required for the 'cellprofiler' extraction method. "
            "Build a pipeline in the CellProfiler GUI and export as .cppipe."
        )

    pipeline_file = Path(pipeline_file)
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")

    # Check that cellprofiler is available
    cp_path = shutil.which(cellprofiler_cmd)
    if cp_path is None:
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

    # Set up staging directories (in cwd, never /tmp on shared HPC)
    staging_dir = Path(output_dir) if output_dir else Path(".cp_staging")
    input_dir = staging_dir / "input"
    cp_output_dir = staging_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    cp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write individual channel images as TIFFs
        for ch_idx, ch_name in enumerate(channel_names):
            tifffile.imwrite(
                input_dir / f"{ch_name}.tif",
                image[ch_idx],
            )

        # Write masks
        tifffile.imwrite(input_dir / "nuclei_mask.tif", nuclei_masks.astype(np.uint32))
        if cell_masks is not None:
            tifffile.imwrite(input_dir / "cell_mask.tif", cell_masks.astype(np.uint32))

        # Run CellProfiler headlessly
        cmd = [
            cellprofiler_cmd,
            "-c",  # headless (no GUI)
            "-r",  # run immediately
            "-p", str(pipeline_file),
            "-i", str(input_dir),
            "-o", str(cp_output_dir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per image
        )

        if result.returncode != 0:
            print(f"CellProfiler failed (exit code {result.returncode}):")
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return pd.DataFrame(columns=["label"])

        # Find and read the output CSV(s)
        # CellProfiler typically writes files like:
        #   MyExpt_Nuclei.csv, MyExpt_Cells.csv, MyExpt_Image.csv
        csv_files = sorted(cp_output_dir.glob("*.csv"))
        if not csv_files:
            # Also check for .txt exports
            csv_files = sorted(cp_output_dir.glob("*.txt"))

        if not csv_files:
            print(f"Warning: No CSV output found in {cp_output_dir}")
            return pd.DataFrame(columns=["label"])

        # Merge all object-level CSVs (skip Image-level)
        dfs = []
        for csv_file in csv_files:
            if "image" in csv_file.stem.lower():
                continue  # Skip image-level measurements
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=["label"])

        # If multiple object tables, merge on ObjectNumber
        if len(dfs) == 1:
            return dfs[0]

        result_df = dfs[0]
        for df in dfs[1:]:
            # Find common merge key
            merge_key = None
            for key in ["ObjectNumber", "label", "ImageNumber"]:
                if key in result_df.columns and key in df.columns:
                    merge_key = key
                    break
            if merge_key:
                result_df = result_df.merge(df, on=merge_key, how="outer", suffixes=("", "_dup"))
            else:
                result_df = pd.concat([result_df, df], axis=1)

        return result_df

    finally:
        # Clean up staging directory if we created it
        if output_dir is None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


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
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        cellprofiler_cmd,
        "-c",  # headless
        "-r",  # run
        "-p", str(pipeline_file),
        "-i", str(input_dir),
        "-o", str(output_dir),
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
