"""Feature extraction backend using CellProfiler headless CLI.

Runs CellProfiler in headless mode via subprocess:
    cellprofiler -c -r -p pipeline.cppipe -i input_dir -o output_dir

The user builds a pipeline in the CellProfiler GUI, exports as .cppipe,
and provides the path; without one, goudacell's default pipeline
(``data/goudacell_default.cppipe``) is filled in for the staged channels and masks.
GoudaCell handles the I/O plumbing: writes image channels and masks as individual
TIFFs, runs CP, and reads the output CSV.

Requires: a CellProfiler install, usually the ``goudacell_cp`` conda env made by
``scripts/setup_cellprofiler_env.sh`` (see :func:`find_cellprofiler`).
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tifffile

from goudacell.environment import CELLPROFILER_ENV

logger = logging.getLogger(__name__)

# Pipeline object name -> (staged mask file stem, goudacell column prefix)
STAGED_OBJECTS = {
    "Nuclei": ("nuclei_mask", "nucleus"),
    "Cells": ("cell_mask", "cell"),
    "Cytoplasm": ("cytoplasm_mask", "cytoplasm"),
}

SETUP_SCRIPT = "scripts/setup_cellprofiler_env.sh"
# CellProfiler versions the default pipeline and the staging are tested with
SUPPORTED_VERSION = "4.2."
DEFAULT_PIPELINE = Path(__file__).parent / "data" / "goudacell_default.cppipe"
# Objects the default pipeline measures neighbors of (cytoplasm borders follow the cells)
NEIGHBOR_OBJECTS = ("Nuclei", "Cells")


def find_cellprofiler() -> Optional[str]:
    """Find a CellProfiler executable without activating any environment.

    Looks, in order, at the ``GOUDACELL_CELLPROFILER`` environment variable,
    ``cellprofiler`` on PATH, and ``bin/cellprofiler`` of the ``goudacell_cp`` conda env
    (under the conda base of ``CONDA_EXE`` or of this interpreter, else as listed by
    ``conda env list --json``).

    Returns:
        The command, or None if no CellProfiler was found.
    """
    return _locate_cellprofiler()[0]


def _locate_cellprofiler() -> Tuple[Optional[str], str]:
    """The CellProfiler command :func:`find_cellprofiler` picks, and where it came from."""
    if os.environ.get("GOUDACELL_CELLPROFILER"):
        return os.environ["GOUDACELL_CELLPROFILER"], "the GOUDACELL_CELLPROFILER env var"
    if shutil.which("cellprofiler"):
        return shutil.which("cellprofiler"), "cellprofiler on PATH"
    source = f"the '{CELLPROFILER_ENV}' conda env"
    conda_exe = os.environ.get("CONDA_EXE")
    bases = [Path(conda_exe).parent.parent] if conda_exe else []
    if Path(sys.prefix).parent.name == "envs":
        bases.append(Path(sys.prefix).parent.parent)
    for base in bases:
        cmd = base / "envs" / CELLPROFILER_ENV / "bin" / "cellprofiler"
        if os.access(cmd, os.X_OK):
            return str(cmd), source
    # Envs outside the base (other envs_dirs) are only known to conda itself
    conda = conda_exe or shutil.which("conda")
    if conda is None:
        return None, ""
    try:
        listed = subprocess.run(
            [conda, "env", "list", "--json"], capture_output=True, text=True, timeout=60
        )
        envs = [Path(env) for env in json.loads(listed.stdout)["envs"]]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None, ""
    for env in envs:
        cmd = env / "bin" / "cellprofiler"
        if env.name == CELLPROFILER_ENV and os.access(cmd, os.X_OK):
            return str(cmd), source
    return None, ""


# The cellprofiler_cmd goudacell 0.3 wrote into every CellProfiler config
LEGACY_DEFAULT_CMD = "cellprofiler"
_legacy_noted = False


def _unset_legacy_default(cellprofiler_cmd: Optional[str]) -> Optional[str]:
    """Treat 0.3's bare ``cellprofiler`` default as unset when it isn't on PATH."""
    global _legacy_noted
    if cellprofiler_cmd != LEGACY_DEFAULT_CMD or shutil.which(LEGACY_DEFAULT_CMD):
        return cellprofiler_cmd
    if not _legacy_noted:
        logger.warning(
            "cellprofiler_cmd: %s (goudacell 0.3's default) is not on PATH; finding "
            "CellProfiler instead (GOUDACELL_CELLPROFILER, then the '%s' conda env)",
            LEGACY_DEFAULT_CMD,
            CELLPROFILER_ENV,
        )
        _legacy_noted = True
    return None


# Command -> version of the CellProfilers that passed check_cellprofiler (failures rerun)
_CHECKED_VERSIONS = {}


def check_cellprofiler(cellprofiler_cmd: Optional[str] = None) -> str:
    """Check that a CellProfiler command runs and is a supported version (4.2.x).

    Runs ``<cmd> --version`` once per command; a passing command is cached.

    Args:
        cellprofiler_cmd: CellProfiler executable. If None, or the bare ``cellprofiler``
            older configs carry while none is on PATH, found by :func:`find_cellprofiler`.

    Returns:
        The checked command.

    Raises:
        RuntimeError: If no CellProfiler is found, the command doesn't exist, isn't
            CellProfiler, or isn't a supported version; the message says what to do.
    """
    cellprofiler_cmd = _unset_legacy_default(cellprofiler_cmd)
    if cellprofiler_cmd in _CHECKED_VERSIONS:
        return cellprofiler_cmd
    located, source = _locate_cellprofiler()
    if cellprofiler_cmd and cellprofiler_cmd != located:
        source = "cellprofiler_cmd, set in the config or the notebook's CP command"
    cellprofiler_cmd = cellprofiler_cmd or located
    fix = (
        f"Run {SETUP_SCRIPT} to create the '{CELLPROFILER_ENV}' conda env, or set "
        "GOUDACELL_CELLPROFILER (or cellprofiler_cmd) to a CellProfiler 4.2 executable."
    )
    if cellprofiler_cmd is None:
        raise RuntimeError(
            "CellProfiler not found (looked at GOUDACELL_CELLPROFILER, cellprofiler on PATH "
            f"and the '{CELLPROFILER_ENV}' conda env). {fix}"
        )
    found = f"CellProfiler command {cellprofiler_cmd!r} (from {source})"
    if shutil.which(cellprofiler_cmd) is None:
        raise RuntimeError(f"{found} doesn't exist or isn't executable. {fix}")
    try:
        run = subprocess.run(
            [cellprofiler_cmd, "--version"], capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise RuntimeError(f"{found} failed to run ({err}). {fix}") from err
    versions = re.findall(r"^(\d+\.\d+\S*)\s*$", run.stdout, re.MULTILINE)
    if run.returncode != 0 or not versions:
        raise RuntimeError(
            f"{found} is not a working CellProfiler: '--version' exited {run.returncode} "
            f"without a version.\n{(run.stderr or run.stdout).strip()[-1000:]}\n{fix}"
        )
    if not versions[-1].startswith(SUPPORTED_VERSION):
        raise RuntimeError(f"{found} is CellProfiler {versions[-1]}, not 4.2.x. {fix}")
    _CHECKED_VERSIONS[cellprofiler_cmd] = versions[-1]
    return cellprofiler_cmd


def default_pipeline(
    channel_names: List[str],
    objects: Tuple[str, ...] = tuple(STAGED_OBJECTS),
    include_texture: bool = True,
    include_correlation: bool = True,
    include_neighbors: bool = True,
) -> str:
    """Fill in goudacell's default pipeline for the staged channels and masks.

    The pipeline loads ``<channel>.tif`` as grayscale images and the staged masks as
    objects, runs MeasureObjectIntensity, MeasureObjectSizeShape, MeasureTexture,
    MeasureColocalization and MeasureObjectNeighbors on them, and exports one CSV per
    object. Write the text to a ``.cppipe`` to open or edit it in the CellProfiler GUI.

    Args:
        channel_names: Channel names, used as file stems and CellProfiler image names.
        objects: Staged objects to load, any of "Nuclei", "Cells", "Cytoplasm".
        include_texture: Keep MeasureTexture.
        include_correlation: Keep MeasureColocalization (needs two channels).
        include_neighbors: Keep MeasureObjectNeighbors (nuclei and cells).

    Returns:
        The pipeline as .cppipe text.
    """
    # (file stem, image type, image name, object name); the unused name keeps its default
    assignments = [(name, "Grayscale image", name, "Cell") for name in channel_names]
    assignments += [(STAGED_OBJECTS[obj][0], "Objects", "DNA", obj) for obj in objects]
    rules = []
    for stem, kind, image_name, object_name in assignments:
        # CellProfiler unescapes backslashes in rule strings, so the regex's backslashes are doubled
        pattern = re.escape(f"{stem}.tif").replace("\\", "\\\\")
        rules += [
            f'    Select the rule criteria:and (file does containregexp "^{pattern}$")',
            f"    Name to assign these images:{image_name}",
            f"    Name to assign these objects:{object_name}",
            f"    Select the image type:{kind}",
            "    Set intensity range from:Image metadata",
            "    Maximum intensity:255.0",
        ]
    fields = {
        "@ASSIGNMENT_COUNT@": str(len(assignments)),
        "@ASSIGNMENTS@": "\n".join(rules),
        "@IMAGES@": ", ".join(channel_names),
        "@OBJECTS@": ", ".join(objects),
    }
    header, *modules = DEFAULT_PIPELINE.read_text().strip().split("\n\n")
    dropped = {
        "MeasureTexture": not include_texture,
        "MeasureColocalization": not include_correlation or len(channel_names) < 2,
        "MeasureObjectNeighbors": not include_neighbors,
    }
    kept = []
    for module in modules:
        name = module.split(":", 1)[0]
        if dropped.get(name):
            continue
        if name == "MeasureObjectNeighbors":
            kept += [
                module.replace("@NEIGHBOR_OBJECT@", obj)
                for obj in NEIGHBOR_OBJECTS
                if obj in objects
            ]
        else:
            kept.append(module)
    kept = [re.sub(r"module_num:\d+", f"module_num:{i}", m, 1) for i, m in enumerate(kept, 1)]
    text = "\n\n".join([header.replace("@MODULE_COUNT@", str(len(kept))), *kept]) + "\n"
    for field, value in fields.items():
        text = text.replace(field, value)
    return text


def extract_features_cellprofiler(
    image: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
    pipeline_file: Optional[Union[str, Path]] = None,
    cellprofiler_cmd: Optional[str] = None,
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
    Without a pipeline, :func:`default_pipeline` is run for these channels and masks.

    Args:
        image: Multichannel image (C, H, W).
        nuclei_masks: Labeled nuclear mask (H, W).
        cell_masks: Optional labeled cell mask (H, W).
        channel_names: Names for each channel. Used as filenames.
        pipeline_file: Path to .cppipe pipeline file. If None, goudacell's default
            pipeline is used.
        cellprofiler_cmd: CellProfiler executable, e.g. the full path to a CP install in
            another conda env. If None, found by :func:`find_cellprofiler`.
        output_dir: Directory for the staged input and CP output, kept afterwards. If
            None, a fresh directory in the current working directory is used and removed
            (never /tmp on shared HPC).
        include_texture: Keep the default pipeline's MeasureTexture (a given pipeline
            controls this itself).
        include_correlation: Keep the default pipeline's MeasureColocalization.
        include_neighbors: Keep the default pipeline's MeasureObjectNeighbors.
        cytoplasm_masks: Optional labeled cytoplasm mask (H, W).
        timeout: Seconds before the CellProfiler run is killed.

    Returns:
        DataFrame with a ``label`` column (the mask label) and the CellProfiler
        measurements of every exported object table.

    Raises:
        FileNotFoundError: If pipeline_file doesn't exist.
        RuntimeError: If the executable is not a supported CellProfiler
            (:func:`check_cellprofiler`), CellProfiler fails, or it exports no object table.
    """
    if pipeline_file is not None:
        pipeline_file = Path(pipeline_file).resolve()
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")

    cellprofiler_cmd = check_cellprofiler(cellprofiler_cmd)

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
        if pipeline_file is None:
            pipeline_file = staging_dir.resolve() / DEFAULT_PIPELINE.name
            pipeline_file.write_text(
                default_pipeline(
                    channel_names,
                    objects=tuple(labels),
                    include_texture=include_texture,
                    include_correlation=include_correlation,
                    include_neighbors=include_neighbors,
                )
            )

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
    cellprofiler_cmd: Optional[str] = None,
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
        cellprofiler_cmd: CellProfiler executable path. If None, found by
            :func:`find_cellprofiler`.
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
        _unset_legacy_default(cellprofiler_cmd) or find_cellprofiler() or "cellprofiler",
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
