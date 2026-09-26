"""Feature extraction: thin adapters over brieflow's vendored phenotype code.

The cp_emulator and cp_measure backends call brieflow's ``extract_phenotype_cp_emulator``
and ``extract_phenotype_cp_measure`` as brieflow-analysis's phenotype notebook does, with
cytoplasms from brieflow's ``identify_cytoplasm_cellpose``. goudacell's selection options map
onto brieflow's: channels onto its per-compartment channel lists, the texture/correlation
toggles onto its feature tables (the skipped groups are not computed, so float images work
without them, as before), compartments and neighbors onto its output columns. The CellProfiler
backend is a goudacell extra.
"""

import contextlib
from typing import List, Optional, Union
from unittest import mock

import numpy as np
import pandas as pd

from goudacell.brieflow.external.cp_emulator import texture_features_multichannel
from goudacell.brieflow.phenotype import extract_phenotype_cp_emulator as bf_emulator
from goudacell.segment import identify_cytoplasm

COMPARTMENTS = ("nucleus", "cell", "cytoplasm")

# Columns brieflow's neighbor_measurements(distances=[1]) adds per compartment
NEIGHBOR_COLUMNS = (
    "number_neighbors_1",
    "percent_touching_1",
    "first_neighbor_distance",
    "second_neighbor_distance",
    "angle_between_neighbors",
)


def extract_features(
    image: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: np.ndarray = None,
    channel_names: Optional[List[str]] = None,
    channels: Optional[List[int]] = None,
    compartments: Optional[List[str]] = None,
    include_texture: bool = True,
    include_correlation: bool = True,
    include_neighbors: bool = True,
    foci_channel: Optional[Union[int, List[int]]] = None,
    method: str = "cp_emulator",
    pipeline_file: Optional[str] = None,
    cellprofiler_cmd: Optional[str] = None,
    custom_features: Optional[dict] = None,
) -> pd.DataFrame:
    """Extract CellProfiler-style features from a segmented image.

    Supports three extraction backends:
    - "cp_emulator" (default): brieflow's ``extract_phenotype_cp_emulator``.
    - "cp_measure": brieflow's ``extract_phenotype_cp_measure`` (requires `.[cp_measure]`).
    - "cellprofiler": Runs CellProfiler headlessly via CLI subprocess.
      Requires a CellProfiler install; runs a .cppipe pipeline file or goudacell's default.

    With the defaults, the cp_emulator table is exactly brieflow's for the same masks.

    Args:
        image: Multichannel image array with shape (C, H, W); (H, W) for one channel.
        nuclei_masks: Labeled segmentation mask for nuclei (H, W).
        cell_masks: Optional labeled cell mask (H, W). With reconciled masks, cytoplasm
            features are also extracted (brieflow ``identify_cytoplasm_cellpose``).
        channel_names: Names for each channel, for all channels or for the selected
            ``channels``. If None, defaults to ["ch0", "ch1", ...].
        channels: Channel indices to extract from. If None, all channels are used.
            cp_emulator measures these channels (brieflow's per-compartment channel
            lists); the other backends get the image sliced to them. Any
            ``foci_channel`` indices refer to positions within this subset.
        compartments: Compartments to measure, any of "nucleus", "cell", "cytoplasm".
            If None, all available are measured.
        include_texture: Compute the Haralick and PFTAS texture features (cp_emulator;
            mahotas needs an integer image for them).
        include_correlation: Compute the between-channel correlation/colocalization
            features (cp_emulator; mahotas needs an integer image), or for cp_measure
            keep their columns.
        include_neighbors: Keep the neighbor measurement columns.
        foci_channel: Optional channel index or list of indices for brieflow's foci
            features (cp_emulator only).
        method: Extraction backend. One of "cp_emulator", "cp_measure",
            or "cellprofiler".
        pipeline_file: Path to .cppipe file (cellprofiler method only); None runs
            goudacell's default pipeline.
        cellprofiler_cmd: CellProfiler executable (cellprofiler method only); None finds
            one (``features_cellprofiler.find_cellprofiler``).
        custom_features: Per-compartment custom features as returned by brieflow's
            ``load_custom_features``, each measured on the full multichannel image
            (channel order of the input). cp_emulator only.

    Returns:
        DataFrame with one row per object; column prefixes indicate the compartment
        ("nucleus_", "cell_", "cytoplasm_"), as brieflow's.
    """
    if method not in ("cp_emulator", "cp_measure", "cellprofiler"):
        raise ValueError(
            f"Unknown extraction method '{method}'. "
            "Supported: 'cp_emulator', 'cp_measure', 'cellprofiler'"
        )
    if custom_features and method != "cp_emulator":
        raise ValueError("Custom features require method 'cp_emulator'")
    if foci_channel is not None and method != "cp_emulator":
        raise ValueError("Foci features require method 'cp_emulator'")

    if image.ndim == 2:
        image = image[np.newaxis, ...]
    if image.ndim != 3:
        raise ValueError(f"Image must be (C, H, W), got shape {image.shape}")
    n_channels = image.shape[0]
    selected = list(range(n_channels)) if channels is None else list(channels)
    names = _full_channel_names(channel_names, selected, n_channels)
    selected_names = [names[i] for i in selected]
    # Between-channel features need two channels (brieflow's would raise on one)
    include_correlation = include_correlation and len(selected) > 1

    # Masks brieflow gets: an unrequested compartment is not measured (foci use the cells)
    wanted = set(compartments or COMPARTMENTS) | set(custom_features or {})
    has_cells = cell_masks is not None and np.sum(cell_masks) > 0
    cytoplasm_masks = None
    if has_cells and "cytoplasm" in wanted:
        cytoplasm_masks = identify_cytoplasm(nuclei_masks, cell_masks)
    if not ("cell" in wanted or foci_channel is not None):
        has_cells = False

    if method == "cellprofiler":
        from goudacell.features_cellprofiler import extract_features_cellprofiler

        # Stage every mask the pipeline may load; compartments only drop columns after
        if cytoplasm_masks is None and cell_masks is not None and np.sum(cell_masks) > 0:
            cytoplasm_masks = identify_cytoplasm(nuclei_masks, cell_masks)
        df = extract_features_cellprofiler(
            image[selected],
            nuclei_masks=nuclei_masks,
            cell_masks=cell_masks,
            channel_names=selected_names,
            pipeline_file=pipeline_file,
            cytoplasm_masks=cytoplasm_masks,
            cellprofiler_cmd=cellprofiler_cmd,
            include_texture=include_texture,
            include_correlation=include_correlation,
            include_neighbors=include_neighbors,
        )
        return _keep_compartments(df, compartments)

    if method == "cp_measure":
        from goudacell.brieflow.phenotype.extract_phenotype_cp_measure import (
            extract_phenotype_cp_measure,
        )

        df = extract_phenotype_cp_measure(
            image[selected],
            nuclei=nuclei_masks,
            cells=cell_masks if has_cells else None,
            cytoplasms=cytoplasm_masks,
            channel_names=selected_names,
        )
    else:
        if isinstance(foci_channel, list):
            foci_channel = [selected[fc] for fc in foci_channel]
        elif foci_channel is not None:
            foci_channel = selected[foci_channel]
        with _feature_groups(include_texture, include_correlation):
            df = bf_emulator.extract_phenotype_cp_emulator(
                image,
                nuclei=nuclei_masks,
                cells=cell_masks if has_cells else None,
                wildcards={},
                cytoplasms=cytoplasm_masks,
                nucleus_channels=selected,
                cell_channels=selected,
                cytoplasm_channels=selected,
                foci_channel=foci_channel,
                channel_names=names,
                custom_features=custom_features,
            )

    drop = [
        c for c in df.columns
        if (not include_correlation and "_coloc_" in c)
        or (not include_neighbors and ("_neighbor_" in c or c.endswith(NEIGHBOR_COLUMNS)))
    ]
    return _keep_compartments(df.drop(columns=drop), compartments)


def _feature_groups(include_texture: bool, include_correlation: bool):
    """Scope brieflow's cp_emulator extractor to the chosen feature groups.

    Hands ``extract_phenotype_cp_emulator`` reduced copies of its module-level feature
    tables for the duration of the call; the brieflow code itself is untouched.
    """
    patches = {}
    if not include_texture:
        patches["grayscale_features_multichannel"] = {
            k: v
            for k, v in bf_emulator.grayscale_features_multichannel.items()
            if k not in texture_features_multichannel
        }
    if not include_correlation:
        patches["correlation_features_multichannel"] = {}
    return mock.patch.multiple(bf_emulator, **patches) if patches else contextlib.nullcontext()


def _full_channel_names(
    channel_names: Optional[List[str]], selected: List[int], n_channels: int
) -> List[str]:
    """One name per image channel, from names given for all or for the selected channels."""
    if channel_names is None:
        return [f"ch{i}" for i in range(n_channels)]
    channel_names = list(channel_names)
    if len(channel_names) == n_channels:
        return channel_names
    if len(channel_names) == len(selected):
        names = [f"ch{i}" for i in range(n_channels)]
        for index, name in zip(selected, channel_names):
            names[index] = name
        return names
    raise ValueError(
        f"channel_names has {len(channel_names)} entries; expected {len(selected)} (one "
        f"per selected channel) or {n_channels} (one per image channel)."
    )


def _keep_compartments(df: pd.DataFrame, compartments: Optional[List[str]]) -> pd.DataFrame:
    """Drop the columns of compartments not in ``compartments`` (None keeps all)."""
    if compartments is None:
        return df
    dropped = tuple(f"{comp}_" for comp in COMPARTMENTS if comp not in compartments)
    return df[[c for c in df.columns if not c.startswith(dropped)]] if dropped else df


def _second_obj_channel_names(fe, image):
    """Channel names for every channel of ``image`` (secondary objects use them all)."""
    n_channels = image.shape[0] if image.ndim == 3 else 1
    if fe.channel_names is not None and len(fe.channel_names) == n_channels:
        return list(fe.channel_names)
    return [f"ch{i}" for i in range(n_channels)]


def _second_obj_foci_channel(fe):
    """Map the feature foci channel to a full-image index for secondary objects."""
    fc = fe.foci_channel
    if isinstance(fc, list):
        if len(fc) != 1:
            raise ValueError("Secondary-object foci take a single foci channel (as brieflow)")
        fc = fc[0]
    if fc is not None and fe.channels:
        fc = fe.channels[fc]
    return fc


def extract_second_obj_features(fe, image, second_obj_masks, second_obj_table):
    """Per-object secondary-object features from brieflow's ``extract_phenotype_second_objs``."""
    from goudacell.brieflow.phenotype.extract_phenotype_second_objs import (
        extract_phenotype_second_objs,
    )

    return extract_phenotype_second_objs(
        image,
        second_objs=second_obj_masks,
        wildcards={},
        second_obj_cell_mapping_df=second_obj_table["second_obj_cell_mapping"],
        foci_channel=_second_obj_foci_channel(fe),
        channel_names=_second_obj_channel_names(fe, image),
    )


def add_num_nuclei(df: pd.DataFrame, nuclei_per_cell: dict) -> pd.DataFrame:
    """Attach the per-cell nuclei count, defaulting to 1 where a cell has no entry.

    Mirrors brieflow's ``scripts/phenotype/extract_phenotype.py`` (``num_nuclei`` column).

    Args:
        df: Feature table with a ``label`` column.
        nuclei_per_cell: ``{cell_label: n_nuclei}`` from segmentation (may be empty).

    Returns:
        The table with a ``num_nuclei`` column.
    """
    labels = df["label"] if "label" in df else pd.Series(dtype=int)
    df["num_nuclei"] = labels.map(pd.Series(nuclei_per_cell, dtype=float)).fillna(1).astype(int)
    return df


def merge_second_obj_summary(df: pd.DataFrame, cell_summary: pd.DataFrame) -> pd.DataFrame:
    """Merge the secondary-object cell summary into the per-cell feature table.

    Mirrors brieflow's ``scripts/phenotype/merge_second_objs_phenotype_cp.py``: a left
    merge on ``label`` = ``cell_id``, dropping ``cell_id``.

    Args:
        df: Per-cell feature table with a ``label`` column.
        cell_summary: The ``cell_summary`` table from secondary-object segmentation.

    Returns:
        The merged table.
    """
    if len(df) == 0 or len(cell_summary) == 0:
        return df
    merged = df.merge(cell_summary, left_on="label", right_on="cell_id", how="left")
    return merged.drop("cell_id", axis=1)


def get_feature_categories() -> dict:
    """Get a dictionary describing available feature categories.

    Returns:
        Dictionary mapping category names to descriptions.
    """
    return {
        "intensity": "Basic intensity statistics (mean, std, min, max, median, etc.)",
        "edge_intensity": "Intensity statistics for edge pixels only",
        "distribution": "Radial intensity distribution features",
        "texture_haralick": "Haralick texture features (13 per channel)",
        "texture_pftas": "PFTAS texture features (54 per channel)",
        "shape": "Morphological shape features (area, perimeter, solidity, etc.)",
        "zernike": "Zernike moment features (30 features)",
        "hu_moments": "Hu moment invariants (7 features)",
        "correlation": "Channel-to-channel correlation features",
        "colocalization": "Colocalization metrics (overlap, Manders, etc.)",
        "neighbors": "Spatial neighbor measurements",
        "foci": "Foci detection features (count, area) - requires foci_channel",
    }
