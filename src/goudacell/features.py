"""Feature extraction for segmented cells.

This module provides CellProfiler-equivalent feature extraction for GoudaCell.
Features include intensity statistics, texture (Haralick, PFTAS), shape
measurements (including Zernike moments), radial distribution, and correlation
metrics between channels.
"""

import warnings
from itertools import combinations, permutations, product
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from goudacell.constants import DEFAULT_METADATA_COLS
from goudacell.cp_emulator import (
    correlation_columns_multichannel,
    correlation_features_multichannel,
    find_foci,
    foci_features,
    grayscale_columns_multichannel,
    grayscale_features_multichannel,
    intensity_columns_multichannel,
    intensity_distribution_columns_multichannel,
    intensity_distribution_features_multichannel,
    intensity_features_multichannel,
    neighbor_measurements,
    shape_columns,
    shape_features,
)
from goudacell.feature_extraction import extract_features_bare
from goudacell.feature_table_utils import feature_table_multichannel
from goudacell.segment import identify_cytoplasm

# Basic features added to all feature extractions
FEATURES_BASIC = {
    "area": lambda r: r.area,
    "i": lambda r: r.centroid[0],
    "j": lambda r: r.centroid[1],
    "label": lambda r: r.label,
    "bounds": lambda r: r.bbox,
}


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
    foci_params: Optional[dict] = None,
    method: str = "cp_emulator",
    pipeline_file: Optional[str] = None,
    cellprofiler_cmd: str = "cellprofiler",
    custom_features: Optional[dict] = None,
) -> pd.DataFrame:
    """Extract CellProfiler-equivalent features from segmented image.

    Supports three extraction backends:
    - "cp_emulator" (default): Built-in reimplementation of CellProfiler features.
    - "cp_measure": Lightweight cp_measure package (requires `.[cp_measure]`).
    - "cellprofiler": Runs CellProfiler headlessly via CLI subprocess.
      Requires cellprofiler in PATH and a .cppipe pipeline file.

    Args:
        image: Multichannel image array with shape (C, H, W) where C is the
            number of channels. For single channel images, pass (1, H, W).
        nuclei_masks: Labeled segmentation mask for nuclei (H, W). Each unique
            integer > 0 represents a distinct nucleus.
        cell_masks: Optional labeled segmentation mask for whole cells (H, W).
            If provided, cytoplasm features are also extracted, on each cell minus its
            same-label nucleus (:func:`goudacell.segment.identify_cytoplasm`); masks
            that are not reconciled yield no cytoplasm, as in brieflow.
        channel_names: Names for each channel. If None, defaults to
            ["ch0", "ch1", ...].
        channels: Channel indices to extract from. If None, all channels are
            used. When given, the image and channel_names are sliced to these
            channels before extraction (applies to all backends). Any
            foci_channel indices then refer to positions within this subset.
        compartments: Which compartments to measure, any of "nucleus", "cell",
            "cytoplasm". If None, all available are measured. cp_emulator only.
        include_texture: Whether to include Haralick and PFTAS texture features.
            These are computationally expensive but informative.
        include_correlation: Whether to include channel correlation features
            (overlap, Manders coefficients, etc.).
        include_neighbors: Whether to include neighbor measurements (count,
            distances, angles).
        foci_channel: Optional channel index or list of indices for foci detection.
            If provided, foci will be detected in the specified channel(s) and
            foci count/area features will be extracted per cell for each channel.
            Only supported with cp_emulator method.
        foci_params: Optional dict of parameters for foci detection:
            - radius: Disk radius for white tophat filter (default: 3)
            - threshold: Threshold for foci detection (default: 10)
            - remove_border_foci: Remove foci touching border (default: True, as brieflow)
        method: Extraction backend. One of "cp_emulator", "cp_measure",
            or "cellprofiler".
        pipeline_file: Path to .cppipe file (cellprofiler method only).
        cellprofiler_cmd: CellProfiler executable (cellprofiler method only).
        custom_features: Per-compartment custom features as returned by
            :func:`goudacell.custom_features.load_custom_features`, each measured on
            the full multichannel image (channel order of the input). cp_emulator only.

    Returns:
        DataFrame with one row per cell and columns for each extracted feature.
        Column prefixes indicate compartment: "nucleus_", "cell_", "cytoplasm_".
        Feature names follow CellProfiler conventions where possible.
    """
    if custom_features and method != "cp_emulator":
        raise ValueError("Custom features require method 'cp_emulator'")

    # Custom features always measure the full image, in its original channel order
    full_image = image[np.newaxis, ...] if image.ndim == 2 else image

    # Restrict to a subset of channels (applies to all backends)
    if channels is not None:
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        image = image[channels]
        if channel_names is not None:
            # Names may be given for the selected channels (same length) or for
            # all original channels (then subset by index).
            if len(channel_names) == len(channels):
                pass
            elif len(channel_names) > max(channels):
                channel_names = [channel_names[i] for i in channels]
            else:
                raise ValueError(
                    f"channel_names has {len(channel_names)} entries; expected "
                    f"{len(channels)} (one per selected channel) or at least "
                    f"{max(channels) + 1} (one per original channel)."
                )

    # Correlation is a between-channel measure — needs at least two channels.
    n_channels_eff = image.shape[0] if image.ndim == 3 else 1
    if include_correlation and n_channels_eff < 2:
        include_correlation = False

    # Dispatch to alternative backends
    if method == "cp_measure":
        from goudacell.features_cp_measure import extract_features_cp_measure

        return extract_features_cp_measure(
            image,
            nuclei_masks=nuclei_masks,
            cell_masks=cell_masks,
            channel_names=channel_names,
            include_texture=include_texture,
            include_correlation=include_correlation,
            include_neighbors=include_neighbors,
        )
    elif method == "cellprofiler":
        from goudacell.features_cellprofiler import extract_features_cellprofiler

        return extract_features_cellprofiler(
            image,
            nuclei_masks=nuclei_masks,
            cell_masks=cell_masks,
            channel_names=channel_names,
            pipeline_file=pipeline_file,
            cellprofiler_cmd=cellprofiler_cmd,
            include_texture=include_texture,
            include_correlation=include_correlation,
            include_neighbors=include_neighbors,
        )
    elif method != "cp_emulator":
        raise ValueError(
            f"Unknown extraction method '{method}'. "
            "Supported: 'cp_emulator', 'cp_measure', 'cellprofiler'"
        )
    # Suppress skimage deprecation warnings for RegionProperties attribute renames
    # (intensity_image -> image_intensity, etc.)
    warnings.filterwarnings(
        "ignore",
        message=r".*RegionProperties\.\w+ is deprecated.*",
        category=FutureWarning,
    )

    # Validate inputs
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # Add channel dimension

    if image.ndim != 3:
        raise ValueError(f"Image must be (C, H, W), got shape {image.shape}")

    n_channels = image.shape[0]

    # Generate default channel names if not provided
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    if len(channel_names) != n_channels:
        raise ValueError(
            f"Number of channel names ({len(channel_names)}) must match "
            f"number of channels ({n_channels})"
        )

    # Check for empty masks
    if np.sum(nuclei_masks) == 0:
        return pd.DataFrame(columns=["label"])

    # Build feature dictionary based on options
    features = _build_feature_dict(include_texture, include_correlation)

    # Create column mapping for renaming
    channel_idx = list(range(n_channels))

    # Decide which compartments to measure
    def want(compartment: str) -> bool:
        return compartments is None or compartment in compartments

    has_cells = cell_masks is not None and np.sum(cell_masks) > 0
    cytoplasm_masks = identify_cytoplasm(nuclei_masks, cell_masks) if has_cells else None
    has_cytoplasm = cytoplasm_masks is not None and np.sum(cytoplasm_masks) > 0

    # Blocks are appended in brieflow's order so the final column order matches
    dfs = []

    for compartment, masks, present in (
        ("nucleus", nuclei_masks, True),
        ("cell", cell_masks, has_cells),
        ("cytoplasm", cytoplasm_masks, has_cytoplasm),
    ):
        if present and want(compartment):
            columns = _make_column_map(
                channel_idx, channel_names, include_texture, include_correlation
            )
            dfs.append(
                _extract_compartment_features(image, masks, features, columns, compartment)
            )

    # Extract foci features if foci channel is provided
    if foci_channel is not None:
        # Use cells if available, otherwise fall back to nuclei
        foci_mask = cell_masks if has_cells else nuclei_masks

        # Get foci detection parameters
        params = foci_params or {}
        radius = params.get("radius", 3)
        threshold = params.get("threshold", 10)
        remove_border = params.get("remove_border_foci", True)

        # Normalize to list for consistent handling
        if isinstance(foci_channel, int):
            foci_channels = [foci_channel]
        else:
            foci_channels = foci_channel

        # Process each foci channel
        for fc in foci_channels:
            # Detect foci in the specified channel
            foci_image = image[fc]
            foci_labeled = find_foci(
                foci_image, radius=radius, threshold=threshold, remove_border_foci=remove_border
            )

            # Extract foci features using the cell/nuclei masks as regions
            dfs.append(
                extract_features_bare(foci_labeled, foci_mask, features=foci_features)
                .set_index("label")
                .add_prefix(f"cell_{channel_names[fc]}_")
            )

    # Extract neighbor measurements
    if include_neighbors:
        for compartment, masks, present in (
            ("nucleus", nuclei_masks, True),
            ("cell", cell_masks, has_cells),
            ("cytoplasm", cytoplasm_masks, has_cytoplasm),
        ):
            if present and want(compartment):
                dfs.append(
                    neighbor_measurements(masks, distances=[1])
                    .set_index("label")
                    .add_prefix(f"{compartment}_")
                )

    # Extract custom features on the compartment each one declares
    if custom_features:
        custom_masks = {"nucleus": nuclei_masks, "cell": cell_masks, "cytoplasm": cytoplasm_masks}

        unknown = sorted(set(custom_features) - set(custom_masks))
        if unknown:
            raise ValueError(f"Custom features declare unknown compartments: {unknown}")

        for compartment, custom_mask in custom_masks.items():
            compartment_features = custom_features.get(compartment)
            if not compartment_features:
                continue

            # A compartment that was never segmented cannot stand in for another one
            if custom_mask is None:
                raise ValueError(
                    f"Custom features {sorted(compartment_features)} are measured on "
                    f"the {compartment} compartment, which is not segmented in this run"
                )
            if np.sum(custom_mask) == 0:
                continue

            custom_df = extract_features_bare(
                full_image, custom_mask, features=compartment_features, multichannel=True
            ).set_index("label")

            collisions = [
                col for col in custom_df.columns if any(col in df.columns for df in dfs)
            ]
            if collisions:
                raise ValueError(
                    f"Custom feature columns collide with built-in features: {collisions}"
                )
            dfs.append(custom_df)

    # Concatenate all features
    if not dfs:
        return pd.DataFrame(columns=["label"])
    result_df = pd.concat(dfs, axis=1, join="outer", sort=False).reset_index()

    # Reorder columns: label, metadata, then nucleus, cell, cytoplasm (brieflow order)
    result_df = _order_columns(result_df)

    return result_df


def _build_feature_dict(include_texture: bool, include_correlation: bool) -> dict:
    """Build the feature dictionary based on options."""
    features = {}

    # Always include intensity and shape features
    if include_texture:
        features.update(grayscale_features_multichannel)
    else:
        # Just intensity without texture
        features.update(intensity_features_multichannel)
        features.update(intensity_distribution_features_multichannel)

    # Correlation before shape, as brieflow, so the column order matches
    if include_correlation:
        features.update(correlation_features_multichannel)

    features.update(shape_features)

    return features


def _make_column_map(
    channels: List[int],
    channel_names: List[str],
    include_texture: bool,
    include_correlation: bool,
) -> dict:
    """Create column name mapping for features."""
    columns = {}

    # Build column map for grayscale features
    if include_texture:
        col_dict = grayscale_columns_multichannel
    else:
        col_dict = {
            **intensity_columns_multichannel,
            **intensity_distribution_columns_multichannel,
        }

    for feat, out in col_dict.items():
        columns.update(
            {
                f"{feat}_{n}": f"{channel_names[ch]}_{renamed}"
                for n, (renamed, ch) in enumerate(product(out, channels))
            }
        )

    # Build column map for correlation features
    if include_correlation:
        for feat, out in correlation_columns_multichannel.items():
            if feat == "lstsq_slope":
                iterator = permutations
            else:
                iterator = combinations
            columns.update(
                {
                    f"{feat}_{n}": renamed.format(
                        first=channel_names[first], second=channel_names[second]
                    )
                    for n, (renamed, (first, second)) in enumerate(
                        product(out, iterator(channels, 2))
                    )
                }
            )

    # Add shape columns
    columns.update(shape_columns)

    return columns


def _extract_compartment_features(
    image: np.ndarray,
    masks: np.ndarray,
    features: dict,
    column_map: dict,
    prefix: str,
) -> pd.DataFrame:
    """Extract features for a single compartment (nucleus, cell, or cytoplasm)."""
    # Add basic features
    all_features = features.copy()
    all_features.update(FEATURES_BASIC)

    # Extract features
    df = feature_table_multichannel(image, masks, all_features)

    # Rename columns and add prefix
    df = df.rename(columns=column_map).set_index("label").add_prefix(f"{prefix}_")

    return df


def _order_columns(
    df: pd.DataFrame, metadata_cols: Optional[List[str]] = None, label_col: str = "label"
) -> pd.DataFrame:
    """Order columns as brieflow's ``order_dataframe_columns``.

    Label first, then the metadata columns present (``DEFAULT_METADATA_COLS``), then any
    other columns, then nucleus, cell and cytoplasm features, each in insertion order.
    """
    if metadata_cols is None:
        metadata_cols = DEFAULT_METADATA_COLS

    ordered_cols = [label_col] if label_col in df.columns else []
    ordered_cols += [c for c in metadata_cols if c in df.columns and c not in ordered_cols]

    remaining = [col for col in df.columns if col not in ordered_cols]
    prefixes = ("nucleus_", "cell_", "cytoplasm_")
    ordered_cols += [col for col in remaining if not col.startswith(prefixes)]
    for prefix in prefixes:
        ordered_cols += [col for col in remaining if col.startswith(prefix)]

    return df[ordered_cols]


def add_num_nuclei(df: pd.DataFrame, nuclei_per_cell: dict) -> pd.DataFrame:
    """Attach the per-cell nuclei count, defaulting to 1 where a cell has no entry.

    Mirrors brieflow's ``extract_phenotype.py`` (``num_nuclei`` column).

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

    Mirrors brieflow's ``merge_second_objs_phenotype_cp.py``: a left merge on
    ``label`` = ``cell_id``, dropping ``cell_id``.

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
