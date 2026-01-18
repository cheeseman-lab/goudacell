"""Feature extraction for segmented cells.

This module provides CellProfiler-equivalent feature extraction for GoudaCell.
Features include intensity statistics, texture (Haralick, PFTAS), shape
measurements (including Zernike moments), radial distribution, and correlation
metrics between channels.
"""

import warnings
from itertools import combinations, permutations, product
from typing import List, Optional

import numpy as np
import pandas as pd

from goudacell.cp_emulator import (
    correlation_columns_multichannel,
    correlation_features_multichannel,
    find_foci,
    foci_columns,
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
from goudacell.feature_table_utils import feature_table, feature_table_multichannel

# Basic features added to all feature extractions
FEATURES_BASIC = {
    "area": lambda r: r.area,
    "i": lambda r: r.centroid[0],
    "j": lambda r: r.centroid[1],
    "label": lambda r: r.label,
}


def extract_features(
    image: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: np.ndarray = None,
    channel_names: Optional[List[str]] = None,
    include_texture: bool = True,
    include_correlation: bool = True,
    include_neighbors: bool = True,
    foci_channel: Optional[int] = None,
    foci_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Extract CellProfiler-equivalent features from segmented image.

    This function extracts comprehensive morphological and intensity features
    from segmented nuclei and optionally cells. Features are extracted per
    channel and include intensity statistics, texture metrics, shape
    measurements, and correlation between channels.

    Args:
        image: Multichannel image array with shape (C, H, W) where C is the
            number of channels. For single channel images, pass (1, H, W).
        nuclei_masks: Labeled segmentation mask for nuclei (H, W). Each unique
            integer > 0 represents a distinct nucleus.
        cell_masks: Optional labeled segmentation mask for whole cells (H, W).
            If provided, cytoplasm features will also be extracted.
        channel_names: Names for each channel. If None, defaults to
            ["ch0", "ch1", ...].
        include_texture: Whether to include Haralick and PFTAS texture features.
            These are computationally expensive but informative.
        include_correlation: Whether to include channel correlation features
            (overlap, Manders coefficients, etc.).
        include_neighbors: Whether to include neighbor measurements (count,
            distances, angles).
        foci_channel: Optional channel index for foci detection. If provided,
            foci will be detected in this channel and foci count/area features
            will be extracted per cell.
        foci_params: Optional dict of parameters for foci detection:
            - radius: Disk radius for white tophat filter (default: 3)
            - threshold: Threshold for foci detection (default: 10)
            - remove_border_foci: Remove foci touching border (default: False)

    Returns:
        DataFrame with one row per cell and columns for each extracted feature.
        Column prefixes indicate compartment: "nucleus_", "cell_", "cytoplasm_".
        Feature names follow CellProfiler conventions where possible.

    Example:
        >>> from goudacell import load_image, segment_nuclei_and_cells, extract_features
        >>> image = load_image("sample.nd2")
        >>> nuclei, cells = segment_nuclei_and_cells(image, ...)
        >>> features_df = extract_features(
        ...     image,
        ...     nuclei_masks=nuclei,
        ...     cell_masks=cells,
        ...     channel_names=["DAPI", "GFP", "RFP"],
        ...     foci_channel=2,  # Detect foci in RFP channel
        ... )
        >>> print(f"Extracted {len(features_df)} cells x {len(features_df.columns)} features")

    Notes:
        Feature categories (~100+ features per compartment):
        - Intensity (17): mean, std, min, max, integrated, median, mad,
          quartiles, edge intensities, mass displacement, center of mass
        - Shape (25+): area, perimeter, solidity, extent, eccentricity,
          axes, orientation, compactness, radius, feret diameter, hu moments,
          zernike moments
        - Texture (67): Haralick (13), PFTAS (54)
        - Distribution (19): frac_at_d, mean_frac, radial_cv, weighted hu moments
        - Correlation: correlation, lstsq_slope, overlap, K coefficients,
          Manders, rank-weighted colocalization
        - Neighbors: number_neighbors, percent_touching, distances, angles
        - Foci (if foci_channel specified): foci_count, foci_area
    """
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
    channels = list(range(n_channels))

    dfs = []

    # Extract nucleus features
    nucleus_columns = _make_column_map(
        channels, channel_names, include_texture, include_correlation
    )
    nucleus_df = _extract_compartment_features(
        image, nuclei_masks, features, nucleus_columns, "nucleus"
    )
    dfs.append(nucleus_df)

    # Extract cell features if masks provided
    if cell_masks is not None and np.sum(cell_masks) > 0:
        cell_columns = _make_column_map(
            channels, channel_names, include_texture, include_correlation
        )
        cell_df = _extract_compartment_features(image, cell_masks, features, cell_columns, "cell")
        dfs.append(cell_df)

        # Extract cytoplasm features (cell - nucleus)
        cytoplasm_masks = _create_cytoplasm_masks(cell_masks, nuclei_masks)
        if np.sum(cytoplasm_masks) > 0:
            cyto_columns = _make_column_map(
                channels, channel_names, include_texture, include_correlation
            )
            cyto_df = _extract_compartment_features(
                image, cytoplasm_masks, features, cyto_columns, "cytoplasm"
            )
            dfs.append(cyto_df)

    # Extract neighbor measurements
    if include_neighbors:
        dfs.append(
            neighbor_measurements(nuclei_masks, distances=[1])
            .set_index("label")
            .add_prefix("nucleus_")
        )

        if cell_masks is not None and np.sum(cell_masks) > 0:
            dfs.append(
                neighbor_measurements(cell_masks, distances=[1])
                .set_index("label")
                .add_prefix("cell_")
            )

    # Extract foci features if foci channel is provided
    if foci_channel is not None:
        # Use cells if available, otherwise fall back to nuclei
        foci_mask = (
            cell_masks if (cell_masks is not None and np.sum(cell_masks) > 0) else nuclei_masks
        )

        # Get foci detection parameters
        params = foci_params or {}
        radius = params.get("radius", 3)
        threshold = params.get("threshold", 10)
        remove_border = params.get("remove_border_foci", False)

        # Detect foci in the specified channel
        foci_image = image[foci_channel]
        foci_labeled = find_foci(
            foci_image, radius=radius, threshold=threshold, remove_border_foci=remove_border
        )

        # Extract foci features using the cell/nuclei masks as regions
        foci_df = feature_table(foci_labeled, foci_mask, foci_features)

        # Rename columns with channel name prefix
        ch_name = channel_names[foci_channel] if channel_names else f"ch{foci_channel}"
        foci_column_map = {feat: f"{ch_name}_{col[0]}" for feat, col in foci_columns.items()}
        foci_df = foci_df.rename(columns=foci_column_map).set_index("label").add_prefix("cell_")
        dfs.append(foci_df)

    # Concatenate all features
    result_df = pd.concat(dfs, axis=1, join="outer", sort=False).reset_index()

    # Reorder columns: label first, then nucleus, cell, cytoplasm
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

    features.update(shape_features)

    if include_correlation:
        features.update(correlation_features_multichannel)

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


def _create_cytoplasm_masks(cell_masks: np.ndarray, nuclei_masks: np.ndarray) -> np.ndarray:
    """Create cytoplasm masks by subtracting nuclei from cells.

    Args:
        cell_masks: Labeled cell segmentation mask.
        nuclei_masks: Labeled nuclei segmentation mask.

    Returns:
        Labeled cytoplasm masks where each cell's cytoplasm has the same
        label as the parent cell.
    """
    cytoplasm = cell_masks.copy()
    cytoplasm[nuclei_masks > 0] = 0
    return cytoplasm


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Order DataFrame columns: label, nucleus features, cell features, cytoplasm features."""
    ordered_cols = []

    # Label first
    if "label" in df.columns:
        ordered_cols.append("label")

    # Get remaining columns
    remaining = [col for col in df.columns if col not in ordered_cols]

    # Group by compartment
    nucleus_cols = sorted([col for col in remaining if col.startswith("nucleus_")])
    cell_cols = sorted([col for col in remaining if col.startswith("cell_")])
    cytoplasm_cols = sorted([col for col in remaining if col.startswith("cytoplasm_")])
    other_cols = sorted(
        [
            col
            for col in remaining
            if not any(col.startswith(p) for p in ["nucleus_", "cell_", "cytoplasm_"])
        ]
    )

    ordered_cols.extend(other_cols)
    ordered_cols.extend(nucleus_cols)
    ordered_cols.extend(cell_cols)
    ordered_cols.extend(cytoplasm_cols)

    return df[ordered_cols]


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
