"""Feature extraction backend using cp_measure package.

Provides CellProfiler-equivalent measurements via the lightweight cp_measure
library (no GUI dependencies). Matches the extraction pattern used by brieflow.

Requires: pip install cp-measure  (or uv pip install -e ".[cp_measure]")
"""

from itertools import permutations
from typing import List, Optional

import numpy as np
import pandas as pd

from cp_measure.bulk import (
    get_core_measurements,
    get_correlation_measurements,
    get_multimask_measurements,
)


def _dict_to_df(features: dict) -> pd.DataFrame:
    """Convert a measurement dict to DataFrame, handling mixed scalar/array values.

    cp_measure functions can return dicts with a mix of scalars (global metrics
    like overlap) and arrays (per-object metrics like neighbor counts). This
    broadcasts scalars to match the longest array.
    """
    if not features:
        return pd.DataFrame()

    # Find the max array length (number of objects)
    max_len = 1
    for v in features.values():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            max_len = max(max_len, len(v))
        elif isinstance(v, (list, tuple)):
            max_len = max(max_len, len(v))

    safe = {}
    for k, v in features.items():
        if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
            safe[k] = np.full(max_len, float(v))
        elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == max_len:
            safe[k] = v
        elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == 1:
            safe[k] = np.full(max_len, v[0])
        else:
            safe[k] = v

    return pd.DataFrame(safe)


def _get_single_object_features(
    image: np.ndarray,
    mask: np.ndarray,
    prefix: str = "",
) -> dict:
    """Extract single-object features (1 image + 1 mask).

    Args:
        image: Single channel image (H, W).
        mask: Segmentation mask (H, W).
        prefix: Prefix for feature names.

    Returns:
        Dictionary of extracted features.
    """
    results = {}
    for name, measure_func in get_core_measurements().items():
        try:
            features = measure_func(mask, image)
            if prefix:
                features = {f"{prefix}{k}": v for k, v in features.items()}
            results.update(features)
        except Exception as e:
            print(f"Warning: Error in {name}: {e}")
    return results


def _get_colocalization_features(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray,
    prefix: str = "",
) -> dict:
    """Extract colocalization features (2 images + 1 mask).

    Args:
        image1: First channel image (H, W).
        image2: Second channel image (H, W).
        mask: Segmentation mask (H, W).
        prefix: Prefix for feature names.

    Returns:
        Dictionary of extracted features.
    """
    results = {}
    for name, measure_func in get_correlation_measurements().items():
        try:
            features = measure_func(image1, image2, mask)
            if prefix:
                features = {f"{prefix}{k}": v for k, v in features.items()}
            results.update(features)
        except Exception as e:
            print(f"Warning: Error in colocalization {name}: {e}")
    return results


def _get_neighbor_features(
    mask1: np.ndarray,
    mask2: np.ndarray,
    prefix: str = "",
) -> dict:
    """Extract neighbor relationship features (2 masks).

    Args:
        mask1: First segmentation mask (H, W).
        mask2: Second segmentation mask (H, W).
        prefix: Prefix for feature names.

    Returns:
        Dictionary of extracted features.
    """
    results = {}
    for name, measure_func in get_multimask_measurements().items():
        try:
            features = measure_func(mask1, mask2)
            if prefix:
                features = {f"{prefix}{k}": v for k, v in features.items()}
            results.update(features)
        except Exception as e:
            print(f"Warning: Error in neighbor {name}: {e}")
    return results


def extract_features_cp_measure(
    image: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
    include_texture: bool = True,
    include_correlation: bool = True,
    include_neighbors: bool = True,
) -> pd.DataFrame:
    """Extract features using the cp_measure package.

    Provides CellProfiler-equivalent measurements using the lightweight cp_measure
    library. Extracts single-object, colocalization, and neighbor features across
    nuclei, cell, and cytoplasm compartments.

    Args:
        image: Multichannel image (C, H, W).
        nuclei_masks: Labeled nuclear mask (H, W).
        cell_masks: Optional labeled cell mask (H, W).
        channel_names: Names for each channel.
        include_texture: Included in core measurements (no separate toggle).
        include_correlation: Whether to include colocalization features.
        include_neighbors: Whether to include neighbor features.

    Returns:
        DataFrame with one row per cell and columns for each feature.
    """
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    if image.ndim != 3:
        raise ValueError(f"Image must be (C, H, W), got shape {image.shape}")

    n_channels = image.shape[0]
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    if np.sum(nuclei_masks) == 0:
        return pd.DataFrame(columns=["label"])

    # Build cytoplasm masks
    cytoplasm_masks = None
    if cell_masks is not None and np.sum(cell_masks) > 0:
        cytoplasm_masks = cell_masks.copy()
        cytoplasm_masks[nuclei_masks > 0] = 0

    all_features = []

    # 1. Single-object features per channel x mask
    for ch_idx, ch_name in enumerate(channel_names):
        channel_data = image[ch_idx]
        for mask, mask_name in [
            (nuclei_masks, "nucleus"),
            (cell_masks, "cell"),
            (cytoplasm_masks, "cytoplasm"),
        ]:
            if mask is not None and np.any(mask > 0):
                features = _get_single_object_features(
                    channel_data, mask, f"{mask_name}_{ch_name}__"
                )
                if features:
                    all_features.append(_dict_to_df(features))

    # 2. Colocalization features
    if include_correlation:
        for (ch1_idx, ch1_name), (ch2_idx, ch2_name) in permutations(
            enumerate(channel_names), 2
        ):
            ch1_data = image[ch1_idx]
            ch2_data = image[ch2_idx]
            for mask, mask_name in [
                (nuclei_masks, "nucleus"),
                (cell_masks, "cell"),
                (cytoplasm_masks, "cytoplasm"),
            ]:
                if mask is not None and np.any(mask > 0):
                    features = _get_colocalization_features(
                        ch1_data, ch2_data, mask,
                        f"{mask_name}_{ch1_name}_{ch2_name}_coloc__",
                    )
                    if features:
                        all_features.append(_dict_to_df(features))

    # 3. Neighbor features
    if include_neighbors:
        for mask, prefix in [
            (nuclei_masks, "nucleus"),
            (cell_masks, "cell"),
            (cytoplasm_masks, "cytoplasm"),
        ]:
            if mask is not None and np.any(mask > 0):
                features = _get_neighbor_features(mask, mask, f"{prefix}_neighbor__")
                if features:
                    all_features.append(_dict_to_df(features))

    if not all_features:
        return pd.DataFrame(columns=["label"])

    # Align all DataFrames to the same index length before concat.
    # Some measurements (e.g., neighbor overlap) return scalar/single-row
    # results even for multi-object masks. Reindex to match the longest.
    max_len = max(len(df) for df in all_features)
    aligned = []
    for df in all_features:
        if len(df) == max_len:
            aligned.append(df.reset_index(drop=True))
        elif len(df) == 1 and max_len > 1:
            # Broadcast single-row results (e.g., global overlap metrics)
            aligned.append(
                pd.concat([df] * max_len, ignore_index=True)
            )
        else:
            # Reindex with NaN fill for mismatched lengths
            aligned.append(df.reindex(range(max_len)).reset_index(drop=True))

    return pd.concat(aligned, axis=1)
