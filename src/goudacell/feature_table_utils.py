"""Utility functions for working with feature tables.

This module provides functions for extracting features from segmented images
using regionprops and applying feature dictionaries to generate feature tables.
"""

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
import skimage.measure


def feature_table(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in data specified by integer labels.

    If provided, the global feature dictionary is applied to the full input data and labels.
    Results are combined in a dataframe with one row per label and one column per feature.

    Args:
        data: Image data array.
        labels: Labeled segmentation mask defining objects to extract features from.
        features: Dictionary of feature names and their corresponding functions.
        global_features: Dictionary of global feature names and their corresponding functions.

    Returns:
        DataFrame containing extracted features with one row per label and one column per feature.
    """
    regions = regionprops(labels, intensity_image=data)

    results = defaultdict(list)

    for region in regions:
        for feature, func in features.items():
            results[feature].append(fix_uint16(func(region)))

    if global_features:
        for feature, func in global_features.items():
            results[feature] = fix_uint16(func(data, labels))

    return pd.DataFrame(results)


def feature_table_multichannel(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in multichannel data.

    If provided, the global feature dictionary is applied to the full input data and labels.
    Results are combined in a dataframe with one row per label and one column per feature.

    Args:
        data: Multichannel image data array.
        labels: Labeled segmentation mask defining objects to extract features from.
        features: Dictionary of feature names and their corresponding functions.
        global_features: Dictionary of global feature names and their corresponding functions.

    Returns:
        DataFrame containing extracted features with one row per label and one column per feature.
    """
    regions = regionprops_multichannel(labels, intensity_image=data)

    results = defaultdict(list)

    for feature, func in features.items():
        result_0 = func(regions[0])
        if isinstance(result_0, Iterable):
            if len(result_0) == 1:
                results[feature] = [func(region)[0] for region in regions]
            else:
                for result in map(func, regions):
                    for index, value in enumerate(result):
                        results[f"{feature}_{index}"].append(value)
        else:
            results[feature] = list(map(func, regions))

    if global_features:
        for feature, func in global_features.items():
            results[feature] = func(data, labels)

    return pd.DataFrame(results)


def regionprops(labeled, intensity_image):
    """Supplement skimage.measure.regionprops with intensity_image_full attribute.

    This adds a field containing the multi-dimensional intensity image for
    each region, enabling per-channel feature extraction.

    Args:
        labeled: Labeled segmentation mask defining objects.
        intensity_image: Intensity image (can be multichannel).

    Returns:
        List of region properties objects with intensity_image_full attribute.
    """
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        base_image = intensity_image[..., 0, :, :]

    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    for region in regions:
        b = region.bbox
        region.intensity_image_full = intensity_image[..., b[0] : b[2], b[1] : b[3]]

    return regions


def regionprops_multichannel(labeled, intensity_image):
    """Format intensity image axes for compatibility with updated skimage regionprops.

    Moves channel axis to last position for compatibility with skimage.measure.regionprops
    that allows multichannel images.

    Args:
        labeled: Labeled segmentation mask defining objects.
        intensity_image: Multichannel intensity image.

    Returns:
        List of region properties objects.
    """
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        base_image = np.moveaxis(
            intensity_image,
            range(intensity_image.ndim - 2),
            range(-1, -(intensity_image.ndim - 1), -1),
        )

    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    return regions


def fix_uint16(x):
    """Fix pandas bug that converts np.uint16 to np.int16.

    Args:
        x: Value to fix.

    Returns:
        Fixed value (int if was uint16, otherwise unchanged).
    """
    if isinstance(x, np.uint16):
        return int(x)
    return x
