"""Phenotype Feature Extraction Module.

This module provides a comprehensive set of functions for extracting
phenotypic features from images, seeking to replicate the feature extraction
capabilities of CellProfiler.

It includes functions for:
1. Intensity Features: Extraction of various intensity-based metrics for cellular regions.
2. Texture Features: Computation of texture features using methods like Haralick and PFTAS.
3. Shape Features: Calculation of morphological features including Zernike moments.
4. Distribution Features: Analysis of intensity distributions within cellular regions.
5. Neighbor Analysis: Functions for analyzing spatial relationships between cells.
6. Colocalization Metrics: Computation of colocalization coefficients for multi-channel images.

Adapted from brieflow's cp_emulator.py for use in GoudaCell.
"""

import warnings
from functools import partial
from itertools import combinations, starmap
from warnings import catch_warnings, simplefilter

import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
from decorator import decorator
from mahotas.features import haralick, pftas, zernike_moments
from mahotas.thresholding import otsu
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt as distance_transform
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from scipy.stats import median_abs_deviation, rankdata
from skimage import img_as_ubyte

# ============================================================================
# INLINED FUNCTIONS FROM feature_utils.py
# ============================================================================


def correlate_channels_masked(r, first, second):
    """Cross-correlation between non-zero pixels of two channels within a masked region.

    Args:
        r: Region properties object containing intensity images for multiple channels.
        first: Index of the first channel.
        second: Index of the second channel.

    Returns:
        Mean cross-correlation coefficient between the non-zero pixels of the two channels.
    """
    A = masked(r, first)
    B = masked(r, second)

    filt = (A > 0) & (B > 0)
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B = B[filt]
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    return corr.mean()


def masked(r, index):
    """Extract masked intensity image for a specific channel index from a region.

    Args:
        r: Region properties object containing intensity images for multiple channels.
        index: Index of the channel to extract.

    Returns:
        Masked intensity image for the specified channel index.
    """
    return r.intensity_image_full[index][r.image]


def correlate_channels_all_multichannel(r):
    """Compute cross-correlation between masked images of all channels within a region.

    Args:
        r: Region properties object containing intensity images for multiple channels.

    Returns:
        Array containing cross-correlation values between all pairs of channels.
    """
    R = np.corrcoef(r.intensity_image[r.image].T)
    return R[np.triu_indices_from(R, k=1)]


# ============================================================================
# CELLPROFILER FEATURE DICTIONARIES
# ============================================================================

# MeasureCorrelation (now named MeasureColocalization in CellProfiler)

correlation_features = {
    "correlation": lambda r: [
        correlate_channels_masked(r, first, second)
        for first, second in combinations(list(range(r.intensity_image_full.shape[-3])), 2)
    ],
    "lstsq_slope": lambda r: [
        lstsq_slope(r, first, second)
        for first, second in combinations(list(range(r.intensity_image_full.shape[-3])), 2)
    ],
    "colocalization": lambda r: cp_colocalization_all_channels(r, mode="old", threshold="otsu"),
}

correlation_features_ch = {
    "correlation": lambda r, ch1, ch2: correlate_channels_masked(r, ch1, ch2),
    "lstsq_slope": lambda r, ch1, ch2: lstsq_slope(r, ch1, ch2),
    "colocalization": lambda r, ch1, ch2: cp_colocalization(
        r, ch1, ch2, mode="old", threshold="otsu"
    ),
}

correlation_features_multichannel = {
    "correlation": lambda r: catch_runtime(correlate_channels_all_multichannel)(r),
    "lstsq_slope": lambda r: lstsq_slope_all_multichannel(r),
    "colocalization": lambda r: cp_colocalization_all_channels(
        r, mode="multichannel", threshold="otsu"
    ),
}

correlation_columns = ["correlation_{first}_{second}", "lstsq_slope_{first}_{second}"]

colocalization_columns = [
    "overlap_{first}_{second}",
    "K_{first}_{second}",
    "K_{second}_{first}",
    "manders_{first}_{second}",
    "manders_{second}_{first}",
    "rwc_{first}_{second}",
    "rwc_{second}_{first}",
]

colocalization_columns_ch = {
    "colocalization_{first}_{second}_0": "overlap_{first}_{second}",
    "colocalization_{first}_{second}_1": "K_{first}_{second}",
    "colocalization_{first}_{second}_2": "K_{second}_{first}",
    "colocalization_{first}_{second}_3": "manders_{first}_{second}",
    "colocalization_{first}_{second}_4": "manders_{second}_{first}",
    "colocalization_{first}_{second}_5": "rwc_{first}_{second}",
    "colocalization_{first}_{second}_6": "rwc_{second}_{first}",
}

correlation_columns_multichannel = {
    "correlation": ["correlation_{first}_{second}"],
    "lstsq_slope": ["lstsq_slope_{first}_{second}"],
    "colocalization": [
        "overlap_{first}_{second}",
        "K_{first}_{second}",
        "K_{second}_{first}",
        "manders_{first}_{second}",
        "manders_{second}_{first}",
        "rwc_{first}_{second}",
        "rwc_{second}_{first}",
    ],
}

# MeasureGranularity - not included (computationally expensive and hard to tune)

GRANULARITY_BACKGROUND = 10
GRANULARITY_BACKGROUND_DOWNSAMPLE = 1
GRANULARITY_DOWNSAMPLE = 1
GRANULARITY_LENGTH = 16

# MeasureObjectIntensity
EDGE_CONNECTIVITY = 2

intensity_features = {
    "int": lambda r: r.intensity_image[r.image].sum(),
    "mean": lambda r: r.intensity_image[r.image].mean(),
    "std": lambda r: np.std(r.intensity_image[r.image]),
    "max": lambda r: r.intensity_image[r.image].max(),
    "min": lambda r: r.intensity_image[r.image].min(),
    "edge_intensity_feature": lambda r: edge_intensity_features(
        r.intensity_image, r.filled_image, mode="inner", connectivity=EDGE_CONNECTIVITY
    ),
    "mass_displacement": lambda r: np.sqrt(
        (
            (
                np.array(r.local_centroid)
                - np.array(catch_runtime(lambda r: r.weighted_local_centroid)(r))
            )
            ** 2
        ).sum()
    ),
    "lower_quartile": lambda r: np.percentile(r.intensity_image[r.image], 25),
    "median": lambda r: np.median(r.intensity_image[r.image]),
    "mad": lambda r: median_abs_deviation(r.intensity_image[r.image], scale=1),
    "upper_quartile": lambda r: np.percentile(r.intensity_image[r.image], 75),
    "center_mass": lambda r: catch_runtime(lambda r: r.weighted_local_centroid)(r),
    "max_location": lambda r: np.unravel_index(np.argmax(r.intensity_image), (r.image).shape),
}

intensity_features_ch = {
    "int": lambda r, ch: r.intensity_image_full[ch, r.image].sum(),
    "mean": lambda r, ch: r.intensity_image_full[ch, r.image].mean(),
    "std": lambda r, ch: np.std(r.intensity_image_full[ch, r.image]),
    "max": lambda r, ch: r.intensity_image_full[ch, r.image].max(),
    "min": lambda r, ch: r.intensity_image_full[ch, r.image].min(),
    "edge_intensity_feature": lambda r, ch: edge_intensity_features(
        r.intensity_image_full[ch],
        r.filled_image,
        mode="inner",
        connectivity=EDGE_CONNECTIVITY,
    ),
    "mass_displacement": lambda r, ch: mass_displacement_grayscale(
        r.local_centroid, r.intensity_image_full[ch] * r.image
    ),
    "lower_quartile": lambda r, ch: np.percentile(r.intensity_image_full[ch, r.image], 25),
    "median": lambda r, ch: np.median(r.intensity_image_full[ch, r.image]),
    "mad": lambda r, ch: median_abs_deviation(r.intensity_image_full[ch, r.image], scale=1),
    "upper_quartile": lambda r, ch: np.percentile(r.intensity_image_full[ch, r.image], 75),
    "center_mass": lambda r, ch: weighted_local_centroid_grayscale(
        r.intensity_image_full[ch] * r.image
    ),
    "max_location": lambda r, ch: np.unravel_index(
        np.argmax(r.intensity_image_full[ch] * r.image), (r.image).shape
    ),
}

intensity_features_multichannel = {
    "int": lambda r: r.intensity_image[r.image, ...].sum(axis=0),
    "mean": lambda r: r.intensity_image[r.image, ...].mean(axis=0),
    "std": lambda r: np.std(r.intensity_image[r.image, ...], axis=0),
    "max": lambda r: r.intensity_image[r.image, ...].max(axis=0),
    "min": lambda r: r.intensity_image[r.image, ...].min(axis=0),
    "edge_intensity_feature": lambda r: edge_intensity_features(
        r.intensity_image, r.filled_image, mode="inner", connectivity=EDGE_CONNECTIVITY
    ),
    "mass_displacement": lambda r: np.sqrt(
        (
            (
                np.array(r.local_centroid)[:, None]
                - catch_runtime(lambda r: r.weighted_local_centroid)(r)
            )
            ** 2
        ).sum(axis=0)
    ),
    "lower_quartile": lambda r: np.percentile(r.intensity_image[r.image, ...], 25, axis=0),
    "median": lambda r: np.median(r.intensity_image[r.image, ...], axis=0),
    "mad": lambda r: median_abs_deviation(r.intensity_image[r.image, ...], scale=1, axis=0),
    "upper_quartile": lambda r: np.percentile(r.intensity_image[r.image, ...], 75, axis=0),
    "center_mass": lambda r: catch_runtime(lambda r: r.weighted_local_centroid)(r).flatten(),
    "max_location": lambda r: np.array(
        np.unravel_index(
            np.argmax(r.intensity_image.reshape(-1, *r.intensity_image.shape[2:]), axis=0),
            (r.image).shape,
        )
    ).flatten(),
}

intensity_columns = {
    "edge_intensity_feature_0": "int_edge",
    "edge_intensity_feature_1": "mean_edge",
    "edge_intensity_feature_2": "std_edge",
    "edge_intensity_feature_3": "max_edge",
    "edge_intensity_feature_4": "min_edge",
    "center_mass_0": "center_mass_r",
    "center_mass_1": "center_mass_c",
    "max_location_0": "max_location_r",
    "max_location_1": "max_location_c",
}

intensity_columns_ch = {
    "{channel}_edge_intensity_feature_0": "{channel}_int_edge",
    "{channel}_edge_intensity_feature_1": "{channel}_mean_edge",
    "{channel}_edge_intensity_feature_2": "{channel}_std_edge",
    "{channel}_edge_intensity_feature_3": "{channel}_max_edge",
    "{channel}_edge_intensity_feature_4": "{channel}_min_edge",
    "{channel}_center_mass_0": "{channel}_center_mass_r",
    "{channel}_center_mass_1": "{channel}_center_mass_c",
    "{channel}_max_location_0": "{channel}_max_location_r",
    "{channel}_max_location_1": "{channel}_max_location_c",
}

intensity_columns_multichannel = {
    "int": ["int"],
    "mean": ["mean"],
    "std": ["std"],
    "max": ["max"],
    "min": ["min"],
    "mass_displacement": ["mass_displacement"],
    "lower_quartile": ["lower_quartile"],
    "median": ["median"],
    "mad": ["mad"],
    "upper_quartile": ["upper_quartile"],
    "edge_intensity_feature": [
        "int_edge",
        "mean_edge",
        "std_edge",
        "max_edge",
        "min_edge",
    ],
    "center_mass": ["center_mass_r", "center_mass_c"],
    "max_location": ["max_location_r", "max_location_c"],
}

# MeasureObjectNeighbors


def neighbor_measurements(labeled, distances=[1, 10], n_cpu=1):
    """Calculate neighbor measurements for labeled objects.

    Args:
        labeled: Labeled segmentation mask.
        distances: List of distances to use for neighbor counting.
        n_cpu: Number of CPUs for parallel processing.

    Returns:
        DataFrame with neighbor measurements per object.
    """
    from pandas import concat

    dfs = [
        object_neighbors(labeled, distance=distance).rename(
            columns=lambda x: x + "_" + str(distance)
        )
        for distance in distances
    ]

    dfs.append(
        closest_objects(labeled, n_cpu=n_cpu).drop(columns=["first_neighbor", "second_neighbor"])
    )

    return concat(dfs, axis=1, join="outer").reset_index()


# MeasureObjectRadialDistribution (now MeasureObjectIntensityDistribution in CellProfiler)

intensity_distribution_features = {
    "intensity_distribution": lambda r: np.array(
        measure_intensity_distribution(r.filled_image, r.image, r.intensity_image, bins=4)
    ).reshape(-1),
    "weighted_hu_moments": lambda r: catch_runtime(lambda r: r.weighted_moments_hu)(r),
}

intensity_distribution_features_ch = {
    "intensity_distribution": lambda r, ch: np.array(
        measure_intensity_distribution(r.filled_image, r.image, r.intensity_image_full[ch], bins=4)
    ).flatten(),
    "weighted_hu_moments": lambda r, ch: weighted_hu_moments_grayscale(
        r.intensity_image_full[ch] * r.image
    ),
}

intensity_distribution_features_multichannel = {
    "intensity_distribution": lambda r: np.concatenate(
        measure_intensity_distribution_multichannel(
            r.filled_image, r.image, r.intensity_image, bins=4
        )
    ),
    "weighted_hu_moments": lambda r: catch_runtime(lambda r: r.weighted_moments_hu)(r).flatten(),
}

intensity_distribution_columns = {
    "intensity_distribution_0": "frac_at_d_0",
    "intensity_distribution_1": "frac_at_d_1",
    "intensity_distribution_2": "frac_at_d_2",
    "intensity_distribution_3": "frac_at_d_3",
    "intensity_distribution_4": "mean_frac_0",
    "intensity_distribution_5": "mean_frac_1",
    "intensity_distribution_6": "mean_frac_2",
    "intensity_distribution_7": "mean_frac_3",
    "intensity_distribution_8": "radial_cv_0",
    "intensity_distribution_9": "radial_cv_1",
    "intensity_distribution_10": "radial_cv_2",
    "intensity_distribution_11": "radial_cv_3",
}

intensity_distribution_columns_ch = {
    "{channel}_intensity_distribution_0": "{channel}_frac_at_d_0",
    "{channel}_intensity_distribution_1": "{channel}_frac_at_d_1",
    "{channel}_intensity_distribution_2": "{channel}_frac_at_d_2",
    "{channel}_intensity_distribution_3": "{channel}_frac_at_d_3",
    "{channel}_intensity_distribution_4": "{channel}_mean_frac_0",
    "{channel}_intensity_distribution_5": "{channel}_mean_frac_1",
    "{channel}_intensity_distribution_6": "{channel}_mean_frac_2",
    "{channel}_intensity_distribution_7": "{channel}_mean_frac_3",
    "{channel}_intensity_distribution_8": "{channel}_radial_cv_0",
    "{channel}_intensity_distribution_9": "{channel}_radial_cv_1",
    "{channel}_intensity_distribution_10": "{channel}_radial_cv_2",
    "{channel}_intensity_distribution_11": "{channel}_radial_cv_3",
}

intensity_distribution_columns_multichannel = {
    "intensity_distribution": [
        "frac_at_d_0",
        "frac_at_d_1",
        "frac_at_d_2",
        "frac_at_d_3",
        "mean_frac_0",
        "mean_frac_1",
        "mean_frac_2",
        "mean_frac_3",
        "radial_cv_0",
        "radial_cv_1",
        "radial_cv_2",
        "radial_cv_3",
    ],
    "weighted_hu_moments": [f"weighted_hu_moments_{n}" for n in range(7)],
}

# MeasureObjectSizeShape

ZERNIKE_DEGREE = 9

shape_features = {
    "area": lambda r: r.area,
    "perimeter": lambda r: r.perimeter,
    "convex_area": lambda r: r.convex_area,
    "form_factor": lambda r: form_factor(r.area, r.perimeter),
    "solidity": lambda r: r.solidity,
    "extent": lambda r: r.extent,
    "euler_number": lambda r: r.euler_number,
    "centroid": lambda r: r.local_centroid,
    "eccentricity": lambda r: r.eccentricity,
    "major_axis": lambda r: r.major_axis_length,
    "minor_axis": lambda r: r.minor_axis_length,
    "orientation": lambda r: r.orientation,
    "compactness": lambda r: 2
    * np.pi
    * (r.moments_central[0, 2] + r.moments_central[2, 0])
    / (r.area**2),
    "radius": lambda r: max_median_mean_radius(r.filled_image),
    "feret_diameter": lambda r: min_max_feret_diameter(r.coords),
    "hu_moments": lambda r: r.moments_hu,
    "zernike": lambda r: zernike_minimum_enclosing_circle(r.coords, degree=ZERNIKE_DEGREE),
}

zernike_nums = [
    "zernike_" + str(radial) + "_" + str(azimuthal)
    for radial in range(ZERNIKE_DEGREE + 1)
    for azimuthal in range(radial % 2, radial + 2, 2)
]

shape_columns = {"zernike_" + str(num): zernike_num for num, zernike_num in enumerate(zernike_nums)}
shape_columns.update(
    {
        "centroid_0": "centroid_r",
        "centroid_1": "centroid_c",
        "radius_0": "max_radius",
        "radius_1": "median_radius",
        "radius_2": "mean_radius",
        "feret_diameter_0": "min_feret_diameter",
        "feret_diameter_1": "max_feret_diameter",
        "feret_diameter_2": "min_feret_r0",
        "feret_diameter_3": "min_feret_c0",
        "feret_diameter_4": "min_feret_r1",
        "feret_diameter_5": "min_feret_c1",
        "feret_diameter_6": "max_feret_r0",
        "feret_diameter_7": "max_feret_c0",
        "feret_diameter_8": "max_feret_r1",
        "feret_diameter_9": "max_feret_c1",
    }
)

# MeasureTexture

texture_features = {
    "pftas": lambda r: masked_pftas(r.intensity_image),
    "haralick_5": lambda r: ubyte_haralick(
        r.intensity_image, ignore_zeros=True, distance=5, return_mean=True
    ),
}

texture_features_ch = {
    "pftas": lambda r, ch: masked_pftas(r.intensity_image_full[ch] * r.image),
    "haralick_5": lambda r, ch: ubyte_haralick(
        r.intensity_image_full[ch] * r.image,
        ignore_zeros=True,
        distance=5,
        return_mean=True,
    ),
}

texture_features_multichannel = {
    "pftas": lambda r: np.array(
        [
            masked_pftas(channel)
            for channel in np.moveaxis(
                r.intensity_image.reshape(*r.intensity_image.shape[:2], -1), -1, 0
            )
        ]
    ).flatten(order="F"),
    "haralick_5": lambda r: np.array(
        [
            ubyte_haralick(channel, ignore_zeros=True, distance=5, return_mean=True)
            for channel in np.moveaxis(
                r.intensity_image.reshape(*r.intensity_image.shape[:2], -1), -1, 0
            )
        ]
    ).flatten(order="F"),
}

texture_columns_multichannel = {
    "pftas": [f"pftas_{n}" for n in range(54)],
    "haralick_5": [f"haralick_5_{n}" for n in range(13)],
}

# ============================================================================
# COMBINED FEATURE DICTIONARIES
# ============================================================================

grayscale_features = {
    **intensity_features,
    **intensity_distribution_features,
    **texture_features,
}

grayscale_features_ch = {
    **intensity_features_ch,
    **intensity_distribution_features_ch,
    **texture_features_ch,
}

grayscale_features_multichannel = {
    **intensity_features_multichannel,
    **intensity_distribution_features_multichannel,
    **texture_features_multichannel,
}

grayscale_columns = {**intensity_columns, **intensity_distribution_columns}

grayscale_columns_ch = {**intensity_columns_ch, **intensity_distribution_columns_ch}

grayscale_columns_multichannel = {
    **intensity_columns_multichannel,
    **intensity_distribution_columns_multichannel,
    **texture_columns_multichannel,
}


# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================


@decorator
def catch_runtime(func, *args, **kwargs):
    """Decorator to catch RuntimeWarnings during function execution."""
    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning)
        return func(*args, **kwargs)


def lstsq_slope(r, first, second):
    """Calculate least squares slope between two channels."""
    A = masked(r, first)
    B = masked(r, second)

    filt = A > 0
    if filt.sum() == 0:
        return np.nan

    A = A[filt]
    B = B[filt]
    slope = np.linalg.lstsq(np.vstack([A, np.ones(len(A))]).T, B, rcond=-1)[0][0]

    return slope


def lstsq_slope_all_multichannel(r):
    """Calculate least squares slopes between all channel pairs."""
    V = r.intensity_image[r.image]

    slopes = []
    for ch in range(r.intensity_image.shape[-1]):
        slopes.extend(
            np.linalg.lstsq(
                np.vstack([V[..., ch], np.ones(V.shape[0])]).T,
                np.delete(V, ch, axis=1),
                rcond=None,
            )[0][0]
        )

    return slopes


def cp_colocalization_all_channels(r, mode="multichannel", **kwargs):
    """Calculate colocalization metrics for all channel pairs."""
    if mode == "multichannel":
        channels = r.intensity_image.shape[-1]
    else:
        channels = r.intensity_image_full.shape[-3]

    results = [
        cp_colocalization(r, first, second, mode=mode, **kwargs)
        for first, second in combinations(list(range(channels)), 2)
    ]

    if mode == "multichannel":
        return np.array(results).flatten(order="F")
    else:
        return np.concatenate(results)


def cp_colocalization(r, first, second, mode="multichannel", **kwargs):
    """Calculate colocalization metrics between two channels."""
    if mode == "multichannel":
        A, B = r.intensity_image[r.image][..., [first, second]].T
    else:
        A = masked(r, first)
        B = masked(r, second)
    return measure_colocalization(A, B, **kwargs)


def measure_colocalization(A, B, threshold="otsu"):
    """Measure overlap, k1/k2, manders, and rank weighted colocalization coefficients.

    References:
        http://www.scian.cl/archivos/uploads/1417893511.1674 starting at slide 35
        Singan et al. (2011) "Dual channel rank-based intensity weighting for quantitative
        co-localization of microscopy images", BMC Bioinformatics, 12:407.

    Args:
        A: First channel intensity values.
        B: Second channel intensity values.
        threshold: Threshold method ('otsu', 'costes', or float 0-1).

    Returns:
        Tuple of 7 colocalization metrics.
    """
    if (A.sum() == 0) | (B.sum() == 0):
        return (np.nan,) * 7

    results = []

    if threshold == "otsu":
        A_thresh, B_thresh = otsu(A), otsu(B)
    elif threshold == "costes":
        A_thresh, B_thresh = costes_threshold(A, B)
    elif isinstance(threshold, float) and (0 <= threshold <= 1):
        A_thresh, B_thresh = (threshold * A.max(), threshold * B.max())
    else:
        raise ValueError("`threshold` must be a float in [0,1] or one of 'otsu', 'costes'")

    A, B = A.astype(float), B.astype(float)

    overlap = (A * B).sum() / np.sqrt((A**2).sum() * (B**2).sum())
    results.append(overlap)

    K1 = (A * B).sum() / (A**2).sum()
    K2 = (A * B).sum() / (B**2).sum()
    results.extend([K1, K2])

    M1 = A[B > B_thresh].sum() / A.sum()
    M2 = B[A > A_thresh].sum() / B.sum()
    results.extend([M1, M2])

    A_ranks = rankdata(A, method="dense")
    B_ranks = rankdata(B, method="dense")

    R = max([A_ranks.max(), B_ranks.max()])
    weight = (R - abs(A_ranks - B_ranks)) / R
    RWC1 = ((A * weight)[B > B_thresh]).sum() / A.sum()
    RWC2 = ((B * weight)[A > A_thresh]).sum() / B.sum()

    results.extend([RWC1, RWC2])

    return results


def costes_threshold(A, B, step=1, pearson_cutoff=0):
    """Costes automatic threshold for colocalization analysis.

    Costes et al. (2004) Biophysical Journal, 86(6) 3993-4003
    """
    A_dtype_max, B_dtype_max = np.iinfo(A.dtype).max, np.iinfo(B.dtype).max
    if A_dtype_max != B_dtype_max:
        raise ValueError("inputs must be of the same dtype")
    A = A / A_dtype_max
    B = B / A_dtype_max

    mask = (A > 0) | (B > 0)
    A = A[mask]
    B = B[mask]

    A_var = np.var(A, ddof=1)
    B_var = np.var(B, ddof=1)

    Z = A + B
    Z_var = np.var(Z, ddof=1)

    covar = 0.5 * (Z_var - (A_var + B_var))

    a = (B_var - A_var) + np.sqrt((B_var - A_var) ** 2 + 4 * (covar**2)) / (2 * covar)
    b = B.mean() - a * A.mean()

    threshold = A.max()

    if (len(np.unique(A)) > 10**4) & (step < 100):
        step = 100

    for threshold in np.unique(A)[::-step]:
        below = (A < threshold) | (B < (a * threshold + b))
        pearson = np.mean(
            (A[below] - A[below].mean())
            * (B[below] - B[below].mean())
            / (A[below].std() * B[below].std())
        )

        if pearson <= pearson_cutoff:
            break

    return threshold * A_dtype_max, (a * threshold + b) * B_dtype_max


def boundaries(labeled, connectivity=1, mode="inner", background=0):
    """Find boundaries of labeled regions, including image edge pixels."""
    from skimage.segmentation import find_boundaries

    kwargs = dict(connectivity=connectivity, mode=mode, background=background)
    pad_width = 1

    padded = np.pad(labeled, pad_width=pad_width, mode="constant", constant_values=background)
    return find_boundaries(padded, **kwargs)[..., pad_width:-pad_width, pad_width:-pad_width]


def edge_intensity_features(intensity_image, filled_image, **kwargs):
    """Calculate intensity statistics for edge pixels."""
    edge_pixels = intensity_image[boundaries(filled_image, **kwargs), ...]

    return np.array(
        [
            edge_pixels.sum(axis=0),
            edge_pixels.mean(axis=0),
            np.std(edge_pixels, axis=0),
            edge_pixels.max(axis=0),
            edge_pixels.min(axis=0),
        ]
    ).flatten()


def weighted_local_centroid_grayscale(intensity_image):
    """Calculate intensity-weighted centroid for a grayscale image."""
    if intensity_image.sum() == 0:
        return (np.nan,) * 2
    wm = skimage.measure.moments(intensity_image, order=3)
    return wm[tuple(np.eye(intensity_image.ndim, dtype=int))] / wm[(0,) * intensity_image.ndim]


def weighted_local_centroid_multichannel(r):
    """Calculate intensity-weighted centroid for multichannel image."""
    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning)
        return r.weighted_local_centroid


def mass_displacement_grayscale(local_centroid, intensity_image):
    """Calculate mass displacement for a grayscale image."""
    weighted_local_centroid = weighted_local_centroid_grayscale(intensity_image)
    return np.sqrt(((np.array(local_centroid) - np.array(weighted_local_centroid)) ** 2).sum())


def closest_objects(labeled, n_cpu=1):
    """Find closest objects for each labeled region."""
    from scipy.spatial import cKDTree

    from goudacell.feature_table_utils import feature_table

    features = {
        "i": lambda r: r.centroid[0],
        "j": lambda r: r.centroid[1],
        "label": lambda r: r.label,
    }

    df = feature_table(labeled, labeled, features)

    # Handle cases with fewer than 3 objects
    if len(df) < 3:
        result_df = df.copy()
        result_df["first_neighbor"] = np.nan
        result_df["first_neighbor_distance"] = np.nan
        result_df["second_neighbor"] = np.nan
        result_df["second_neighbor_distance"] = np.nan
        result_df["angle_between_neighbors"] = np.nan

        if len(df) == 2:
            result_df["first_neighbor"] = result_df.index[::-1].values
            points = result_df[["i", "j"]].values
            distance = np.sqrt(((points[0] - points[1]) ** 2).sum())
            result_df["first_neighbor_distance"] = distance

        return result_df.drop(columns=["i", "j"]).set_index("label")

    kdt = cKDTree(df[["i", "j"]])
    distances, indexes = kdt.query(df[["i", "j"]], 3, workers=n_cpu)

    df["first_neighbor"], df["first_neighbor_distance"] = indexes[:, 1], distances[:, 1]
    df["second_neighbor"], df["second_neighbor_distance"] = (
        indexes[:, 2],
        distances[:, 2],
    )

    first_neighbors = df[["i", "j"]].values[df["first_neighbor"].values]
    second_neighbors = df[["i", "j"]].values[df["second_neighbor"].values]

    angles = [
        angle(v, p0, p1)
        for v, p0, p1 in zip(df[["i", "j"]].values, first_neighbors, second_neighbors)
    ]

    df["angle_between_neighbors"] = np.array(angles) * (180 / np.pi)

    return df.drop(columns=["i", "j"]).set_index("label")


def object_neighbors(labeled, distance=1):
    """Calculate neighbor statistics at a given distance."""
    from pandas import DataFrame
    from skimage.measure import regionprops

    outlined = boundaries(labeled, connectivity=EDGE_CONNECTIVITY, mode="inner") * labeled

    regions = regionprops(labeled)
    bboxes = [r.bbox for r in regions]
    labels = [r.label for r in regions]

    neighbors_disk = skimage.morphology.disk(distance)
    perimeter_disk = cp_disk(distance + 0.5)

    info_dicts = [
        neighbor_info(labeled, outlined, label, bbox, distance, neighbors_disk, perimeter_disk)
        for label, bbox in zip(labels, bboxes)
    ]

    return DataFrame(info_dicts).set_index("label")


def neighbor_info(
    labeled, outlined, label, bbox, distance, neighbors_disk=None, perimeter_disk=None
):
    """Calculate neighbor info for a single object."""
    if neighbors_disk is None:
        neighbors_disk = skimage.morphology.disk(distance)
    if perimeter_disk is None:
        perimeter_disk = cp_disk(distance + 0.5)

    label_mask = subimage(labeled, bbox, pad=distance)
    outline_mask = subimage(outlined, bbox, pad=distance) == label

    dilated = skimage.morphology.binary_dilation(label_mask == label, footprint=neighbors_disk)
    neighbors = np.unique(label_mask[dilated])
    neighbors = neighbors[(neighbors != 0) & (neighbors != label)]
    n_neighbors = len(neighbors)

    dilated_neighbors = skimage.morphology.binary_dilation(
        (label_mask != label) & (label_mask != 0), footprint=perimeter_disk
    )
    percent_touching = (outline_mask & dilated_neighbors).sum() / outline_mask.sum()

    return {
        "label": label,
        "number_neighbors": n_neighbors,
        "percent_touching": percent_touching,
    }


def subimage(stack, bbox, pad=0):
    """Extract a rectangular region from a stack with optional padding."""
    i0, j0, i1, j1 = bbox + np.array([-pad, -pad, pad, pad])

    sub = np.zeros(stack.shape[:-2] + (i1 - i0, j1 - j0), dtype=stack.dtype)

    i0_, j0_ = max(i0, 0), max(j0, 0)
    i1_, j1_ = min(i1, stack.shape[-2]), min(j1, stack.shape[-1])
    s = (
        Ellipsis,
        slice(i0_ - i0, (i0_ - i0) + i1_ - i0_),
        slice(j0_ - j0, (j0_ - j0) + j1_ - j0_),
    )

    sub[s] = stack[..., i0_:i1_, j0_:j1_]
    return sub


def cp_disk(radius):
    """Create a disk structuring element."""
    iradius = int(radius)
    x, y = np.mgrid[-iradius : iradius + 1, -iradius : iradius + 1]
    radius2 = radius * radius
    strel = np.zeros(x.shape)
    strel[x * x + y * y <= radius2] = 1
    return strel


@catch_runtime
def measure_intensity_distribution(filled_image, image, intensity_image, bins=4):
    """Measure radial intensity distribution."""
    if intensity_image.sum() == 0:
        return (np.nan,) * 12

    binned, center = binned_rings(filled_image, image, bins)

    frac_at_d = (
        np.array([intensity_image[binned == bin_ring].sum() for bin_ring in range(1, bins + 1)])
        / intensity_image[image].sum()
    )

    frac_pixels_at_d = (
        np.array([(binned == bin_ring).sum() for bin_ring in range(1, bins + 1)]) / image.sum()
    )

    mean_frac = frac_at_d / frac_pixels_at_d

    wedges = radial_wedges(image, center)

    mean_binned_wedges = np.array(
        [
            np.array(
                [
                    intensity_image[(wedges == wedge) & (binned == bin_ring)].mean()
                    for wedge in range(1, 9)
                ]
            )
            for bin_ring in range(1, bins + 1)
        ]
    )
    radial_cv = np.nanstd(mean_binned_wedges, axis=1) / np.nanmean(mean_binned_wedges, axis=1)

    return frac_at_d, mean_frac, radial_cv


@catch_runtime
def measure_intensity_distribution_multichannel(filled_image, image, intensity_image, bins=4):
    """Measure radial intensity distribution for multichannel images."""
    if all((intensity_image[image, ...].sum(axis=0)) == 0):
        return (np.nan,) * 12 * intensity_image.shape[-1]

    binned, center = binned_rings(filled_image, image, bins)

    frac_at_d = np.array(
        [intensity_image[binned == bin_ring, ...].sum(axis=0) for bin_ring in range(1, bins + 1)]
    ) / intensity_image[image, ...].sum(axis=0)

    frac_pixels_at_d = (
        np.array([(binned == bin_ring).sum() for bin_ring in range(1, bins + 1)]) / image.sum()
    )

    mean_frac = frac_at_d.reshape(bins, -1) / frac_pixels_at_d[:, None]

    wedges = radial_wedges(image, center)

    mean_binned_wedges = np.array(
        [
            np.array(
                [
                    intensity_image[(wedges == wedge) & (binned == bin_ring), ...].mean(axis=0)
                    for wedge in range(1, 9)
                ]
            )
            for bin_ring in range(1, bins + 1)
        ]
    )
    radial_cv = np.nanstd(mean_binned_wedges, axis=1) / np.nanmean(mean_binned_wedges, axis=1)

    return frac_at_d.flatten(), mean_frac.flatten(), radial_cv.flatten()


def binned_rings(filled_image, image, bins):
    """Separate image into radial bins normalized by edge distance."""
    normalized_distance, center = normalized_distance_to_center(filled_image)

    binned = np.ceil(normalized_distance * bins)
    binned[binned == 0] = 1

    return np.multiply(np.ceil(binned), image), center


def normalized_distance_to_center(filled_image):
    """Calculate distance to center normalized by edge distance."""
    distance_to_edge = distance_transform(np.pad(filled_image, 1, "constant"))[1:-1, 1:-1]

    max_distance = distance_to_edge.max()

    center = tuple(np.median(np.where(distance_to_edge == max_distance), axis=1).astype(int))

    mask = np.ones(filled_image.shape)
    mask[center[0], center[1]] = 0

    distance_to_center = distance_transform(mask)

    return distance_to_center / (distance_to_center + distance_to_edge), center


def radial_wedges(image, center):
    """Divide shape into 8 radial wedges of 45 degrees each."""
    i, j = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]

    positive_i, positive_j = (i > center[0], j > center[1])
    abs_i_greater_j = abs(i - center[0]) > abs(j - center[1])

    return ((positive_i + positive_j * 2 + abs_i_greater_j * 4 + 1) * image).astype(int)


def weighted_hu_moments_grayscale(masked_intensity_image):
    """Calculate weighted Hu moments for a grayscale image."""
    if masked_intensity_image.sum() == 0:
        return (np.nan,) * 7
    return skimage.measure.moments_hu(
        skimage.measure.moments_normalized(skimage.measure.moments_central(masked_intensity_image))
    )


def max_median_mean_radius(filled_image):
    """Calculate max, median, and mean radius from distance transform."""
    transformed = distance_transform(np.pad(filled_image, 1, "constant"))[1:-1, 1:-1][filled_image]

    return (transformed.max(), np.median(transformed), transformed.mean())


def min_max_feret_diameter(coords):
    """Calculate min and max Feret diameters."""
    try:
        hull_vertices = coords[ConvexHull(coords).vertices]
        antipodes = get_antipodes(hull_vertices)
        point_distances = pdist(hull_vertices)

        argmin, argmax = (antipodes[:, 6].argmin(), point_distances.argmax())
        results = (
            (antipodes[argmin, 6], point_distances[argmax])
            + (
                np.mean([antipodes[argmin, 0], antipodes[argmin, 2]]),
                np.mean([antipodes[argmin, 1], antipodes[argmin, 3]]),
            )
            + tuple(antipodes[argmin, 4:6])
        )
        for v in tuple(combinations(hull_vertices, r=2))[argmax]:
            results += tuple(v)
    except Exception:
        results = (np.nan,) * 10

    return results


def get_antipodes(vertices):
    """Rotating calipers algorithm for finding antipodal pairs."""
    antipodes = []
    for v_index, vertex in enumerate(vertices):
        current_distance = 0
        candidates = vertices[circular_index(v_index + 1, v_index - 2, len(vertices))]

        for c_index, candidate in enumerate(candidates):
            d = perpendicular_distance(vertex, vertices[v_index - 1], candidate)

            if d < current_distance:
                antipodes.append(
                    np.concatenate(
                        [
                            vertex,
                            vertices[v_index - 1],
                            candidates[c_index - 1],
                            current_distance[None],
                        ]
                    )
                )
                break

            elif d >= current_distance:
                if d == current_distance:
                    antipodes.append(
                        np.concatenate(
                            [
                                vertex,
                                vertices[v_index - 1],
                                candidates[c_index - 1],
                                current_distance[None],
                            ]
                        )
                    )
                    if c_index == len(candidates) - 1:
                        antipodes.append(
                            np.concatenate(
                                [
                                    vertex,
                                    vertices[v_index - 1],
                                    candidates[c_index],
                                    current_distance[None],
                                ]
                            )
                        )
                current_distance = d

    return np.array(antipodes)


def circular_index(first, last, length):
    """Generate circular indices."""
    if last < first:
        last += length
        return np.arange(first, last + 1) % length
    elif last == first:
        return np.roll(range(length), -first)
    else:
        return np.arange(first, last + 1)


def perpendicular_distance(line_p0, line_p1, p0):
    """Calculate perpendicular distance from point to line."""
    if line_p0[0] == line_p1[0]:
        return abs(line_p0[0] - p0[0])
    elif line_p0[1] == line_p1[1]:
        return abs(line_p0[1] - p0[1])
    else:
        return abs(
            (
                (line_p1[1] - line_p0[1]) * (line_p0[0] - p0[0])
                - (line_p1[0] - line_p0[0]) * (line_p0[1] - p0[1])
            )
            / np.sqrt((line_p1[1] - line_p0[1]) ** 2 + (line_p1[0] - line_p0[0]) ** 2)
        )


def zernike_minimum_enclosing_circle(coords, degree=9):
    """Calculate Zernike moments using minimum enclosing circle."""
    if coords.shape[0] < 3:
        return np.array([np.nan] * 30)

    try:
        image, center, diameter = minimum_enclosing_circle_shift(coords)

        if image is None or diameter <= 0:
            return np.array([np.nan] * 30)

        return zernike_moments(image, radius=diameter / 2, degree=degree, cm=center)

    except QhullError:
        return np.array([np.nan] * 30)


def form_factor(area, perimeter):
    """Calculate form factor (isoperimetric quotient)."""
    if perimeter == 0:
        return np.nan
    else:
        return 4 * np.pi * area / (perimeter**2)


def minimum_enclosing_circle_shift(coords, pad=1):
    """Shift coordinates to fit in minimum enclosing circle."""
    diameter, center = minimum_enclosing_circle(coords)

    if diameter is None or center is None:
        return None, None, None

    shift = np.round(diameter / 2 - center)
    shifted = np.zeros((int(np.ceil(diameter) + pad), int(np.ceil(diameter) + pad)))
    coords_shifted = (coords + shift).astype(int)
    shifted[coords_shifted[:, 0], coords_shifted[:, 1]] = 1
    center_shifted = center + shift

    return shifted, center_shifted, np.ceil(diameter)


def minimum_enclosing_circle(coords):
    """Find minimum enclosing circle using iterative algorithm."""
    try:
        hull_vertices = coords[ConvexHull(coords).vertices]

        s0 = hull_vertices[0]
        s1 = hull_vertices[1]

        iterations = 0

        while True:
            remaining = hull_vertices[
                (hull_vertices != s0).max(axis=1) & (hull_vertices != s1).max(axis=1)
            ]

            angles = np.array(list(map(partial(angle, p0=s0, p1=s1), remaining)))

            min_angle = angles.min()

            if min_angle >= np.pi / 2:
                diameter = np.sqrt(((s0 - s1) ** 2).sum())
                center = (s0 + s1) / 2
                break

            vertex = remaining[np.argmin(angles)]

            remaining_angles = np.array(
                list(starmap(angle, zip([s1, s0], [s0, vertex], [vertex, s1])))
            )

            if remaining_angles.max() <= np.pi / 2:
                diameter, center = circumscribed_circle(s0, s1, vertex)
                break

            keep = [s0, s1][np.argmax(remaining_angles)]

            s0 = keep
            s1 = vertex

            iterations += 1

            if iterations == len(hull_vertices):
                diameter = center = None

    except QhullError:
        diameter = center = None

    return diameter, center


def angle(vertex, p0, p1):
    """Calculate angle at vertex formed by points p0 and p1."""
    v0 = p0 - vertex
    v1 = p1 - vertex

    cosine_angle = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
    return np.arccos(cosine_angle)


def circumscribed_circle(p0, p1, p2):
    """Calculate circumscribed circle of three points."""
    P = np.array([p0, p1, p2])

    Sx = (1 / 2) * np.linalg.det(
        np.concatenate(
            [(P**2).sum(axis=1).reshape(3, 1), P[:, 1].reshape(3, 1), np.ones((3, 1))],
            axis=1,
        )
    )
    Sy = (1 / 2) * np.linalg.det(
        np.concatenate(
            [P[:, 0].reshape(3, 1), (P**2).sum(axis=1).reshape(3, 1), np.ones((3, 1))],
            axis=1,
        )
    )
    a = np.linalg.det(np.concatenate([P, np.ones((3, 1))], axis=1))
    b = np.linalg.det(np.concatenate([P, (P**2).sum(axis=1).reshape(3, 1)], axis=1))

    center = np.array([Sx, Sy]) / a
    diameter = 2 * np.sqrt((b / a) + (np.array([Sx, Sy]) ** 2).sum() / (a**2))
    return diameter, center


def masked_pftas(intensity_image):
    """Calculate PFTAS features with Otsu thresholding."""
    T = otsu(intensity_image, ignore_zeros=True)
    return pftas(intensity_image, T=T)


@catch_runtime
def ubyte_haralick(intensity_image, **kwargs):
    """Calculate Haralick features on unsigned byte image."""
    with catch_warnings():
        simplefilter("ignore", category=UserWarning)
        ubyte_image = img_as_ubyte(intensity_image)
    try:
        features = haralick(ubyte_image, **kwargs)
    except ValueError:
        features = [np.nan] * 13

    return features


# ============================================================================
# FOCI DETECTION AND FEATURES
# ============================================================================


def log_ndi(data, sigma=1):
    """Apply Laplacian of Gaussian to each image in a stack of shape (..., I, J).

    Args:
        data: Input data array.
        sigma: Standard deviation of the Gaussian kernel. Default is 1.

    Returns:
        Array after applying Laplacian of Gaussian.
    """
    if data.ndim == 2:
        # Single 2D image
        arr_ = -1 * ndi.gaussian_laplace(data.astype(float), sigma)
        arr_ = np.clip(arr_, 0, 65535) / 65535
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return skimage.img_as_uint(arr_)
    else:
        # Stack of images - apply to each frame
        h, w = data.shape[-2:]
        reshaped = data.reshape((-1, h, w))
        results = []
        for frame in reshaped:
            arr_ = -1 * ndi.gaussian_laplace(frame.astype(float), sigma)
            arr_ = np.clip(arr_, 0, 65535) / 65535
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results.append(skimage.img_as_uint(arr_))
        return np.array(results).reshape(data.shape)


def apply_watershed(img, smooth=4):
    """Apply the watershed algorithm to refine segmentation.

    Args:
        img: Input binary image.
        smooth: Size of Gaussian kernel used to smooth the distance map. Default is 4.

    Returns:
        Labeled image after watershed segmentation.
    """
    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(img)

    if smooth > 0:
        # Apply Gaussian smoothing to the distance transform
        distance = skimage.filters.gaussian(distance, sigma=smooth)

    # Identify local maxima in the distance transform
    local_max_coords = skimage.feature.peak_local_max(
        distance, footprint=np.ones((3, 3)), exclude_border=False
    )

    # Create a boolean mask for peaks
    local_max = np.zeros_like(distance, dtype=bool)
    local_max[tuple(local_max_coords.T)] = True

    # Label the local maxima
    markers = ndi.label(local_max)[0]

    # Apply watershed algorithm to the distance transform
    result = skimage.segmentation.watershed(-distance, markers, mask=img)

    return result.astype(np.uint16)


def remove_border_objects(labels, mask, dilate=5):
    """Remove labeled regions that touch the border of the given mask.

    Args:
        labels: Labeled image.
        mask: Mask indicating the border regions.
        dilate: Number of dilation iterations to apply to the mask. Default is 5.

    Returns:
        Labeled image with border regions removed.
    """
    # Dilate the mask to ensure regions touching the border are included
    mask = skimage.morphology.binary_dilation(mask, np.ones((dilate, dilate)))

    # Identify labels that need to be removed
    remove = np.unique(labels[mask])

    # Remove the identified labels from the labeled image
    labels = labels.copy()
    labels.flat[np.in1d(labels, remove)] = 0

    return labels


def count_labels(labels, return_list=False):
    """Count the unique non-zero labels in a labeled segmentation mask.

    Args:
        labels: Labeled segmentation mask.
        return_list: Whether to return the list of unique labels along with the count.

    Returns:
        Number of unique non-zero labels. If return_list is True, returns a tuple
        containing the count and the list of unique labels.
    """
    # Get unique labels in the segmentation mask
    uniques = np.unique(labels)
    # Remove the background label (0)
    ls = np.delete(uniques, np.where(uniques == 0))
    # Count the unique non-zero labels
    num_labels = len(ls)
    # Return the count or both count and list of unique labels based on return_list flag
    if return_list:
        return num_labels, ls
    return num_labels


def find_foci(data, radius=3, threshold=10, remove_border_foci=False):
    """Detect foci in the given image using a white tophat filter.

    Args:
        data: Input image data.
        radius: Radius of the disk used in the white tophat filter. Default is 3.
        threshold: Threshold value for identifying foci in the processed image.
            Default is 10.
        remove_border_foci: Flag to remove foci touching the image border.
            Default is False.

    Returns:
        Labeled segmentation mask of foci.
    """
    # Apply white tophat filter to highlight foci
    tophat = skimage.morphology.white_tophat(data, footprint=skimage.morphology.disk(radius))

    # Apply Laplacian of Gaussian to the filtered image
    tophat_log = log_ndi(tophat, sigma=radius)

    # Threshold the image to create a binary mask
    mask = tophat_log > threshold

    # Remove small objects from the mask
    mask = skimage.morphology.remove_small_objects(mask, min_size=(radius**2))

    # Label connected components in the mask
    labeled = skimage.measure.label(mask)

    # Apply watershed algorithm to refine segmentation
    labeled = apply_watershed(labeled, smooth=1)

    if remove_border_foci:
        # Remove foci touching the border
        border_mask = data > 0
        labeled = remove_border_objects(labeled, ~border_mask)

    return labeled


# Foci feature dictionary
foci_features = {
    "foci_count": lambda r: count_labels(r.intensity_image),
    "foci_area": lambda r: (r.intensity_image > 0).sum(),
}

# Column names for foci features
foci_columns = {
    "foci_count": ["foci_count"],
    "foci_area": ["foci_area"],
}
