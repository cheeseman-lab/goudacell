"""Visualization utilities for GoudaCell."""

import numpy as np
from matplotlib.colors import ListedColormap


def make_mask_cmap(masks: np.ndarray, seed: int = None) -> ListedColormap:
    """Create a random colormap for mask visualization.

    Args:
        masks: Segmentation mask array with integer labels.
        seed: Random seed for reproducible colors.

    Returns:
        ListedColormap with random colors per label, transparent background.
    """
    if seed is not None:
        np.random.seed(seed)
    n_labels = masks.max() + 1
    colors = np.random.rand(n_labels, 4)
    colors[:, 3] = 1.0  # Full opacity
    colors[0] = [0, 0, 0, 0]  # Background transparent
    return ListedColormap(colors)
