"""Visualization utilities for GoudaCell."""

from typing import List

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


def plot_sweep(
    display_img: np.ndarray,
    results: List,
    param_label: str = "",
    title: str = "",
):
    """Plot a parameter sweep as mask overlays, one panel per value.

    Args:
        display_img: Grayscale background image to overlay masks on.
        results: List of SweepResult (from segment.parameter_sweep).
        param_label: Short label for the swept parameter (e.g. "diameter").
        title: Overall figure title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        ax.imshow(display_img, cmap="gray")
        ax.imshow(result.masks, cmap=make_mask_cmap(result.masks), alpha=0.4)
        ax.set_title(f"{param_label}={result.value} ({result.count})")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_sweep_1d(values, counts, stable=None, param_label: str = "", title: str = ""):
    """Plot object count vs. a swept parameter, shading the reproducible range.

    Args:
        values: Swept parameter values.
        counts: Object count at each value.
        stable: Optional dict from reproducible_range_1d (lo/hi/setpoint).
        param_label: X-axis label.
        title: Figure title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values, counts, "-o", color="#333")
    if stable is not None:
        ax.axvspan(stable["lo"], stable["hi"], alpha=0.2, color="green", label="reproducible range")
        ax.axvline(
            stable["setpoint"], color="green", ls="--", label=f"setpoint = {stable['setpoint']}"
        )
        ax.legend()
    ax.set_xlabel(param_label)
    ax.set_ylabel("object count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_sweep_grid(
    counts,
    x_values,
    y_values,
    x_label: str = "",
    y_label: str = "",
    plateau=None,
    title: str = "",
):
    """Plot a 2D count grid as a heatmap, outlining the reproducible plateau.

    Args:
        counts: 2D array of object counts, indexed ``counts[y, x]``.
        x_values: Values along the x axis.
        y_values: Values along the y axis.
        x_label: X-axis label.
        y_label: Y-axis label.
        plateau: Optional dict from reproducible_plateau_2d (mask + iy/ix).
        title: Figure title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    counts = np.asarray(counts)
    fig, ax = plt.subplots(figsize=(1.2 * len(x_values) + 3, 1.0 * len(y_values) + 2))
    im = ax.imshow(counts, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels(y_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for y in range(counts.shape[0]):
        for x in range(counts.shape[1]):
            ax.text(x, y, int(counts[y, x]), ha="center", va="center", color="w", fontsize=8)

    if plateau is not None:
        ax.contour(plateau["mask"].astype(float), levels=[0.5], colors="red", linewidths=1.5)
        ax.scatter(
            [plateau["ix"]], [plateau["iy"]], marker="*", s=220,
            color="red", edgecolor="white", label="setpoint",
        )
        ax.legend()

    fig.colorbar(im, ax=ax, label="object count")
    fig.tight_layout()
    return fig


def plot_sweep_montage(
    display_img,
    cells,
    x_values,
    y_values,
    x_param: str = "",
    y_param: str = "",
    title: str = "",
):
    """Show a grid of segmentation overlays, one per swept parameter combination.

    Args:
        display_img: Grayscale background image (the swept target's channel).
        cells: List of GridCell (from segment.sweep_grid).
        x_values: Values along the x axis (columns).
        y_values: Values along the y axis (rows); use [None] for a 1D sweep.
        x_param: Label for the x parameter.
        y_param: Label for the y parameter.
        title: Figure title.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n_rows = max(len(y_values), 1)
    n_cols = len(x_values)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False
    )
    for row in axes:
        for ax in row:
            ax.axis("off")

    for cell in cells:
        ax = axes[cell.iy][cell.ix]
        ax.imshow(display_img, cmap="gray")
        ax.imshow(cell.masks, cmap=make_mask_cmap(cell.masks), alpha=0.4)
        if cell.y is None:
            label = f"{x_param}={cell.x} ({cell.count})"
        else:
            label = f"{x_param}={cell.x}, {y_param}={cell.y} ({cell.count})"
        ax.set_title(label, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
