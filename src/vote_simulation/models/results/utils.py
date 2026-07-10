from __future__ import annotations

import os
from typing import Any, NamedTuple

import numpy as np


def _plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    ax: Any | None = None,
    vmin: float = 0,
    vmax: float = 100,
    *,
    annotate: bool = True,
    annotation_fmt: str = ".0f",
    colorbar_label: str = "Distance",
    show: bool = True,
    save_path: str | None = None,
) -> Any:
    """Render a matrix as a heatmap.

    Shared by :class:`SimulationStepResult` and :class:`SimulationSeriesResult`.
    """
    import matplotlib.pyplot as plt

    rule_count = len(labels)
    longest_label = max((len(lbl) for lbl in labels), default=1)
    figure_size = max(6.0, 0.45 * rule_count + 0.18 * longest_label)
    annotation_fontsize = max(4, min(10, int(240 / max(rule_count, 1))))

    if ax is None:
        _, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)

    image = ax.imshow(matrix, cmap="Reds", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xticks(range(rule_count), labels=labels)
    ax.set_yticks(range(rule_count), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")
    ax.set_title(title)
    ax.set_xlabel("Rules")
    ax.set_ylabel("Rules")
    ax.set_xticks(np.arange(-0.5, rule_count, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rule_count, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for row_index, col_index in np.ndindex(matrix.shape):
            raw = matrix[row_index, col_index]
            value = raw.item()  # native Python int or float
            ax.text(
                col_index,
                row_index,
                format(value, annotation_fmt),
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color="black",
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    colorbar.set_ticks([vmin, vmax])
    colorbar.set_ticklabels([str(vmin), str(vmax)])
    colorbar.set_label(colorbar_label)

    if save_path is not None:
        # Handle plain filenames (no directory component) gracefully.
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        import matplotlib.figure as _mpl_fig

        _fig = ax.figure
        if isinstance(_fig, _mpl_fig.Figure):
            _fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


# ------------------------------------------------------------------
# Shared helpers used by both series and total result
# ------------------------------------------------------------------


def _save_figure(fig: Any, save_path: str) -> None:
    """Save a matplotlib figure, creating parent directories as needed.

    Args:
        fig: A matplotlib Figure object.
        save_path: Destination file path.
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path)


def _mds_project(distance_matrix: np.ndarray, n_components: int = 2) -> MdsProjection:
    """Project a precomputed square distance matrix using MDS.

    Args:
        distance_matrix: Square ``float64`` dissimilarity matrix.
        n_components: Number of output dimensions (2 or 3).

    Returns:
        :class:`MdsProjection` with coordinates and normalised stress.
    """
    from sklearn.manifold import MDS

    mds = MDS(
        n_components=n_components,
        metric="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=4,
    )
    coords = mds.fit_transform(distance_matrix)
    return MdsProjection(coords=coords, stress=float(mds.stress_))


def _plot_rules_2d_scatter(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    ax: Any | None = None,
    *,
    show: bool = True,
    save_path: str | None = None,
) -> Any:
    """Plot rules as labeled points in a 2D MDS scatter plot.

    Args:
        coords: ``(n_rules, 2)`` coordinate array from MDS.
        labels: Rule short codes, one per point.
        title: Plot title (may contain newlines).
        ax: Optional matplotlib Axes. A new figure is created when *None*.
        show: Whether to call ``plt.show()`` at the end.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Axes used for plotting.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure as MplFigure

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        fig.patch.set_facecolor("white")

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=60,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            fontweight="medium",
            color="#222222",
        )

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("MDS 1", fontsize=9, color="#555555")
    ax.set_ylabel("MDS 2", fontsize=9, color="#555555")
    ax.tick_params(labelsize=8, colors="#888888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    if save_path is not None:
        fig = ax.get_figure()
        if isinstance(fig, MplFigure):
            _save_figure(fig, save_path)

    if show:
        plt.show()

    return ax


def _plot_rules_3d_scatter(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    ax: Any | None = None,
    *,
    show: bool = True,
    save_path: str | None = None,
) -> Any:
    """Plot rules as labeled points in a 3D MDS scatter plot.

    Args:
        coords: ``(n_rules, 3)`` coordinate array from MDS.
        labels: Rule short codes, one per point.
        title: Plot title (may contain newlines).
        ax: Optional matplotlib 3D Axes. A new figure is created when *None*.
        show: Whether to call ``plt.show()`` at the end.
        save_path: Optional file path to save the figure.

    Returns:
        The matplotlib Axes used for plotting.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure as MplFigure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=60,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    for i, label in enumerate(labels):
        ax.text(
            coords[i, 0],
            coords[i, 1],
            coords[i, 2],
            label,
            fontsize=8,
            fontweight="medium",
            color="#222222",
        )

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("MDS 1", fontsize=9, color="#555555")
    ax.set_ylabel("MDS 2", fontsize=9, color="#555555")
    ax.set_zlabel("MDS 3", fontsize=9, color="#555555")
    ax.tick_params(labelsize=8, colors="#888888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    if save_path is not None:
        fig = ax.get_figure()
        if isinstance(fig, MplFigure):
            _save_figure(fig, save_path)

    if show:
        plt.show()

    return ax


class MdsProjection(NamedTuple):
    """Result of an MDS dimensionality reduction.

    Attributes:
        coords: Array of shape ``(n_rules, n_components)`` with projected coordinates.
        stress: Normalized Kruskal stress (0 = perfect, 1 = poor).
    """

    coords: np.ndarray
    stress: float
