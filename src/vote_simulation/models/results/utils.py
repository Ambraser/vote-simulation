

from __future__ import annotations

import os
from builtins import max as builtins_max
from builtins import min as builtins_min
from dataclasses import dataclass, field
from textwrap import indent
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from vote_simulation.models.distance import Distance
from vote_simulation.models.distance.distance import JaccardDistance



import os


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
    longest_label = builtins_max((len(lbl) for lbl in labels), default=1)
    figure_size = builtins_max(6.0, 0.45 * rule_count + 0.18 * longest_label)
    annotation_fontsize = builtins_max(4, builtins_min(10, int(220 / builtins_max(rule_count, 1))))

    if ax is None:
        _, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)

    image = ax.imshow(matrix, cmap="Reds", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xticks(range(rule_count), labels=labels)
    ax.set_yticks(range(rule_count), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")
    # plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)
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

    if show:
        plt.show()
    if save_path is not None:
        # check if the directory exists, create it if not
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    return ax



class MdsProjection(NamedTuple):
    """Result of an MDS dimensionality reduction.

    Attributes:
        coords: Array of shape ``(n_rules, n_components)`` with projected coordinates.
        stress: Normalized Kruskal stress (0 = perfect, 1 = poor).
    """

    coords: np.ndarray
    stress: float