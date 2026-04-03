"""Data models for simulation outputs across multiple iterations."""

from __future__ import annotations

from builtins import max as builtins_max
from builtins import min as builtins_min
from dataclasses import dataclass, field
import os
from textwrap import indent
from typing import Any

import numpy as np
import pandas as pd

from vote_simulation.models.distance import Distance
from vote_simulation.models.distance.distance import JaccardDistance


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
    """Render a matrix as a white-to-blue heatmap.

    Shared by :class:`SimulationStepResult` and :class:`SimulationSeriesResult`.
    """
    import matplotlib.pyplot as plt

    rule_count = len(labels)
    longest_label = builtins_max((len(lbl) for lbl in labels), default=1)
    figure_size = builtins_max(6.0, 0.45 * rule_count + 0.18 * longest_label)
    annotation_fontsize = builtins_max(4, builtins_min(10, int(220 / builtins_max(rule_count, 1))))

    # white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["#FFFFFF", "#0055FF"])

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
        #check if the directory exists, create it if not
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    return ax


@dataclass(slots=True)
class SimulationStepResult:
    """Result of a simulation step.

    The comparison matrix is stored as a symmetric ``float32`` 2D array so that
    any distance metric (binary, Jaccard, etc.) can be used.
    """

    data_source: str
    winners_by_rule: dict[str, list[str]] = field(default_factory=dict)
    distance_metric: Distance = field(default_factory=JaccardDistance)
    _rule_order: list[str] = field(default_factory=list, init=False, repr=False)
    _rule_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _winner_sets_by_rule: dict[str, frozenset[str]] = field(default_factory=dict, init=False, repr=False)
    _distance_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Normalize any pre-populated data and build the matrix once."""

        initial_items = list(self.winners_by_rule.items())
        self.winners_by_rule = {}
        self._rule_order = []
        self._rule_index = {}
        self._winner_sets_by_rule = {}
        self._distance_matrix = np.zeros((0, 0), dtype=np.float32)

        for rule_code, winners in initial_items:
            self.add_method_result(rule_code, winners)

    @property
    def rule_codes(self) -> list[str]:
        """Ordered rule codes matching the matrix axes."""

        return list(self._rule_order)

    @property
    def dist_matrix(self) -> np.ndarray:
        """Read-only 2D matrix of pairwise distances between rules."""

        matrix = self._distance_matrix.view()
        matrix.flags.writeable = False
        return matrix

    @property
    def distance_matrix_frame(self) -> pd.DataFrame:
        """Distance matrix as a labeled DataFrame for display and analysis."""

        idx = pd.Index(self._rule_order)
        return pd.DataFrame(self._distance_matrix, index=idx, columns=idx, copy=False)

    def add_method_result(self, rule_code: str, winners: list[str]) -> None:
        """Add or update winners for one voting method in this step.

        Args:
            rule_code: str - Code of the voting method (e.g., "STV", "IRV", "Borda")
            winners: list[str] - List of winner labels for the given method. Can be multiple in case of ties.
        """

        normalized_code = rule_code.strip().upper()
        normalized_winners = list(dict.fromkeys(winners))
        winner_set = frozenset(normalized_winners)

        self.winners_by_rule[normalized_code] = normalized_winners
        self._winner_sets_by_rule[normalized_code] = winner_set

        if normalized_code in self._rule_index:
            self._refresh_rule_distances(normalized_code)
            return

        self._append_rule(normalized_code)

    def save_to_file(self, file_path: str) -> None:
        """Save the step result to a parquet file.

        Args:
            file_path: str - Path to the parquet file where the step result will be saved.
            The resulting file will have columns "Rule" and "Winner",
            where each row corresponds to one winner for a given rule.
        """
        # Convert the winners_by_rule dictionary to a DataFrame
        df = pd.DataFrame(
            [(rule, winner) for rule, winners in self.winners_by_rule.items() for winner in winners],
            columns=pd.Index(["Rule", "Winner"]),
        )

        # Save the DataFrame to a parquet file
        df.to_parquet(file_path, index=False)

    def load_from_file(self, file_path: str) -> None:
        """Load the step result from a parquet file.

        The parquet file is expected to have columns "Rule" and "Winner",
        where each row corresponds to one winner for a given rule.

        Args:
            file_path: str - Path to the parquet file containing the step result.
        """
        df = pd.read_parquet(file_path)
        loaded_winners = df.groupby("Rule")["Winner"].apply(list).to_dict()

        self.winners_by_rule = {}
        self._rule_order = []
        self._rule_index = {}
        self._winner_sets_by_rule = {}
        self._distance_matrix = np.zeros((0, 0), dtype=np.float32)

        for rule_code, winners in loaded_winners.items():
            self.add_method_result(str(rule_code), winners)

    def format_distance_matrix(self) -> str:
        """Return a printable matrix with row and column labels."""

        if self._distance_matrix.size == 0:
            return "<empty matrix>"

        return self.distance_matrix_frame.to_string()

    def __str__(self) -> str:
        """String representation with a readable matrix block."""

        winners_str = (
            "\n".join(f"- {rule}: {', '.join(winners)}" for rule, winners in self.winners_by_rule.items()) or "- <none>"
        )

        return (
            f"Data Source: {self.data_source}\n"
            f"Winners by rule:\n{indent(winners_str, '  ')}\n"
            f"Distance Matrix:\n{indent(self.format_distance_matrix(), '  ')}"
        )

    def compute_distance_matrix(self) -> np.ndarray:
        """Rebuild the full distance matrix from winners and return it."""

        ordered_items = [(rule_code, self.winners_by_rule[rule_code]) for rule_code in self._rule_order]
        self.winners_by_rule = {}
        self._rule_order = []
        self._rule_index = {}
        self._winner_sets_by_rule = {}
        self._distance_matrix = np.zeros((0, 0), dtype=np.float32)

        for rule_code, winners in ordered_items:
            self.add_method_result(rule_code, winners)

        return self.dist_matrix

    def plot_distance_matrix(
        self,
        ax: Any | None = None,
        *,
        annotate: bool = True,
        show: bool = True,
    ) -> Any:
        """Plot the 0/1 distance matrix as a white-to-blue heatmap."""

        if not self._rule_order:
            raise ValueError("Cannot plot an empty distance matrix.")

        return _plot_heatmap(
            self._distance_matrix,
            self._rule_order,
            title=f"Rule distance matrix\n{self.data_source}",
            ax=ax,
            vmin=0,
            vmax=1,
            annotate=annotate,
            annotation_fmt=".2f",
            colorbar_label="Distance",
            show=show,
        )

    def _append_rule(self, rule_code: str) -> None:
        """Append a new rule and update only the new row/column."""

        previous_size = len(self._rule_order)
        new_size = previous_size + 1
        new_matrix = np.zeros((new_size, new_size), dtype=np.float32)

        if previous_size:
            new_matrix[:previous_size, :previous_size] = self._distance_matrix

        self._rule_order.append(rule_code)
        self._rule_index[rule_code] = previous_size
        self._distance_matrix = new_matrix
        self._refresh_rule_distances(rule_code)

    def _refresh_rule_distances(self, rule_code: str) -> None:
        """Refresh only one rule row/column in the symmetric matrix."""

        row_index = self._rule_index[rule_code]
        self._distance_matrix[row_index, row_index] = 0.0
        winner_set = self._winner_sets_by_rule[rule_code]
        metric = self.distance_metric

        for other_rule, other_index in self._rule_index.items():
            if other_rule == rule_code:
                continue

            distance = metric.compute(winner_set, self._winner_sets_by_rule[other_rule])
            self._distance_matrix[row_index, other_index] = distance
            self._distance_matrix[other_index, row_index] = distance


@dataclass(slots=True)
class SimulationSeriesResult:
    """Aggregation of simulation steps.

    Maintains a running ``uint32`` sum of per-step binary distance matrices so
    that the mean can be computed with a single division at any time, regardless
    of how many iterations have been added.
    """

    steps: list[SimulationStepResult] = field(default_factory=list)
    # Accumulator fields
    _rule_order: list[str] = field(default_factory=list, init=False, repr=False)
    _rule_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _matrix_sum: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64),
        init=False,
        repr=False,
    )
    _iteration_count: int = field(default=0, init=False, repr=False)

    def add_step(self, step_result: SimulationStepResult) -> None:
        """Add one step result to the series and accumulate its distance matrix."""

        self.steps.append(step_result)
        self._accumulate_step(step_result)

    @property
    def step_count(self) -> int:
        """Number of recorded steps (equals the iteration count)."""

        return self._iteration_count

    @property
    def mean_distance_matrix(self) -> np.ndarray:
        """Mean pairwise distance matrix over all accumulated steps.

        Returns a ``float32`` array of shape ``(n_rules, n_rules)``.
        Values are in ``[0, 100]``: 0 means every step agreed, 100 means they never did.
        """

        if self._iteration_count == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return (100.0 * self._matrix_sum / self._iteration_count).astype(np.float32)

    @property
    def mean_distance_matrix_frame(self) -> pd.DataFrame:
        """Mean distance matrix as a labeled DataFrame."""

        matrix = self.mean_distance_matrix
        idx = pd.Index(self._rule_order)
        return pd.DataFrame(matrix, index=idx, columns=idx)

    def plot_mean_distance_matrix(
        self,
        ax: Any | None = None,
        folder_save_path: str | None = None,
        *,
        annotate: bool = True,
        show: bool = True,
    ) -> Any:
        """Plot the mean distance matrix as a white-to-blue heatmap.

        Cell values show the fraction of iterations where two rules disagreed.
        """

        if self._iteration_count == 0:
            raise ValueError("Cannot plot: no steps have been added yet.")
        
        if folder_save_path is not None:
            os.makedirs(folder_save_path, exist_ok=True)
            save_path = os.path.join(folder_save_path, f"{self._iteration_count}_mean_distance_matrix.png")

        return _plot_heatmap(
            self.mean_distance_matrix,
            self._rule_order,
            f"Mean rule distance matrix\n({self._iteration_count} iterations)",
            ax,
            annotate=annotate,
            annotation_fmt=".1f",
            colorbar_label="Mean distance (%)",
            show=show,
            save_path=save_path,
        )
    
    def map_rules_2d(self) -> np.ndarray:
        """Project rules into 2D using Multi-Dimensional Scaling (MDS).

        Uses the mean distance matrix as a precomputed dissimilarity matrix
        so that pairwise distances in the 2D plane approximate the original
        rule-to-rule distances.

        Returns:
            np.ndarray: Array of shape ``(n_rules, 2)`` with 2D coordinates.

        Raises:
            ValueError: If no steps have been added yet.
        """
        if self._iteration_count == 0:
            raise ValueError("Cannot project: no steps have been added yet.")

        from sklearn.manifold import MDS

        distance_matrix = self.mean_distance_matrix
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
        coords = mds.fit_transform(distance_matrix)
        return coords

    def plot_rules_2d(
        self,
        ax: Any | None = None,
        *,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Plot rules as labeled points in a 2D MDS projection.

        Distances between points approximate mean pairwise rule distances.

        Args:
            ax: Optional matplotlib Axes to draw on. A new figure is created
                when *None*.
            show: Whether to call ``plt.show()`` at the end.
            save_path: Optional path to save the plot as an image file.

        Returns:
            The matplotlib Axes used for plotting.
        """
        import matplotlib.pyplot as plt

        coords = self.map_rules_2d()
        labels = self._rule_order

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            fig.patch.set_facecolor("white")

        # scatter points
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=60,
            #c="#3366CC",
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        # label each point with its rule short code
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

        ax.set_title(
            f"Rule proximity map  ({self._iteration_count} iterations)",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("MDS 1", fontsize=9, color="#555555")
        ax.set_ylabel("MDS 2", fontsize=9, color="#555555")
        ax.tick_params(labelsize=8, colors="#888888")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

        if show:
            plt.show()

        return ax

    def save_to_file(self, file_path: str) -> None:
        """Save the series result to a parquet file.

        Args:
            file_path: str - Path to the parquet file where the series result will be saved.
            The resulting file will have columns "DataSource", "Rule", and "Winner", where each row corresponds
            to one winner for a given rule and data source.
        """
        df = pd.DataFrame(
            [
                {"DataSource": step.data_source, "Rule": rule, "Winner": winner}
                for step in self.steps
                for rule, winners in step.winners_by_rule.items()
                for winner in winners
            ]
        )
        df.to_parquet(file_path, index=False)

    def load_from_file(self, file_path: str) -> None:
        """Load the series result from a parquet file and rebuild the accumulator.

        Args:
            file_path: str - Path to the parquet file containing the series result.
            The file is expected to have columns "DataSource", "Rule", and "Winner".
        """
        df = pd.read_parquet(file_path)
        self.steps = []
        self._rule_order = []
        self._rule_index = {}
        self._matrix_sum = np.zeros((0, 0), dtype=np.float64)
        self._iteration_count = 0

        for data_source, group in df.groupby("DataSource"):
            step_result = SimulationStepResult(data_source=str(data_source))
            for rule, winners in group.groupby("Rule")["Winner"]:
                step_result.add_method_result(str(rule), winners.tolist())
            self.add_step(step_result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accumulate_step(self, step: SimulationStepResult) -> None:
        """Add one step's binary distance matrix to the running sum."""

        step_rules = step._rule_order
        if not step_rules:
            return

        if not self._rule_order:
            # First step: establish canonical rule order for the whole series
            self._rule_order = list(step_rules)
            self._rule_index = dict(step._rule_index)
            n = len(step_rules)
            self._matrix_sum = np.zeros((n, n), dtype=np.float64)

        # Build permutation mapping step column index → series column index.
        # This handles the (rare) case where a step's rules are in a different order.
        perm = np.array([self._rule_index[r] for r in step_rules], dtype=np.intp)
        self._matrix_sum[np.ix_(perm, perm)] += step._distance_matrix
        self._iteration_count += 1
