"""Data models for simulation outputs across multiple iterations."""

from __future__ import annotations

from builtins import max as builtins_max
from builtins import min as builtins_min
from dataclasses import dataclass, field
import os
from textwrap import indent
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from vote_simulation.models.distance import Distance
from vote_simulation.models.distance.distance import JaccardDistance


# ---------------------------------------------------------------------------
# MDS projection result
# ---------------------------------------------------------------------------


class MdsProjection(NamedTuple):
    """Result of an MDS dimensionality reduction.

    Attributes:
        coords: Array of shape ``(n_rules, n_components)`` with projected coordinates.
        stress: Normalized Kruskal stress (0 = perfect, 1 = poor).
    """

    coords: np.ndarray
    stress: float


# ---------------------------------------------------------------------------
# Lightweight configuration descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResultConfig:
    """Describes the simulation context attached to a result.

    Supports single‑valued **and** multi‑valued configurations so that a
    :class:`SimulationSeriesResult` that spans several generation models,
    voter counts or candidate counts can express that in its metadata.

    All collection fields use :class:`frozenset` for immutability and
    O(1) membership checks.
    """

    gen_models: frozenset[str] = field(default_factory=frozenset)
    n_voters: frozenset[int] = field(default_factory=frozenset)
    n_candidates: frozenset[int] = field(default_factory=frozenset)
    n_iterations: int = 0

    # -- Factories --------------------------------------------------------

    @staticmethod
    def single(
        gen_model: str = "",
        n_voters: int = 0,
        n_candidates: int = 0,
        n_iterations: int = 0,
    ) -> ResultConfig:
        """Create a config for a single (model, voters, candidates) combo."""
        return ResultConfig(
            gen_models=frozenset({gen_model}) if gen_model else frozenset(),
            n_voters=frozenset({n_voters}) if n_voters else frozenset(),
            n_candidates=frozenset({n_candidates}) if n_candidates else frozenset(),
            n_iterations=n_iterations,
        )

    # -- Merge / combine --------------------------------------------------

    def merge(self, other: ResultConfig) -> ResultConfig:
        """Return the union of two configs (idempotent & commutative)."""
        return ResultConfig(
            gen_models=self.gen_models | other.gen_models,
            n_voters=self.n_voters | other.n_voters,
            n_candidates=self.n_candidates | other.n_candidates,
            n_iterations=builtins_max(self.n_iterations, other.n_iterations),
        )

    # -- Labels -----------------------------------------------------------

    @property
    def label(self) -> str:
        """Short slug suitable for directory / file names.

        Example: ``"VMF_HC_v101_c3"`` or ``"IC_UNI_v11_101_c3_14"``.
        """
        models = "_".join(sorted(self.gen_models)) or "UNKNOWN"
        voters = "_".join(str(v) for v in sorted(self.n_voters)) or "0"
        candidates = "_".join(str(c) for c in sorted(self.n_candidates)) or "0"
        base = f"{models}_v{voters}_c{candidates}"
        if self.n_iterations:
            base += f"_i{self.n_iterations}"
        return base

    @property
    def description(self) -> str:
        """Human‑readable description for plot titles.

        Automatically switches between singular and plural phrasing depending
        on how many distinct values are present.
        """
        parts: list[str] = []
        if self.gen_models:
            if len(self.gen_models) == 1:
                parts.append(next(iter(self.gen_models)))
            else:
                parts.append(f"Models: {', '.join(sorted(self.gen_models))}")
        if self.n_voters:
            if len(self.n_voters) == 1:
                parts.append(f"{next(iter(self.n_voters))} voters")
            else:
                parts.append(f"Voters: {', '.join(str(v) for v in sorted(self.n_voters))}")
        if self.n_candidates:
            if len(self.n_candidates) == 1:
                parts.append(f"{next(iter(self.n_candidates))} cand.")
            else:
                parts.append(f"Candidates: {', '.join(str(c) for c in sorted(self.n_candidates))}")
        return " · ".join(parts) if parts else ""

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, str]:
        """Serialize to a ``{key: csv_string}`` mapping."""
        return {
            "gen_models": ",".join(sorted(self.gen_models)),
            "n_voters": ",".join(str(v) for v in sorted(self.n_voters)),
            "n_candidates": ",".join(str(c) for c in sorted(self.n_candidates)),
            "n_iterations": str(self.n_iterations),
        }

    @staticmethod
    def from_dict(data: dict[str, str]) -> ResultConfig:
        """Deserialize from a ``{key: csv_string}`` mapping."""
        gen_models = frozenset(m for m in data.get("gen_models", "").split(",") if m)
        n_voters = frozenset(int(v) for v in data.get("n_voters", "").split(",") if v)
        n_candidates = frozenset(int(c) for c in data.get("n_candidates", "").split(",") if c)
        n_iterations = int(data["n_iterations"]) if data.get("n_iterations") else 0
        return ResultConfig(
            gen_models=gen_models, n_voters=n_voters,
            n_candidates=n_candidates, n_iterations=n_iterations,
        )

    def __bool__(self) -> bool:
        return bool(self.gen_models or self.n_voters or self.n_candidates)


# ---------------------------------------------------------------------------
# Shared plotting helper
# ---------------------------------------------------------------------------


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
    config: ResultConfig = field(default_factory=ResultConfig)
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

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def mean_distance(self) -> float:
        """Mean of all off-diagonal pairwise distances (O(1) numpy ops)."""
        n = len(self._rule_order)
        if n < 2:
            return 0.0
        total = float(np.sum(self._distance_matrix))  # diag is 0 so no need to subtract
        return total / (n * (n - 1))

    @property
    def most_distant_rules(self) -> tuple[str, str, float]:
        """Pair of rules with the maximum distance.

        Returns:
            ``(rule_a, rule_b, distance)`` or ``("", "", 0.0)`` if fewer
            than two rules are present.
        """
        n = len(self._rule_order)
        if n < 2:
            return ("", "", 0.0)
        idx = int(np.argmax(self._distance_matrix))
        i, j = divmod(idx, n)
        return (self._rule_order[i], self._rule_order[j], float(self._distance_matrix[i, j]))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(self, file_path: str) -> None:
        """Save the step result to a parquet file.

        Configuration metadata is stored via pyarrow schema metadata so that
        the payload columns remain compact ("Rule" + "Winner" only).

        Args:
            file_path: Path to the output parquet file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            (rule, winner)
            for rule, winners in self.winners_by_rule.items()
            for winner in winners
        ]
        df = pd.DataFrame(rows, columns=["Rule", "Winner"])
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Inject config into schema metadata (prefixed to avoid collisions).
        existing_meta = table.schema.metadata or {}
        config_meta = {
            f"vote_sim:{k}".encode(): v.encode()
            for k, v in self.config.to_dict().items()
        }
        table = table.replace_schema_metadata({**existing_meta, **config_meta})

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        pq.write_table(table, file_path)

    def load_from_file(self, file_path: str) -> None:
        """Load the step result from a parquet file.

        Reads configuration metadata from the parquet schema when available.

        Args:
            file_path: Path to the parquet file containing the step result.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(file_path)
        meta = table.schema.metadata or {}
        config_dict = {
            k.decode().removeprefix("vote_sim:"): v.decode()
            for k, v in meta.items()
            if k.decode().startswith("vote_sim:")
        }
        self.config = ResultConfig.from_dict(config_dict) if config_dict else ResultConfig()

        df = table.to_pandas()
        loaded_winners = df.groupby("Rule")["Winner"].apply(list).to_dict()

        self.winners_by_rule = {}
        self._rule_order = []
        self._rule_index = {}
        self._winner_sets_by_rule = {}
        self._distance_matrix = np.zeros((0, 0), dtype=np.float32)

        for rule_code, winners in loaded_winners.items():
            self.add_method_result(str(rule_code), winners)

    @staticmethod
    def delete_file(file_path: str) -> bool:
        """Delete a saved step result file.

        Returns:
            ``True`` if the file existed and was deleted, ``False`` otherwise.
        """
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False

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

        header = f"Data Source: {self.data_source}"
        if self.config:
            header += f"\nConfig: {self.config.description}"

        n = len(self._rule_order)
        metrics = ""
        if n >= 2:
            r1, r2, d = self.most_distant_rules
            metrics = (
                f"\nMean distance: {self.mean_distance:.4f}"
                f"\nMost distant: {r1} <-> {r2} ({d:.4f})"
            )

        return (
            f"{header}\n"
            f"Winners by rule:\n{indent(winners_str, '  ')}{metrics}\n"
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
        save_path: str | None = None,
        *,
        annotate: bool = True,
        show: bool = True,
    ) -> Any:
        """Plot the distance matrix as a heatmap.

        When *save_path* is given the plot is written to disk.  If *save_path*
        is a **directory**, the filename is derived automatically from the
        attached :attr:`config`.
        """

        if not self._rule_order:
            raise ValueError("Cannot plot an empty distance matrix.")

        subtitle = self.config.description if self.config else ""
        title = "Rule distance matrix"
        if subtitle:
            title += f"\n{subtitle}"
        else:
            title += f"\n{self.data_source}"

        resolved_save: str | None = None
        if save_path is not None:
            resolved_save = self._resolve_save_path(save_path, "step_distance_matrix.png")

        return _plot_heatmap(
            self._distance_matrix,
            self._rule_order,
            title=title,
            ax=ax,
            vmin=0,
            vmax=1,
            annotate=annotate,
            annotation_fmt=".2f",
            colorbar_label="Distance",
            show=show,
            save_path=resolved_save,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_save_path(self, path: str, default_filename: str) -> str:
        """If *path* is a directory, append a config-based filename."""
        if os.path.isdir(path) or path.endswith(os.sep):
            subdir = self.config.label if self.config else "unknown"
            out = os.path.join(path, subdir, default_filename)
        else:
            out = path
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        return out

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

    Maintains a running ``float64`` sum of per-step distance matrices so
    that the mean can be computed with a single division at any time, regardless
    of how many iterations have been added.

    The aggregated :attr:`config` is automatically updated on each
    :meth:`add_step` call and reflects the union of all per-step configs.
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
    _config: ResultConfig = field(default_factory=ResultConfig, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_step(self, step_result: SimulationStepResult) -> None:
        """Add one step result to the series and accumulate its distance matrix."""

        self.steps.append(step_result)
        self._accumulate_step(step_result)
        if step_result.config:
            self._config = self._config.merge(step_result.config)

    @property
    def config(self) -> ResultConfig:
        """Aggregated configuration across all added steps."""
        return self._config

    @config.setter
    def config(self, value: ResultConfig) -> None:
        self._config = value

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

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def mean_distance(self) -> float:
        """Scalar mean of all off-diagonal cells in the mean distance matrix.

        Value in ``[0, 100]``.
        """
        n = len(self._rule_order)
        if n < 2 or self._iteration_count == 0:
            return 0.0
        mean_mat = self.mean_distance_matrix
        total = float(np.sum(mean_mat))  # diag is 0
        return total / (n * (n - 1))

    @property
    def most_distant_rules(self) -> tuple[str, str, float]:
        """Pair of rules with the maximum mean distance.

        Returns:
            ``(rule_a, rule_b, distance)`` or ``("", "", 0.0)`` when fewer
            than two rules are present.
        """
        n = len(self._rule_order)
        if n < 2 or self._iteration_count == 0:
            return ("", "", 0.0)
        mean_mat = self.mean_distance_matrix
        idx = int(np.argmax(mean_mat))
        i, j = divmod(idx, n)
        return (self._rule_order[i], self._rule_order[j], float(mean_mat[i, j]))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _build_title(self, prefix: str) -> str:
        """Build a plot title from *prefix*, config description and iteration count."""
        desc = self._config.description if self._config else ""
        iters = f"{self._iteration_count} iterations"
        if desc:
            return f"{prefix}\n{desc} \u00b7 {iters}"
        return f"{prefix}\n({iters})"

    def _resolve_save_path(self, base_path: str, default_filename: str) -> str:
        """Derive a full save path, inserting a config‑based sub-directory."""
        if os.path.isdir(base_path) or base_path.endswith(os.sep):
            subdir = self._config.label if self._config else "unknown"
            out = os.path.join(base_path, subdir, default_filename)
        else:
            out = base_path
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        return out

    def plot_mean_distance_matrix(
        self,
        ax: Any | None = None,
        folder_save_path: str | None = None,
        *,
        annotate: bool = True,
        show: bool = True,
    ) -> Any:
        """Plot the mean distance matrix as a heatmap.

        Cell values show the percentage of iterations where two rules disagreed.
        When multi‑config (several models / voter counts / candidate counts),
        the title mentions all of them.
        """
        if self._iteration_count == 0:
            raise ValueError("Cannot plot: no steps have been added yet.")

        save_path: str | None = None
        if folder_save_path is not None:
            save_path = self._resolve_save_path(
                folder_save_path,
                f"{self._iteration_count}_mean_distance_matrix.png",
            )

        result = _plot_heatmap(
            self.mean_distance_matrix,
            self._rule_order,
            self._build_title("Mean rule distance matrix"),
            ax,
            annotate=annotate,
            annotation_fmt=".1f",
            colorbar_label="Mean distance (%)",
            show=show,
            save_path=save_path,
        )

        # Auto-save series parquet alongside the plot
        if save_path is not None:
            parquet_path = os.path.join(
                os.path.dirname(save_path),
                f"{self._iteration_count}_series.parquet",
            )
            self.save_to_file(parquet_path)

        return result
    
    def map_rules_2d(self) -> MdsProjection:
        """Project rules into 2D using Multi-Dimensional Scaling (MDS).

        Uses the mean distance matrix as a precomputed dissimilarity matrix
        so that pairwise distances in the 2D plane approximate the original
        rule-to-rule distances.

        Returns:
            :class:`MdsProjection` with 2D coordinates and normalized stress.

        Raises:
            ValueError: If no steps have been added yet.
        """
        if self._iteration_count == 0:
            raise ValueError("Cannot project: no steps have been added yet.")

        from sklearn.manifold import MDS

        distance_matrix = self.mean_distance_matrix
        mds = MDS(n_components=2, metric="precomputed", random_state=42, normalized_stress="auto", n_init=4, init="random")
        coords = mds.fit_transform(distance_matrix)
        return MdsProjection(coords=coords, stress=float(mds.stress_))
    
    def map_rules_3d(self) -> MdsProjection:
        """Project rules into 3D using Multi-Dimensional Scaling (MDS).

        Uses the mean distance matrix as a precomputed dissimilarity matrix
        so that pairwise distances in the 3D space approximate the original
        rule-to-rule distances.

        Returns:
            :class:`MdsProjection` with 3D coordinates and normalized stress.

        Raises:
            ValueError: If no steps have been added yet.
        """
        if self._iteration_count == 0:
            raise ValueError("Cannot project: no steps have been added yet.")

        from sklearn.manifold import MDS

        distance_matrix = self.mean_distance_matrix
        mds = MDS(n_components=3, metric="precomputed", random_state=42, normalized_stress="auto", n_init=4, init="random")
        coords = mds.fit_transform(distance_matrix)
        return MdsProjection(coords=coords, stress=float(mds.stress_))
    
    def plot_rules_3d(
        self,
        ax: Any | None = None,
        *,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Plot rules as labeled points in a 3D MDS projection.

        Distances between points approximate mean pairwise rule distances.
        The normalized MDS stress is shown on the plot.

        Args:
            ax: Optional matplotlib Axes to draw on. A new figure is created
                when *None*.
            show: Whether to call ``plt.show()`` at the end.
            save_path: Optional path (file or directory) to save the plot."""
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        projection = self.map_rules_3d()
        coords, stress = projection.coords, projection.stress
        labels = self._rule_order

        if ax is None:
            fig = plt.figure(figsize=(8, 6), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            fig.patch.set_facecolor("white")

        # scatter points
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=60,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        # label each point with its rule short code
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

        title = self._build_title("Rule proximity map (3D)")
        title += f"\nMDS stress: {stress:.4f}"
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
            resolved = self._resolve_save_path(
                save_path,
                f"{self._iteration_count}_rules_3d.png",
            )
            ax.figure.savefig(resolved)
            # Auto-save series parquet alongside the plot
            parquet_path = os.path.join(
                os.path.dirname(resolved),
                f"{self._iteration_count}_series.parquet",
            )
            self.save_to_file(parquet_path)

        if show:
            plt.show()

        return ax

    def plot_rules_2d(
        self,
        ax: Any | None = None,
        *,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Plot rules as labeled points in a 2D MDS projection.

        Distances between points approximate mean pairwise rule distances.
        The normalized MDS stress is shown on the plot.

        Args:
            ax: Optional matplotlib Axes to draw on. A new figure is created
                when *None*.
            show: Whether to call ``plt.show()`` at the end.
            save_path: Optional path (file or directory) to save the plot.

        Returns:
            The matplotlib Axes used for plotting."""

        import matplotlib.pyplot as plt

        projection = self.map_rules_2d()
        coords, stress = projection.coords, projection.stress
        labels = self._rule_order

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            fig.patch.set_facecolor("white")

        # scatter points
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=60,
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

        title = self._build_title("Rule proximity map")
        title += f"\nMDS stress: {stress:.4f}"
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
            resolved = self._resolve_save_path(
                save_path,
                f"{self._iteration_count}_rules_2d.png",
            )
            ax.figure.savefig(resolved)
            # Auto-save series parquet alongside the plot
            parquet_path = os.path.join(
                os.path.dirname(resolved),
                f"{self._iteration_count}_series.parquet",
            )
            self.save_to_file(parquet_path)

        if show:
            plt.show()

        return ax

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(self, file_path: str) -> None:
        """Save the series result to a parquet file.

        Per-step config is stored in columns ``GenModel``, ``NVoters``,
        ``NCandidates`` so that each row is self-describing.  The aggregated
        series config is stored in schema metadata.

        Args:
            file_path: Path to the output parquet file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows: list[dict[str, str | int]] = []
        for step in self.steps:
            # Flatten per-step config to single values (or empty string / 0)
            gm = ",".join(sorted(step.config.gen_models)) if step.config.gen_models else ""
            nv = ",".join(str(v) for v in sorted(step.config.n_voters)) if step.config.n_voters else ""
            nc = ",".join(str(c) for c in sorted(step.config.n_candidates)) if step.config.n_candidates else ""
            for rule, winners in step.winners_by_rule.items():
                for winner in winners:
                    rows.append({
                        "DataSource": step.data_source,
                        "GenModel": gm,
                        "NVoters": nv,
                        "NCandidates": nc,
                        "Rule": rule,
                        "Winner": winner,
                    })

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Store aggregated config in schema metadata
        existing_meta = table.schema.metadata or {}
        config_meta = {
            f"vote_sim:{k}".encode(): v.encode()
            for k, v in self._config.to_dict().items()
        }
        table = table.replace_schema_metadata({**existing_meta, **config_meta})

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        pq.write_table(table, file_path)

    def load_from_file(self, file_path: str) -> None:
        """Load the series result from a parquet file and rebuild the accumulator.

        Reads per-step config from row columns and aggregated config from
        schema metadata.  Backwards-compatible with files lacking config columns.

        Args:
            file_path: Path to the parquet file containing the series result.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(file_path)

        # --- Aggregated config from metadata ---
        meta = table.schema.metadata or {}
        config_dict = {
            k.decode().removeprefix("vote_sim:"): v.decode()
            for k, v in meta.items()
            if k.decode().startswith("vote_sim:")
        }

        df = table.to_pandas()

        self.steps = []
        self._rule_order = []
        self._rule_index = {}
        self._matrix_sum = np.zeros((0, 0), dtype=np.float64)
        self._iteration_count = 0
        self._config = ResultConfig()

        has_config_cols = {"GenModel", "NVoters", "NCandidates"}.issubset(df.columns)

        for data_source, group in df.groupby("DataSource", sort=False):
            step_config = ResultConfig()
            if has_config_cols:
                row0 = group.iloc[0]
                gm = str(row0["GenModel"]) if row0["GenModel"] else ""
                nv_str = str(row0["NVoters"]) if row0["NVoters"] else ""
                nc_str = str(row0["NCandidates"]) if row0["NCandidates"] else ""
                step_config = ResultConfig(
                    gen_models=frozenset(m for m in gm.split(",") if m),
                    n_voters=frozenset(int(v) for v in nv_str.split(",") if v),
                    n_candidates=frozenset(int(c) for c in nc_str.split(",") if c),
                )

            step_result = SimulationStepResult(
                data_source=str(data_source), config=step_config,
            )
            for rule, winners in group.groupby("Rule", sort=False)["Winner"]:
                step_result.add_method_result(str(rule), winners.tolist())
            self.add_step(step_result)

        # If schema metadata had config, prefer it (more complete for aggregates)
        if config_dict:
            self._config = ResultConfig.from_dict(config_dict)

    @staticmethod
    def delete_file(file_path: str) -> bool:
        """Delete a saved series result file.

        Returns:
            ``True`` if the file existed and was deleted, ``False`` otherwise.
        """
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accumulate_step(self, step: SimulationStepResult) -> None:
        """Add one step's distance matrix to the running sum."""

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
