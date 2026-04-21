from __future__ import annotations

import os
from dataclasses import dataclass, field
from textwrap import indent
from typing import Any

import numpy as np
import pandas as pd

from vote_simulation.models.distance import Distance
from vote_simulation.models.distance.distance import JaccardDistance
from vote_simulation.models.results.result_config import ResultConfig
from vote_simulation.models.results.utils import _plot_heatmap
from vote_simulation.models.rules.winner_metrics import METRIC_FIELDS, WinnerMetrics


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
    _metrics_by_rule: dict[str, WinnerMetrics] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize any pre-populated data and build the matrix once."""

        initial_items = list(self.winners_by_rule.items())
        self.winners_by_rule = {}
        self._rule_order = []
        self._rule_index = {}
        self._winner_sets_by_rule = {}
        self._distance_matrix = np.zeros((0, 0), dtype=np.float32)
        self._metrics_by_rule = {}

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

    def add_method_result_with_metrics(
        self,
        rule_code: str,
        winners: list[str],
        metrics: WinnerMetrics,
    ) -> None:
        """Add winners *and* pre-computed :class:`WinnerMetrics` for one rule.

        This is the enriched variant of :meth:`add_method_result` used by the
        simulation engine so that winner-quality metrics can be aggregated
        across iterations with no extra recomputation.

        Args:
            rule_code: Voting rule code (e.g. ``"COPE"``).
            winners: List of co-winner labels.
            metrics: Pre-computed :class:`WinnerMetrics` for this step.
        """
        self.add_method_result(rule_code, winners)
        normalized_code = rule_code.strip().upper()
        self._metrics_by_rule[normalized_code] = metrics

    @property
    def metrics_by_rule(self) -> dict[str, WinnerMetrics]:
        """Mapping from rule code to its :class:`WinnerMetrics` for this step.

        Only rules registered via :meth:`add_method_result_with_metrics` will
        appear here.  Rules loaded from disk (without metrics) return an empty
        dict for their key.
        """
        return dict(self._metrics_by_rule)

    @property
    def metrics_frame(self) -> pd.DataFrame:
        """Metrics for all rules in this step as a tidy DataFrame.

        Returns a DataFrame indexed by ``rule`` with one column per metric
        field (see :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS`).
        Rules without metrics are omitted.
        """
        if not self._metrics_by_rule:
            return pd.DataFrame(columns=list(METRIC_FIELDS))
        rows = [
            {"rule": code, **m.to_dict()}
            for code, m in self._metrics_by_rule.items()
        ]
        return pd.DataFrame(rows).set_index("rule")

    # ------------------------------------------------------------------
    # Distance metrics
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

    # Persistence

    def save_to_file(self, file_path: str) -> None:
        """Save the step result to a parquet file.

        Configuration metadata is stored via pyarrow schema metadata so that
        the payload columns remain compact ("Rule" + "Winner" only).

        Args:
            file_path: Path to the output parquet file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [(rule, winner) for rule, winners in self.winners_by_rule.items() for winner in winners]
        df = pd.DataFrame(rows, columns=pd.Index(["Rule", "Winner"]))
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Inject config into schema metadata (prefixed to avoid collisions).
        existing_meta = table.schema.metadata or {}
        config_meta = {f"vote_sim:{k}".encode(): v.encode() for k, v in self.config.to_dict().items()}
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
        self._metrics_by_rule = {}

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
            metrics = f"\nMean distance: {self.mean_distance:.4f}\nMost distant: {r1} <-> {r2} ({d:.4f})"

        mat_str = f"\nDistance Matrix:\n{indent(self.format_distance_matrix(), '  ')}" if self._rule_order else ""
        return (
            f"{header}\nWinners by rule:\n{indent(winners_str, '  ')}{metrics}{mat_str}\n"
        )

    def compute_distance_matrix(self) -> np.ndarray:
        """Rebuild the full distance matrix from winners and return it."""

        ordered_items = [(rule_code, self.winners_by_rule[rule_code]) for rule_code in self._rule_order]
        self.winners_by_rule = {}
        self._rule_order = []
        self._rule_index = {}
        self._winner_sets_by_rule = {}
        self._distance_matrix = np.zeros((0, 0), dtype=np.float32)
        self._metrics_by_rule = {}

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

    # Path helpers

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
