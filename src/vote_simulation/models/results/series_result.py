"""Data models for simulation outputs across multiple iterations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from vote_simulation.models.results.result_config import ResultConfig
from vote_simulation.models.results.step_result import SimulationStepResult
from vote_simulation.models.results.utils import (
    MdsProjection,
    _mds_project,
    _plot_heatmap,
    _plot_rules_2d_scatter,
    _plot_rules_3d_scatter,
)
from vote_simulation.models.rules.winner_metrics import METRIC_FIELDS, metrics_to_array


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
    # Per-rule metric accumulators (sum and sum-of-squares for online mean/std)
    _metrics_sum: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _metrics_sum_sq: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _metrics_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_step(self, step_result: SimulationStepResult) -> None:
        """Add one step result to the series and accumulate its distance matrix."""

        self.steps.append(step_result)
        self._accumulate_step(step_result)
        if step_result.config:
            self._config = self._config.merge(step_result.config)

    def add_rules_to_steps(self, new_rule_codes: list[str]) -> None:
        """Apply additional rules to all existing steps and update the series.

        Does not re-run existing rules, only computes distances for new rules.
        Rebuilds the accumulated distance matrix with all rules (old + new).

        Args:
            new_rule_codes: List of additional rule codes to apply to each step.

        Raises:
            ImportError: If ``vote_simulation.models.rules`` is not available.
        """
        if not new_rule_codes:
            return

        from vote_simulation.models.data_generation.data_instance import DataInstance
        from vote_simulation.models.rules import get_rule_builder

        # Apply new rules to each step
        for step in self.steps:
            if not step.data_source:
                print("Warning: Step without data_source, skipping rule application")
                continue

            try:
                di = DataInstance(step.data_source)
                profile = di.profile

                for code in new_rule_codes:
                    normalized = code.strip().upper()
                    if normalized in step.winners_by_rule:
                        continue  # Skip if rule already exists

                    try:
                        builder = get_rule_builder(normalized)
                        rule = builder(profile, None)
                        winners = rule.cowinners_
                        try:
                            metrics = rule.compute_metrics()
                            step.add_method_result_with_metrics(normalized, winners, metrics)
                        except Exception:
                            step.add_method_result(normalized, winners)
                    except Exception as e:
                        print(f"Error applying rule '{normalized}' to step: {e}")
                        step.add_method_result(normalized, [f"ERROR: {e}"])
            except Exception as e:
                print(f"Error loading data source '{step.data_source}': {e}")

        # Rebuild the aggregated distance matrix and metric accumulators
        self._rule_order = []
        self._rule_index = {}
        self._matrix_sum = np.zeros((0, 0), dtype=np.float64)
        self._iteration_count = 0
        self._metrics_sum = {}
        self._metrics_sum_sq = {}
        self._metrics_count = {}

        for step in self.steps:
            self._accumulate_step(step)

        # Update config to include new rules
        if new_rule_codes:
            new_rules = frozenset(c.strip().upper() for c in new_rule_codes)
            self._config = ResultConfig(
                gen_models=self._config.gen_models,
                n_voters=self._config.n_voters,
                n_candidates=self._config.n_candidates,
                rules_codes=self._config.rules_codes | new_rules,
                n_iterations=self._config.n_iterations,
            )

    def filter_rules(self, rule_codes: list[str]) -> SimulationSeriesResult:
        """Return a new series restricted to the specified rule codes.

        Only the accumulator state is copied — ``steps`` are not included in
        the returned object.  All plot and export methods work identically on
        the filtered result.

        Args:
            rule_codes: Rule codes to keep.  Unknown codes are silently
                ignored.  Codes are normalised (stripped, upper-cased) before
                matching.

        Returns:
            A new :class:`SimulationSeriesResult` with only the selected
            rules.  Returns an empty instance if none of the requested rules
            are present.
        """
        if not rule_codes:
            return SimulationSeriesResult()
        normalized = {r.strip().upper() for r in rule_codes}
        selected = [r for r in self._rule_order if r in normalized]
        if not selected:
            return SimulationSeriesResult()

        new = SimulationSeriesResult()
        new._rule_order = list(selected)
        new._rule_index = {r: i for i, r in enumerate(selected)}
        new._iteration_count = self._iteration_count
        new._config = self._config

        # Extract the sub-matrix for selected rules
        idx = np.array([self._rule_index[r] for r in selected], dtype=np.intp)
        new._matrix_sum = self._matrix_sum[np.ix_(idx, idx)].copy()

        # Filter per-rule metric accumulators
        new._metrics_sum = {r: v.copy() for r, v in self._metrics_sum.items() if r in normalized}
        new._metrics_sum_sq = {r: v.copy() for r, v in self._metrics_sum_sq.items() if r in normalized}
        new._metrics_count = {r: v for r, v in self._metrics_count.items() if r in normalized}

        return new

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

    @property
    def metrics_summary_frame(self) -> pd.DataFrame:
        """Per-rule winner-metric statistics aggregated across all iterations.

        Returns a :class:`~pandas.DataFrame` indexed by ``rule`` with two
        columns per metric field — one for the mean and one for the standard
        deviation across all accumulated steps:

        ``<field>_mean``, ``<field>_std``  for each field in
        :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS`.

        Rules for which no metrics were recorded (e.g. loaded from a parquet
        file without metrics) are omitted from the frame.

        An empty DataFrame is returned when no metrics have been accumulated.
        """
        if not self._metrics_sum:
            col_names = [f"{f}_{s}" for f in METRIC_FIELDS for s in ("mean", "std")]
            return pd.DataFrame(columns=pd.Index(np.asarray(col_names, dtype=object)))

        rows = []
        for rule in self._rule_order:
            if rule not in self._metrics_sum:
                continue
            count = self._metrics_count[rule]
            mean_arr = self._metrics_sum[rule] / count
            mean_sq_arr = self._metrics_sum_sq[rule] / count
            # population std — safe against floating precision below zero
            std_arr = np.sqrt(np.maximum(0.0, mean_sq_arr - mean_arr**2))
            row: dict[str, object] = {"rule": rule}
            for i, field_name in enumerate(METRIC_FIELDS):
                row[f"{field_name}_mean"] = float(mean_arr[i])
                row[f"{field_name}_std"] = float(std_arr[i])
            rows.append(row)

        if not rows:
            col_names = [f"{f}_{s}" for f in METRIC_FIELDS for s in ("mean", "std")]
            return pd.DataFrame(columns=pd.Index(np.asarray(col_names, dtype=object)))
        return pd.DataFrame(rows).set_index("rule")

    # ------------------------------------------------------------------
    # Distance metrics
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

    def export_mean_distance_matrix_csv(self, folder_save_path: str) -> str:
        """Export the mean distance matrix as CSV.

        Uses the same folder semantics as :meth:`plot_mean_distance_matrix`:
        if *folder_save_path* is a directory (or ends with a path separator),
        the file is written under a config-based sub-directory.

        Args:
            folder_save_path: Destination file path or folder path.

        Returns:
            The resolved CSV file path.
        """
        if self._iteration_count == 0:
            raise ValueError("Cannot export: no steps have been added yet.")

        save_path = self._resolve_save_path(
            folder_save_path,
            f"{self._iteration_count}_mean_distance_matrix.csv",
        )
        self.mean_distance_matrix_frame.to_csv(save_path, index=True)
        return save_path

    def export_metrics_summary_csv(
        self,
        folder_save_path: str,
        *,
        stat: str = "mean",
        metrics: list[str] | None = None,
    ) -> str:
        """Export per-rule winner metrics as CSV.

        Args:
            folder_save_path: Destination file path or folder path.
            stat: Statistic to export, ``"mean"`` or ``"std"``.
            metrics: Optional subset of metric field names.

        Returns:
            The resolved CSV file path.
        """
        if stat not in {"mean", "std"}:
            raise ValueError(f"Invalid stat '{stat}'. Expected 'mean' or 'std'.")

        frame = self.metrics_summary_frame
        if frame.empty:
            raise ValueError("No winner-metric data found. Check that metrics were computed during simulation.")

        fields = list(metrics) if metrics is not None else list(METRIC_FIELDS)
        invalid = [m for m in fields if m not in METRIC_FIELDS]
        if invalid:
            raise ValueError(f"Unknown metric fields: {invalid}")

        cols = [f"{field}_{stat}" for field in fields if f"{field}_{stat}" in frame.columns]
        if not cols:
            raise ValueError("No valid metric columns available for export.")

        save_path = self._resolve_save_path(
            folder_save_path,
            f"{self._iteration_count}_metrics_{stat}.csv",
        )
        frame.loc[:, cols].to_csv(save_path, index=True)
        return save_path

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
        return _mds_project(self.mean_distance_matrix.astype(np.float64), n_components=2)

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
        return _mds_project(self.mean_distance_matrix.astype(np.float64), n_components=3)

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
            save_path: Optional path (file or directory) to save the plot.

        Returns:
            The matplotlib Axes used for plotting.
        """
        projection = self.map_rules_3d()
        title = self._build_title("Rule proximity map (3D)")
        title += f"\nMDS stress: {projection.stress:.4f}"

        resolved: str | None = None
        if save_path is not None:
            resolved = self._resolve_save_path(save_path, f"{self._iteration_count}_rules_3d.png")

        ax = _plot_rules_3d_scatter(projection.coords, self._rule_order, title, ax, show=show, save_path=resolved)

        if resolved is not None:
            self.save_to_file(os.path.join(os.path.dirname(resolved), f"{self._iteration_count}_series.parquet"))

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

        Args:
            ax: Optional matplotlib Axes to draw on. A new figure is created
                when *None*.
            show: Whether to call ``plt.show()`` at the end.
            save_path: Optional path (file or directory) to save the plot.

        Returns:
            The matplotlib Axes used for plotting.
        """
        projection = self.map_rules_2d()
        title = self._build_title("Rule proximity map")

        resolved: str | None = None
        if save_path is not None:
            resolved = self._resolve_save_path(save_path, f"{self._iteration_count}_rules_2d.png")

        ax = _plot_rules_2d_scatter(projection.coords, self._rule_order, title, ax, show=show, save_path=resolved)

        if resolved is not None:
            self.save_to_file(os.path.join(os.path.dirname(resolved), f"{self._iteration_count}_series.parquet"))

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
                    rows.append(
                        {
                            "DataSource": step.data_source,
                            "GenModel": gm,
                            "NVoters": nv,
                            "NCandidates": nc,
                            "Rule": rule,
                            "Winner": winner,
                        }
                    )

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Store aggregated config in schema metadata
        existing_meta = table.schema.metadata or {}
        config_meta = {f"vote_sim:{k}".encode(): v.encode() for k, v in self._config.to_dict().items()}

        # Persist metric accumulators so they survive a round-trip through disk.
        # Serialised as JSON under the key "vote_sim:metrics".
        import json as _json

        if self._metrics_sum:
            metrics_payload = {
                "sum": {rule: arr.tolist() for rule, arr in self._metrics_sum.items()},
                "sum_sq": {rule: arr.tolist() for rule, arr in self._metrics_sum_sq.items()},
                "count": {rule: int(c) for rule, c in self._metrics_count.items()},
            }
            metrics_meta = {b"vote_sim:metrics": _json.dumps(metrics_payload).encode()}
        else:
            metrics_meta = {}

        table = table.replace_schema_metadata({**existing_meta, **config_meta, **metrics_meta})

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
            if k.decode().startswith("vote_sim:") and k != b"vote_sim:metrics"
        }

        df = table.to_pandas()

        self.steps = []
        self._rule_order = []
        self._rule_index = {}
        self._matrix_sum = np.zeros((0, 0), dtype=np.float64)
        self._iteration_count = 0
        self._config = ResultConfig()
        self._metrics_sum = {}
        self._metrics_sum_sq = {}
        self._metrics_count = {}

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
                data_source=str(data_source),
                config=step_config,
            )
            for rule, winners in group.groupby("Rule", sort=False)["Winner"]:
                step_result.add_method_result(str(rule), winners.tolist())
            self.add_step(step_result)

        # If schema metadata had config, prefer it (more complete for aggregates)
        if config_dict:
            self._config = ResultConfig.from_dict(config_dict)

        # Restore metric accumulators if they were persisted in the metadata.
        # This is backward-compatible: files without "vote_sim:metrics" will
        # simply have empty metric accumulators (metrics not available).
        import json as _json

        metrics_raw = meta.get(b"vote_sim:metrics")
        if metrics_raw:
            try:
                payload = _json.loads(metrics_raw.decode())
                self._metrics_sum = {
                    rule: np.array(arr, dtype=np.float64) for rule, arr in payload.get("sum", {}).items()
                }
                self._metrics_sum_sq = {
                    rule: np.array(arr, dtype=np.float64) for rule, arr in payload.get("sum_sq", {}).items()
                }
                self._metrics_count = {rule: int(c) for rule, c in payload.get("count", {}).items()}
            except Exception:
                # Corrupted metadata — silently ignore; metrics will be unavailable.
                pass

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
        """Add one step's distance matrix and winner metrics to the running sums."""

        step_rules = step._rule_order
        if not step_rules:
            return

        if not self._rule_order:
            # First step: establish canonical rule order for the whole series
            self._rule_order = list(step_rules)
            self._rule_index = dict(step._rule_index)
            n = len(step_rules)
            self._matrix_sum = np.zeros((n, n), dtype=np.float64)
        else:
            # Expand the matrix if this step introduces rules not seen before.
            new_rules = [r for r in step_rules if r not in self._rule_index]
            if new_rules:
                old_n = len(self._rule_order)
                for r in new_rules:
                    self._rule_index[r] = len(self._rule_order)
                    self._rule_order.append(r)
                new_n = len(self._rule_order)
                expanded = np.zeros((new_n, new_n), dtype=np.float64)
                expanded[:old_n, :old_n] = self._matrix_sum
                self._matrix_sum = expanded

        # Build permutation mapping step column index → series column index.
        # This handles the (rare) case where a step's rules are in a different order,
        # or when a step introduces rules not present in earlier steps.
        perm = np.array([self._rule_index[r] for r in step_rules], dtype=np.intp)
        self._matrix_sum[np.ix_(perm, perm)] += step._distance_matrix
        self._iteration_count += 1

        # Accumulate winner metrics (only for rules that carry metrics)
        n_fields = len(METRIC_FIELDS)
        for rule_code, wm in step._metrics_by_rule.items():
            arr = metrics_to_array(wm)
            if rule_code not in self._metrics_sum:
                self._metrics_sum[rule_code] = np.zeros(n_fields, dtype=np.float64)
                self._metrics_sum_sq[rule_code] = np.zeros(n_fields, dtype=np.float64)
                self._metrics_count[rule_code] = 0
            self._metrics_sum[rule_code] += arr
            self._metrics_sum_sq[rule_code] += arr * arr
            self._metrics_count[rule_code] += 1
