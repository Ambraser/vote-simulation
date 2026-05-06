"""Total result model for vote_simulation.

Aggregates multiple :class:`SimulationSeriesResult` instances across a
parameter space defined by *generative model*, *number of voters*, and
*number of candidates*.

Each series is uniquely keyed by ``(gen_model, n_voters, n_candidates)``.
Provides filtering, scalar-metric pivots, and comparison plots.

Typical workflow::

    total = SimulationTotalResult()
    for model in ["UNI", "IC"]:
        for n_v in [101, 1001]:
            for n_c in [3, 14]:
                series = simulation_instance(model, n_v, n_c, rules)
                total.add_series(series)

    # Fix model, compare across voters × candidates
    uni = total.filter(gen_model="UNI")
    uni.plot_metric_heatmap(row_param="n_voters", col_param="n_candidates")

    # Side-by-side rule distance heatmaps varying candidates
    total.filter(gen_model="UNI", n_voters=1001) \\
         .plot_comparison_grid("n_candidates")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from vote_simulation.models.results.series_result import SimulationSeriesResult
from vote_simulation.models.results.utils import _plot_heatmap
from vote_simulation.models.rules.winner_metrics import METRIC_FIELDS

# ------------------------------------------------------------------
# Key & constants
# ------------------------------------------------------------------


class SeriesKey(NamedTuple):
    """Identifies one series in the ``(model, voters, candidates)`` space."""

    gen_model: str
    n_voters: int
    n_candidates: int


_PARAM_NAMES: frozenset[str] = frozenset({"gen_model", "n_voters", "n_candidates"})


def _extract_key(series: SimulationSeriesResult) -> SeriesKey:
    """Derive a :class:`SeriesKey` from a series result's config.

    Each config field must contain exactly one value.
    """
    cfg = series.config
    if len(cfg.gen_models) != 1:
        raise ValueError(f"Series config must have exactly one gen_model, got {cfg.gen_models!r}")
    if len(cfg.n_voters) != 1:
        raise ValueError(f"Series config must have exactly one n_voters value, got {cfg.n_voters!r}")
    if len(cfg.n_candidates) != 1:
        raise ValueError(f"Series config must have exactly one n_candidates value, got {cfg.n_candidates!r}")
    return SeriesKey(
        gen_model=next(iter(cfg.gen_models)),
        n_voters=next(iter(cfg.n_voters)),
        n_candidates=next(iter(cfg.n_candidates)),
    )


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


@dataclass(slots=True)
class SimulationTotalResult:
    """Collection of series results spanning a parameter space.

    Each series is uniquely keyed by ``(gen_model, n_voters, n_candidates)``.
    Provides filtering, scalar-metric pivots, and comparison plots.
    """

    _entries: dict[SeriesKey, SimulationSeriesResult] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_series(self, series: SimulationSeriesResult) -> None:
        """Register a series result.

        Raises:
            ValueError: If a series with the same key is already present.
        """
        key = _extract_key(series)
        if key in self._entries:
            raise ValueError(f"Duplicate series key {key!r}. Use replace_series() to overwrite.")
        self._entries[key] = series

    def replace_series(self, series: SimulationSeriesResult) -> None:
        """Add or overwrite a series at its config key."""
        key = _extract_key(series)
        self._entries[key] = series

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def series_count(self) -> int:
        """Number of stored series."""
        return len(self._entries)

    @property
    def keys(self) -> list[SeriesKey]:
        """Sorted list of all series keys."""
        return sorted(self._entries)

    @property
    def gen_models(self) -> list[str]:
        """Sorted distinct generative-model codes."""
        return sorted({k.gen_model for k in self._entries})

    @property
    def voter_counts(self) -> list[int]:
        """Sorted distinct voter counts."""
        return sorted({k.n_voters for k in self._entries})

    @property
    def candidate_counts(self) -> list[int]:
        """Sorted distinct candidate counts."""
        return sorted({k.n_candidates for k in self._entries})

    def get_series(self, gen_model: str, n_voters: int, n_candidates: int) -> SimulationSeriesResult:
        """Retrieve a single series by its parameter triple.

        Raises:
            KeyError: If no matching series exists.
        """
        key = SeriesKey(gen_model, n_voters, n_candidates)
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(f"No series for {key!r}") from None

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self):  # noqa: ANN204
        """Iterate over ``(key, series)`` pairs in sorted key order."""
        yield from sorted(self._entries.items())

    def __repr__(self) -> str:
        return (
            f"SimulationTotalResult("
            f"series={self.series_count}, "
            f"models={self.gen_models}, "
            f"voters={self.voter_counts}, "
            f"candidates={self.candidate_counts})"
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        *,
        gen_model: str | None = None,
        n_voters: int | None = None,
        n_candidates: int | None = None,
    ) -> SimulationTotalResult:
        """Return a new instance containing only the matching series.

        Series objects are shared (shallow copy), not deep-copied.
        """
        result = SimulationTotalResult()
        for key, series in self._entries.items():
            if gen_model is not None and key.gen_model != gen_model:
                continue
            if n_voters is not None and key.n_voters != n_voters:
                continue
            if n_candidates is not None and key.n_candidates != n_candidates:
                continue
            result._entries[key] = series
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def summary_frame(self) -> pd.DataFrame:
        """One-row-per-series DataFrame with key fields and scalar metrics.

        Columns: ``gen_model``, ``n_voters``, ``n_candidates``,
        ``step_count``, ``n_iterations``, ``mean_distance``,
        ``most_distant_rule_a``, ``most_distant_rule_b``,
        ``most_distant_distance``.
        """
        rows: list[dict[str, Any]] = []
        for key in sorted(self._entries):
            series = self._entries[key]
            r_a, r_b, dist = series.most_distant_rules
            rows.append(
                {
                    "gen_model": key.gen_model,
                    "n_voters": key.n_voters,
                    "n_candidates": key.n_candidates,
                    "step_count": series.step_count,
                    "n_iterations": series.config.n_iterations,
                    "mean_distance": series.mean_distance,
                    "most_distant_rule_a": r_a,
                    "most_distant_rule_b": r_b,
                    "most_distant_distance": dist,
                }
            )
        return pd.DataFrame(rows)

    def metrics_comparison_frame(
        self,
        metric_field: str,
        rule_code: str,
        stat: str = "mean",
    ) -> pd.DataFrame:
        """Cross-parameter table of one winner metric for one rule.

        Builds a long-form :class:`~pandas.DataFrame` with one row per
        ``(gen_model, n_voters, n_candidates)`` entry, containing the
        requested aggregated statistic for ``rule_code``.

        Parameters
        ----------
        metric_field:
            One of the fields in
            :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS`
            (e.g. ``"social_acceptability"``, ``"rank_mean"``).
        rule_code:
            The normalised rule code to extract (e.g. ``"COPE"``).
        stat:
            Which statistic column to read from
            :attr:`~vote_simulation.models.results.series_result.SimulationSeriesResult.metrics_summary_frame`:
            ``"mean"`` (default) or ``"std"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``gen_model``, ``n_voters``, ``n_candidates``,
            ``<metric_field>_<stat>``.  Series without the requested rule's
            metrics are omitted.
        """
        col = f"{metric_field}_{stat}"
        rule_up = rule_code.strip().upper()
        rows: list[dict[str, Any]] = []
        for key in sorted(self._entries):
            series = self._entries[key]
            frame = series.metrics_summary_frame
            if frame.empty or rule_up not in frame.index or col not in frame.columns:
                continue
            rows.append(
                {
                    "gen_model": key.gen_model,
                    "n_voters": key.n_voters,
                    "n_candidates": key.n_candidates,
                    col: float(frame.loc[rule_up, col]),
                }
            )
        return pd.DataFrame(rows)

    def metrics_pivot(
        self,
        metric_field: str,
        rule_code: str,
        row_param: str = "n_voters",
        col_param: str = "n_candidates",
        stat: str = "mean",
    ) -> tuple[pd.DataFrame, str]:
        """Pivot :meth:`metrics_comparison_frame` into a 2D matrix.

        Analogous to :meth:`metric_matrix` but for winner metrics instead of
        rule-distance metrics.

        Parameters
        ----------
        metric_field:
            Metric field name (see :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS`).
        rule_code:
            The rule to inspect.
        row_param / col_param:
            Which of the three key parameters (``gen_model``, ``n_voters``,
            ``n_candidates``) to use as row/column axes.
        stat:
            ``"mean"`` or ``"std"``.

        Returns
        -------
        (pivot_df, fixed_description)
            The pivot DataFrame and a description of the fixed third parameter.
        """
        self._validate_axis_params(row_param, col_param)
        col = f"{metric_field}_{stat}"
        df = self.metrics_comparison_frame(metric_field, rule_code, stat=stat)
        if df.empty:
            return pd.DataFrame(), ""
        third = next(iter(_PARAM_NAMES - {row_param, col_param}))
        third_vals = {str(v) for v in df[third].unique()}
        fixed_desc = f"{third}={', '.join(sorted(third_vals))}" if third_vals else ""
        pivot = df.pivot_table(index=row_param, columns=col_param, values=col, aggfunc="mean")
        return pivot, fixed_desc

    def metric_matrix(
        self,
        row_param: str = "n_voters",
        col_param: str = "n_candidates",
        *,
        metric: str = "mean_distance",
    ) -> tuple[pd.DataFrame, str]:
        """Pivot a scalar metric into a 2D matrix.

        Args:
            row_param: Key field for the row axis.
            col_param: Key field for the column axis.
            metric: Column name from :meth:`summary_frame`
                (e.g. ``"mean_distance"``, ``"most_distant_distance"``).

        Returns:
            ``(pivot_df, fixed_description)`` — the pivot DataFrame and a
            human-readable description of the fixed (third) parameter.

        Raises:
            ValueError: If the third parameter has multiple distinct values.
        """
        self._validate_axis_params(row_param, col_param)

        third = next(iter(_PARAM_NAMES - {row_param, col_param}))
        third_vals = {getattr(k, third) for k in self._entries}
        if len(third_vals) > 1:
            raise ValueError(
                f"Parameter '{third}' has {len(third_vals)} distinct values "
                f"{third_vals}. Call .filter({third}=<value>) first."
            )

        fixed_desc = f"{third}={next(iter(third_vals))}" if third_vals else ""

        df = self.summary_frame()
        pivot = df.pivot_table(
            index=row_param,
            columns=col_param,
            values=metric,
            aggfunc="mean",
        )
        return pivot, fixed_desc

    def rule_pair_frame(self, rule_a: str, rule_b: str) -> pd.DataFrame:
        """Mean distance between two rules in every series.

        Returns a DataFrame with columns ``gen_model``, ``n_voters``,
        ``n_candidates``, ``distance``.
        """
        a, b = rule_a.strip().upper(), rule_b.strip().upper()
        rows: list[dict[str, Any]] = []
        for key in sorted(self._entries):
            series = self._entries[key]
            mat = series.mean_distance_matrix_frame
            if a in mat.index and b in mat.columns:
                dist = float(mat.loc[a, b])
            else:
                dist = float("nan")
            rows.append(
                {
                    "gen_model": key.gen_model,
                    "n_voters": key.n_voters,
                    "n_candidates": key.n_candidates,
                    "distance": dist,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_metric_heatmap(
        self,
        row_param: str = "n_voters",
        col_param: str = "n_candidates",
        *,
        metric: str = "mean_distance",
        ax: Any | None = None,
        annotate: bool = True,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Heatmap of a scalar metric pivoted across two parameters.

        The remaining (third) parameter must have a single distinct value
        — use :meth:`filter` first if needed.
        """
        import matplotlib.pyplot as plt

        pivot, fixed_desc = self.metric_matrix(row_param, col_param, metric=metric)
        matrix = pivot.to_numpy(dtype=np.float64)
        row_labels = [str(v) for v in pivot.index]
        col_labels = [str(v) for v in pivot.columns]

        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        margin = (vmax - vmin) * 0.1 or 1.0
        vmin = max(0.0, vmin - margin)
        vmax = vmax + margin

        fig_w = max(6.0, 1.2 * len(col_labels) + 2)
        fig_h = max(4.0, 1.0 * len(row_labels) + 2)
        if ax is None:
            _, ax = plt.subplots(
                figsize=(fig_w, fig_h),
                constrained_layout=True,
            )

        image = ax.imshow(
            matrix,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xticks(range(len(col_labels)), labels=col_labels)
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
        ax.set_xlabel(col_param)
        ax.set_ylabel(row_param)

        title = metric.replace("_", " ").title()
        if fixed_desc:
            title += f"\n({fixed_desc})"
        ax.set_title(title, fontsize=11)

        if annotate:
            fs = max(6, min(12, int(160 / max(matrix.size, 1))))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=fs,
                            color="black",
                        )

        cbar = ax.figure.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            shrink=0.9,
        )
        cbar.set_label(metric.replace("_", " ").title())

        if save_path is not None:
            fig = ax.figure
            assert isinstance(fig, plt.Figure)
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path)
        if show:
            plt.show()

        return ax

    def plot_rule_pair_heatmap(
        self,
        rule_a: str,
        rule_b: str,
        row_param: str = "n_voters",
        col_param: str = "n_candidates",
        *,
        ax: Any | None = None,
        annotate: bool = True,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Heatmap of one rule-pair distance across two parameters."""
        import matplotlib.pyplot as plt

        self._validate_axis_params(row_param, col_param)

        df = self.rule_pair_frame(rule_a, rule_b)
        third = next(iter(_PARAM_NAMES - {row_param, col_param}))
        third_vals = df[third].unique()
        if len(third_vals) > 1:
            raise ValueError(f"Parameter '{third}' has {len(third_vals)} values. Call .filter({third}=<value>) first.")

        pivot = df.pivot_table(
            index=row_param,
            columns=col_param,
            values="distance",
            aggfunc="mean",
        )
        matrix = pivot.to_numpy(dtype=np.float64)
        row_labels = [str(v) for v in pivot.index]
        col_labels = [str(v) for v in pivot.columns]

        fig_w = max(6.0, 1.2 * len(col_labels) + 2)
        fig_h = max(4.0, 1.0 * len(row_labels) + 2)
        if ax is None:
            _, ax = plt.subplots(
                figsize=(fig_w, fig_h),
                constrained_layout=True,
            )

        image = ax.imshow(
            matrix,
            cmap="Reds",
            vmin=0,
            vmax=100,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xticks(range(len(col_labels)), labels=col_labels)
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
        ax.set_xlabel(col_param)
        ax.set_ylabel(row_param)

        a_up, b_up = rule_a.strip().upper(), rule_b.strip().upper()
        title = f"Distance: {a_up} \u2194 {b_up}"
        if len(third_vals) == 1:
            title += f"\n({third}={third_vals[0]})"
        ax.set_title(title, fontsize=11)

        if annotate:
            fs = max(6, min(12, int(160 / max(matrix.size, 1))))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=fs,
                            color="black",
                        )

        cbar = ax.figure.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            shrink=0.9,
        )
        cbar.set_label("Mean distance (%)")

        if save_path is not None:
            fig = ax.figure
            assert isinstance(fig, plt.Figure)
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path)
        if show:
            plt.show()

        return ax

    def plot_metrics_rules_matrix(
        self,
        row_param: str = "n_voters",
        col_param: str = "n_candidates",
        *,
        stat: str = "mean",
        metrics: list[str] | None = None,
        annotate: bool = True,
        fmt: str = ".3f",
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Heatmap with rules as columns and metrics as rows.

        Each cell shows the aggregated statistic (*stat*) for a given
        ``(metric, rule)`` pair, averaged over all series currently in the
        object (use :meth:`filter` to restrict the scope first).

        The color scale is **normalised per row** (per metric) so that the
        gradient is uniform within each row and rules can be directly compared
        regardless of the absolute scale of each metric.

        Args:
            row_param: Used only to derive a description of fixed parameters
                in the title.  Filtering is the recommended way to narrow down
                the data.
            col_param: Same as *row_param*.
            stat: ``"mean"`` (default) or ``"std"``.
            metrics: Explicit list of metric fields (rows).  When *None* all
                fields in :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS`
                are included.
            annotate: Whether to print raw values inside each cell.
            fmt: Format string used for annotations (e.g. ``\".3f\"``).
            show: Whether to call ``plt.show()`` at the end.
            save_path: Optional file path to save the figure.

        Returns:
            The matplotlib Axes used for plotting.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        fields = list(metrics) if metrics is not None else list(METRIC_FIELDS)

        # Collect per-rule means across all series
        # Discover all rules present in any series
        all_rules: list[str] = []
        for series in self._entries.values():
            frame = series.metrics_summary_frame
            if not frame.empty:
                for r in frame.index:
                    if r not in all_rules:
                        all_rules.append(r)

        if not all_rules:
            raise ValueError(
                "No winner-metric data found. "
                "Check that metrics were computed during simulation."
            )

        # Build raw matrix: shape (n_metrics, n_rules)
        # Average over all series
        n_m = len(fields)
        n_r = len(all_rules)
        raw = np.full((n_m, n_r), np.nan)

        for s_idx, series in enumerate(self._entries.values()):
            frame = series.metrics_summary_frame
            if frame.empty:
                continue
            for ri, rule in enumerate(all_rules):
                if rule not in frame.index:
                    continue
                for mi, field in enumerate(fields):
                    col = f"{field}_{stat}"
                    if col not in frame.columns:
                        continue
                    val = float(frame.loc[rule, col])
                    if np.isnan(raw[mi, ri]):
                        raw[mi, ri] = val
                    else:
                        raw[mi, ri] = (raw[mi, ri] * s_idx + val) / (s_idx + 1)

        # Drop metric rows that are entirely NaN
        valid_mask = ~np.all(np.isnan(raw), axis=1)
        fields = [f for f, v in zip(fields, valid_mask) if v]
        raw = raw[valid_mask]

        if len(fields) == 0:
            raise ValueError("No valid metric data to plot.")

        # Row-normalise: each row scaled to [0, 1]
        row_min = np.nanmin(raw, axis=1, keepdims=True)
        row_max = np.nanmax(raw, axis=1, keepdims=True)
        row_range = row_max - row_min
        row_range[row_range == 0] = 1.0  # avoid division by zero for constant rows
        normed = (raw - row_min) / row_range

        fig_w = max(8.0, 1.4 * n_r + 2)
        fig_h = max(5.0, 0.6 * len(fields) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        im = ax.imshow(normed, cmap="YlGnBu", vmin=0, vmax=1,
                       interpolation="nearest", aspect="auto")

        ax.set_xticks(range(n_r), labels=all_rules, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(fields)), labels=[f.replace("_", " ") for f in fields], fontsize=9)
        ax.set_xlabel("Rule", fontsize=10)
        ax.set_ylabel("Metric", fontsize=10)

        # Build title from fixed params info
        n_series = self.series_count
        fixed_parts = []
        if len(self.gen_models) == 1:
            fixed_parts.append(f"model={self.gen_models[0]}")
        if len(self.voter_counts) == 1:
            fixed_parts.append(f"n_voters={self.voter_counts[0]}")
        if len(self.candidate_counts) == 1:
            fixed_parts.append(f"n_candidates={self.candidate_counts[0]}")
        desc = " · ".join(fixed_parts) if fixed_parts else f"{n_series} series averaged"
        ax.set_title(
            f"Winner metrics ({stat}) per rule\n{desc}",
            fontsize=11,
        )

        if annotate:
            fs = max(6, min(10, int(120 / max(n_r, 1))))
            for mi in range(len(fields)):
                for ri in range(n_r):
                    val = raw[mi, ri]
                    if not np.isnan(val):
                        cell_norm = normed[mi, ri]
                        txt_color = "white" if cell_norm > 0.65 else "black"
                        ax.text(ri, mi, format(val, fmt),
                                ha="center", va="center",
                                fontsize=fs, color=txt_color)

        cbar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap="YlGnBu"),
            ax=ax, fraction=0.03, pad=0.02, shrink=0.8,
        )
        cbar.set_label("Normalised value (per metric)", fontsize=9)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["min", "mid", "max"])

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path)
        if show:
            plt.show()
        return ax

    def plot_winner_metric_heatmap(
        self,
        metric_field: str,
        rule_code: str,
        row_param: str = "n_voters",
        col_param: str = "n_candidates",
        *,
        stat: str = "mean",
        ax: Any | None = None,
        annotate: bool = True,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Heatmap of one winner metric for one rule, pivoted across two parameters.

        Analogous to :meth:`plot_metric_heatmap` but for
        :class:`~vote_simulation.models.rules.WinnerMetrics` fields instead of
        rule-distance metrics.

        Args:
            metric_field: One of the fields in
                :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS`
                (e.g. ``"social_acceptability"``, ``"rank_mean"``).
            rule_code: Normalised rule code to inspect (e.g. ``"COPE"``).
            row_param: Row axis — ``"gen_model"``, ``"n_voters"``, or
                ``"n_candidates"``.
            col_param: Column axis (same choices).
            stat: ``"mean"`` (default) or ``"std"``.
            ax: Optional matplotlib Axes.  A new figure is created when *None*.
            annotate: Whether to print cell values on the heatmap.
            show: Whether to call ``plt.show()`` at the end.
            save_path: Optional file path to save the figure.

        Returns:
            The matplotlib Axes used for plotting.
        """
        import matplotlib.pyplot as plt

        pivot, fixed_desc = self.metrics_pivot(
            metric_field, rule_code,
            row_param=row_param, col_param=col_param, stat=stat,
        )
        if pivot.empty:
            raise ValueError(
                f"No data for metric '{metric_field}' / rule '{rule_code}'. "
                "Check that metrics were computed during simulation."
            )

        matrix = pivot.to_numpy(dtype=np.float64)
        row_labels = [str(v) for v in pivot.index]
        col_labels = [str(v) for v in pivot.columns]

        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        margin = (vmax - vmin) * 0.1 or 0.01
        vmin = max(0.0, vmin - margin)
        vmax = min(1.0, vmax + margin) if metric_field not in {"rank_mean", "rank_median", "rank_var", "utility_mean", "utility_median", "utility_var", "n_cowinners"} else vmax + margin

        fig_w = max(6.0, 1.2 * len(col_labels) + 2)
        fig_h = max(4.0, 1.0 * len(row_labels) + 2)
        if ax is None:
            _, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        image = ax.imshow(
            matrix,
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xticks(range(len(col_labels)), labels=col_labels)
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
        ax.set_xlabel(col_param)
        ax.set_ylabel(row_param)

        rule_up = rule_code.strip().upper()
        title = f"{metric_field.replace('_', ' ').title()} ({stat}) — {rule_up}"
        if fixed_desc:
            title += f"\n({fixed_desc})"
        ax.set_title(title, fontsize=11)

        if annotate:
            fs = max(6, min(12, int(160 / max(matrix.size, 1))))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=fs, color="black")

        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
        cbar.set_label(f"{metric_field.replace('_', ' ').title()} ({stat})")

        if save_path is not None:
            fig = ax.figure
            assert isinstance(fig, plt.Figure)
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path)
        if show:
            plt.show()
        return ax

    def plot_winner_metrics_grid(
        self,
        rule_code: str,
        row_param: str = "n_voters",
        col_param: str = "n_candidates",
        *,
        stat: str = "mean",
        metrics: list[str] | None = None,
        annotate: bool = True,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Grid of heatmaps — one panel per winner metric — for a single rule.

        Iterates over every field in
        :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS` (or
        the subset given by *metrics*) and draws each as a heatmap panel,
        using the same axes as :meth:`plot_winner_metric_heatmap`.

        Args:
            rule_code: Normalised rule code to inspect (e.g. ``"COPE"``).
            row_param: Row axis — ``"gen_model"``, ``"n_voters"``, or
                ``"n_candidates"``.
            col_param: Column axis (same choices).
            stat: ``"mean"`` (default) or ``"std"``.
            metrics: Explicit list of metric fields to plot.  When *None* all
                fields in :data:`~vote_simulation.models.rules.winner_metrics.METRIC_FIELDS`
                are included.
            annotate: Whether to print cell values on each panel.
            show: Whether to call ``plt.show()`` at the end.
            save_path: Optional file path to save the whole figure.

        Returns:
            The 2-D NumPy array of matplotlib Axes.
        """
        import math
        import matplotlib.pyplot as plt

        fields = list(metrics) if metrics is not None else list(METRIC_FIELDS)

        # Drop metrics with no data
        available = [
            f for f in fields
            if not self.metrics_comparison_frame(f, rule_code, stat=stat).empty
        ]
        if not available:
            raise ValueError(
                f"No winner-metric data found for rule '{rule_code}'. "
                "Check that metrics were computed during simulation."
            )

        n = len(available)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            constrained_layout=True,
        )
        # Normalise axes to a flat list
        axes_flat: list[Any] = np.array(axes).flatten().tolist()

        rule_up = rule_code.strip().upper()
        fig.suptitle(
            f"Winner metrics ({stat}) — {rule_up}",
            fontsize=13,
            fontweight="bold",
        )

        for ax_i, field in zip(axes_flat, available):
            try:
                self.plot_winner_metric_heatmap(
                    field, rule_code,
                    row_param=row_param, col_param=col_param,
                    stat=stat, ax=ax_i, annotate=annotate, show=False,
                )
            except ValueError:
                ax_i.set_visible(False)

        # Hide unused axes
        for ax_i in axes_flat[len(available):]:
            ax_i.set_visible(False)

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path)
        if show:
            plt.show()
        return axes

    def plot_comparison_grid(
        self,
        vary_param: str = "n_candidates",
        *,
        annotate: bool = True,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Side-by-side rule-distance heatmaps, one per value of *vary_param*.

        The other two parameters must each have a single distinct value
        — use :meth:`filter` first.

        Example::

            total.filter(gen_model="UNI", n_voters=1001) \\
                 .plot_comparison_grid("n_candidates")
        """
        import matplotlib.pyplot as plt

        if vary_param not in _PARAM_NAMES:
            raise ValueError(f"Invalid param '{vary_param}'. Choose from {sorted(_PARAM_NAMES)}")

        fixed_params = _PARAM_NAMES - {vary_param}

        vary_values = sorted({getattr(k, vary_param) for k in self._entries})
        n = len(vary_values)
        if n == 0:
            raise ValueError("No series to plot.")

        fig, axes = plt.subplots(
            1,
            n,
            figsize=(6 * n + 1, 6),
            constrained_layout=True,
        )
        if n == 1:
            axes = [axes]

        for ax_i, val in zip(axes, vary_values, strict=True):
            matching = [s for k, s in self._entries.items() if getattr(k, vary_param) == val]
            if not matching:
                continue
            # Average distance matrices when multiple series match
            avg_matrix = np.mean(
                np.stack([s.mean_distance_matrix for s in matching]),
                axis=0,
            )
            labels = matching[0]._rule_order
            subtitle = f"{vary_param}={val}"
            if len(matching) > 1:
                subtitle += f" (avg. {len(matching)})"
            _plot_heatmap(
                avg_matrix,
                labels,
                subtitle,
                ax=ax_i,
                annotate=annotate,
                annotation_fmt=".1f",
                colorbar_label="Mean distance (%)",
                show=False,
            )

        # Build description of fixed parameters
        fixed_parts: list[str] = []
        for p in sorted(fixed_params):
            vals = sorted({getattr(k, p) for k in self._entries}, key=str)
            if len(vals) == 1:
                fixed_parts.append(f"{p}={vals[0]}")
            else:
                fixed_parts.append(f"{p}: averaged")
        fixed_desc = " \u00b7 ".join(fixed_parts)

        fig.suptitle(
            f"Rule distance comparison\n{fixed_desc}",
            fontsize=13,
        )

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path)
        if show:
            plt.show()

        return axes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_dir(self, dir_path: str) -> None:
        """Persist every series as ``<label>.parquet`` inside *dir_path*."""
        os.makedirs(dir_path, exist_ok=True)
        for _key, series in self._entries.items():
            filename = f"{series.config.label}.parquet"
            series.save_to_file(os.path.join(dir_path, filename))

    @classmethod
    def load_from_dir(cls, dir_path: str) -> SimulationTotalResult:
        """Reconstruct a :class:`SimulationTotalResult` from a folder of parquet files.

        Each ``.parquet`` file is loaded as a :class:`SimulationSeriesResult`
        and registered via :meth:`add_series`.
        """
        import glob as _glob

        total = cls()
        paths = sorted(_glob.glob(os.path.join(dir_path, "*.parquet")))
        if not paths:
            raise ValueError(f"No .parquet files found in {dir_path!r}")
        for path in paths:
            series = SimulationSeriesResult()
            series.load_from_file(path)
            total.add_series(series)
        return total

    @staticmethod
    def delete_dir(dir_path: str) -> bool:
        """Remove a saved total-result directory.

        Returns ``True`` if the directory existed and was deleted.
        """
        import shutil

        try:
            shutil.rmtree(dir_path)
            return True
        except FileNotFoundError:
            return False

    # Private helpers

    @staticmethod
    def _validate_axis_params(row_param: str, col_param: str) -> None:
        """Ensure *row_param* and *col_param* are valid and distinct."""
        if row_param not in _PARAM_NAMES:
            raise ValueError(f"Invalid row_param '{row_param}'. Choose from {sorted(_PARAM_NAMES)}")
        if col_param not in _PARAM_NAMES:
            raise ValueError(f"Invalid col_param '{col_param}'. Choose from {sorted(_PARAM_NAMES)}")
        if row_param == col_param:
            raise ValueError("row_param and col_param must be different.")