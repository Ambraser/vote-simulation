"""Tests for SimulationTotalResult."""

import os
import tempfile

import matplotlib
import numpy as np
import pytest

from vote_simulation.models.results.result_config import ResultConfig
from vote_simulation.models.results.series_result import SimulationSeriesResult
from vote_simulation.models.results.step_result import SimulationStepResult
from vote_simulation.models.results.total_result import (
    SeriesKey,
    SimulationTotalResult,
    _extract_key,
)

matplotlib.use("Agg")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_series(model: str, n_v: int, n_c: int, n_steps: int = 3) -> SimulationSeriesResult:
    """Create a minimal series with deterministic winners."""
    cfg = ResultConfig.single(gen_model=model, n_voters=n_v, n_candidates=n_c)
    series = SimulationSeriesResult()
    for i in range(n_steps):
        step = SimulationStepResult(
            data_source=f"test_{model}_{n_v}_{n_c}_{i}",
            config=cfg,
        )
        step.add_method_result("PLU1", [f"c{i % n_c}"])
        step.add_method_result("BORD", [f"c{(i + 1) % n_c}"])
        step.add_method_result("RV", ["c0"])
        series.add_step(step)
    return series


def _make_total_4() -> SimulationTotalResult:
    """Build a total with 4 series: 2 models × 2 candidate counts, fixed voters."""
    total = SimulationTotalResult()
    total.add_series(_make_series("UNI", 100, 3))
    total.add_series(_make_series("UNI", 100, 5))
    total.add_series(_make_series("IC", 100, 3))
    total.add_series(_make_series("IC", 100, 5))
    return total


# ------------------------------------------------------------------
# Key extraction
# ------------------------------------------------------------------


class TestExtractKey:
    def test_valid_single_valued_config(self) -> None:
        series = _make_series("UNI", 100, 3)
        key = _extract_key(series)
        assert key == SeriesKey("UNI", 100, 3)

    def test_empty_config_raises(self) -> None:
        series = SimulationSeriesResult()
        with pytest.raises(ValueError, match="exactly one gen_model"):
            _extract_key(series)

    def test_multi_model_config_raises(self) -> None:
        series = SimulationSeriesResult()
        # Manually build a multi-valued config
        s1 = SimulationStepResult(
            data_source="a",
            config=ResultConfig.single(gen_model="UNI", n_voters=10, n_candidates=3),
        )
        s1.add_method_result("PLU1", ["c0"])
        s2 = SimulationStepResult(
            data_source="b",
            config=ResultConfig.single(gen_model="IC", n_voters=10, n_candidates=3),
        )
        s2.add_method_result("PLU1", ["c0"])
        series.add_step(s1)
        series.add_step(s2)
        # config now has gen_models={"UNI", "IC"}
        with pytest.raises(ValueError, match="exactly one gen_model"):
            _extract_key(series)


# ------------------------------------------------------------------
# Mutation & accessors
# ------------------------------------------------------------------


class TestMutation:
    def test_add_series(self) -> None:
        total = SimulationTotalResult()
        s = _make_series("UNI", 100, 3)
        total.add_series(s)
        assert total.series_count == 1

    def test_duplicate_key_raises(self) -> None:
        total = SimulationTotalResult()
        total.add_series(_make_series("UNI", 100, 3))
        with pytest.raises(ValueError, match="Duplicate"):
            total.add_series(_make_series("UNI", 100, 3))

    def test_replace_series_overwrites(self) -> None:
        total = SimulationTotalResult()
        s1 = _make_series("UNI", 100, 3, n_steps=2)
        s2 = _make_series("UNI", 100, 3, n_steps=5)
        total.add_series(s1)
        total.replace_series(s2)
        assert total.series_count == 1
        assert total.get_series("UNI", 100, 3).step_count == 5


class TestAccessors:
    def test_keys(self) -> None:
        total = _make_total_4()
        assert total.keys == [
            SeriesKey("IC", 100, 3),
            SeriesKey("IC", 100, 5),
            SeriesKey("UNI", 100, 3),
            SeriesKey("UNI", 100, 5),
        ]

    def test_gen_models(self) -> None:
        total = _make_total_4()
        assert total.gen_models == ["IC", "UNI"]

    def test_voter_counts(self) -> None:
        total = _make_total_4()
        assert total.voter_counts == [100]

    def test_candidate_counts(self) -> None:
        total = _make_total_4()
        assert total.candidate_counts == [3, 5]

    def test_get_series(self) -> None:
        total = _make_total_4()
        s = total.get_series("UNI", 100, 3)
        assert s.step_count == 3

    def test_get_series_missing_raises(self) -> None:
        total = _make_total_4()
        with pytest.raises(KeyError):
            total.get_series("MISSING", 100, 3)

    def test_len(self) -> None:
        total = _make_total_4()
        assert len(total) == 4

    def test_contains(self) -> None:
        total = _make_total_4()
        assert SeriesKey("UNI", 100, 3) in total
        assert SeriesKey("MISSING", 1, 1) not in total

    def test_iter_sorted(self) -> None:
        total = _make_total_4()
        keys_from_iter = [k for k, _s in total]
        assert keys_from_iter == sorted(keys_from_iter)

    def test_repr(self) -> None:
        total = _make_total_4()
        r = repr(total)
        assert "series=4" in r
        assert "UNI" in r


# ------------------------------------------------------------------
# Filtering
# ------------------------------------------------------------------


class TestFilter:
    def test_filter_by_model(self) -> None:
        total = _make_total_4()
        uni = total.filter(gen_model="UNI")
        assert uni.series_count == 2
        assert uni.gen_models == ["UNI"]

    def test_filter_by_candidates(self) -> None:
        total = _make_total_4()
        c3 = total.filter(n_candidates=3)
        assert c3.series_count == 2
        assert c3.candidate_counts == [3]

    def test_filter_multiple_criteria(self) -> None:
        total = _make_total_4()
        one = total.filter(gen_model="IC", n_candidates=5)
        assert one.series_count == 1

    def test_filter_no_match_empty(self) -> None:
        total = _make_total_4()
        empty = total.filter(gen_model="MISSING")
        assert empty.series_count == 0

    def test_filter_shares_series_references(self) -> None:
        total = _make_total_4()
        filtered = total.filter(gen_model="UNI")
        # Same object, not a copy
        assert filtered.get_series("UNI", 100, 3) is total.get_series("UNI", 100, 3)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


class TestMetrics:
    def test_summary_frame_shape(self) -> None:
        total = _make_total_4()
        df = total.summary_frame()
        assert len(df) == 4
        assert set(df.columns) == {
            "gen_model",
            "n_voters",
            "n_candidates",
            "step_count",
            "n_iterations",
            "mean_distance",
            "most_distant_rule_a",
            "most_distant_rule_b",
            "most_distant_distance",
        }

    def test_summary_frame_values(self) -> None:
        total = _make_total_4()
        df = total.summary_frame()
        # All series have 3 steps
        assert (df["step_count"] == 3).all()
        # Mean distances should be non-negative
        assert (df["mean_distance"] >= 0).all()

    def test_metric_matrix_shape(self) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        pivot, desc = total.metric_matrix("n_voters", "n_candidates")
        assert pivot.shape == (1, 2)  # 1 voter count, 2 candidate counts
        assert "gen_model=UNI" in desc

    def test_metric_matrix_multiple_third_values_averages(self) -> None:
        # When the third dimension has multiple values, metric_matrix averages them
        total = _make_total_4()
        pivot, desc = total.metric_matrix("n_voters", "n_candidates")
        assert pivot.shape == (1, 2)  # 1 voter count, 2 candidate counts
        assert "avg" in desc

    def test_metric_matrix_invalid_params(self) -> None:
        total = _make_total_4()
        with pytest.raises(ValueError, match="Invalid"):
            total.metric_matrix("bad_param", "n_candidates")
        with pytest.raises(ValueError, match="must be different"):
            total.metric_matrix("n_voters", "n_voters")

    def test_rule_pair_frame(self) -> None:
        total = _make_total_4()
        df = total.rule_pair_frame("PLU1", "BORD")
        assert len(df) == 4
        assert "distance" in df.columns
        # Distances should be valid (not NaN since those rules exist)
        assert not df["distance"].isna().any()

    def test_rule_pair_frame_missing_rule_gives_nan(self) -> None:
        total = _make_total_4()
        df = total.rule_pair_frame("PLU1", "NONEXISTENT")
        assert df["distance"].isna().all()

    def test_mean_distance_matrix_frame_returns_dataframe(self) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        frame = total.mean_distance_matrix_frame()
        assert list(frame.index) == list(frame.columns)
        assert frame.shape[0] >= 2

    def test_export_mean_distance_matrix_csv(self) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mean_distance.csv")
            out = total.export_mean_distance_matrix_csv(path)
            assert out == path
            assert os.path.isfile(path)
            loaded = np.genfromtxt(path, delimiter=",", dtype=str)
            assert loaded.shape[0] > 1

    def test_export_metrics_rules_matrix_csv(self) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.csv")
            out = total.export_metrics_rules_matrix_csv(path)
            assert out == path
            assert os.path.isfile(path)
            loaded = np.genfromtxt(path, delimiter=",", dtype=str)
            assert loaded.shape[0] > 1


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


class TestPlotting:
    def test_plot_metric_heatmap_returns_axes(self) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        ax = total.plot_metric_heatmap("n_voters", "n_candidates", show=False)
        assert ax is not None
        title = ax.get_title()
        assert "Mean Distance" in title

    def test_plot_rule_pair_heatmap_returns_axes(self) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        ax = total.plot_rule_pair_heatmap(
            "PLU1",
            "BORD",
            "n_voters",
            "n_candidates",
            show=False,
        )
        assert ax is not None
        assert "PLU1" in ax.get_title()

    def test_plot_comparison_grid_returns_axes(self) -> None:
        total = _make_total_4().filter(gen_model="UNI", n_voters=100)
        axes = total.plot_comparison_grid("n_candidates", show=False)
        assert len(axes) == 2

    def test_plot_comparison_grid_averages_unfixed(self) -> None:
        total = _make_total_4()
        # gen_model has 2 values → should average across models, not raise
        axes = total.plot_comparison_grid("n_candidates", show=False)
        assert len(axes) == 2

    def test_plot_metric_heatmap_save(self) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "heatmap.png")
            total.plot_metric_heatmap(
                "n_voters",
                "n_candidates",
                show=False,
                save_path=path,
            )
            assert os.path.isfile(path)

    def test_plot_mean_distance_matrix_title_shows_averaged_voters(self) -> None:
        total = SimulationTotalResult()
        total.add_series(_make_series("UNI", 10, 3))
        total.add_series(_make_series("UNI", 20, 3))
        total.add_series(_make_series("UNI", 50, 3))

        ax = total.filter(gen_model="UNI", n_candidates=3).plot_mean_distance_matrix(show=False)
        assert "n_votants=avg(10,20,50)" in ax.get_title()

    def test_plot_mean_distance_matrix_title_shows_averaged_candidates(self) -> None:
        total = SimulationTotalResult()
        total.add_series(_make_series("UNI", 100, 3))
        total.add_series(_make_series("UNI", 100, 5))
        total.add_series(_make_series("UNI", 100, 8))

        ax = total.filter(gen_model="UNI", n_voters=100).plot_mean_distance_matrix(show=False)
        assert "n_candidates=avg(3,5,8)" in ax.get_title()

    def test_plot_mean_distance_matrix_save_basename(self, tmp_path) -> None:
        total = _make_total_4().filter(gen_model="UNI")
        previous_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            total.plot_mean_distance_matrix(show=False, save_path="mean_matrix.png")
            assert os.path.isfile("mean_matrix.png")
            assert os.path.getsize("mean_matrix.png") > 0
        finally:
            os.chdir(previous_cwd)


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_round_trip(self) -> None:
        total = _make_total_4()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "total")
            total.save_to_dir(save_dir)

            # Check files were created
            files = os.listdir(save_dir)
            assert len(files) == 4
            assert all(f.endswith(".parquet") for f in files)

            # Reload
            loaded = SimulationTotalResult.load_from_dir(save_dir)
            assert loaded.series_count == 4
            assert loaded.gen_models == total.gen_models
            assert loaded.voter_counts == total.voter_counts
            assert loaded.candidate_counts == total.candidate_counts

            # Verify metrics match
            orig_df = total.summary_frame()
            load_df = loaded.summary_frame()
            np.testing.assert_array_almost_equal(
                orig_df["mean_distance"].values,
                load_df["mean_distance"].values,
            )

    def test_load_from_empty_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No .parquet"):
                SimulationTotalResult.load_from_dir(tmpdir)

    def test_delete_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "to_delete")
            _make_total_4().save_to_dir(save_dir)
            assert os.path.isdir(save_dir)
            assert SimulationTotalResult.delete_dir(save_dir) is True
            assert not os.path.isdir(save_dir)

    def test_delete_missing_dir_returns_false(self) -> None:
        assert SimulationTotalResult.delete_dir("/tmp/nonexistent_total_result") is False
