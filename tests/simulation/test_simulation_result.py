import os
import tempfile

import matplotlib
import numpy as np

from vote_simulation.models.results.series_result import ResultConfig, SimulationSeriesResult, SimulationStepResult

matplotlib.use("Agg")


def test_add_method_result_builds_symmetric_uint8_matrix() -> None:
    step = SimulationStepResult(data_source="sample.parquet")

    step.add_method_result("borda", ["A"])
    step.add_method_result("stv", ["A"])
    step.add_method_result("irv", ["B"])

    expected = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )

    assert step.rule_codes == ["BORDA", "STV", "IRV"]
    assert step.dist_matrix.dtype == np.float32
    assert np.array_equal(step.dist_matrix, expected)


def test_updating_existing_rule_refreshes_only_its_distances() -> None:
    step = SimulationStepResult(
        data_source="sample.parquet",
        winners_by_rule={
            "BORDA": ["A"],
            "STV": ["A"],
            "IRV": ["B"],
        },
    )

    step.add_method_result("stv", ["B"])

    expected = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(step.dist_matrix, expected)


def test_string_representation_contains_pretty_matrix() -> None:
    step = SimulationStepResult(data_source="sample.parquet")
    step.add_method_result("borda", ["A"])
    step.add_method_result("stv", ["B"])

    rendered = str(step)

    assert "Data Source: sample.parquet" in rendered
    assert "Winners by rule:" in rendered
    assert "Distance Matrix:" in rendered
    assert "BORDA" in rendered
    assert "STV" in rendered


def test_plot_distance_matrix_returns_axes() -> None:
    step = SimulationStepResult(data_source="sample.parquet")
    step.add_method_result("borda", ["A"])
    step.add_method_result("stv", ["B"])

    ax = step.plot_distance_matrix(show=False)

    assert "Rule distance matrix" in ax.get_title()
    assert ax.get_aspect() == 1.0
    assert [tick.get_text() for tick in ax.get_xticklabels()[:2]] == ["BORDA", "STV"]


def test_series_step_count_and_mean_diagonal_is_zero() -> None:
    """Mean of binary matrices should have a zero diagonal."""

    series = SimulationSeriesResult()
    for _ in range(5):
        step = SimulationStepResult(data_source="iter.parquet")
        step.add_method_result("BORDA", ["A"])
        step.add_method_result("STV", ["B"])
        series.add_step(step)

    assert series.step_count == 5
    mean = series.mean_distance_matrix
    assert mean.dtype == np.float32
    assert np.all(np.diag(mean) == 0)


def test_series_mean_values_correct() -> None:
    """Mean matrix should average properly over iterations."""

    series = SimulationSeriesResult()

    # 3 steps where BORDA and STV always disagree → mean distance = 1.0
    step_always = SimulationStepResult(data_source="a.parquet")
    step_always.add_method_result("BORDA", ["A"])
    step_always.add_method_result("STV", ["B"])
    series.add_step(step_always)
    series.add_step(step_always)
    series.add_step(step_always)

    # 1 step where they agree → mean distance = 0.75 over 4 steps
    step_agree = SimulationStepResult(data_source="b.parquet")
    step_agree.add_method_result("BORDA", ["A"])
    step_agree.add_method_result("STV", ["A"])
    series.add_step(step_agree)

    mean = series.mean_distance_matrix
    borda_idx = series._rule_index["BORDA"]
    stv_idx = series._rule_index["STV"]
    # 3/4 steps disagree → 75 on the [0,100] scale
    assert abs(float(mean[borda_idx, stv_idx]) - 75.0) < 0.1


def test_series_plot_mean_distance_matrix_returns_axes() -> None:
    series = SimulationSeriesResult()
    step = SimulationStepResult(data_source="x.parquet")
    step.add_method_result("BORDA", ["A"])
    step.add_method_result("STV", ["B"])
    series.add_step(step)

    ax = series.plot_mean_distance_matrix(show=False)

    assert "Mean rule distance matrix" in ax.get_title()
    assert ax.get_aspect() == 1.0


# ======================================================================
# ResultConfig tests
# ======================================================================


def test_result_config_single_factory() -> None:
    cfg = ResultConfig.single(gen_model="IC", n_voters=101, n_candidates=3)
    assert cfg.gen_models == frozenset({"IC"})
    assert cfg.n_voters == frozenset({101})
    assert cfg.n_candidates == frozenset({3})
    assert cfg.label == "IC_v101_c3"
    assert "IC" in cfg.description
    assert "101 voters" in cfg.description
    assert "3 cand." in cfg.description


def test_result_config_merge() -> None:
    a = ResultConfig.single(gen_model="IC", n_voters=11, n_candidates=3)
    b = ResultConfig.single(gen_model="UNI", n_voters=101, n_candidates=14)
    merged = a.merge(b)
    assert merged.gen_models == frozenset({"IC", "UNI"})
    assert merged.n_voters == frozenset({11, 101})
    assert merged.n_candidates == frozenset({3, 14})
    assert "Models:" in merged.description
    assert "Voters:" in merged.description
    assert "Candidates:" in merged.description


def test_result_config_round_trip() -> None:
    cfg = ResultConfig.single(gen_model="VMF_HC", n_voters=1001, n_candidates=14)
    d = cfg.to_dict()
    restored = ResultConfig.from_dict(d)
    assert restored == cfg


def test_result_config_empty_is_falsy() -> None:
    assert not ResultConfig()
    assert ResultConfig.single(gen_model="IC")


# ======================================================================
# StepResult – metrics tests
# ======================================================================


def test_step_mean_distance() -> None:
    step = SimulationStepResult(data_source="s.parquet")
    step.add_method_result("A", ["X"])
    step.add_method_result("B", ["X"])
    step.add_method_result("C", ["Y"])
    # A-B: 0, A-C: 1, B-C: 1  → mean off-diag = (0+1+1+0+1+1)/(3*2) = 4/6
    assert abs(step.mean_distance - 4 / 6) < 1e-5


def test_step_most_distant_rules() -> None:
    step = SimulationStepResult(data_source="s.parquet")
    step.add_method_result("BORDA", ["A"])
    step.add_method_result("STV", ["A"])
    step.add_method_result("IRV", ["B"])
    r1, r2, d = step.most_distant_rules
    assert {r1, r2} == {"BORDA", "IRV"} or {r1, r2} == {"STV", "IRV"}
    assert abs(d - 1.0) < 1e-5


def test_step_metrics_with_one_rule() -> None:
    step = SimulationStepResult(data_source="s.parquet")
    step.add_method_result("ONLY", ["A"])
    assert step.mean_distance == 0.0
    assert step.most_distant_rules == ("", "", 0.0)


# ======================================================================
# StepResult – save / load / delete with config
# ======================================================================


def test_step_save_load_preserves_config() -> None:
    cfg = ResultConfig.single(gen_model="IC", n_voters=101, n_candidates=3)
    step = SimulationStepResult(data_source="test.parquet", config=cfg)
    step.add_method_result("BORDA", ["A"])
    step.add_method_result("STV", ["B"])

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        step.save_to_file(path)
        loaded = SimulationStepResult(data_source="")
        loaded.load_from_file(path)

        assert loaded.config.gen_models == frozenset({"IC"})
        assert loaded.config.n_voters == frozenset({101})
        assert loaded.config.n_candidates == frozenset({3})
        assert loaded.winners_by_rule == step.winners_by_rule
        assert np.array_equal(loaded.dist_matrix, step.dist_matrix)
    finally:
        os.unlink(path)


def test_step_delete_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    assert os.path.isfile(path)
    assert SimulationStepResult.delete_file(path) is True
    assert not os.path.isfile(path)
    assert SimulationStepResult.delete_file(path) is False


# ======================================================================
# StepResult – __str__ includes config & metrics
# ======================================================================


def test_step_str_includes_config_and_metrics() -> None:
    cfg = ResultConfig.single(gen_model="IC", n_voters=11, n_candidates=3)
    step = SimulationStepResult(data_source="test.parquet", config=cfg)
    step.add_method_result("A", ["X"])
    step.add_method_result("B", ["Y"])
    s = str(step)
    assert "Config:" in s
    assert "Mean distance:" in s
    assert "Most distant:" in s


# ======================================================================
# SeriesResult – metrics tests
# ======================================================================


def test_series_mean_distance() -> None:
    series = SimulationSeriesResult()
    step = SimulationStepResult(data_source="a.parquet")
    step.add_method_result("BORDA", ["A"])
    step.add_method_result("STV", ["B"])
    series.add_step(step)

    # mean_distance_matrix has 100 on off-diagonal → mean_distance = 100
    assert abs(series.mean_distance - 100.0) < 0.1


def test_series_most_distant_rules() -> None:
    series = SimulationSeriesResult()
    step = SimulationStepResult(data_source="a.parquet")
    step.add_method_result("BORDA", ["A"])
    step.add_method_result("STV", ["A"])
    step.add_method_result("IRV", ["B"])
    series.add_step(step)

    r1, r2, d = series.most_distant_rules
    assert d > 0
    assert "IRV" in {r1, r2}


# ======================================================================
# SeriesResult – config aggregation
# ======================================================================


def test_series_config_aggregation() -> None:
    series = SimulationSeriesResult()
    cfg1 = ResultConfig.single(gen_model="IC", n_voters=11, n_candidates=3)
    cfg2 = ResultConfig.single(gen_model="UNI", n_voters=101, n_candidates=14)

    step1 = SimulationStepResult(data_source="a", config=cfg1)
    step1.add_method_result("X", ["A"])
    step2 = SimulationStepResult(data_source="b", config=cfg2)
    step2.add_method_result("X", ["B"])

    series.add_step(step1)
    series.add_step(step2)

    assert series.config.gen_models == frozenset({"IC", "UNI"})
    assert series.config.n_voters == frozenset({11, 101})
    assert series.config.n_candidates == frozenset({3, 14})


# ======================================================================
# SeriesResult – save / load / delete with config
# ======================================================================


def test_series_save_load_preserves_config() -> None:
    cfg = ResultConfig.single(gen_model="VMF_HC", n_voters=1001, n_candidates=3)
    series = SimulationSeriesResult()
    for i in range(3):
        step = SimulationStepResult(data_source=f"iter_{i}.parquet", config=cfg)
        step.add_method_result("BORDA", ["A"])
        step.add_method_result("STV", ["B"] if i % 2 else ["A"])
        series.add_step(step)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    try:
        series.save_to_file(path)
        loaded = SimulationSeriesResult()
        loaded.load_from_file(path)

        assert loaded.step_count == 3
        assert loaded.config.gen_models == frozenset({"VMF_HC"})
        assert loaded.config.n_voters == frozenset({1001})
        assert loaded.config.n_candidates == frozenset({3})
        assert np.allclose(loaded.mean_distance_matrix, series.mean_distance_matrix, atol=0.1)
    finally:
        os.unlink(path)


def test_series_delete_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name

    assert os.path.isfile(path)
    assert SimulationSeriesResult.delete_file(path) is True
    assert not os.path.isfile(path)
    assert SimulationSeriesResult.delete_file(path) is False


# ======================================================================
# SeriesResult – plot titles include config description
# ======================================================================


def test_series_plot_title_includes_config() -> None:
    cfg = ResultConfig.single(gen_model="IC", n_voters=101, n_candidates=3)
    series = SimulationSeriesResult()
    step = SimulationStepResult(data_source="x.parquet", config=cfg)
    step.add_method_result("BORDA", ["A"])
    step.add_method_result("STV", ["B"])
    series.add_step(step)

    ax = series.plot_mean_distance_matrix(show=False)
    title = ax.get_title()
    assert "IC" in title
    assert "101 voters" in title
    assert "3 cand." in title


def test_series_plot_multi_config_title() -> None:
    series = SimulationSeriesResult()
    cfg1 = ResultConfig.single(gen_model="IC", n_voters=11, n_candidates=3)
    cfg2 = ResultConfig.single(gen_model="UNI", n_voters=101, n_candidates=14)

    step1 = SimulationStepResult(data_source="a", config=cfg1)
    step1.add_method_result("X", ["A"])
    step1.add_method_result("Y", ["B"])
    step2 = SimulationStepResult(data_source="b", config=cfg2)
    step2.add_method_result("X", ["A"])
    step2.add_method_result("Y", ["B"])

    series.add_step(step1)
    series.add_step(step2)

    ax = series.plot_mean_distance_matrix(show=False)
    title = ax.get_title()
    assert "Models:" in title
    assert "Voters:" in title
    assert "Candidates:" in title
