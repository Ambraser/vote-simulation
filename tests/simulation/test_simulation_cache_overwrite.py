"""Regression tests for series cache overwrite/invalidation behavior."""

from __future__ import annotations

from pathlib import Path

from vote_simulation.models.results.series_result import SimulationSeriesResult
from vote_simulation.simulation import simulation as sim


def test_simulation_instance_overwrites_cache_when_rules_removed(tmp_path: Path) -> None:
    """A reduced rule list must overwrite previous cached series content."""
    base = str(tmp_path)

    sim.simulation_instance(
        gen_code="UNI",
        n_v=9,
        n_c=3,
        rule_codes=["PLU1", "BORD"],
        n_iteration=2,
        seed=42,
        base_path=base,
        reload=False,
        show_progress=False,
    )

    reduced = sim.simulation_instance(
        gen_code="UNI",
        n_v=9,
        n_c=3,
        rule_codes=["PLU1"],
        n_iteration=2,
        seed=42,
        base_path=base,
        reload=False,
        show_progress=False,
    )

    assert set(reduced.config.rules_codes) == {"PLU1"}

    cache_path = tmp_path / "results" / "UNI_v9_c3_i2.parquet"
    loaded = SimulationSeriesResult()
    loaded.load_from_file(str(cache_path))
    assert set(loaded.config.rules_codes) == {"PLU1"}


def test_simulation_instance_recomputes_when_seed_changes(tmp_path: Path, monkeypatch) -> None:
    """Changing seed should invalidate cache and rerun obtain_data_instance."""
    base = str(tmp_path)
    call_count = 0
    original_obtain = sim.obtain_data_instance

    def _wrapped_obtain(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_obtain(*args, **kwargs)

    monkeypatch.setattr(sim, "obtain_data_instance", _wrapped_obtain)

    sim.simulation_instance(
        gen_code="UNI",
        n_v=9,
        n_c=3,
        rule_codes=["PLU1"],
        n_iteration=3,
        seed=1,
        base_path=base,
        reload=False,
        show_progress=False,
    )

    call_count = 0
    sim.simulation_instance(
        gen_code="UNI",
        n_v=9,
        n_c=3,
        rule_codes=["PLU1"],
        n_iteration=3,
        seed=2,
        base_path=base,
        reload=False,
        show_progress=False,
    )

    assert call_count == 3
