from pathlib import Path

import pytest

from vote_simulation.simulation.configuration import load_simulation_config


def test_load_simulation_config_resolves_relative_data_path(tmp_path: Path):
    """Relative data_file is resolved from config file directory."""

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    data_file = tmp_path / "dataset.csv"
    data_file.write_text("candidate,voter1\nAlice,1\n", encoding="utf-8")

    config_file = config_dir / "simulation.toml"
    config_file.write_text(
        """
        [simulation]
        data_file = "../dataset.csv"
        rule_codes = ["plu1", " bord "]
        """.strip(),
        encoding="utf-8",
    )

    config = load_simulation_config(config_file)

    assert config.data_path == str(data_file.resolve())
    assert config.rule_codes == ["PLU1", "BORD"]


def test_load_simulation_config_missing_simulation_section(tmp_path: Path):
    """Missing [simulation] section raises an explicit error."""

    config_file = tmp_path / "simulation.toml"
    config_file.write_text('title = "oops"', encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing \[simulation\] section"):
        load_simulation_config(config_file)
