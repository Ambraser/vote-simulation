"""Configuration loading for vote simulations."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SimulationConfig:
    """Validated simulation configuration."""

    data_file: str
    rule_codes: list[str]
    # TODO : add every parameters


DEFAULT_CONFIG_PATH = Path("config/simulation.toml")


def load_simulation_config(config_path: str | Path) -> SimulationConfig:
    """Load and validate a simulation config file.

    Expected structure:

    [simulation]
    data_file = "..."
    rule_codes = ["PLU1", "BORD"]
    """

    path = Path(config_path)
    # Check path
    if not path.is_file():
        raise ValueError(f"Configuration file not found: {path}")

    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    simulation = payload.get("simulation")
    # simulation section must exist and be a dict
    if not isinstance(simulation, dict):
        raise ValueError("Invalid configuration: missing [simulation] section")

    data_file = simulation.get("data_file")
    if not isinstance(data_file, str) or not data_file.strip():
        raise ValueError("Invalid configuration: simulation.data_file must be a non-empty string")

    rule_codes = simulation.get("rule_codes")
    if not isinstance(rule_codes, list) or not rule_codes:
        raise ValueError("Invalid configuration: simulation.rule_codes must be a non-empty list")

    normalized_rule_codes = [str(code).strip().upper() for code in rule_codes if str(code).strip()]
    if not normalized_rule_codes:
        raise ValueError("Invalid configuration: simulation.rule_codes cannot be empty")

    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = (path.parent / data_path).resolve()

    return SimulationConfig(data_file=str(data_path), rule_codes=normalized_rule_codes)
