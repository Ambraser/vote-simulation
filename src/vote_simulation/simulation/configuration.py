"""Configuration loading for vote simulations."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SimulationConfig:
    """Validated simulation configuration."""

    rule_codes: list[str]
    data_path: str | None = None  # optional, if not provided, data will be generated
    candidates: list[int] | None = None
    voters: list[int] | None = None
    iterations: int = 1
    seed: int = 0
    input_folder_path: str | None = None  # optional, for batch simulation on multiple files



DEFAULT_CONFIG_PATH = Path("config/simulation.toml")


def load_simulation_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> SimulationConfig:
    """Load and validate a simulation config file."""
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

    rule_codes = simulation.get("rule_codes")
    if not isinstance(rule_codes, list) or not rule_codes:
        raise ValueError("Invalid configuration: simulation.rule_codes must be a non-empty list")

    normalized_rule_codes = [str(code).strip().upper() for code in rule_codes if str(code).strip()]
    if not normalized_rule_codes:
        raise ValueError("Invalid configuration: simulation.rule_codes cannot be empty")

    candidates = simulation.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Invalid configuration: simulation.candidates must be a non-empty list")
        if not all(isinstance(c, int) and c > 0 for c in candidates):
            raise ValueError("Invalid configuration: all simulation.candidates must be positive integers")

    voters = simulation.get("voters")
    if voters is not None:
        if not isinstance(voters, list) or not voters:
            raise ValueError("Invalid configuration: simulation.voters must be a non-empty list")
        if not all(isinstance(v, int) and v > 0 for v in voters):
            raise ValueError("Invalid configuration: all simulation.voters must be positive integers")

    iterations = simulation.get("iterations", 1)
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("Invalid configuration: simulation.iterations must be a positive integer")

    seed = simulation.get("seed", 0)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid configuration: simulation.seed must be a non-negative integer")

    data_path = simulation.get("data_path", simulation.get("data_file"))
    if data_path is not None and (not isinstance(data_path, str) or not data_path.strip()):
        raise ValueError("Invalid configuration: simulation.data_path must be a non-empty string if provided")
    if isinstance(data_path, str):
        data_path = (
            str((path.parent / data_path).resolve())
            if not Path(data_path).is_absolute()
            else str(Path(data_path).resolve())
        )

    input_folder_path = simulation.get("input_folder_path")
    if input_folder_path is not None and (not isinstance(input_folder_path, str) or not input_folder_path.strip()):
        raise ValueError("Invalid configuration: simulation.input_folder_path must be a non-empty string if provided")
    if isinstance(input_folder_path, str):
        input_folder_path = (
            str((path.parent / input_folder_path).resolve())
            if not Path(input_folder_path).is_absolute()
            else str(Path(input_folder_path).resolve())
        )

    return SimulationConfig(
        data_path=data_path,
        rule_codes=normalized_rule_codes,
        candidates=candidates,
        voters=voters,
        iterations=iterations,
        seed=seed,
        input_folder_path=input_folder_path,
    )
