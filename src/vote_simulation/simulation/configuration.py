"""Configuration loading for vote simulations."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SimulationConfig:
    """Validated simulation configuration."""

    rule_codes: list[str]
    candidates: list[int] | None = None
    voters: list[int] | None = None
    iterations: int = 1
    seed: int = 0
    generative_models: list[str] = field(default_factory=list)  # e.g. ["UNI", "IC"]
    output_base_path: str = "data"  # root folder for gen/ and sim_result/
    input_folder_path: str | None = None  # folder with pre-existing vote files for batch mode
    generator_params: dict[str, dict[str, object]] = field(default_factory=dict)  # per-model extra params


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

    # Check validity of rules codes
    rule_codes = simulation.get("rule_codes")
    if not isinstance(rule_codes, list) or not rule_codes:
        raise ValueError("Invalid configuration: simulation.rule_codes must be a non-empty list")

    normalized_rule_codes = [str(code).strip().upper() for code in rule_codes if str(code).strip()]
    if not normalized_rule_codes:
        raise ValueError("Invalid configuration: simulation.rule_codes cannot be empty")

    # Check validity of candidates, voters, iterations
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

    # Check validity of seed
    seed = simulation.get("seed", 0)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid configuration: simulation.seed must be a non-negative integer")

    # --- Generative models ---
    raw_gen_models = simulation.get("generative_models")
    generative_models: list[str] = []
    if raw_gen_models is not None:
        if not isinstance(raw_gen_models, list):
            raise ValueError("Invalid configuration: simulation.generative_models must be a list")
        generative_models = [str(m).strip().upper() for m in raw_gen_models if str(m).strip()]

    # --- Output base path ---
    # Relative paths are resolved from the current working directory (the project root),
    # consistent with how the UI and write_temp_toml resolve them.
    output_base_path = simulation.get("output_base_path", "data")
    if not isinstance(output_base_path, str) or not output_base_path.strip():
        output_base_path = "data"
    if not Path(output_base_path).is_absolute():
        output_base_path = str(Path(output_base_path).resolve())

    # --- Per-model generator params (optional TOML sub-tables) ---
    generator_params: dict[str, dict[str, object]] = {}
    gen_params_section = payload.get("generator_params")
    if isinstance(gen_params_section, dict):
        for model_key, params in gen_params_section.items():
            if isinstance(params, dict):
                generator_params[model_key.strip().upper()] = dict(params)

    # --- Input folder path (optional, for batch mode) ---
    raw_input_folder = simulation.get("input_folder_path")
    input_folder_path: str | None = None
    if raw_input_folder is not None:
        input_folder_path = str(raw_input_folder).strip() or None
        if input_folder_path and not Path(input_folder_path).is_absolute():
            input_folder_path = str(Path(input_folder_path).resolve())

    return SimulationConfig(
        rule_codes=normalized_rule_codes,
        candidates=candidates,
        voters=voters,
        iterations=iterations,
        seed=seed,
        generative_models=generative_models,
        output_base_path=output_base_path,
        input_folder_path=input_folder_path,
        generator_params=generator_params,
    )
