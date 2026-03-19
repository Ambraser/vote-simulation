import os
from csv import reader
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from svvamp import Profile

from vote_simulation.models.rules import get_rule_builder
from vote_simulation.models.simulation_result import SimulationStepResult
from vote_simulation.simulation.configuration import load_simulation_config


def get_csv(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the data from the file path.

    Args:
        file_path (str): The file path of the CSV data.
    """
    try:
        candidates_list: list[str] = []
        rows: list[list[float]] = []

        with open(file_path, encoding="utf-8", newline="") as fh:
            csv_reader = reader(fh)
            next(csv_reader, None)

            for row in csv_reader:
                if len(row) < 2:
                    raise ValueError("CSV file must contain at least one data column.")
                candidates_list.append(row[0].strip('"'))
                rows.append([float(value) for value in row[1:]])

        if not rows:
            raise ValueError("CSV file must contain at least one row.")

        candidates = np.asarray(candidates_list, dtype=str)
        data = np.asarray(rows, dtype=np.float64).T  # rows = voters, columns = candidates

    except Exception as e:
        raise ValueError(f"Error reading the file : {e}") from e

    return candidates, data


def get_parquet(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the data from the file path.

    Args:
        file_path (str): The file path of the Parquet data.
    """
    # TODO : implement parquet file support
    raise NotImplementedError("Parquet file support is not implemented yet.")


def get_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the data from the file path.

    Args:
        file_path (str): The file path of the CSV or Parquet data.

    Returns:
        candidates (np.ndarray): 1-D array of candidate names.
        data (np.ndarray): 2-D array of shape (n_voters, n_candidates).
    """
    if not os.path.isfile(file_path):
        raise ValueError("Invalid file path. Please provide a valid file path.")

    if file_path.endswith(".csv"):
        return get_csv(file_path)

    if file_path.endswith(".parquet"):
        return get_parquet(file_path)

    raise ValueError("Unable to load data from provided file path.")


def build_profile(candidates: np.ndarray, data: np.ndarray) -> Profile:
    """Build an `svvamp.Profile` from candidate labels and utility matrix."""
    return Profile(preferences_ut=data, labels_candidates=candidates.tolist())


def extract_winners(rule: Any, profile: Profile) -> list[str]:
    """Extract winner labels from a `svvamp` rule instance."""
    labels: list[Any] = profile.labels_candidates

    if hasattr(rule, "winner_indices_"):
        return [str(labels[int(index)]) for index in rule.winner_indices_]

    if hasattr(rule, "w_"):
        winner = rule.w_
        if isinstance(winner, (float, np.floating)) and np.isnan(winner):
            raise ValueError("Rule did not determine a winner.")
        return [str(labels[int(winner)])]

    if hasattr(rule, "winner_"):
        return [str(rule.winner_)]

    raise TypeError(f"Unexpected rule type: {type(rule)!r}")


def sim(file_path: str, rule_code: str) -> None:
    """Execute a step of the simulation

    Args:
        file_path (String): The file path of the data
        rule_code (String): The code of the rule to apply
    """

    candidates, data = get_data(file_path)
    profile = build_profile(candidates, data)

    rule_code = rule_code.strip().upper()

    try:
        rule_builder = get_rule_builder(rule_code)
        rule = rule_builder(profile, None)
        if isinstance(rule, NotImplementedError):
            raise rule
        if not hasattr(rule, "w_") and not hasattr(rule, "winner_indices_") and not hasattr(rule, "winner_"):
            raise TypeError(f"Unexpected rule type for '{rule_code}': {type(rule)!r}")

        print(f"{rule_code.upper()} winner: {extract_winners(rule, profile)}")
    except Exception as e:
        print(f"Error building rule '{rule_code}': {e}")


def simulation(config_path: str) -> SimulationStepResult:
    """Run the vote simulation based on the provided configuration.

    Args:
        config_path (str): The file path of the simulation configuration.
    """
    config = load_simulation_config(config_path)

    if config.data_path is None:
        raise ValueError("Configuration must include data_path for simulation")

    candidates, data = get_data(config.data_path)
    profile = build_profile(candidates, data)

    step_result = SimulationStepResult(data_source=config.data_path)

    print("Simulation results:")
    for rule_code in config.rule_codes:
        try:
            normalized_code = rule_code.strip().upper()
            rule_builder = get_rule_builder(normalized_code)
            rule = rule_builder(profile, None)

            if isinstance(rule, NotImplementedError):
                raise rule
            if not hasattr(rule, "w_") and not hasattr(rule, "winner_indices_") and not hasattr(rule, "winner_"):
                raise TypeError(f"Unexpected rule type for '{normalized_code}': {type(rule)!r}")

            winners = extract_winners(rule, profile)
            step_result.add_method_result(normalized_code, winners)
            # print(f"{normalized_code} winner: {winners}")
        except Exception as e:
            print(f"Error building rule '{rule_code}': {e}")

    output_dir = Path("data/sim")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"simulation_{timestamp}.parquet"

    step_result.save_to_file(str(output_file))
    print(f"Saved simulation step to: {output_file}")

    return step_result


def simulation_batch(config_path: str):
    """Run vote simulations on all files in a folder specified in the configuration.

    Args:
        config_path (str): The file path of the simulation configuration.
    """
    config = load_simulation_config(config_path)

    if not config.input_folder_path:
        raise ValueError(
            "Configuration does not contain 'input_folder_path' parameter. Please add it to run batch simulations."
        )

    input_folder = Path(config.input_folder_path)
    if not input_folder.is_dir():
        raise ValueError(f"Input folder not found: {input_folder}")

    # Find all CSV and Parquet files in the folder
    data_files = list(input_folder.glob("*.csv")) + list(input_folder.glob("*.parquet"))

    if not data_files:
        print(f"No CSV or Parquet files found in {input_folder}")
        return

    print(f"Found {len(data_files)} data files to process in {input_folder}")

    output_dir = Path("data/sim")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(data_files):
        print(f"\n{'=' * 60}")
        print(f"Processing: {file_path.name}")
        print(f"{'=' * 60}")

        try:
            candidates, data = get_data(str(file_path))
            profile = build_profile(candidates, data)

            step_result = SimulationStepResult(data_source=str(file_path))

            print("Simulation results:")
            for rule_code in config.rule_codes:
                try:
                    normalized_code = rule_code.strip().upper()
                    rule_builder = get_rule_builder(normalized_code)
                    rule = rule_builder(profile, None)

                    if isinstance(rule, NotImplementedError):
                        raise rule
                    if (
                        not hasattr(rule, "w_")
                        and not hasattr(rule, "winner_indices_")
                        and not hasattr(rule, "winner_")
                    ):
                        raise TypeError(f"Unexpected rule type for '{normalized_code}': {type(rule)!r}")

                    winners = extract_winners(rule, profile)
                    step_result.add_method_result(normalized_code, winners)
                    print(f"  {normalized_code}: {winners}")
                except Exception as e:
                    print(f"  Error building rule '{rule_code}': {e}")

            output_file = output_dir / f"simulation_{file_path.stem}.parquet"
            step_result.save_to_file(str(output_file))
            print(f"Saved results to: {output_file}")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("Batch simulation completed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    simulation_batch("config/simulation.toml")
