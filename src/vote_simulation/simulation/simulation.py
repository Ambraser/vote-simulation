import os
from csv import reader
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from whalrus import BallotLevels, Rule, ScaleRange

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
    raise NotImplementedError("Parquet file support is not implemented yet.")


def get_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the data from the file path.

    Args:
        file_path (str): The file path of the CSV data.

    Returns:
        candidates (np.ndarray): 1-D array of candidate names.
        data (np.ndarray): 2-D array of shape (n_voters, n_candidates).
    """
    if file_path.endswith(".csv"):
        try:
            return get_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}") from e

    if file_path.endswith(".parquet"):
        return get_parquet(file_path)

    if not os.path.isfile(file_path):
        raise ValueError("Invalid file path. Please provide a valid file path.")

    if not file_path.endswith(".csv"):
        raise ValueError("Unsupported file type. Supported file type is : .csv")

    raise ValueError("Unable to load data from provided file path.")


def sim(file_path: str, rule_code: str):
    """Execute a step of the simulation

    Args:
        file_path (String): The file path of the data
        rule_code (String): The code of the rule to apply
    """

    candidates, data = get_data(file_path)
    candidate_names = candidates.tolist()

    ballots = [
        BallotLevels(
            dict(zip(candidate_names, voter_scores.tolist(), strict=False)),
            candidates=set(candidate_names),
            scale=ScaleRange(low=0, high=1),
        )
        for voter_scores in data
    ]

    rule_code = rule_code.strip().upper()

    try:
        rule_builder = get_rule_builder(rule_code)
        rule = rule_builder(ballots, set(candidate_names))
        if isinstance(rule, NotImplementedError):
            raise rule
        if not isinstance(rule, Rule):
            raise TypeError(f"Unexpected rule type for '{rule_code}': {type(rule)!r}")

        print(f"{rule_code.upper()} winner: {rule.cowinners_}")
    except Exception as e:
        print(f"Error building rule '{rule_code}': {e}")


def simulation(config_path: str):
    """Run the vote simulation based on the provided configuration.

    Args:
        config_path (str): The file path of the simulation configuration.
    """
    config = load_simulation_config(config_path)

    candidates, data = get_data(config.data_file)
    candidate_names = candidates.tolist()
    ballots = [
        BallotLevels(
            dict(zip(candidate_names, voter_scores.tolist(), strict=False)),
            candidates=set(candidate_names),
            scale=ScaleRange(low=0, high=1),
        )
        for voter_scores in data
    ]

    step_result = SimulationStepResult(data_source=config.data_file)

    print("Simulation results:")
    for rule_code in config.rule_codes:
        try:
            normalized_code = rule_code.strip().upper()
            rule_builder = get_rule_builder(normalized_code)
            rule = rule_builder(ballots, set(candidate_names))

            if isinstance(rule, NotImplementedError):
                raise rule
            if not isinstance(rule, Rule):
                raise TypeError(f"Unexpected rule type for '{normalized_code}': {type(rule)!r}")

            winners = [str(candidate) for candidate in rule.cowinners_]
            step_result.add_method_result(normalized_code, winners)
            print(f"{normalized_code} winner: {winners}")
        except Exception as e:
            print(f"Error building rule '{rule_code}': {e}")

    output_dir = Path("data/sim")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"simulation_{timestamp}.parquet"

    step_result.save_to_file(str(output_file))
    print(f"Saved simulation step to: {output_file}")

    reloaded_step = SimulationStepResult(data_source=step_result.data_source)
    reloaded_step.load_from_file(str(output_file))

    print("Reloaded simulation results:")
    for rule_code, winners in reloaded_step.winners_by_rule.items():
        print(f"{rule_code} winner: {winners}")

    return reloaded_step


if __name__ == "__main__":
    simulation("config/simulation.toml")
