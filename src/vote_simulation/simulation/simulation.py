from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.rules import get_rule_builder
from vote_simulation.models.simulation_result import SimulationStepResult
from vote_simulation.simulation.configuration import load_simulation_config


def sim(file_path: str, rule_code: str) -> None:
    """Execute a step of the simulation

    Args:
        file_path (String): The file path of the data
        rule_code (String): The code of the rule to apply
    """

    data_instance = DataInstance(file_path)
    profile = data_instance.profile

    rule_code = rule_code.strip().upper()

    try:
        rule_builder = get_rule_builder(rule_code)
        rule = rule_builder(profile, None)
        if isinstance(rule, NotImplementedError):
            raise rule
        if not hasattr(rule, "w_") and not hasattr(rule, "winner_indices_") and not hasattr(rule, "winner_"):
            raise TypeError(f"Unexpected rule type for '{rule_code}': {type(rule)!r}")

        print(f"{rule_code.upper()} winner: {rule.cowinners_}")
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

    data_instance = DataInstance(config.data_path)
    profile = data_instance.profile

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

            winners = rule.cowinners_
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

    for file_path in tqdm(sorted(data_files)):
        try:
            try:
                data_instance = DataInstance(str(file_path))
                profile = data_instance.profile
            except Exception:
                continue

            step_result = SimulationStepResult(data_source=str(file_path))

            # print("Simulation results:")
            for rule_code in config.rule_codes:
                try:
                    normalized_code = rule_code.strip().upper()
                    rule_builder = get_rule_builder(normalized_code)
                    rule = rule_builder(profile, None)

                    if isinstance(rule, NotImplementedError):
                        raise rule

                    winners = rule.cowinners_
                    step_result.add_method_result(normalized_code, winners)
                except Exception:
                    pass

            output_file = output_dir / f"simulation_{file_path.stem}.parquet"
            step_result.save_to_file(str(output_file))

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("Batch simulation completed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    #simulation_batch("config/simulation.toml")
    simulation("config/simulation.toml")
