"""Data models for simulation outputs across multiple iterations."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class SimulationStepResult:
    """Result of a simulation"""

    data_source: str  # File path of the source
    winners_by_rule: dict[str, list[str]] = field(
        default_factory=dict
    )  # Mapping of rule code to list of winners for the simulation

    def add_method_result(self, rule_code: str, winners: list[str]) -> None:
        """Add or update winners for one voting method in this step."""

        normalized_code = rule_code.strip().upper()
        self.winners_by_rule[normalized_code] = winners

    def save_to_file(self, file_path: str) -> None:
        """Save the step result to a parquet file."""
        # Convert the winners_by_rule dictionary to a DataFrame
        df = pd.DataFrame(
            [(rule, winner) for rule, winners in self.winners_by_rule.items() for winner in winners],
            columns=["Rule", "Winner"],
        )

        # Save the DataFrame to a parquet file
        df.to_parquet(file_path, index=False)

    def load_from_file(self, file_path: str) -> None:
        """Load the step result from a parquet file."""
        df = pd.read_parquet(file_path)
        self.winners_by_rule = df.groupby("Rule")["Winner"].apply(list).to_dict()

    def __str__(self) -> str:
        """String representation of the step result."""
        winners_str = ", ".join(f"{rule}: {', '.join(winners)}" for rule, winners in self.winners_by_rule.items())
        return f"Data Source: {self.data_source}, Winners: {winners_str}"


@dataclass(slots=True)
class SimulationSeriesResult:
    """Aggregation of simulation steps."""

    steps: list[SimulationStepResult] = field(default_factory=list)

    def add_step(self, step_result: SimulationStepResult) -> None:
        """Add one step result to the series."""

        self.steps.append(step_result)

    @property
    def step_count(self) -> int:
        """Number of recorded steps."""

        return len(self.steps)

    def save_to_file(self, file_path: str) -> None:
        """Save the series result to a parquet file."""
        # Convert the series of steps into a DataFrame
        df = pd.DataFrame(
            [
                {"DataSource": step.data_source, "Rule": rule, "Winner": winner}
                for step in self.steps
                for rule, winners in step.winners_by_rule.items()
                for winner in winners
            ]
        )

        # Save the DataFrame to a parquet file
        df.to_parquet(file_path, index=False)

    def load_from_file(self, file_path: str) -> None:
        """Load the series result from a parquet file."""
        df = pd.read_parquet(file_path)
        self.steps = []
        for data_source, group in df.groupby("DataSource"):
            data_source = str(data_source)
            step_result = SimulationStepResult(data_source=data_source)
            for rule, winners in group.groupby("Rule")["Winner"]:
                rule = str(rule)
                step_result.winners_by_rule[rule] = winners.tolist()
            self.steps.append(step_result)
