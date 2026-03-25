"""DataInstance: load or generate election profiles and persist them."""

from __future__ import annotations

import os
from csv import reader
from pathlib import Path

import numpy as np
import pandas as pd
from svvamp import Profile


class DataInstance:
    """Encapsulates an election profile (utility matrix + candidate labels).

    `DataInstance`` can be created in three ways:

     **From an existing file** (CSV or Parquet)::

           di = DataInstance("path/to/data.csv")

     **From a generator** (wraps svvamp ``GeneratorProfile*``)::

           di = DataInstance.from_generator(
               model_code="UNI", n_v=101, n_c=5, seed=42, iteration=0
           )

     **From a raw Profile**::

           di = DataInstance.from_profile(profile)
    """

    # ------------------------------------------------------------------ init

    def __init__(self, file_path: str):
        try:
            self.candidates, self.data = self.get_data(file_path)
            self.profile = self.build_profile(self.candidates, self.data)
            self.file_path = file_path
        except Exception as e:
            raise ValueError(f"Error initializing DataInstance: {e}") from e

    # --------------------------------------------------------- class methods

    @classmethod
    def from_generator(
        cls,
        model_code: str,
        n_v: int,
        n_c: int,
        *,
        seed: int = 0,
        iteration: int = 0,
        **extra_params: object,
    ) -> DataInstance:
        """Generate an election profile using a registered generator.

        Args:
            model_code: Registered generator short code (e.g. ``"UNI"``).
            n_v: Number of voters.
            n_c: Number of candidates.
            seed: Base random seed for reproducibility.
            iteration: Iteration index (added to *seed*).
            **extra_params: Model-specific keyword arguments forwarded to
                the generator builder.

        Returns:
            A new ``DataInstance`` whose profile was generated in-memory.
        """
        from vote_simulation.models.data_generation.generator_registry import (
            get_generator_builder,
        )

        builder = get_generator_builder(model_code)
        profile: Profile = builder(n_v, n_c, seed=seed, iteration=iteration, **extra_params)

        instance = object.__new__(cls)
        instance.candidates = np.asarray(profile.labels_candidates, dtype=str)
        instance.data = np.asarray(profile.preferences_ut, dtype=np.float64)
        instance.profile = profile
        instance.file_path = ""  # not loaded from disk
        return instance

    @classmethod
    def from_profile(cls, profile: Profile, file_path: str = "") -> DataInstance:
        """Wrap an existing ``svvamp.Profile`` into a ``DataInstance``."""
        instance = object.__new__(cls)
        instance.candidates = np.asarray(profile.labels_candidates, dtype=str)
        instance.data = np.asarray(profile.preferences_ut, dtype=np.float64)
        instance.profile = profile
        instance.file_path = file_path
        return instance

    # -------------------------------------------------------------- loaders

    def get_csv(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load candidate labels and utility matrix from a CSV file.

        Args:
            file_path: Path to the CSV file.
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

    def get_parquet(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load candidate labels and utility matrix from a Parquet file.

        The Parquet file is expected to have one column per candidate
        (column name = candidate label) and one row per voter.

        Args:
            file_path: Path to the Parquet file.
        """
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                raise ValueError("Parquet file is empty.")
            candidates = np.asarray(df.columns.tolist(), dtype=str)
            data = df.to_numpy(dtype=np.float64)  # (n_voters, n_candidates)
        except Exception as e:
            raise ValueError(f"Error reading parquet file: {e}") from e
        return candidates, data

    def get_data(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load data from a CSV or Parquet file.

        Args:
            file_path: Path to the data file.

        Returns:
            candidates: 1-D array of candidate names.
            data: 2-D array of shape ``(n_voters, n_candidates)``.
        """
        if not os.path.isfile(file_path):
            raise ValueError("Invalid file path. Please provide a valid file path.")

        if file_path.endswith(".csv"):
            return self.get_csv(file_path)

        if file_path.endswith(".parquet"):
            return self.get_parquet(file_path)

        raise ValueError("Unable to load data from provided file path.")

    # profile builder

    def build_profile(self, candidates: np.ndarray, data: np.ndarray) -> Profile:
        """Build a ``svvamp.Profile`` from candidate labels and utility matrix."""
        return Profile(preferences_ut=data, labels_candidates=candidates.tolist())

    def save_parquet(self, file_path: str) -> str:
        """Persist the utility matrix to a Parquet file.

        Creates parent directories if needed. The file contains one column
        per candidate and one row per voter.

        Args:
            file_path: Destination path (should end in ``.parquet``).

        Returns:
            The resolved absolute path of the written file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data, columns=self.candidates.tolist())
        df.to_parquet(str(path), index=False)
        return str(path.resolve())

    def save_csv(self, file_path: str) -> str:
        """Persist the utility matrix to a CSV file (same layout as input).

        Args:
            file_path: Destination path (should end in ``.csv``).

        Returns:
            The resolved absolute path of the written file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data, columns=self.candidates.tolist())
        df.to_csv(str(path), index=False)
        return str(path.resolve())

    # --------------------------------------------------------- properties

    @property
    def n_voters(self) -> int:
        """Number of voters in this instance."""
        return int(self.data.shape[0])

    @property
    def n_candidates(self) -> int:
        """Number of candidates in this instance."""
        return int(self.data.shape[1])
