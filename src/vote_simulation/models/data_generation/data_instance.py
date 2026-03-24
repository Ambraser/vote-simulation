import os
from csv import reader

import numpy as np
from svvamp import Profile


class DataInstance:
    """
    Class representing a data instance for vote simulation.
    Encapsulates methods to load, build profiles, and manage election data.

    """

    def __init__(self, file_path: str):
        try:
            self.candidates, self.data = self.get_data(file_path)
            self.profile = self.build_profile(self.candidates, self.data)
            self.file_path = file_path
        except Exception as e:
            raise ValueError(f"Error initializing DataInstance: {e}") from e

    def get_csv(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
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

    def get_parquet(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the data from the file path.

        Args:
            file_path (str): The file path of the Parquet data.
        """
        # TODO : implement parquet file support
        raise NotImplementedError("Parquet file support is not implemented yet.")

    def get_data(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
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
            return self.get_csv(file_path)

        if file_path.endswith(".parquet"):
            return self.get_parquet(file_path)

        raise ValueError("Unable to load data from provided file path.")

    def build_profile(self, candidates: np.ndarray, data: np.ndarray) -> Profile:
        """Build an `svvamp.Profile` from candidate labels and utility matrix."""
        return Profile(preferences_ut=data, labels_candidates=candidates.tolist())
