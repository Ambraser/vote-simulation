import os

from vote_simulation.models.rules.registry import _RULE_BUILDERS
from whalrus import * 
import numpy as np
from vote_simulation.models.rules import get_rule_builder

def get_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the data from the file path.

    Args:
        file_path (str): The file path of the CSV data.

    Returns:
        candidates (np.ndarray): 1-D array of candidate names.
        data (np.ndarray): 2-D array of shape (n_voters, n_candidates).
    """
    if not os.path.isfile(file_path):
        raise ValueError("Invalid file path. Please provide a valid file path.")

    if not file_path.endswith(".csv"):
        raise ValueError("Unsupported file type. Supported file type is : .csv")

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            header = fh.readline().strip()

        column_count = len(header.split(","))
        if column_count < 2:
            raise ValueError("CSV file must contain at least one data column.")

        # Candidates
        candidates = np.genfromtxt(
            file_path,
            delimiter=",",
            skip_header=1,
            usecols=0,
            dtype=str,
        )
        candidates = np.char.strip(candidates, '"')  # remove surrounding quotes if any

        # Data
        data = np.genfromtxt(
            file_path,
            delimiter=",",
            skip_header=1,
            usecols=tuple(range(1, column_count)),
            dtype=np.float64,
        ).T  # transpose: rows = voters, columns = candidates
        if data.ndim == 1:
            data = data.reshape(1, -1)

    except Exception as e:
        raise ValueError(f"Error reading the file : {e}")

    return candidates, data


def sim(file_path:str, rule_code:str) : 
    """Execute a step of the simulation 

    Args:
        file_path (String): The file path of the data
        rule_code (String): The code of the rule to apply
    """
    
    candidates, data = get_data(file_path)
    candidate_names = candidates.tolist()

    ballots = [
        BallotLevels(
            dict(zip(candidate_names, voter_scores.tolist())),
            candidates=set(candidate_names),
            scale=ScaleRange(low=0, high=1),
        )
        for voter_scores in data
    ]

    for rule_code in _RULE_BUILDERS:
        try : 
            rule_builder = get_rule_builder(rule_code)
            rule = rule_builder(ballots, set(candidate_names))
    
            print(f"{rule_code.upper()} winner: {rule.winner_}")
        except Exception as e:
            print(f"Error building rule '{rule_code}': {e}")

if __name__ == "__main__" :
    sim("/home/ambraser/Desktop/Stage/Code/R/data_simulation_2025/data_simu_DIV/simu_DIV_v_9_c_3_i_1.csv", "PLU")

    


    
    


