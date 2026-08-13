from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.notebook import tqdm

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.results.result_config import ResultConfig
from vote_simulation.models.results.series_result import SimulationSeriesResult
from vote_simulation.models.results.total_result import SimulationTotalResult
from vote_simulation.models.rules.registry import _ensure_profile
from vote_simulation.simulation.simulation import run_rules_on_instance

def _clean_incomplete_data(profile: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with incomplete data (ie incomplete row caracterized by having at least an empty column)

    Args:
        profile (pd.DataFrame): matrix of preferences (n_voters, n_candidates)

    Returns:
        pd.DataFrame: matrix of preferences with incomplete rows removed
    """
    profile = profile.dropna(axis=0, how="any")
    return profile


def save_cleaned_data(profile: pd.DataFrame, folder_path: str = "output.csv") -> None:
    """Save cleaned data to a new CSV file"""
    profile.to_csv(folder_path, index=False)


def normalize_between_0_and_1(arr: np.ndarray) -> np.ndarray:
    """if value > 1 exist, divide by ten (one of two possible cases)"""
    if np.any(arr > 1):
        arr = arr / 10
    return arr


def deterministic_step_seed(
    base_seed: int,
    election_name: str,
    n_voters: int,
    n_candidates: int,
) -> int:
    """Build a stable per-election seed independent from run order."""
    payload = f"{base_seed}|{election_name}|{n_voters}|{n_candidates}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _load_election_csv(csv_path: str) -> DataInstance:
    """Load a voter candidate CSV into a DataInstance.

    Expected format: header row (first column = unnamed row index,
    remaining columns = candidate labels); one voter per subsequent row.
    """
    df = pd.read_csv(csv_path, index_col=0)
    profile = _clean_incomplete_data(df)
    profile = _ensure_profile(profile.to_numpy(dtype=np.float64))
    return DataInstance.from_profile(profile, file_path=csv_path)


def _extract_data(
    data_folder: str,
    candidate_forks: list[int] | None = None,
    display_summary: bool = False,
) -> list[dict]:
    """Scan a folder for election CSVs, apply a candidate-count filter, and print a summary."""
    folder = Path(data_folder)
    all_csv_files = sorted(folder.glob("*.csv"))

    catalogue: list[dict] = []
    for csv_path in all_csv_files:
        df_head = pd.read_csv(csv_path, index_col=0, nrows=0)  # header only
        n_candidates = len(df_head.columns)
        n_voters = sum(1 for _ in open(csv_path)) - 1  # subtract header row
        catalogue.append(
            {
                "path": str(csv_path),
                "name": csv_path.stem,
                "n_voters": n_voters,
                "n_candidates": n_candidates,
            }
        )

    fork_set = set(candidate_forks) if candidate_forks else None
    selected = [e for e in catalogue if fork_set is None or e["n_candidates"] in fork_set]

    counts = Counter(e["n_candidates"] for e in selected)
    if display_summary:
        print(f"Total CSV files found : {len(catalogue)}")
        print(f"After filter          : {len(selected)} file(s)")
        print(f"By candidate count    : {dict(sorted(counts.items()))}")
        print()
        for e in selected:
            print(f"  {e['name']:40s}  voters={e['n_voters']:5d}  candidates={e['n_candidates']} ")
    return selected

def _clean_data(selected: list[dict], display_summary: bool = False) -> pd.DataFrame:
    comparison_rows = []

    for entry in selected:
        csv_path = entry["path"]
        df_raw = pd.read_csv(csv_path, index_col=0)
        df_clean = _clean_incomplete_data(df_raw)

        n_raw = len(df_raw)
        n_clean = len(df_clean)
        n_dropped = n_raw - n_clean
        pct_dropped = 100 * n_dropped / n_raw if n_raw > 0 else 0.0

        try:
            comparison_rows.append(
                {
                    "election": entry["name"],
                    "n_candidates": entry["n_candidates"],
                    "voters_raw": n_raw,
                    "voters_clean": n_clean,
                    "rows_dropped": n_dropped,
                    "pct_dropped (%)": round(pct_dropped, 2),
                    "min_rating_raw": df_raw.to_numpy(dtype=float).min(),
                    "max_rating_raw": df_raw.to_numpy(dtype=float).max(),
                    "min_rating_clean": df_clean.min(axis=None).min() if not df_clean.empty else float("nan"),
                    "max_rating_clean": df_clean.max(axis=None).max() if not df_clean.empty else float("nan"),
                    "has_missing_raw": bool(df_raw.isnull().values.any()),
                    "has_missing_clean": bool(df_clean.isnull().values.any()),
                }
            )
        except Exception as e:
            print(f"Error processing {entry['name']}: {e}")

    comparison_df = pd.DataFrame(comparison_rows).set_index("election")

    if display_summary:
        print("=== Raw vs Cleaned — overview ===")
        display(comparison_df)

    if display_summary:
        print("\n=== Numeric summary (raw voters / cleaned voters / rows dropped) ===")
        display(comparison_df[["voters_raw", "voters_clean", "rows_dropped", "pct_dropped (%)"]].describe())

    n_affected = (comparison_df["rows_dropped"] > 0).sum()
    total_dropped = comparison_df["rows_dropped"].sum()
    if display_summary:
        print(f"\n{n_affected}/{len(comparison_df)} elections had incomplete rows — {total_dropped} rows dropped in total.")

    return comparison_df, selected

def load_data_clean(
    data_folder: str,
    candidate_forks: list[int] | None = None,
    display_summary:bool = False
) -> pd.DataFrame:
    """Load and clean election CSVs from a folder, returning a summary DataFrame."""
    selected = _extract_data(data_folder, candidate_forks, display_summary)
    comparison_df, selected = _clean_data(selected, display_summary)
    return comparison_df, selected


def run_simulations(
    selected: list[dict],
    rule_codes: list[str],
    compute_metrics: bool = True,
    repro_seed: int | None = None,
) -> SimulationTotalResult:
    """Run simulations on a list of election data entries.
    Args:
        selected: A list of dictionaries, each representing an election data entry.
        rule_codes: A list of rule codes to apply in the simulations.
        compute_metrics: Whether to compute metrics for each simulation step.
        repro_seed: An optional seed for reproducibility.

    Returns:
        A SimulationTotalResult object containing the results of all simulations.
    """
    total_result = SimulationTotalResult()

    if repro_seed is not None:
        np.random.seed(repro_seed)

    for entry in tqdm(selected, desc="Elections", leave=True):
        n_c = entry["n_candidates"]
        n_v = entry["n_voters"]
        name = entry["name"]
        #print(f"Name : {name}")
        if n_v == 0:
            print(f"  [SKIP] {entry['name']}: no voters found.")
            continue

        step_config = ResultConfig.single(
            gen_model=name,
            n_voters=n_v,
            n_candidates=n_c,
            rules_codes=rule_codes,
        )

        try:
            di = _load_election_csv(entry["path"])
        except Exception as exc:
            print(f"  [SKIP] {entry['name']}: {exc}")
            continue

        step_seed = None
        rng_state = None
        if repro_seed is not None:
            step_seed = deterministic_step_seed(repro_seed, name, n_v, n_c)
            rng_state = np.random.get_state()
            np.random.seed(step_seed)

        step = run_rules_on_instance(
            di,
            rule_codes,
            config=step_config,
            compute_metrics=compute_metrics,
        )

        if rng_state is not None:
            np.random.set_state(rng_state)

        # one series per (gen_model, n_voters, n_candidates) — merge if already exists
        try:
            existing = total_result.get_series("IRL", n_v, n_c)
            existing.add_step(step)
        except KeyError:
            series = SimulationSeriesResult()
            series.add_step(step)
            series.config = ResultConfig.single(
                gen_model=name,
                n_voters=n_v,
                n_candidates=n_c,
                n_iterations=1,
                rules_codes=rule_codes,
            )
            total_result.add_series(series)

    print(f"\nSimulation complete — {total_result.series_count} series")
    return total_result