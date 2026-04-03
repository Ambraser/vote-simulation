"""Core simulation engine.

Workflow
--------
1. Read the TOML configuration.
2. For each generative model × (n_voters, n_candidates) × iteration:
   a. Check if the generated profile already exists on disk → load it.
   b. If not, generate it via the generator registry and persist it.
3. Apply every rule to each profile and collect winners.
4. Persist the simulation results to ``sim_result/``.

The directory layout follows::

    <output_base>/
      gen/<MODEL>_v<NV>_c<NC>/
        iter_0001.parquet
        …
      sim_result/<MODEL>_v<NV>_c<NC>/
        iter_0001.parquet
        …
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.rules import RuleResult, get_rule_builder
from vote_simulation.models.simulation_result import SimulationSeriesResult, SimulationStepResult
from vote_simulation.simulation.configuration import SimulationConfig, load_simulation_config

# utils


def _gen_dir(base: str, model: str, n_v: int, n_c: int) -> Path:
    """Return the directory for generated data: ``<base>/gen/<MODEL>_v<NV>_c<NC>``."""
    return Path(base) / "gen" / f"{model}_v{n_v}_c{n_c}"


def _sim_dir(base: str, model: str, n_v: int, n_c: int) -> Path:
    """Return the directory for results: ``<base>/sim_result/<MODEL>_v<NV>_c<NC>``."""
    return Path(base) / "sim_result" / f"{model}_v{n_v}_c{n_c}"


def _iter_filename(iteration: int) -> str:
    """Return the filename for a given iteration index (1-based display, 0-based index)."""
    return f"iter_{iteration + 1:04d}.parquet"


# data obtain-or-generate


def obtain_data_instance(
    model: str,
    n_v: int,
    n_c: int,
    *,
    iteration: int = 0,
    seed: int = 161,
    base_path: str = "data",
    extra_params: dict[str, object] | None = None,
) -> DataInstance:
    """Load a cached profile or generate + persist it.

    If the parquet file already exists the profile is loaded from disk;
    otherwise it is generated and saved for future reuse.

    Args:
        model: Generative model code (e.g. "UNI", "IC").
        n_v: Number of voters.
        n_c: Number of candidates.
        iteration: Iteration index.
        seed: Random seed for generation (will be combined with iteration index for variability).
        base_path: Root folder for generated data (see config.output_base_path).
        extra_params: Optional dict of extra parameters to pass to the generator (per-model).
    """
    gen_path = _gen_dir(base_path, model, n_v, n_c) / _iter_filename(iteration)

    if gen_path.is_file():
        return DataInstance(str(gen_path))

    # Generate
    di = DataInstance.from_generator(
        model_code=model,
        n_v=n_v,
        n_c=n_c,
        seed=seed,
        iteration=iteration,
        **(extra_params or {}),
    )
    di.save_parquet(str(gen_path))
    di.file_path = str(gen_path)
    return di


def run_rules_on_instance(
    data_instance: DataInstance,
    rule_codes: list[str],
) -> SimulationStepResult:
    """
    Apply every rule and collect winners into a ``SimulationStepResult``.

    Args:
        data_instance: The profile data to run the rules on.
        rule_codes: List of rule codes to apply (e.g. ["RV", "MJ", "AP_T"]).
    """
    profile = data_instance.profile
    step = SimulationStepResult(data_source=data_instance.file_path)

    for code in rule_codes:
        normalized = code.strip().upper()
        try:
            builder = get_rule_builder(normalized)
            rule: RuleResult = builder(profile, None)
            winners = rule.cowinners_
            step.add_method_result(normalized, winners)
            # print(f"Applied rule '{normalized}': winners = {winners}")
        except Exception as e:  # noqa: BLE001
            print(f"Error applying rule '{normalized}': {e}")
            step.add_method_result(normalized, [f"ERROR: {e}"])
    return step


# ===================================================================
# Public entry-points
# ===================================================================


def sim(file_path: str, rule_code: str) -> None:
    """Execute a single rule on a single file

    ²"""
    data_instance = DataInstance(file_path)
    profile = data_instance.profile
    rule_code = rule_code.strip().upper()

    try:
        rule_builder = get_rule_builder(rule_code)
        rule: RuleResult = rule_builder(profile, None)
        if not hasattr(rule, "w_") and not hasattr(rule, "winner_indices_") and not hasattr(rule, "winner_"):
            raise TypeError(f"Unexpected rule type for '{rule_code}': {type(rule)!r}")
        print(f"{rule_code.upper()} winner: {rule.cowinners_}")
    except Exception as e:
        print(f"Error building rule '{rule_code}': {e}")


#  generate data
# --------------------------------------------------------------------------


def generate_data(config_path: str) -> list[str]:
    """Generate (or retrieve cached) profiles for every combination defined in the config.

    Returns:
        List of file paths of generated/cached parquet files.
    """
    config = load_simulation_config(config_path)
    _validate_generation_config(config)

    paths: list[str] = []
    total = len(config.generative_models) * len(config.voters or []) * len(config.candidates or []) * config.iterations
    with tqdm(total=total, desc="Generating profiles") as pbar:
        for model in config.generative_models:
            extra = config.generator_params.get(model, {})
            for n_v in config.voters or []:
                for n_c in config.candidates or []:
                    for it in range(config.iterations):
                        di = obtain_data_instance(
                            model=model,
                            n_v=n_v,
                            n_c=n_c,
                            iteration=it,
                            seed=config.seed,
                            base_path=config.output_base_path,
                            extra_params=extra,
                        )
                        paths.append(di.file_path)
                        pbar.update(1)
    print(f"Generated / loaded {len(paths)} profiles.")
    return paths


# full pipeline


def simulation_from_config(config_path: str) -> None:
    """Full pipeline: generate profiles, apply rules, save results.

    For every ``(model, n_voters, n_candidates, iteration)`` combination:
    1. Obtain (generate or load) the profile.
    2. Run all requested rules.
    3. Save the result in ``sim_result/<MODEL>_v<NV>_c<NC>/iter_XXXX.parquet``.

    Args:
        config_path: Path to the TOML configuration file (see docs for the template).
    """
    config = load_simulation_config(config_path)
    _validate_generation_config(config)

    total = len(config.generative_models) * len(config.voters or []) * len(config.candidates or []) * config.iterations
    print(f"Running full simulation: {total} profile(s) × {len(config.rule_codes)} rule(s)")

    with tqdm(total=total, desc="Simulating") as pbar:
        for model in config.generative_models:
            extra = config.generator_params.get(model, {})
            for n_v in config.voters or []:
                for n_c in config.candidates or []:
                    for it in range(config.iterations):
                        # 1) Obtain data
                        di = obtain_data_instance(
                            model=model,
                            n_v=n_v,
                            n_c=n_c,
                            iteration=it,
                            seed=config.seed,
                            base_path=config.output_base_path,
                            extra_params=extra,
                        )

                        # 2) Apply rules
                        step = run_rules_on_instance(di, config.rule_codes)

                        # 3) Save result
                        result_path = _sim_dir(config.output_base_path, model, n_v, n_c) / _iter_filename(it)
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        step.save_to_file(str(result_path))

                        pbar.update(1)

    print("Full simulation completed.")

def simulation_instance(
        gen_code:str, 
        n_v:int,
        n_c:int, 
        rule_codes:list[str],
        n_iteration:int = 1000, 
        seed:int = 161,
    ) -> SimulationSeriesResult:
    """Run the workflow on a single instance

    Args:
        gen_code (str): Generative model code (list can be found in doc)
        n_v (int): number of voters 
        n_c (int): number of candidates
        rule_codes (list[str]): list of rule_codes to apply
        n_iteration (int, optional): number of iterations. Defaults to 1000.
        seed (int, optional): seed for reproductability. Defaults to 161.

    Returns:
        SimulationSeriesResult: Result of the sim instance
    """
    print("Running simulation")
    #total = n_iteration
    series = SimulationSeriesResult()
    with tqdm(total=n_iteration, desc="Simulating") as pbar:
        for it in range(n_iteration):
            di = obtain_data_instance(
                model=gen_code,
                n_v=n_v,
                n_c=n_c,
                iteration=it,
                seed=seed,
                base_path="data",
            )
            step = run_rules_on_instance(di, rule_codes)
            series.add_step(step)
            pbar.update(1)
    print("Simulation completed.")
    return series



def simulation_full(config_path: str) -> None:
    """Full pipeline: generate profiles, apply rules, save results.

    Alias for :func:`simulation_from_config`.
    """
    return simulation_from_config(config_path)




def simulation_from_file(file_path: str, rule_codes: list[str]) -> SimulationStepResult:
    """Run simulation on a single file with specified rules."""
    data_instance = DataInstance(file_path)
    step_result = run_rules_on_instance(data_instance, rule_codes)
    return step_result


def simulation_series(folder_path: str, rule_codes: list[str]) -> SimulationSeriesResult:
    """Run simulations on all files in a folder and return a :class:`SimulationSeriesResult`.

    Each file is processed as a :class:`SimulationStepResult` and accumulated
    into the series via :meth:`SimulationSeriesResult.add_step`, which keeps the
    running sum matrix up to date incrementally.

    Args:
    folder_path: Path to the folder containing input CSV or Parquet files.
    rule_codes: List of rule codes to apply to each file (e.g. ["RV", "MJ", "AP_T"]).

    Returns:
    A :class:`SimulationSeriesResult` containing all the step results and the aggregated mean distance matrix.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Input folder not found: {folder}")

    data_files = list(folder.glob("*.csv")) + list(folder.glob("*.parquet"))
    if not data_files:
        print(f"No CSV or Parquet files found in {folder}")
        return SimulationSeriesResult()

    print(f"Found {len(data_files)} data files to process in {folder}")

    series = SimulationSeriesResult()
    for file_path in tqdm(sorted(data_files)):
        try:
            step_result = simulation_from_file(str(file_path), rule_codes)
            series.add_step(step_result)
        except Exception as e:  # noqa: BLE001
            print(f"Error processing file '{file_path}': {e}")

    print(f"\n{'=' * 60}")
    print(f"Series simulation completed — {series.step_count} iteration(s)")
    print(f"{'=' * 60}")
    return series


# validation


def _validate_generation_config(config: SimulationConfig) -> None:
    """Ensure the config has all fields needed for generative simulation."""
    if not config.generative_models:
        raise ValueError("Configuration must include at least one generative_models entry.")
    if not config.voters:
        raise ValueError("Configuration must include a 'voters' list for generative simulation.")
    if not config.candidates:
        raise ValueError("Configuration must include a 'candidates' list for generative simulation.")


if __name__ == "__main__":
    simulation_from_config("config/simulation.toml")
