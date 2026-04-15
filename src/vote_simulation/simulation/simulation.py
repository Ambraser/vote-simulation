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

from svvamp import Profile
from tqdm import tqdm

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.results.series_result import ResultConfig, SimulationSeriesResult, SimulationStepResult
from vote_simulation.models.results.total_result import SimulationTotalResult
from vote_simulation.models.rules import RuleResult, get_rule_builder
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
    config: ResultConfig | None = None,
) -> SimulationStepResult:
    """
    Apply every rule and collect winners into a ``SimulationStepResult``.

    Args:
        data_instance: The profile data to run the rules on.
        rule_codes: List of rule codes to apply (e.g. ["RV", "MJ", "AP_T"]).
        config: Optional :class:`ResultConfig` attached to the step.
    """
    profile = data_instance.profile
    step = SimulationStepResult(
        data_source=data_instance.file_path,
        config=config or ResultConfig(),
    )

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


def simulation_step(
    profile: Profile, rule_codes: list[str], config: ResultConfig | None = None
) -> SimulationStepResult:
    """Run a single profile through all rules and return a :class:`SimulationStepResult`.

    Args:
        profile: The profile data to run the rules on.
        candidates: List of candidate names.
        rule_codes: List of rule codes to apply (e.g. ["RV", "MJ", "AP_T"]).
        config: Optional :class:`ResultConfig` attached to the step.
    """
    step_config = config or ResultConfig()

    data = DataInstance.from_profile(profile)

    step_result = run_rules_on_instance(data, rule_codes, config=step_config)

    return step_result


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
                    step_cfg = ResultConfig.single(
                        gen_model=model,
                        n_voters=n_v,
                        n_candidates=n_c,
                    )
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
                        step = run_rules_on_instance(di, config.rule_codes, config=step_cfg)

                        # 3) Save result
                        result_path = _sim_dir(config.output_base_path, model, n_v, n_c) / _iter_filename(it)
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        step.save_to_file(str(result_path))

                        pbar.update(1)

    print("Full simulation completed.")


def simulation_instance(
    gen_code: str,
    n_v: int,
    n_c: int,
    rule_codes: list[str],
    n_iteration: int = 1000,
    seed: int = 161,
    base_path: str = "data",
    reload: bool = False,
) -> SimulationSeriesResult:
    """Run the workflow on a single (model, voters, candidates) instance.

    Each step receives a :class:`ResultConfig` so that the series
    automatically aggregates the simulation context.

    Before running, checks if a cached result already exists at
    ``<base_path>/results/<config_label>.parquet``.  When found (and the
    step count matches *n_iteration*), the cached series is returned
    directly, skipping re-computation.

    Args:
        gen_code: Generative model code (list can be found in doc).
        n_v: Number of voters.
        n_c: Number of candidates.
        rule_codes: List of rule codes to apply.
        n_iteration: Number of iterations. Defaults to 1000.
        seed: Seed for reproducibility. Defaults to 161.
        base_path: Root folder for generated data. Defaults to ``"data"``.

    Returns:
        SimulationSeriesResult with attached :attr:`config`.
    """
    step_config = ResultConfig.single(
        gen_model=gen_code.strip().upper(),
        n_voters=n_v,
        n_candidates=n_c,
        n_iterations=n_iteration,
    )

    # --- Cache check ---
    cache_path = Path(base_path) / "results" / f"{step_config.label}.parquet"
    if not reload:
        cache_path = Path(base_path) / "results" / f"{step_config.label}.parquet"
        if cache_path.is_file():
            cached = SimulationSeriesResult()
            cached.load_from_file(str(cache_path))
            if cached.step_count == n_iteration:
                print(f"Cache hit: loaded {cached.step_count} steps from {cache_path}")
                return cached
            print(f"Cache stale ({cached.step_count} steps vs {n_iteration} requested) — re-running.")

    print(f"Running simulation: {step_config.description} × {n_iteration} iterations")
    series = SimulationSeriesResult()
    with tqdm(total=n_iteration, desc="Simulating") as pbar:
        for it in range(n_iteration):
            di = obtain_data_instance(
                model=gen_code,
                n_v=n_v,
                n_c=n_c,
                iteration=it,
                seed=seed,
                base_path=base_path,
            )
            step = run_rules_on_instance(di, rule_codes, config=step_config)
            series.add_step(step)
            pbar.update(1)

    # --- Persist for future cache hits ---
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.save_to_file(str(cache_path))
    print(f"Simulation completed — cached to {cache_path}")
    return series


def simulation_series_from_config(config_path: str, reload: bool = False) -> SimulationTotalResult:
    """Run simulation instances for every combination in the config.

    Iterates over each ``(model, n_voters, n_candidates)`` triplet defined
    in the TOML configuration, delegates to :func:`simulation_instance`,
    and collects all resulting series into a :class:`SimulationTotalResult`.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        A :class:`SimulationTotalResult` containing one series per
        ``(model, voters, candidates)`` combination.
    """
    config = load_simulation_config(config_path)
    _validate_generation_config(config)

    total_result = SimulationTotalResult()
    n_combos = len(config.generative_models) * len(config.voters or []) * len(config.candidates or [])
    with tqdm(total=n_combos, desc="Running simulation series") as pbar:
        for model in config.generative_models:
            for n_v in config.voters or []:
                for n_c in config.candidates or []:
                    series = simulation_instance(
                        gen_code=model,
                        n_v=n_v,
                        n_c=n_c,
                        rule_codes=config.rule_codes,
                        n_iteration=config.iterations,
                        seed=config.seed,
                        base_path=config.output_base_path,
                        reload=reload,
                    )
                    total_result.add_series(series)
                    pbar.update(1)

    print(f"Completed {total_result.series_count} simulation series.")
    return total_result


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
