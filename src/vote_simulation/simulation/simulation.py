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
from vote_simulation.models.simulation_result import SimulationStepResult
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


# ---------------------------------------------------- data obtain-or-generate


def obtain_data_instance(
    model: str,
    n_v: int,
    n_c: int,
    *,
    iteration: int,
    seed: int,
    base_path: str,
    extra_params: dict[str, object] | None = None,
) -> DataInstance:
    """Load a cached profile or generate + persist it.

    If the parquet file already exists the profile is loaded from disk;
    otherwise it is generated and saved for future reuse.
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


# apply rules to a DI


def run_rules_on_instance(
    data_instance: DataInstance,
    rule_codes: list[str],
) -> SimulationStepResult:
    """Apply every rule and collect winners into a ``SimulationStepResult``."""
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


def simulation(config_path: str) -> SimulationStepResult:
    """Run the vote simulation on a *single* data file (legacy behaviour)."""
    config = load_simulation_config(config_path)

    if config.data_path is None:
        raise ValueError("Configuration must include data_path for simulation")

    data_instance = DataInstance(config.data_path)
    step_result = run_rules_on_instance(data_instance, config.rule_codes)

    output_dir = Path(config.output_base_path) / "sim"
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import UTC, datetime

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"simulation_{timestamp}.parquet"
    step_result.save_to_file(str(output_file))
    print(f"Saved simulation step to: {output_file}")
    return step_result


#  generate


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
                        # Obtain data
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


def simulation_full(config_path: str) -> None:
    """Full pipeline: generate profiles, apply rules, save results.

    Alias for :func:`simulation_from_config`.
    """
    return simulation_from_config(config_path)


def simulation_batch(config_path: str) -> None:
    """Run vote simulations on all files in a folder specified in the configuration."""
    config = load_simulation_config(config_path)

    if not config.input_folder_path:
        raise ValueError(
            "Configuration does not contain 'input_folder_path' parameter. Please add it to run batch simulations."
        )

    input_folder = Path(config.input_folder_path)
    if not input_folder.is_dir():
        raise ValueError(f"Input folder not found: {input_folder}")

    data_files = list(input_folder.glob("*.csv")) + list(input_folder.glob("*.parquet"))
    if not data_files:
        print(f"No CSV or Parquet files found in {input_folder}")
        return

    print(f"Found {len(data_files)} data files to process in {input_folder}")

    output_dir = Path(config.output_base_path) / "sim"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(sorted(data_files)):
        try:
            data_instance = DataInstance(str(file_path))
        except Exception:
            continue

        step_result = run_rules_on_instance(data_instance, config.rule_codes)
        output_file = output_dir / f"simulation_{file_path.stem}.parquet"
        step_result.save_to_file(str(output_file))

    print(f"\n{'=' * 60}")
    print("Batch simulation completed")
    print(f"{'=' * 60}")


# ----------------------------------------------------------- validation


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
    # sim("data/gen/UNI_v101_c3/iter_0001.parquet", "PLU1")
#
# data = DataInstance("data/gen/VMF_HC_v101_c14/iter_0001.parquet")
# print("Candidates:", data.candidates)
# print("Data shape:", data.data.shape)

# print(_RULE_BUILDERS.keys())
# run_rules_on_instance(data, rule_codes=_RULE_BUILDERS.keys())
