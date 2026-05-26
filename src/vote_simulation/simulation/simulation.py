"""Core simulation engine.

Workflow
--------
1. Read the TOML configuration.
2. For each generative model x n_voters x n_candidates x iteration:
   a. Check if the generated profile already exists on disk : load it.
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
from typing import Any

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
    reload: bool = False,
) -> DataInstance:
    """Load a cached profile or generate + persist it.

    If the parquet file already exists AND *reload* is False, the profile
    is loaded from disk; otherwise it is (re-)generated with the given seed
    and saved for future reuse.

    Args:
        model: Generative model code (e.g. "UNI", "IC").
        n_v: Number of voters.
        n_c: Number of candidates.
        iteration: Iteration index.
        seed: Random seed for generation (combined with iteration for uniqueness).
        base_path: Root folder for generated data (see config.output_base_path).
        extra_params: Optional dict of extra parameters to pass to the generator (per-model).
        reload: When True, ignore any existing cached file and regenerate from
            the seed.  Use this whenever the seed changes to ensure the new
            seed is actually applied.
    """
    gen_path = _gen_dir(base_path, model, n_v, n_c) / _iter_filename(iteration)

    # Single stat() call instead of is_file() + stat() (2 syscalls → 1).
    if not reload:
        try:
            if gen_path.stat().st_size > 0:
                return DataInstance(str(gen_path))
        except FileNotFoundError:
            pass

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
    *,
    compute_metrics: bool = True,
) -> SimulationStepResult:
    """
    Apply every rule and collect winners into a ``SimulationStepResult``.

    Args:
        data_instance: The profile data to run the rules on.
        rule_codes: List of rule codes to apply (e.g. ["RV", "MJ", "AP_T"]).
        config: Optional :class:`ResultConfig` attached to the step.
        compute_metrics: Whether to compute :class:`~vote_simulation.models.rules.WinnerMetrics`
            for each rule.  Set to ``False`` to skip metric computation and
            reduce runtime when quality metrics are not needed.
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
            if compute_metrics:
                try:
                    metrics = rule.compute_metrics()
                    step.add_method_result_with_metrics(normalized, winners, metrics)
                except Exception:
                    # Rule wrappers outside SvvampRuleWrapper don't carry metrics — degrade gracefully.
                    step.add_method_result(normalized, winners)
            else:
                step.add_method_result(normalized, winners)
        except Exception as e:  # noqa: BLE001
            print(f"Error applying rule '{normalized}': {e}")
            step.add_method_result(normalized, [f"ERROR: {e}"])
    return step


# Public entry-points


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


def generate_data(config_path: str, show_progress: bool = True) -> list[str]:
    """Generate (or retrieve cached) profiles for every combination defined in the config.

    Returns:
        List of file paths of generated/cached parquet files.
    """
    config = load_simulation_config(config_path)
    _validate_generation_config(config)

    paths: list[str] = []
    total = len(config.generative_models) * len(config.voters or []) * len(config.candidates or []) * config.iterations
    with tqdm(total=total, desc="Generating profiles", disable=not show_progress) as pbar:
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
    profile: Profile,
    rule_codes: list[str],
    config: ResultConfig | None = None,
    *,
    compute_metrics: bool = True,
) -> SimulationStepResult:
    """Run a single profile through all rules and return a :class:`SimulationStepResult`.

    Args:
        profile: The profile data to run the rules on.
        candidates: List of candidate names.
        rule_codes: List of rule codes to apply (e.g. ["RV", "MJ", "AP_T"]).
        config: Optional :class:`ResultConfig` attached to the step.
        compute_metrics: Whether to compute :class:`~vote_simulation.models.rules.WinnerMetrics`
            for each rule.  Defaults to ``True``.
    """
    step_config = config or ResultConfig()

    data = DataInstance.from_profile(profile)

    step_result = run_rules_on_instance(data, rule_codes, config=step_config, compute_metrics=compute_metrics)

    return step_result


def simulation_from_config(
    config_path: str,
    show_progress: bool = True,
    *,
    reload: bool = False,
    compute_metrics: bool = True,
) -> None:
    """Full pipeline: generate profiles, apply rules, save results.

    For every ``(model, n_voters, n_candidates, iteration)`` combination:
    1. Obtain (generate or load) the profile.
    2. Run all requested rules.
    3. Save the result in ``sim_result/<MODEL>_v<NV>_c<NC>/iter_XXXX.parquet``.

    Args:
        config_path: Path to the TOML configuration file (see docs for the template).
        show_progress: Whether to display a progress bar.
        reload: When ``True``, ignore any cached result files and recompute
            every iteration from scratch.  When ``False`` (default), iterations
            whose result file already exists are skipped.
        compute_metrics: Whether to compute :class:`~vote_simulation.models.rules.WinnerMetrics`
            for each rule.  Defaults to ``True``.
    """
    config = load_simulation_config(config_path)
    _validate_generation_config(config)

    # Pre-fetch rule builders once before any loop.
    # config.rule_codes are already normalised by load_simulation_config, so no
    # .strip().upper() needed here.  Avoids len(rules) × total_iterations
    # redundant dict lookups + string operations in the hot path.
    # Unknown codes are warned once and skipped (graceful degradation).
    builders: list[tuple[str, Any]] = []
    for code in config.rule_codes:
        try:
            builders.append((code, get_rule_builder(code)))
        except ValueError:
            print(f"Warning: unknown rule code '{code}' — skipped.")

    total = len(config.generative_models) * len(config.voters or []) * len(config.candidates or []) * config.iterations
    print(f"Running full simulation: {total} profile(s) × {len(builders)} rule(s)")

    with tqdm(total=total, desc="Simulating", disable=not show_progress) as pbar:
        for model in config.generative_models:
            extra = config.generator_params.get(model, {})
            for n_v in config.voters or []:
                for n_c in config.candidates or []:
                    # Create output directory ONCE per (model, n_v, n_c) combo,
                    # not 1 000 times inside the iteration loop.
                    sim_dir = _sim_dir(config.output_base_path, model, n_v, n_c)
                    sim_dir.mkdir(parents=True, exist_ok=True)

                    step_cfg = ResultConfig.single(
                        gen_model=model,
                        n_voters=n_v,
                        n_candidates=n_c,
                        rules_codes=config.rule_codes,
                    )
                    for it in range(config.iterations):
                        result_path = sim_dir / _iter_filename(it)

                        # Single stat() call instead of is_file() + stat() (2 syscalls → 1).
                        if not reload:
                            try:
                                if result_path.stat().st_size > 0:
                                    pbar.update(1)
                                    continue
                            except FileNotFoundError:
                                pass

                        # 1) Obtain (generate or load) profile
                        di = obtain_data_instance(
                            model=model,
                            n_v=n_v,
                            n_c=n_c,
                            iteration=it,
                            seed=config.seed,
                            base_path=config.output_base_path,
                            extra_params=extra,
                            reload=reload,
                        )

                        # 2) Apply rules using pre-fetched builders (inline hot path)
                        profile = di.profile
                        step = SimulationStepResult(
                            data_source=di.file_path,
                            config=step_cfg,
                        )
                        for code, builder in builders:
                            try:
                                rule = builder(profile, None)
                                winners = rule.cowinners_
                                if compute_metrics:
                                    try:
                                        metrics = rule.compute_metrics()
                                        step.add_method_result_with_metrics(code, winners, metrics)
                                    except Exception:
                                        step.add_method_result(code, winners)
                                else:
                                    step.add_method_result(code, winners)
                            except Exception as e:  # noqa: BLE001
                                print(f"Error applying rule '{code}': {e}")
                                step.add_method_result(code, [f"ERROR: {e}"])

                        # 3) Save result
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
    show_progress: bool = True,
    *,
    extra_params: dict[str, object] | None = None,
    compute_metrics: bool = True,
) -> SimulationSeriesResult:
    """Run the workflow on a single (model, voters, candidates) instance.

    Each step receives a :class:`ResultConfig` so that the series
    automatically aggregates the simulation context.

    Cache logic:
    1. Checks for a cached result at ``<base_path>/results/<base_label>.parquet``
       (where base_label excludes rules).
    2. If found with matching step count and same base parameters:
       - If rules are identical: returns cached series (no recomputation).
       - If rules differ: loads cached series and applies new rules incrementally.
    3. If not found or stale: recomputes from scratch.

    Args:
        gen_code: Generative model code (list can be found in doc).
        n_v: Number of voters.
        n_c: Number of candidates.
        rule_codes: List of rule codes to apply.
        n_iteration: Number of iterations. Defaults to 1000.
        seed: Seed for reproducibility. Defaults to 161.
        base_path: Root folder for generated data. Defaults to ``"data"``.
        reload: Force re-computation (ignore cache). Defaults to False.
        show_progress: Whether to display progress bars. Defaults to True.
        extra_params: Optional dict of extra generator parameters (e.g.
            ``{"vmf_concentration": 100.0}``). Passed to
            :func:`obtain_data_instance` on cache miss. Defaults to ``None``.
        compute_metrics: Whether to compute :class:`~vote_simulation.models.rules.WinnerMetrics`
            for each rule.  Defaults to ``True``.  Set to ``False`` to skip
            metric computation when only winner distances are needed.
    Returns:
        SimulationSeriesResult with attached :attr:`config` including all rules.
    """
    # Build the base config (cache key, excludes rules)
    gen_code = gen_code.strip().upper()
    base_config = ResultConfig.single(
        gen_model=gen_code,
        n_voters=n_v,
        n_candidates=n_c,
        n_iterations=n_iteration,
    )

    # Pre-build rule builders once before any loop (mirrors simulation_from_config).
    # Unknown codes are warned once and skipped; avoids per-iteration exceptions,
    # error-message formatting (sorted string of 68+ codes), and queue.put() calls.
    valid_builders: list[tuple[str, Any]] = []
    for code in [r.strip().upper() for r in rule_codes]:
        try:
            valid_builders.append((code, get_rule_builder(code)))
        except ValueError:
            print(f"Warning: unknown rule code '{code}' — skipped.")
    valid_rule_codes = [code for code, _ in valid_builders]

    # full_config uses only valid rules so the cache key is accurate.
    full_config = ResultConfig.single(
        gen_model=gen_code,
        n_voters=n_v,
        n_candidates=n_c,
        n_iterations=n_iteration,
        rules_codes=valid_rule_codes,
    )

    # --- Cache check with partial-load support ---
    cache_path = Path(base_path) / "results" / f"{base_config.label}.parquet"

    if not reload and cache_path.is_file():
        cached = SimulationSeriesResult()
        cached.load_from_file(str(cache_path))

        #        if cached.step_count != n_iteration:
        # print(f"Cache stale ({cached.step_count} steps vs {n_iteration} requested) — re-running.")
        # elif not cached.config.matches_base(base_config):
        # print("Cache config mismatch — re-running.")
        #        else:
        # Cache is valid for base parameters
        cached_rules = set(cached.config.rules_codes)
        requested_rules = set(valid_rule_codes)

        if cached_rules == requested_rules:
            # Perfect match! Return cached series
            # print(f"Cache hit: loaded {cached.step_count} steps from {cache_path}")
            return cached
        elif cached_rules < requested_rules:
            # Partial match: cached has subset of requested rules
            new_rules = sorted(requested_rules - cached_rules)
            # print(
            #    f"Partial cache hit: {cached.step_count} steps with rules "
            #    f"{sorted(cached_rules)}. Adding {new_rules}..."
            # )
            cached.add_rules_to_steps(new_rules)
            # Update config to match requested rules
            cached.config = ResultConfig.single(
                gen_model=gen_code,
                n_voters=n_v,
                n_candidates=n_c,
                n_iterations=n_iteration,
                rules_codes=valid_rule_codes,
            )
            # Save updated series with new rules
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.save_to_file(str(cache_path))
            # print(f"Updated cache saved to {cache_path}")
            return cached
        # else:
        # Cached has rules we don't want (or extra rules not matching scenarios)
        # print(
        #     f"Cache rule mismatch: cached has {sorted(cached_rules)}, "
        #     f"but requested {sorted(requested_rules)} — re-running."
        # )

    # --- No valid cache: compute from scratch ---
    # print(
    #    f"Running simulation: {base_config.description} × {n_iteration} iterations with {len(valid_rule_codes)} rules"
    # )
    series = SimulationSeriesResult()
    with tqdm(total=n_iteration, desc="Simulating", disable=not show_progress) as pbar:
        for it in range(n_iteration):
            di = obtain_data_instance(
                model=gen_code,
                n_v=n_v,
                n_c=n_c,
                iteration=it,
                seed=seed,
                base_path=base_path,
                extra_params=extra_params or {},
                reload=reload,
            )
            # Inline hot loop using pre-built builders (mirrors simulation_from_config).
            # Avoids per-iteration get_rule_builder() calls and exception overhead.
            profile = di.profile
            step = SimulationStepResult(data_source=di.file_path, config=base_config)
            for code, builder in valid_builders:
                try:
                    rule: RuleResult = builder(profile, None)
                    winners = rule.cowinners_
                    if compute_metrics:
                        try:
                            metrics = rule.compute_metrics()
                            step.add_method_result_with_metrics(code, winners, metrics)
                        except Exception:
                            step.add_method_result(code, winners)
                    else:
                        step.add_method_result(code, winners)
                except Exception as e:  # noqa: BLE001
                    print(f"Error applying rule '{code}': {e}")
                    step.add_method_result(code, [f"ERROR: {e}"])
            series.add_step(step)
            pbar.update(1)

    # Set the full config on the series (including rules)
    series.config = full_config

    # --- Persist for future cache hits ---
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.save_to_file(str(cache_path))
    # print(f"Simulation completed — cached to {cache_path}")
    return series


def simulation_series_from_config_2(
    config_path: str,
    reload: bool = False,
    *,
    compute_metrics: bool = True,
) -> SimulationTotalResult:
    """Run simulation instances for every combination in the config.

    Iterates over each ``(model, n_voters, n_candidates)`` triplet defined
    in the TOML configuration, delegates to :func:`simulation_instance`,
    and collects all resulting series into a :class:`SimulationTotalResult`.

    Args:
        config_path: Path to the TOML configuration file.
        reload: Force re-computation (ignore cache). Defaults to False.
        compute_metrics: Whether to compute :class:`~vote_simulation.models.rules.WinnerMetrics`
            for each rule.  Defaults to ``True``.

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
                        show_progress=False,  # inner progress is handled by simulation_instance
                        compute_metrics=compute_metrics,
                    )
                    total_result.add_series(series)
                    pbar.update(1)

    print(f"Completed {total_result.series_count} simulation series.")
    return total_result


def simulation_series_from_config(
    config_path: str,
    reload: bool = False,
    *,
    compute_metrics: bool = True,
) -> SimulationTotalResult:
    """Run simulation instances for every combination in the config.

    Iterates over each ``(model, n_voters, n_candidates)`` triplet defined
    in the TOML configuration, delegates to :func:`simulation_instance`,
    and collects all resulting series into a :class:`SimulationTotalResult`.

    Séquentiel — le travail est CPU-bound Python (svvamp) ; le GIL empêche
    toute parallélisation réelle via ThreadPoolExecutor.
    En mode UI (Streamlit), le tqdm externe est intercepté par ``_PatchedTqdm``
    pour mettre à jour la barre de progression sans toucher à la hot loop.

    Args:
        config_path: Path to the TOML configuration file.
        reload: Force re-computation (ignore cache). Defaults to False.
        compute_metrics: Whether to compute :class:`~vote_simulation.models.rules.WinnerMetrics`
            for each rule.  Defaults to ``True``.

    Returns:
        A :class:`SimulationTotalResult` containing one series per
        ``(model, voters, candidates)`` combination.
    """
    config = load_simulation_config(config_path)
    _validate_generation_config(config)

    total_result = SimulationTotalResult()
    combos = [
        (model, config.generator_params.get(model, {}), n_v, n_c)
        for model in config.generative_models
        for n_v in (config.voters or [])
        for n_c in (config.candidates or [])
    ]
    n_combos = len(combos)

    with tqdm(total=n_combos, desc="Running simulation series") as pbar:
        for model, extra, n_v, n_c in combos:
            total_result.add_series(
                simulation_instance(
                    gen_code=model,
                    n_v=n_v,
                    n_c=n_c,
                    rule_codes=config.rule_codes,
                    n_iteration=config.iterations,
                    seed=config.seed,
                    base_path=config.output_base_path,
                    reload=reload,
                    show_progress=False,
                    extra_params=extra,
                    compute_metrics=compute_metrics,
                )
            )
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
