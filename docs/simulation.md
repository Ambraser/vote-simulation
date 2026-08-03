# Simulation workflow

This page explains how the simulation pipeline is structured and where to look depending on what you want to change.

## End-to-end flow

The full workflow is:

1. load a TOML configuration,
2. validate the requested simulation mode,
3. generate or reuse election profiles,
4. apply every requested voting rule,
5. save one result file per iteration.

## Directory layout

```text
<output_base>/
├── gen/
│   └── <MODEL>_v<NV>_c<NC>/
│       ├── iter_0001.parquet
│       └── ...
├── sim_result/
│   └── <MODEL>_v<NV>_c<NC>/
│       ├── iter_0001.parquet
│       └── ...
└── results/
    └── <MODEL>_v<NV>_c<NC>.parquet   ← series cache
```

## Main public entry points

| Function | Returns | Description |
|---|---|---|
| `simulation_from_config(config_path)` | `None` | Full pipeline: generate → apply rules → save per-iteration Parquet files under `sim_result/` |
| `simulation_series_from_config(config_path)` | `SimulationTotalResult` | Same as above but returns results in memory; caches series under `results/` |
| `simulation_instance(gen_code, n_v, n_c, rule_codes, ...)` | `SimulationSeriesResult` | Run one `(model, voters, candidates)` combination; supports incremental rule addition from cache |
| `generate_data(config_path)` | `list[str]` | Generate (or reuse cached) profiles only, no rules applied |
| `obtain_data_instance(model, n_v, n_c, ...)` | `DataInstance` | Load a single cached profile or generate and persist it |
| `run_rules_on_instance(data_instance, rule_codes)` | `SimulationStepResult` | Apply a list of rules to a single `DataInstance` |
| `simulation_step(profile, rule_codes)` | `SimulationStepResult` | Apply a list of rules to an svvamp `Profile` directly |

## Configuration reference

::: vote_simulation.simulation.configuration

## Simulation engine reference

::: vote_simulation.simulation.simulation