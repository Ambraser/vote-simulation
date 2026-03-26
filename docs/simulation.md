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
└── sim_result/
	└── <MODEL>_v<NV>_c<NC>/
		├── iter_0001.parquet
		└── ...
```

Legacy single-file and batch modes write their outputs under `sim/`.

## Main public entry points

- `generate_data()` for generation only
- `simulation_full()` for the full pipeline
- `simulation()` for one file
- `simulation_batch()` for one folder
- `obtain_data_instance()` for the cache-or-generate step

## Configuration reference

::: vote_simulation.simulation.configuration

## Simulation engine reference

::: vote_simulation.simulation.simulation