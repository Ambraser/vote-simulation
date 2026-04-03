# Usage

The project is currently easiest to use through its Python API.

## Recommended workflow

The typical workflow is:

1. prepare a TOML config file,
2. generate or load election profiles,
3. run one or more voting rules,
4. inspect the produced Parquet files.

The default example config lives in `config/simulation.toml`.

## Minimal import

```python
from vote_simulation.simulation import simulation_full
```

## Run the full pipeline

This is the main entry point when you want to:

- generate synthetic data,
- run all configured rules,
- save results to disk.

```python
from vote_simulation.simulation import simulation_full

simulation_full("config/simulation.toml")
```

With the default configuration, the pipeline:

- reads `config/simulation.toml`,
- creates or reuses generated profiles under `data/gen/`,
- applies every configured rule,
- writes results under `data/sim_result/`.

## Generate data only

If you only want the synthetic election profiles:

```python
from vote_simulation.simulation import generate_data

paths = generate_data("config/simulation.toml")
print(paths[:3])
```

This is useful when you want to inspect or reuse generated datasets before running rules.


Expected config shape:

```toml
[simulation]
rule_codes = ["PLU1", "BORD", "SCHU"]
output_base_path = "../data"
```

The result is written under `data/sim/`.


## Example generative configuration

```toml
[simulation]
output_base_path = "../data/"
generative_models = ["VMF_HC"]
rule_codes = ["PLU1", "BORD", "SCHU"]
candidates = [3, 14]
voters = [11, 101]
iterations = 10
seed = 42

[generator_params.VMF_HC]
vmf_concentration = 10.0
```

## Meaning of the main configuration keys

### Common keys

- `rule_codes`: list of voting rule identifiers to execute.
- `output_base_path`: root directory where outputs are stored.
- `seed`: base seed used for reproducibility.

### Full generative simulation

- `generative_models`: generator codes such as `UNI`, `IC`, or `VMF_HC`.
- `voters`: voter counts to evaluate.
- `candidates`: candidate counts to evaluate.
- `iterations`: number of repetitions for each combination.
- `generator_params`: optional per-generator parameters.

## Output structure

The full pipeline writes files with this structure:

```text
data/
├── gen/
│   └── <MODEL>_v<VOTERS>_c<CANDIDATES>/
│       ├── iter_0001.parquet
│       └── ...
└── sim_result/
	└── <MODEL>_v<VOTERS>_c<CANDIDATES>/
		├── iter_0001.parquet
		└── ...
```

## Notes

- Cached generated profiles are reused automatically if the target Parquet file already exists.
- Rule and generator codes are normalized to uppercase.
- The CLI entrypoint exists in the repository, but the stable documented workflow for now is the Python API.
