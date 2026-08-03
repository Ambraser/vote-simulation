# Usage

The project is currently easiest to use through its Python API.

## Recommended workflow

The typical workflow is:

1. prepare a TOML config file,
2. generate or load election profiles,
3. run one or more voting rules,
4. inspect the produced Parquet files.

The default example config lives in `config/simulation.toml`.

## Main entry points

### Run the full pipeline

Generates profiles, applies all configured rules, and saves one result file per iteration.

```python
from vote_simulation.simulation.simulation import simulation_from_config

simulation_from_config("config/simulation.toml")
```

- Creates or reuses generated profiles under `data/gen/`.
- Applies every configured rule.
- Writes per-iteration results under `data/sim_result/`.

### Run and get a `SimulationTotalResult` back

Use `simulation_series_from_config` when you want to keep the results in memory for analysis and plotting:

```python
from vote_simulation.simulation.simulation import simulation_series_from_config

total = simulation_series_from_config("config/simulation.toml")
print(total)
```

The series results are cached under `data/results/` and reused on subsequent calls.

### Generate data only

Data generation happens automatically inside `simulation_from_config`, but you can also call it on its own:

```python
from vote_simulation.simulation.simulation import generate_data

paths = generate_data("config/simulation.toml")
print(paths[:3])
```


## Example configuration

```toml
[simulation]
output_base_path = "data"
generative_models = ["VMF_HC"]
rule_codes = ["PLU1", "BORD", "SCHU"]
candidates = [3, 14]
voters = [11, 101]
iterations = 10
seed = 42

# Optional per-model parameters
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

```text
data/
├── gen/
│   └── <MODEL>_v<VOTERS>_c<CANDIDATES>/
│       ├── iter_0001.parquet
│       └── ...
├── sim_result/
│   └── <MODEL>_v<VOTERS>_c<CANDIDATES>/
│       ├── iter_0001.parquet
│       └── ...
└── results/
    └── <MODEL>_v<VOTERS>_c<CANDIDATES>.parquet   ← series cache
```

## Notes

- Generated profiles are cached and reused automatically. Pass `reload=True` to force regeneration.
- Series results from `simulation_instance` / `simulation_series_from_config` are also cached under `data/results/`. Adding new rules to an existing run only computes the new ones.
- Rule and generator codes are normalised to uppercase. Unknown codes are skipped with a warning.
