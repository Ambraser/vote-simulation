# Getting started
## Install latest stable release

To install vote_simulation, run this command in your terminal:

```sh
uv add vote_simulation
```

Or if you prefer to use `pip`:

```sh
pip install vote_simulation
```

## From source

The source files for vote_simulation can be downloaded from the [Github repo](https://github.com/Damidas0/vote-simulation).

You can either clone the public repository:

```sh
git clone https://github.com/Damidas0/vote-simulation
```

Or download the [tarball](https://github.com/Damidas0/vote_simulation/tarball/main):

```sh
curl -OJL https://github.com/Damidas0/vote_simulation/tarball/main
```

Once you have a copy of the source, you can install it with:

```sh
cd vote_simulation
uv sync
```

## Recommended workflow

The typical workflow is:

1. prepare a TOML config file,
2. run the simulation (data generation is handled automatically),
3. inspect the produced Parquet files.

The default example config lives in `config/simulation.toml`.

### TOML config file

```toml
[simulation]
output_base_path = "data"
generative_models = ["VMF_HC"]
rule_codes = ["PLU1", "BORD", "SCHU"]
candidates = [3, 14]
voters = [11, 101]
iterations = 10
seed = 42
```

Copy this into a `simulation.toml` file and adjust as needed.

### Run the pipeline

```python
from vote_simulation.simulation.simulation import simulation_from_config

simulation_from_config("config/simulation.toml")
```

### Data generation only

Data generation is handled automatically inside `simulation_from_config`, but you can also call it on its own:

```python
from vote_simulation.simulation.simulation import generate_data

paths = generate_data("config/simulation.toml")
```

### Output structure

The pipeline writes files with this structure:

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
