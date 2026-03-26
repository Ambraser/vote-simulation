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
2. generate or load election profiles,
3. run one or more voting rules,
4. inspect the produced Parquet files.

The default example config lives in `config/simulation.toml`.

### TOML config file 

```toml
[simulation]
output_base_path = "data/"
generative_models = ["VMF_HC"]
rule_codes = ["PLU1", "BORD", "SCHU"]
candidates = [3, 14]
voters = [11, 101]
iterations = 10
seed = 161
```

You can copy paste this content into a toml file at the root of your project. 

### Data generation 
This part is optionnal, because it is automatically handle, but it can be called on it's own. 





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



### Simulation 

To run you're first simulation : 

```python
simulation_from_config(config_file.toml)
```

