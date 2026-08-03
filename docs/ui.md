# Graphical interface

`vote_simulation` ships a browser-based interface built with [Streamlit](https://streamlit.io/).
It covers the full workflow — configuration, data generation, simulation, and result exploration — without writing any Python.

## Launch

```sh
vote-sim-ui
```

This starts a local Streamlit server and opens the browser automatically.
The server stops when you close the browser tab, or by typing `stop` + Enter in the terminal.

You can also start it with a specific port:

```sh
vote-sim-ui --server.port 8502
```

Or directly with Streamlit:

```sh
streamlit run src/vote_simulation/ui/app.py
```

---

## Global bar

At the top of every tab, a persistent bar shows:

- **Status** — current state (`Ready`, `Running`, `Done`, or an error message).
- **Active config** — path of the loaded TOML file, or `No configuration loaded`.
- **Full run** button — runs the entire pipeline (data generation + simulation) in one click using the current configuration.

---

## Tab 1 — Configuration

Set up or load a simulation configuration.

### Import / Export

- **Upload a TOML file** — loads an existing `simulation.toml` and propagates all values to the other tabs.
- **Download** — exports the current settings as a TOML file.
- **Reset** — clears the configuration back to an empty state.

### Simulation parameters

| Field | Description |
|---|---|
| Output base path | Root directory for `gen/`, `sim_result/`, and `results/` |
| Seed | Integer seed for reproducibility |

Changes here are reflected immediately across all tabs.

### TOML preview

Live read-only view of the TOML that would be written from the current state.

---

## Tab 2 — Data Generation

Generate synthetic election profiles.

### Generative models

Select one or more generator codes from the full registry (includes DDD R-based generators if R is available). Each code is shown as `CODE — Human-readable name`.

### Simulation combinations

| Field | Description |
|---|---|
| Voters | Comma-separated list of voter counts, e.g. `11, 101, 1001` |
| Candidates | Comma-separated list of candidate counts, e.g. `3, 14` |
| Iterations | Number of profiles to generate per (model, voters, candidates) combination |

### Launch

Click **Generate** to start. A progress bar and live log track the generation.
Already-cached profiles are reused automatically (no regeneration unless the seed changes).
Click **Stop** to interrupt.

---

## Tab 3 — Simulation

Apply voting rules to the generated profiles.

### Data source

Choose between:
- **Generated data** — uses the profiles produced in Tab 2 (default).
- **Custom folder** — point to a folder of pre-existing Parquet vote files.

### Rule selection

Rules are grouped by family (score-based, elimination-based, Condorcet-based, etc.).
Each family can be expanded to select individual rules.

- **Select all / Deselect all** — toggles all rules at once.
- **All — `<family>` / None — `<family>`** — toggles all rules within one family.

Selected rules are shown as `CODE — Rule name`.

### Launch

Click **Run simulation** (or use the global **Full run** button).
A progress bar and live log track the run. Already-computed iterations are skipped.

---

## Tab 4 — Results

Explore simulation results using the native `SimulationSeriesResult` and `SimulationTotalResult` methods.

### Loading results

Results are loaded automatically from the path set in Tab 1.
The scanner reads both:
- `sim_result/<MODEL>_v<NV>_c<NC>/iter_XXXX.parquet` — per-iteration files from `simulation_from_config`.
- `results/<MODEL>_v<NV>_c<NC>.parquet` — series cache from `simulation_instance` / `simulation_series_from_config`.

### Series view (single combination)

Select one `(model, voters, candidates)` combination to explore a `SimulationSeriesResult`:

| Plot | Method |
|---|---|
| Mean distance matrix | `series.plot_mean_distance_matrix()` |
| Rules projected (2D MDS) | `series.plot_rules_2d()` |
| Rules projected (3D MDS) | `series.plot_rules_3d()` |
| Distance matrix table | `series.mean_distance_matrix_frame()` |
| Winner metrics summary | `series.metrics_summary_frame()` |

### Total view (multiple combinations)

Load all available series into a `SimulationTotalResult` to compare across the parameter space:

| Plot / table | Method |
|---|---|
| Summary table | `total.summary_frame()` |
| Mean distance heatmap | `total.plot_mean_distance_matrix()` |
| Metric heatmap | `total.plot_metric_heatmap()` |
| Comparison grid | `total.plot_comparison_grid()` |
| Rule-pair heatmap | `total.plot_rule_pair_heatmap()` |

### Data management

- **Save total to disk** — persists the current `SimulationTotalResult` for later reuse.
- **Delete series** — removes the Parquet files for a selected series from `results/` and `sim_result/`.
- **Export plots** — download any displayed figure as a PNG.
