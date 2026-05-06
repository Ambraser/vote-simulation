# Total Result

The **`SimulationTotalResult`** class aggregates multiple [`SimulationSeriesResult`](simulation_results.md) instances across a parameter space defined by three axes:

- **Generative model** (e.g. `UNI`, `IC`, `VMF_HC`)
- **Number of voters** (e.g. `11`, `101`, `1001`)
- **Number of candidates** (e.g. `3`, `14`)

Each series is uniquely keyed by its `(gen_model, n_voters, n_candidates)` triplet.
The class enables **filtering**, **metric pivoting**, **comparison plots**, and **persistence** across the entire parameter space.

---

## Quick start

### From a TOML config (recommended)

The simplest way to build a `SimulationTotalResult` is via `simulation_series_from_config`, which iterates over every combination defined in your config file:

```python
from vote_simulation.simulation.simulation import simulation_series_from_config

total = simulation_series_from_config("config/simulation.toml")
print(total)
# SimulationTotalResult(series=6, models=['VMF_HC'], voters=[11, 101, 1001], candidates=[3, 14])
```

### Manual construction

You can also build it by hand, calling `simulation_instance` for each point in the parameter grid:

```python
from vote_simulation.models.results.total_result import SimulationTotalResult
from vote_simulation.simulation.simulation import simulation_instance

rules = ["PLU1", "PLU2", "BORD", "RV", "COPE", "HARE", "MJ"]

total = SimulationTotalResult()
for model in ["UNI", "IC"]:
    for n_v in [101, 1001]:
        for n_c in [3, 14]:
            series = simulation_instance(model, n_v, n_c, rules, n_iteration=100)
            total.add_series(series)
```

!!! warning "Duplicate keys"
    `add_series()` raises `ValueError` if a series with the same `(model, voters, candidates)` key is already present.
    Use `replace_series()` to overwrite an existing entry.

---

## Series key

Each series is indexed by a `SeriesKey` named tuple:

```python
from vote_simulation.models.results.total_result import SeriesKey

key = SeriesKey(gen_model="UNI", n_voters=101, n_candidates=3)
```

You can check membership and retrieve series by key:

```python
SeriesKey("UNI", 101, 3) in total   # True
series = total.get_series("UNI", 101, 3)
```

---

## Accessors

| Property / method | Returns | Description |
|---|---|---|
| `series_count` | `int` | Number of stored series |
| `keys` | `list[SeriesKey]` | Sorted list of all keys |
| `gen_models` | `list[str]` | Distinct model codes |
| `voter_counts` | `list[int]` | Distinct voter counts |
| `candidate_counts` | `list[int]` | Distinct candidate counts |
| `get_series(model, nv, nc)` | `SimulationSeriesResult` | Look up one series |
| `len(total)` | `int` | Same as `series_count` |
| `key in total` | `bool` | Membership test |

Iteration yields `(SeriesKey, SimulationSeriesResult)` pairs in sorted order:

```python
for key, series in total:
    print(f"{key} → {series.step_count} steps, mean_dist={series.mean_distance:.1f}%")
```

---

## Filtering

`filter()` returns a **new** `SimulationTotalResult` sharing the same series objects (shallow copy).
All three parameters are optional keyword-only arguments — combine as needed:

```python
# Keep only UNI model
uni = total.filter(gen_model="UNI")

# Keep only 14-candidate runs
c14 = total.filter(n_candidates=14)

# Combine: IC model with 3 candidates
ic3 = total.filter(gen_model="IC", n_candidates=3)
```

Filtering is the key step before plotting or pivoting: most methods require that the "third" parameter (the one not on the axes) has a single distinct value.

---

## Metrics

### Summary DataFrame

`summary_frame()` returns one row per series with scalar metrics:

```python
df = total.summary_frame()
print(df.to_string(index=False))
```

Columns: `gen_model`, `n_voters`, `n_candidates`, `step_count`, `n_iterations`, `mean_distance`, `most_distant_rule_a`, `most_distant_rule_b`, `most_distant_distance`.

### Metric matrix (pivot)

`metric_matrix()` pivots a scalar metric into a 2D table across two parameters.
The third parameter must be fixed to a single value (filter first if needed).

```python
# Fix voters → pivot models vs candidates
pivot, desc = total.filter(n_voters=101).metric_matrix(
    row_param="gen_model",
    col_param="n_candidates",
    metric="mean_distance",
)
print(f"Fixed: {desc}")
print(pivot)
```

Output:

```
Fixed: n_voters=101
n_candidates         3          14
gen_model
IC            17.81      50.91
UNI           24.95      50.62
```

Valid axis parameters: `"gen_model"`, `"n_voters"`, `"n_candidates"`.

### Rule-pair distance

`rule_pair_frame()` extracts the mean distance between two specific rules across all series:

```python
df = total.rule_pair_frame("PLU1", "BORD")
print(df.to_string(index=False))
```

```
gen_model  n_voters  n_candidates  distance
       IC       101             3     24.00
       IC       101            14     72.17
      UNI       101             3     24.00
      UNI       101            14     72.17
```

---

## Winner metrics

`SimulationTotalResult` exposes two methods to analyse **winner quality metrics** (see [`WinnerMetrics`](../rules.md)) aggregated across the entire parameter space.

### Available metric fields

The following fields are available in every `WinnerMetrics` object and can be used as the `metric_field` argument:

| Field | Type | Description |
|---|---|---|
| `social_acceptability` | `float` | Fraction of voters who prefer the winner(s) to every other candidate |
| `utility_mean` | `float` | Mean cardinal utility of the co-winner(s) across all voters |
| `utility_median` | `float` | Median cardinal utility of the co-winner(s) |
| `utility_var` | `float` | Variance of cardinal utility of the co-winner(s) |
| `rank_mean` | `float` | Mean Borda rank of the co-winner(s) (0 = worst, n_candidates−1 = best) |
| `rank_median` | `float` | Median Borda rank of the co-winner(s) |
| `rank_var` | `float` | Variance of Borda rank of the co-winner(s) |
| `freq_first` | `float` | Fraction of voters who rank a co-winner first |
| `freq_last` | `float` | Fraction of voters who rank a co-winner last |
| `has_tie` | `bool` | Whether there are multiple co-winners |
| `n_cowinners` | `int` | Number of co-winners |

### metrics_comparison_frame

`metrics_comparison_frame()` returns a long-form DataFrame with one row per `(gen_model, n_voters, n_candidates)` entry, containing the aggregated statistic for one rule and one metric:

```python
df = total.metrics_comparison_frame("social_acceptability", "COPE")
print(df.to_string(index=False))
```

```
gen_model  n_voters  n_candidates  social_acceptability_mean
       IC        50             3                       0.71
       IC        50             7                       0.63
      UNI        50             3                       0.68
      UNI        50             7                       0.59
```

Parameters:

- `metric_field` — one of the fields listed in the table above
- `rule_code` — normalised rule code (e.g. `"COPE"`, `"AP_T0"`)
- `stat` — `"mean"` (default) or `"std"`

Series without the requested rule's metrics are silently omitted.

### metrics_pivot

`metrics_pivot()` is the 2-D counterpart of `metrics_comparison_frame()`: it pivots the long-form table into a matrix, analogously to `metric_matrix()` for rule-distance metrics.

```python
pivot, fixed = total.metrics_pivot(
    "social_acceptability", "COPE",
    row_param="n_voters",
    col_param="n_candidates",
)
print(f"Fixed: {fixed}")
print(pivot.to_string())
```

```
Fixed: gen_model=IC, UNI
n_candidates        3         7
n_voters
50            0.695     0.610
200           0.721     0.638
```

```python
pivot2, fixed2 = total.metrics_pivot(
    "rank_mean", "AP_T0",
    row_param="n_voters",
    col_param="n_candidates",
)
print(f"Fixed: {fixed2}")
print(pivot2.to_string())
```

Parameters mirror `metric_matrix()`:

| Parameter | Type | Description |
|---|---|---|
| `metric_field` | `str` | Metric field name (see table above) |
| `rule_code` | `str` | Rule to inspect |
| `row_param` | `str` | Row axis — `"gen_model"`, `"n_voters"`, or `"n_candidates"` |
| `col_param` | `str` | Column axis — `"gen_model"`, `"n_voters"`, or `"n_candidates"` |
| `stat` | `str` | `"mean"` (default) or `"std"` |

!!! tip "Filter before pivoting"
    If the third parameter has more than one distinct value, `metrics_pivot()` averages over it implicitly. Use `.filter(...)` beforehand when you want a strict 2-D slice.

---

## Plotting

All plot methods accept `show=False` to skip `plt.show()` and `save_path` to write to disk.
They return the matplotlib `Axes` object for further customization.

### Metric heatmap

Visualize a scalar metric across two parameter axes:

```python
total.filter(n_voters=101).plot_metric_heatmap(
    row_param="gen_model",
    col_param="n_candidates",
    metric="mean_distance",
    show=False,
    save_path="results/metric_heatmap.png",
)
```

This produces a color-coded matrix where each cell shows the `mean_distance` for that `(model, candidates)` combination.

### Rule-pair heatmap

Focus on a single pair of rules:

```python
total.filter(n_voters=101).plot_rule_pair_heatmap(
    "PLU1", "BORD",
    row_param="gen_model",
    col_param="n_candidates",
    show=False,
    save_path="results/rule_pair.png",
)
```

### Comparison grid

Side-by-side full rule-distance heatmaps, one per value of a varying parameter:

```python
total.filter(gen_model="UNI", n_voters=101).plot_comparison_grid(
    "n_candidates",
    show=False,
    save_path="results/comparison_grid.png",
)
```

This is particularly useful to visually compare how rule agreement changes when the number of candidates increases, with all other variables held constant.

!!! tip "Filter before plotting"
    `plot_comparison_grid` requires that the two non-varying parameters are each fixed to a single value.
    Chain `.filter(...)` before calling it.

---

## Persistence

### Save to directory

Each series is saved as a separate parquet file named after its config label:

```python
total.save_to_dir("results/total/")
# Creates:
#   results/total/UNI_v101_c3_i100.parquet
#   results/total/UNI_v101_c14_i100.parquet
#   ...
```

### Load from directory

```python
loaded = SimulationTotalResult.load_from_dir("results/total/")
print(loaded)
```

### Delete

```python
SimulationTotalResult.delete_dir("results/total/")
```

---

## Complete example

```python
from vote_simulation.models.results.total_result import SimulationTotalResult
from vote_simulation.simulation.simulation import simulation_instance

# 1) Build
rules = ["PLU1", "PLU2", "BORD", "RV", "COPE", "HARE", "MJ"]
total = SimulationTotalResult()
for model in ["UNI", "IC"]:
    for n_c in [3, 14]:
        series = simulation_instance(model, 101, n_c, rules, n_iteration=50)
        total.add_series(series)

# 2) Explore
print(total)
print(total.summary_frame())

# 3) Filter + pivot
pivot, desc = total.filter(n_voters=101).metric_matrix(
    "gen_model", "n_candidates",
)
print(pivot)

# 4) Plot
total.filter(n_voters=101).plot_metric_heatmap(
    "gen_model", "n_candidates", show=False,
    save_path="metric_heatmap.png",
)

total.filter(gen_model="UNI", n_voters=101).plot_comparison_grid(
    "n_candidates", show=False,
    save_path="comparison.png",
)

# 5) Persist
total.save_to_dir("results/total/")
loaded = SimulationTotalResult.load_from_dir("results/total/")
assert loaded.series_count == total.series_count
```

---

## API reference

::: vote_simulation.models.results.total_result
