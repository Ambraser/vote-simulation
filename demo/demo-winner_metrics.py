"""Demo — winner quality metrics across the full simulation workflow.

This script demonstrates how :class:`WinnerMetrics` flows through every
layer of the pipeline:

    1. One election (rule wrapper)  →  WinnerMetrics
    2. One simulation step          →  metrics_frame
    3. One simulation series        →  metrics_summary_frame (mean ± std)
    4. Total result                 →  metrics_comparison_frame / metrics_pivot

Run from the repository root::

    python demo/demo-winner_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the source tree is on the path when running the script directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from svvamp import Profile

from vote_simulation.models.results.result_config import ResultConfig
from vote_simulation.models.results.series_result import SimulationSeriesResult
from vote_simulation.models.results.step_result import SimulationStepResult
from vote_simulation.models.results.total_result import SimulationTotalResult
from vote_simulation.models.rules import get_rule_builder
from vote_simulation.models.rules.rule_approval import ApprovalResult
from vote_simulation.models.rules.rule_copeland import CopelandResult

# ---------------------------------------------------------------------------
# 1. Single-election level
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("1 · Single election — raw WinnerMetrics")
print("=" * 60)


rng = np.random.default_rng(42)
pref_ut = rng.normal(size=(50, 5))
profile = Profile(
    preferences_ut=pref_ut,
    labels_candidates=["Alice", "Bob", "Carol", "Dave", "Eve"],
)

ap = ApprovalResult(profile, approval_threshold=0.0)
cop = CopelandResult(profile)

print(f"\nApproval co-winners : {ap.cowinners_}")
print(f"Copeland co-winners : {cop.cowinners_}")

for name, rule in [("Approval", ap), ("Copeland", cop)]:
    m = rule.compute_metrics()
    print(f"\n{name} WinnerMetrics:")
    for k, v in m.to_dict().items():
        print(f"  {k:25s}: {v}")

# ---------------------------------------------------------------------------
# 2. Step level  →  metrics_frame
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2 · SimulationStepResult.metrics_frame")
print("=" * 60)


step = SimulationStepResult(data_source="demo")
for code, rule in [("AP_T0", ap), ("COPE", cop)]:
    metrics = rule.compute_metrics()
    step.add_method_result_with_metrics(code, rule.cowinners_, metrics)

print(f"\nStep winners:\n{step.winners_by_rule}")
print(f"\nStep metrics_frame:\n{step.metrics_frame.T.to_string()}")

# ---------------------------------------------------------------------------
# 3. Series level  →  metrics_summary_frame
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3 · SimulationSeriesResult.metrics_summary_frame")
print("=" * 60)

rng2 = np.random.default_rng(7)
series = SimulationSeriesResult()
config = ResultConfig.single(gen_model="UNI", n_voters=100, n_candidates=5, n_iterations=30)

for it in range(30):
    ut = rng2.normal(size=(100, 5))
    prof = Profile(preferences_ut=ut)
    step_i = SimulationStepResult(data_source=f"iter_{it:04d}", config=config)

    for code in ["AP_T0", "COPE"]:
        builder = get_rule_builder(code)
        rule_result = builder(prof, None)
        m = rule_result.compute_metrics()
        step_i.add_method_result_with_metrics(code, rule_result.cowinners_, m)

    series.add_step(step_i)

summary = series.metrics_summary_frame
print(f"\nmetrics_summary_frame (shape {summary.shape}):")
# Display a clean subset of columns
display_cols = [
    "social_acceptability_mean",
    "social_acceptability_std",
    "utility_mean_mean",
    "rank_mean_mean",
    "freq_first_mean",
    "freq_last_mean",
    "has_tie_mean",
    "n_cowinners_mean",
]
print(summary[display_cols].to_string())

# ---------------------------------------------------------------------------
# 4. Total result  →  metrics_comparison_frame & metrics_pivot
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4 · SimulationTotalResult.metrics_comparison_frame + metrics_pivot")
print("=" * 60)


total = SimulationTotalResult()

for gen_model in ["UNI", "IC"]:
    for n_v in [50, 200]:
        for n_c in [3, 7]:
            rng_inner = np.random.default_rng(hash((gen_model, n_v, n_c)) & 0xFFFFFFFF)
            ser = SimulationSeriesResult()
            cfg = ResultConfig.single(gen_model=gen_model, n_voters=n_v, n_candidates=n_c, n_iterations=20)
            for _ in range(20):
                ut = rng_inner.normal(size=(n_v, n_c))
                prof = Profile(preferences_ut=ut)
                s = SimulationStepResult(data_source="sim", config=cfg)
                for code in ["AP_T0", "COPE"]:
                    rb = get_rule_builder(code)
                    rr = rb(prof, None)
                    mm = rr.compute_metrics()
                    s.add_method_result_with_metrics(code, rr.cowinners_, mm)
                ser.add_step(s)
            total.add_series(ser)

print(f"\nTotal result: {total}")

comp = total.metrics_comparison_frame("social_acceptability", "COPE")
print("\nmetrics_comparison_frame('social_acceptability', 'COPE'):")
print(comp.to_string())

pivot, fixed = total.metrics_pivot("social_acceptability", "COPE", row_param="n_voters", col_param="n_candidates")
print(f"\nmetrics_pivot (social_acceptability, COPE) — {fixed}:")
print(pivot.to_string())

pivot2, fixed2 = total.metrics_pivot("rank_mean", "AP_T0", row_param="n_voters", col_param="n_candidates")
print(f"\nmetrics_pivot (rank_mean, AP_T0) — {fixed2}:")
print(pivot2.to_string())

print("\n✓ Demo completed successfully.")
