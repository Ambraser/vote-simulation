from svvamp import RuleApproval, RuleKApproval

from vote_simulation.models.rules.registry import make_rule_builder, register_rule
from vote_simulation.simulation.simulation import simulation_instance

if __name__ == "__main__":
    register_rule("AP_K2", make_rule_builder(lambda profile: RuleKApproval(k=2)(profile)))
    register_rule("AP_K3", make_rule_builder(lambda profile: RuleKApproval(k=3)(profile)))
    register_rule("AP_K4", make_rule_builder(lambda profile: RuleKApproval(k=4)(profile)))
    register_rule("AP_K5", make_rule_builder(lambda profile: RuleKApproval(k=5)(profile)))
    register_rule("AP_K6", make_rule_builder(lambda profile: RuleKApproval(k=6)(profile)))
    register_rule("AP_K7", make_rule_builder(lambda profile: RuleKApproval(k=7)(profile)))
    register_rule("AP_K8", make_rule_builder(lambda profile: RuleKApproval(k=8)(profile)))
    register_rule("AP_K9", make_rule_builder(lambda profile: RuleKApproval(k=9)(profile)))
    register_rule("AP_K10", make_rule_builder(lambda profile: RuleKApproval(k=10)(profile)))
    register_rule("AP_K11", make_rule_builder(lambda profile: RuleKApproval(k=11)(profile)))
    register_rule("AP_K12", make_rule_builder(lambda profile: RuleKApproval(k=12)(profile)))
    register_rule("AP_T05", make_rule_builder(lambda profile: RuleApproval(approval_threshold=0.5)(profile)))
    register_rule("AP_T06", make_rule_builder(lambda profile: RuleApproval(approval_threshold=0.6)(profile)))
    register_rule("AP_T07", make_rule_builder(lambda profile: RuleApproval(approval_threshold=0.7)(profile)))
    register_rule("AP_T08", make_rule_builder(lambda profile: RuleApproval(approval_threshold=0.8)(profile)))
    register_rule("AP_T09", make_rule_builder(lambda profile: RuleApproval(approval_threshold=0.9)(profile)))

    rules_codes = [
        "RV",
        "MJ",
        "AP_K2",
        "AP_T05",
        "AP_T08",
        "BUCK_I",
        "BUCK_R",
        "BORD",
        "STAR",
        "BLAC",
        "SCHU",
        "COPE",
        "MMAX",
        "NANS",
        "COOM",
        "HARE",
        "PLU2",
        "PLU1",
    ]

    # 1) Run simulation (with cache — first run computes, second run loads)
    series = simulation_instance("UNI", 1001, 3, rules_codes, n_iteration=1000)

    # 2) Config info – automatically aggregated from steps
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Config label: {series.config.label}")
    print(f"Config description: {series.config.description}")
    print(f"Gen models: {sorted(series.config.gen_models)}")
    print(f"Voters: {sorted(series.config.n_voters)}")
    print(f"Candidates: {sorted(series.config.n_candidates)}")
    print(f"Iterations: {series.config.n_iterations}")

    # 3) Metrics
    print("\n" + "=" * 60)
    print("SERIES METRICS")
    print("=" * 60)
    print(f"Step count: {series.step_count}")
    print(f"Mean distance: {series.mean_distance:.2f}%")
    r1, r2, d = series.most_distant_rules
    print(f"Most distant pair:  {r1} <-> {r2}  ({d:.2f}%)")

    # 4) MDS projections with stress scores
    print("\n" + "=" * 60)
    print("MDS PROJECTIONS")
    print("=" * 60)
    proj_2d = series.map_rules_2d()
    print(f"2D stress: {proj_2d.stress:.4f}")
    proj_3d = series.map_rules_3d()
    print(f"3D stress: {proj_3d.stress:.4f}")

    # Step-level metrics on the first step
    first_step = series.steps[0]
    print("\n" + "-" * 40)
    print(f"STEP #1 METRICS (source: {first_step.data_source})")
    print("-" * 40)
    print(f" Step config: {first_step.config.description}")
    print(f" Mean distance: {first_step.mean_distance:.4f}")
    sr1, sr2, sd = first_step.most_distant_rules
    print(f" Most distant pair:  {sr1} <-> {sr2}  ({sd:.4f})")

    # 5) Save & reload (round-trip demo)
    import os

    save_path = "demo/results/UNI_v1001_c3_series.parquet"
    series.save_to_file(save_path)
    print(f"\n  Series saved to {save_path}  ({os.path.getsize(save_path) / 1024:.1f} KB)")

    from vote_simulation.models.results.series_result import SimulationSeriesResult

    loaded = SimulationSeriesResult()
    loaded.load_from_file(save_path)
    print(f"  Loaded back: {loaded.step_count} steps, config = {loaded.config.label}")
    print(f"  Mean distance after reload: {loaded.mean_distance:.2f}%")

    # 6) Plots – auto-saves series parquet alongside each plot
    series.plot_mean_distance_matrix(folder_save_path="demo/results/", show=False)
    print(f"\n  Heatmap + series parquet saved under demo/results/{series.config.label}/")

    series.plot_rules_2d(save_path="demo/results/", show=False)
    print(f"  2D MDS plot (stress={proj_2d.stress:.4f}) saved")

    series.plot_rules_3d(save_path="demo/results/", show=False)
    print(f"  3D MDS plot (stress={proj_3d.stress:.4f}) saved")

    # 7) Delete demo (show API, then re-save)
    deleted = SimulationSeriesResult.delete_file(save_path)
    print(f"\n  Deleted {save_path}: {deleted}")
    series.save_to_file(save_path)
    print("  Re-saved for future use.")
