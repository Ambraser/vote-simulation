from vote_simulation.models.rules.registry import make_rule_builder, register_rule
from svvamp import RuleApproval, RuleKApproval
from vote_simulation.simulation.simulation import simulation_from_file, simulation_instance, simulation_series

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
        #"AP_T",
        "AP_K2",
        #"AP_K3",
        #"AP_K4",
        #"AP_K5",
        #"AP_K6",
        #"AP_K7",
        #"AP_K8",
        #"AP_K9",
        #"AP_K10",
        #"AP_K11",
        #"AP_K12",
        "AP_T05",
        #"AP_T06",
        #"AP_T07",
        "AP_T08",
        #"AP_T09",
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

    #step_result = simulation_from_file("data/gen/IC_v1001_c14/iter_0001.parquet", rules_codes)
    # print(step_result)
    #step_result.plot_distance_matrix()

    #series = simulation_series("data/gen/IC_v1001_c14", rules_codes)
    series = simulation_instance("UNI", 1001, 3, rules_codes, n_iteration=1000)
    series.plot_mean_distance_matrix(folder_save_path="demo/results/", show=False)
    #series.plot_rules_2d()

#    series_result =

"""rules_codes = [
    "AP_T",
    "AP_K",
    "BALD",
    "BLAC",
    "BORD",
    "CAIR",
    "CSUM",
    "CVIR",
    "BUCK_I",
    "BUCK_R",
    "COOM",
    "COPE",
    "DODG_C",
    "DODG_S",
    "EXHB",
    "HARE",
    "IRV",
    "IRVA",
    "IRVD",
    "ICRV",
    "KIMR",
    "MJ",
    "MMAX",
    "NANS",
    "PLU1",
    "PLU2",
    "RPAR",
    "RV",
    "SCHU",
    "SIRV",
    "SPCY",
    "STAR",
    "TIDE",
    "VETO",
    "WOOD",
    "YOUN"]
    """
