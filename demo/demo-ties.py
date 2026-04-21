from svvamp import Profile

from vote_simulation.models.rules.registry import get_all_rules_codes
from vote_simulation.simulation.simulation import simulation_step

if __name__ == "__main__":
    rule_codes = [
        "PV-BALD",
        "PV-BALDPUT",
        "PV-BALDTB",
        "PV-BEN",
        "PV-BENPUT",
        "PV-BENTB",
        "PV-BLC",
        "PV-BTRIR",
        "PV-BTRIRPUT",
        "PV-CIRV",
        "PV-CIRVPUT",
        "PV-COMB",
        "PV-COMBPUT",
        # "PV-COMBTB", "PV-CPGB", "PV-CPLB", "PV-CPLUR", "PV-DAUN",
        # "PV-IR", "PV-IRCL", "PV-IRPUT", "PV-IRTB", "PV-NANS", "PV-NANSW",
        "PV-PWRPUT",
        "PV-RAYN",
        "PV-SMMIN",
        "PV-WOOD",
        "AP_K2",
        "MJ",
        "PLU1",
        "PLU2",
    ]

    ballots = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]



    profile = Profile(preferences_ut=ballots, labels_candidates=["A", "B", "C"])
    rule_codes = ["COPE"]

    
    result = simulation_step(profile=profile, rule_codes=rule_codes)


    #print(result)
    #result.print_summary()
    #result.plot_matrix_heatmap()
