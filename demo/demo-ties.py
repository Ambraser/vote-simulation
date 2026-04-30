from svvamp import Profile

from vote_simulation.simulation.simulation import simulation_step

if __name__ == "__main__":
    rule_codes = [
        "MJ",
        "PLU1",
        "PLU2",
    ]

    ballots = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    profile = Profile(preferences_ut=ballots, labels_candidates=["A", "B", "C"])
    profile.demo()
    rule_codes = ["COPE"]

    result = simulation_step(profile=profile, rule_codes=rule_codes)

    print(result)
    # result.print_summary()
    # result.plot_matrix_heatmap()
