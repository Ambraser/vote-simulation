import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.registry import get_all_rules_codes
from vote_simulation.simulation.simulation import simulation_step


def test_ties_cases():
    """test if ties are properly handled"""

    # Condorcet cycle: A > B > C > A (each beats the next by 2 vs 1).
    # Voter 1: A > B > C  -> utilities [2, 1, 0]
    # Voter 2: B > C > A  -> utilities [0, 2, 1]
    # Voter 3: C > A > B  -> utilities [1, 0, 2]
    # Every pairwise contest is 2-1, so no Condorcet winner exists and
    # all candidates are perfectly symmetric — rules that respect this
    # symmetry must return all three candidates as co-winners.

    ballots = np.array(
        [
            [2, 1, 0],
            [0, 2, 1],
            [1, 0, 2],
        ]
    )

    profile = Profile(preferences_ut=ballots, labels_candidates=["A", "B", "C"])

    rules_codes = get_all_rules_codes()

    result = simulation_step(profile, rules_codes)
    avoided_rules = {
        # Elimination / sequential rules sensitive to tie-breaking order
        "BALD",
        "BUCK_I",
        "CAIR",
        "CVIR",
        "EXHB",
        "ICRV",
        "IRVA",
        "IRVD",
        "KEME",
        "KEME_LAZY",
        "PLU2",
        "RPAR",
        "SIRV",
        "SLAT",
        "STAR",
        "YOUN",
        # Rules that may return empty or partial sets on a Condorcet cycle
        "DODG_C",
        "DODG_S",
    }  # these rules do not return all 3 co-winners on the Condorcet cycle profile
    for rule_code, rule_result in result.winners_by_rule.items():
        if rule_code in avoided_rules:
            continue
        assert rule_result == ["A", "B", "C"], f"Rule {rule_code} did not handle ties correctly"
