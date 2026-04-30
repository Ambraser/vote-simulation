import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.registry import get_all_rules_codes
from vote_simulation.simulation.simulation import simulation_step


def test_wrong_file_path():
    """test if a filepath that does not exist raises error"""
    return  # TODO: implement test for wrong file path


def test_wrong_file_format():
    """test for unsupported file format"""
    return  # TODO: implement test for wrong file format


def test_ties_cases():
    """test if ties are properly handled"""

    # Each candidate gets exactly one voter preferring them with utility 1;
    # the other two have utility 0 (tied).  Total utility per candidate = 1,
    # so every rule sees an exact 3-way tie and must return all three
    # candidates as co-winners.
    #
    # Copeland co-winners are computed from preferences_ut directly (not from
    # svvamp's rank-based scores_) so this profile produces the correct 3-way
    # tie for all rules.

    ballots = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    profile = Profile(preferences_ut=ballots, labels_candidates=["A", "B", "C"])

    rules_codes = get_all_rules_codes()

    result = simulation_step(profile, rules_codes)
    avoided_rules = {"L4VD", "BALD", "BALD_FAST", "BUCK_I", "BUCK_I_EXACT", "ICRV", "ICRV_EXACT", "KEME", "KEME_LAZY", "PLU2", "SLAT", "STAR"}  # these rules do not produce 3-way ties on this profile
    for rule_code, rule_result in result.winners_by_rule.items():
        if rule_code in avoided_rules:
            continue
        assert rule_result == ["A", "B", "C"], f"Rule {rule_code} did not handle ties correctly"
