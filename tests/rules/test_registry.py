from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.rules import get_rule_builder
from vote_simulation.models.rules.registry import _RULE_BUILDERS, RuleInput, _ensure_profile

REFERENCE_DATA_INSTANCE = DataInstance("tests/rules/simu_UNI_v_101_c_14_i_693.csv")
TIE_DATA_INSTANCE = DataInstance("tests/rules/simu_UNI_v_2_c_2_tie.csv")

REFERENCE_PROFILE = REFERENCE_DATA_INSTANCE.profile
TIE_PROFILE = TIE_DATA_INSTANCE.profile
TIE_COWINNERS = list(TIE_DATA_INSTANCE.candidates)

EXPECTED_REFERENCE_COWINNERS = [
    ("COPE", ["Candidate 1", "Candidate 4"]),
    ("DODG_S", []),
    ("BALD", ["Candidate 4"]),
    ("BLAC", ["Candidate 1"]),
    ("BORD", ["Candidate 1"]),
    ("CAIR", ["Candidate 1"]),
    ("CSUM", ["Candidate 4"]),
    ("CVIR", ["Candidate 1"]),
    ("BUCK_I", ["Candidate 1"]),
    ("BUCK_R", ["Candidate 1"]),
    ("COOM", ["Candidate 1"]),
    ("EXHB", ["Candidate 1"]),
    ("HARE", ["Candidate 1"]),
    ("KIMR", ["Candidate 6"]),
    ("MJ", ["Candidate 6"]),
    ("MMAX", ["Candidate 4"]),
    ("NANS", ["Candidate 4"]),
    ("PLU1", ["Candidate 1"]),
    ("PLU2", ["Candidate 1"]),
    ("RPAR", ["Candidate 4"]),
    ("RV", ["Candidate 1"]),
    ("SCHU", ["Candidate 4"]),
    ("SIRV", ["Candidate 1"]),
    ("SPCY", ["Candidate 4"]),
    ("STAR", ["Candidate 1"]),
    ("TIDE", ["Candidate 4"]),
    ("VETO", ["Candidate 7"]),
    ("WOOD", ["Candidate 1"]),
    ("YOUN", []),
]
IMPLEMENTED_RULE_CODES = sorted(rule_name for rule_name in _RULE_BUILDERS if rule_name != "L4VD")

# Rules that cannot detect ties via scores_ due to algorithmic limitations.
# Kemeny: finding all optimal orders is NP-hard — svvamp returns a single lexicographic order.
# ICRV: Condorcet-check rounds pick a single winner even on symmetric profiles.
_TIE_XFAIL_RULES = {"ICRV", "ICRV_EXACT", "KEME", "KEME_LAZY", "SLAT"}
_TIE_PARAMETRIZE = [
    pytest.param(code, marks=pytest.mark.xfail(reason="known tie-detection limitation"))
    if code in _TIE_XFAIL_RULES
    else code
    for code in IMPLEMENTED_RULE_CODES
]


def assert_rule_cowinners(
    rule_code: str,
    expected_cowinners: Sequence[str],
    *,
    input_profile: RuleInput = REFERENCE_PROFILE,
):
    """Build a rule and assert its co-winners."""
    rule_instance = get_rule_builder(rule_code)(input_profile, None)
    assert sorted(rule_instance.cowinners_) == sorted(expected_cowinners)
    return rule_instance


def test_register_rule():
    """Try all registered rules on a small dummy dataset."""
    dummy_ballots = [
        {"Alice": 5, "Bob": 3, "Charlie": 1},
        {"Alice": 4, "Bob": 4, "Charlie": 2},
        {"Alice": 3, "Bob": 5, "Charlie": 1},
    ]
    dummy_candidates = {"Alice", "Bob", "Charlie"}

    for rule_name in _RULE_BUILDERS:
        rule_builder = get_rule_builder(rule_name)
        if rule_name == "L4VD":
            with pytest.raises(NotImplementedError):
                rule_builder(dummy_ballots, dummy_candidates)
            continue
        rule_instance = rule_builder(dummy_ballots, dummy_candidates)
        assert rule_instance is NotImplementedError or (
            hasattr(rule_instance, "w_")
            or hasattr(rule_instance, "winner_indices_")
            or hasattr(rule_instance, "winner_")
            or hasattr(rule_instance, "cowinners_")
        )
#
#def test_ensure_profile_preserves_equal_score_ties_from_ballots():
#    """Registry conversion should keep equal scores at the same rank."""
#    profile = _ensure_profile(
#        [
#            {"Candidate 1": 0.0, "Candidate 2": 0.0, "Candidate 3": -1.0},
#            {"Candidate 1": 1.0, "Candidate 2": 0.0, "Candidate 3": 0.0},
#        ],
#        None,
#    )
#
#    preferences_rk = np.asarray(profile.preferences_rk, dtype=int)
#    assert preferences_rk[0, 0] == preferences_rk[0, 1]
#    assert preferences_rk[1, 1] == preferences_rk[1, 2]
#    assert preferences_rk[0, 0] < preferences_rk[0, 2]
#    assert preferences_rk[1, 0] < preferences_rk[1, 1]


def test_ensure_profile_accepts_list_ballots_with_candidates():
    """Registry conversion should also accept plain utility matrices."""
    profile = _ensure_profile(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        {"A", "B", "C"},
    )

    assert list(profile.labels_candidates) == ["A", "B", "C"]
    np.testing.assert_allclose(np.asarray(profile.preferences_ut, dtype=float), np.eye(3))


def test_ensure_profile_generates_default_labels_for_matrix_ballots():
    """Matrix ballots without explicit labels should get deterministic default labels."""
    profile = _ensure_profile(np.array([[2.0, 1.0], [0.0, 3.0]], dtype=float), None)

    assert list(profile.labels_candidates) == ["Candidate 1", "Candidate 2"]

@pytest.mark.parametrize(
    ("rule_code", "expected_cowinners"),
    EXPECTED_REFERENCE_COWINNERS,
    ids=[rule_code for rule_code, _ in EXPECTED_REFERENCE_COWINNERS],
)
def test_reference_rule_cowinners(rule_code: str, expected_cowinners: list[str]):
    """Each rule should keep returning the expected winners on the reference dataset."""
    assert_rule_cowinners(rule_code, expected_cowinners)


def test_approval_threshold():
    """The approval rule should keep its configured threshold."""
    rule_instance = assert_rule_cowinners("AP_T", ["Candidate 1"])
    assert cast(Any, rule_instance).approval_threshold == 0.7


def test_k_approval():
    """The k-approval rule should keep its configured k value."""
    rule_instance = assert_rule_cowinners("AP_K", ["Candidate 1"])
    assert cast(Any, rule_instance).k == 2


@pytest.mark.parametrize("rule_code", _TIE_PARAMETRIZE)
def test_rules_report_two_way_ties(rule_code: str):
    """Every implemented rule should expose both co-winners on a tied profile."""
    assert_rule_cowinners(rule_code, TIE_COWINNERS, input_profile=TIE_PROFILE)
