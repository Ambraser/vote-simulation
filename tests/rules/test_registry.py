from collections.abc import Sequence
from typing import Any, cast

import pytest

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.rules import get_rule_builder
from vote_simulation.models.rules.registry import _RULE_BUILDERS, RuleInput

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


@pytest.mark.parametrize("rule_code", IMPLEMENTED_RULE_CODES)
def test_rules_report_two_way_ties(rule_code: str):
    """Every implemented rule should expose both co-winners on a tied profile."""
    assert_rule_cowinners(rule_code, TIE_COWINNERS, input_profile=TIE_PROFILE)
