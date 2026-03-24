import pytest

from vote_simulation.models.data_generation.data_instance import DataInstance
from vote_simulation.models.rules import get_rule_builder
from vote_simulation.models.rules.registry import _RULE_BUILDERS

data_instance = DataInstance("tests/rules/simu_UNI_v_101_c_14_i_693.csv")
dummy_candidates = data_instance.candidates
dummy_ballots = data_instance.data

profile = data_instance.profile


def test_register_rule():
    """try all registered rules and test them with dummy dataset"""

    # Test each registered rule with a dummy dataset
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
        )


def test_cope_rule():
    """Test the COPE rule with a dummy dataset."""
    rule_builder = get_rule_builder("COPE")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1", "Candidate 4"]

def test_dodg_s_rule():
    """Test the DODG rule with a dummy dataset."""
    rule_builder = get_rule_builder("DODG_S")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == []


def test_approval_threshold(): 
    """Test that the correct threshold is used for the APPROVAL rule."""
    rule_builder = get_rule_builder("AP_T")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.approval_threshold == 0.7
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_k_approval():
    """Test that the correct k is used for the K_APPROVAL rule."""
    rule_builder = get_rule_builder("AP_K")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.k == 2
    assert rule_instance.cowinners_ == ["Candidate 1"]


def test_bald():
    """Test the BALD rule with a dummy dataset."""
    rule_builder = get_rule_builder("BALD")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_blac():
    """Test the BLAC rule with a dummy dataset."""
    rule_builder = get_rule_builder("BLAC")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_borda():
    """Test the BORDA rule with a dummy dataset."""
    rule_builder = get_rule_builder("BORD")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_condorcet_abs_irv():
    """Test the CONDORCET_ABS_IRV rule with a dummy dataset."""
    rule_builder = get_rule_builder("CAIR")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_condorcet_sum_defeat():
    """Test the CONDORCET_SUM_DEFEAT rule with a dummy dataset."""
    rule_builder = get_rule_builder("CSUM")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_condorcet_vtb_irv():
    """Test the CONDORCET_VTB_IRV rule with a dummy dataset."""
    rule_builder = get_rule_builder("CVIR")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_bucklin():
    """Test the BUCKLIN rule with a dummy dataset."""
    rule_builder = get_rule_builder("BUCK_I")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_bucklin_round():
    """Test the BUCKLIN_ROUND rule with a dummy dataset."""
    rule_builder = get_rule_builder("BUCK_R")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_coombs():
    """Test the COOMBS rule with a dummy dataset."""
    rule_builder = get_rule_builder("COOM")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]


def test_exhaustive_ballot():
    """Test the EXHAUSTIVE_BALLOT rule with a dummy dataset."""
    rule_builder = get_rule_builder("EXHB")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]


def test_hare():
    """Test the HARE rule with a dummy dataset."""
    rule_builder = get_rule_builder("HARE")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_kim_roush():
    """Test the KIM_ROUSH rule with a dummy dataset."""
    rule_builder = get_rule_builder("KIMR")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 6"]

def test_majority_judgment():
    """Test the MAJORITY_JUDGMENT rule with a dummy dataset."""
    rule_builder = get_rule_builder("MJ")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 6"]


def test_MMAX_rule():
    """Test the MMAX rule with a dummy dataset."""
    rule_builder = get_rule_builder("MMAX")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_nanson():
    """Test the NANSON rule with a dummy dataset."""
    rule_builder = get_rule_builder("NANS")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_plurality():
    """Test the PLURALITY rule with a dummy dataset."""
    rule_builder = get_rule_builder("PLU1")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_tworounds():
    """Test the TWO_ROUNDS rule with a dummy dataset."""
    rule_builder = get_rule_builder("PLU2")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_ranked_pair():
    """Test the RANKED_PAIR rule with a dummy dataset."""
    rule_builder = get_rule_builder("RPAR")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_range_voting():
    """Test the RANGE_VOTING rule with a dummy dataset."""
    rule_builder = get_rule_builder("RV")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_schultz_rule():
    """Test the SCHULTZ rule with a dummy dataset."""
    rule_builder = get_rule_builder("SCHU")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_smith_irv():
    """Test the SMITH_IRV rule with a dummy dataset."""
    rule_builder = get_rule_builder("SIRV")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_split_cycle():
    """Test the SPLIT_CYCLE rule with a dummy dataset."""
    rule_builder = get_rule_builder("SPCY")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_star():
    """Test the STAR rule with a dummy dataset."""
    rule_builder = get_rule_builder("STAR")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_tideman() :
    """Test the TIDEMAN rule with a dummy dataset."""
    rule_builder = get_rule_builder("TIDE")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 4"]

def test_antiplurality():
    """Test the ANTI_PLURALITY rule with a dummy dataset."""
    rule_builder = get_rule_builder("VETO")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 7"]

def test_woodall():
    """Test the WOOD rule with a dummy dataset."""
    rule_builder = get_rule_builder("WOOD")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == ["Candidate 1"]

def test_young():
    """Test the YOUNG rule with a dummy dataset."""
    rule_builder = get_rule_builder("YOUN")
    rule_instance = rule_builder(profile, None)
    assert rule_instance.cowinners_ == []
