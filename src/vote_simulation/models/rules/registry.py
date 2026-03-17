"""Rule index for mapping short codes to whalrus rule factories."""

from collections.abc import Callable

from whalrus import (
    Priority,
    RuleApproval,
    RuleBaldwin,
    RuleBlack,
    RuleBorda,
    RuleBucklinByRounds,
    RuleBucklinInstant,
    RuleCondorcet,
    RuleCoombs,
    RuleCopeland,
    RuleIRV,
    RuleKApproval,
    RuleMajorityJudgment,
    RuleMaximin,
    RuleNanson,
    RulePlurality,
    RuleRangeVoting,
    RuleSchulze,
    RuleSimplifiedDodgson,
    RuleTwoRound,
)

RuleBuilder = Callable[[list, set[str]], object]
# Index
_RULE_BUILDERS: dict[str, RuleBuilder] = {}


def register_rule(code: str, builder: RuleBuilder) -> None:
    """Register a rule builder under a short code."""
    normalized_code = code.strip().upper()
    _RULE_BUILDERS[normalized_code] = builder


def get_rule_builder(code: str) -> RuleBuilder:
    """Return rule builder from code

    Args:
        code (str): rule encoding (detailed index in documentation)

    Raises:
        ValueError: if wrong code

    Returns:
        RuleBuilder: rule applied
    """
    normalized_code = code.strip().upper()
    try:
        return _RULE_BUILDERS[normalized_code]
    except KeyError as error:
        available = ", ".join(sorted(_RULE_BUILDERS))
        raise ValueError(f"Unknown rule code: '{code}'. Available codes: {available}") from error


# ALREADY EXISTING IN WHALRUS
def _build_plurality_1(ballots: list, candidates: set[str]) -> object:
    """Plurality rule : code PLU1"""
    return RulePlurality(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("PLU1", _build_plurality_1)  # Alias for PLU


def _build_two_rounds(ballots: list, candidates: set[str]) -> object:
    """Two rounds rule : code PLU2"""
    return RuleTwoRound(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("PLU2", _build_two_rounds)  # Alias for PLU


def _build_black(ballots: list, candidates: set[str]) -> object:
    """Black rule : code BLAC"""
    return RuleBlack(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("BLAC", _build_black)  # Alias for BLAC


def _build_borda(ballots: list, candidates: set[str]) -> object:
    """Borda rule : code BORD"""
    return RuleBorda(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("BORD", _build_borda)


def _build_condorcet(ballots: list, candidates: set[str]) -> object:
    """Condorcet rule : code COND"""
    return RuleCondorcet(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("COND", _build_condorcet)


def _build_coombs(ballots: list, candidates: set[str]) -> object:
    """Coombs rule : code COOM"""
    return RuleCoombs(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("COOM", _build_coombs)


def _build_l4vd(ballots: list, candidates: set[str]) -> object:
    """L4VD rule : code L4VD"""
    raise NotImplementedError("L4VD rule is not implemented yet")


register_rule("L4VD", _build_l4vd)  # TODO: implement L4VD rule


def _build_rv(ballots: list, candidates: set[str]) -> object:
    """Range voting rule : code RV"""
    return RuleRangeVoting(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("RV", _build_rv)


def _build_copeland(ballots: list, candidates: set[str]) -> object:
    """Copeland rule : code COPE"""
    return RuleCopeland(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("COPE", _build_copeland)  # Alias for COPE


def _build_majority_judgment(ballots: list, candidates: set[str]) -> object:
    """Majority judgment rule : code MJ"""
    return RuleMajorityJudgment(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("MJ", _build_majority_judgment)


def _build_bucklin_rounds(ballots: list, candidates: set[str]) -> object:
    """Bucklin by rounds rule : code BUCK_R"""
    return RuleBucklinByRounds(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("BUCK_R", _build_bucklin_rounds)


def _build_star(ballots: list, candidates: set[str]) -> object:
    """STAR rule : code STAR"""
    return NotImplementedError("STAR rule is not implemented yet")


register_rule("STAR", _build_star)  # TODO: implement STAR rule


def _build_dodgson(ballots: list, candidates: set[str]) -> object:
    """Dodgson rule : code DODG"""
    return NotImplementedError("DODGSON rule is not implemented yet")


register_rule("DODG", _build_dodgson)  # TODO: implement DODGSON rule


def _build_simplified_dodgson(ballots: list, candidates: set[str]) -> object:
    """Simplified Dodgson rule : code DODG_S"""
    return RuleSimplifiedDodgson(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("DODG_S", _build_simplified_dodgson)  # Alias for DODG_S


def _build_nanson(ballots: list, candidates: set[str]) -> object:
    """Nanson rule : code NANS"""
    return RuleNanson(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("NANS", _build_nanson)


# APPROVAL is not properly defined yet
def _build_approval(ballots: list, candidates: set[str]) -> object:
    """Approval rule : code AP"""
    return RuleApproval(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("AP", _build_approval)


def _build_baldwin(ballots: list, candidates: set[str]) -> object:
    """Baldwin rule :"""
    return RuleBaldwin(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("BALD", _build_baldwin)


def _build_bucklin_instant(ballots: list, candidates: set[str]) -> object:
    """Bucklin instant rule : code BUCK_I"""
    return RuleBucklinInstant(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("BUCK_I", _build_bucklin_instant)


def _build_irv(ballots: list, candidates: set[str]) -> object:
    """Instant-runoff voting or HARE rule : code HARE"""
    return RuleIRV(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("HARE", _build_irv)


def _build_minimax(ballots: list, candidates: set[str]) -> object:
    """Minimax rule : code MMAX"""
    return RuleMaximin(
        ballots, candidates=candidates, tie_break=Priority.ASCENDING
    )  # TODO : check if minimax and maximin are the same rule or if we need to implement a specific minimax rule


register_rule("MMAX", _build_minimax)


def _build_schulze(ballots: list, candidates: set[str]) -> object:
    """Schulze rule"""
    return RuleSchulze(ballots, candidates=candidates, tie_break=Priority.ASCENDING)


register_rule("SCHU", _build_schulze)


def _build_k_approval(ballots: list, candidates: set[str]) -> object:
    """K-approval rule : code AP_K"""
    return RuleKApproval(ballots, candidates=candidates, k=2, tie_break=Priority.ASCENDING)  # Example with k=2


register_rule("AP_K", _build_k_approval)

""" TO CHECK LATER ON """
'''
def _build_iterated_elimination(ballots: list, candidates: set[str]) -> object:
    """ Iterated elimination rule : code IE"""
    return RuleIteratedElimination(ballots, candidates=candidates)





def _build_kim_roush(ballots: list, candidates: set[str]) -> object:
    """ Kim-Roush rule"""
    return RuleKimRoush(ballots, candidates=candidates)


def _build_ranked_pairs(ballots: list, candidates: set[str]) -> object:
    """ Ranked pairs rule"""
    return RuleRankedPairs(ballots, candidates=candidates)


def _build_score(ballots: list, candidates: set[str]) -> object:
    """ Score rule : code SCORE"""
    return RuleScore(ballots, candidates=candidates)

register_rule("SCORE", _build_score)  # Alias for SCORE







def _build_score_num(ballots: list, candidates: set[str]) -> object:
    """ Score num rule"""
    return RuleScoreNum(ballots, candidates=candidates)

def _build_score_num_average(ballots: list, candidates: set[str]) -> object:
    """ Score num average rule"""
    return RuleScoreNumAverage(ballots, candidates=candidates)

def _build_score_num_row_sum(ballots: list, candidates: set[str]) -> object:
    """ Score num row sum rule"""
    return RuleScoreNumRowSum(ballots, candidates=candidates)

def _build_score_positional(ballots: list, candidates: set[str]) -> object:
    """ Score positional rule"""
    return RuleScorePositional(ballots, candidates=candidates)

def _build_sequential_elimination(ballots: list, candidates: set[str]) -> object:
    """ Sequential elimination rule"""
    return RuleSequentialElimination(ballots, candidates=candidates)

def _build_sequential_tie_break(ballots: list, candidates: set[str]) -> object:
    """ Sequential tie break rule"""
    return RuleSequentialElimination(ballots, candidates=candidates, tie_break=Priority.ASCENDING)




def _build_veto(ballots: list, candidates: set[str]) -> object:
    """ Veto rule"""
    return RuleVeto(ballots, candidates=candidates)


#register_rule("AP_R", _build_ap_r)  # Placeholder
register_rule("AP_T", lambda ballots, candidates: None)  # Placeholder

register_rule("AP_H", lambda ballots, candidates: None)  # Placeholder

'''
