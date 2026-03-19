"""Rule index for mapping short codes to `svvamp` rule factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from svvamp import (
    Profile,
    RuleApproval,
    RuleBaldwin,
    RuleBlack,
    RuleBorda,
    RuleBucklin,
    RuleCoombs,
    RuleCopeland,
    RuleDodgson,
    RuleIRV,
    RuleIteratedBucklin,
    RuleKApproval,
    RuleMajorityJudgment,
    RuleMaximin,
    RuleNanson,
    RulePlurality,
    RuleRangeVoting,
    RuleSchulze,
    RuleTwoRound,
)

RuleInput = Profile | Sequence[Mapping[str, float]]
RuleBuilder = Callable[[RuleInput, set[str] | None], object]
# Index
_RULE_BUILDERS: dict[str, RuleBuilder] = {}


def _infer_labels(ballots: Sequence[Mapping[str, float]], candidates: set[str] | None) -> list[str]:
    if ballots:
        first_ballot = ballots[0]
        if first_ballot:
            return [str(label) for label in first_ballot.keys()]
    if candidates:
        return sorted(str(candidate) for candidate in candidates)
    raise ValueError("Unable to infer candidate labels from empty ballots.")


def _ensure_profile(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> Profile:
    if isinstance(profile_or_ballots, Profile):
        return profile_or_ballots

    labels = _infer_labels(profile_or_ballots, candidates)
    matrix = np.asarray(
        [[float(ballot[label]) for label in labels] for ballot in profile_or_ballots],
        dtype=np.float64,
    )
    return Profile(preferences_ut=matrix, labels_candidates=labels)


def _grade_bounds(profile: Profile) -> tuple[float, float]:
    return float(np.min(profile.preferences_ut)), float(np.max(profile.preferences_ut))


def _build_with_rule(rule_factory: Callable[[Profile], object]) -> RuleBuilder:
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> object:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return rule_factory(profile)

    return builder


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


register_rule("PLU1", _build_with_rule(lambda profile: RulePlurality()(profile)))


register_rule("PLU2", _build_with_rule(lambda profile: RuleTwoRound()(profile)))


register_rule("BLAC", _build_with_rule(lambda profile: RuleBlack()(profile)))


register_rule("BORD", _build_with_rule(lambda profile: RuleBorda()(profile)))


# register_rule("COND", _build_with_rule(lambda profile: RuleCondorcet()(profile)))


register_rule("COOM", _build_with_rule(lambda profile: RuleCoombs()(profile)))


def _build_l4vd(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> object:
    """L4VD rule : code L4VD"""
    raise NotImplementedError("L4VD rule is not implemented yet")


register_rule("L4VD", _build_l4vd)  # TODO: implement L4VD rule


def _build_rv(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> object:
    """Range voting rule : code RV"""
    profile = _ensure_profile(profile_or_ballots, candidates)
    min_grade, max_grade = _grade_bounds(profile)
    return RuleRangeVoting(min_grade=min_grade, max_grade=max_grade, rescale_grades=False)(profile)


register_rule("RV", _build_rv)


register_rule("COPE", _build_with_rule(lambda profile: RuleCopeland()(profile)))


def _build_majority_judgment(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> object:
    """Majority judgment rule : code MJ"""
    profile = _ensure_profile(profile_or_ballots, candidates)
    min_grade, max_grade = _grade_bounds(profile)
    return RuleMajorityJudgment(min_grade=min_grade, max_grade=max_grade, rescale_grades=False)(profile)


register_rule("MJ", _build_majority_judgment)


register_rule("BUCK_R", _build_with_rule(lambda profile: RuleBucklin()(profile)))


register_rule("DODG_S", _build_with_rule(lambda profile: RuleDodgson()(profile)))


register_rule("NANS", _build_with_rule(lambda profile: RuleNanson()(profile)))


register_rule("AP", _build_with_rule(lambda profile: RuleApproval()(profile)))


register_rule("BALD", _build_with_rule(lambda profile: RuleBaldwin()(profile)))


register_rule("BUCK_I", _build_with_rule(lambda profile: RuleIteratedBucklin()(profile)))


register_rule("HARE", _build_with_rule(lambda profile: RuleIRV()(profile)))


register_rule("MMAX", _build_with_rule(lambda profile: RuleMaximin()(profile)))


register_rule("SCHU", _build_with_rule(lambda profile: RuleSchulze()(profile)))


register_rule("AP_K", _build_with_rule(lambda profile: RuleKApproval(k=2)(profile)))

# register_rule(
#    "STAR", _build_with_rule(lambda profile: RuleSTAR(min_grade=0.0, max_grade=1.0, rescale_grades=False)(profile))
# )

register_rule("DODG_C", _build_with_rule(lambda profile: RuleDodgson()(profile)))


""" TO CHECK LATER ON """
'''

def _build_kim_roush(ballots: list, candidates: set[str]) -> object:
    """ Kim-Roush rule"""
    return RuleKimRoush(ballots, candidates=candidates)


def _build_ranked_pairs(ballots: list, candidates: set[str]) -> object:
    """ Ranked pairs rule"""
    return RuleRankedPairs(ballots, candidates=candidates)


def _build_score(ballots: list, candidates: set[str]) -> object:
    """ Score rule : code SCORE"""
    return RuleScore(ballots, candidates=candidates)






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
