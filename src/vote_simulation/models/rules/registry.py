"""Rule index for mapping short codes to `svvamp` rule factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

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
from svvamp.rules.rule_condorcet_abs_irv import RuleCondorcetAbsIRV
from svvamp.rules.rule_condorcet_sum_defeats import RuleCondorcetSumDefeats
from svvamp.rules.rule_condorcet_vtb_irv import RuleCondorcetVtbIRV
from svvamp.rules.rule_exhaustive_ballot import RuleExhaustiveBallot
from svvamp.rules.rule_icrv import RuleICRV
from svvamp.rules.rule_irv_average import RuleIRVAverage
from svvamp.rules.rule_irv_duels import RuleIRVDuels
from svvamp.rules.rule_kemeny import RuleKemeny
from svvamp.rules.rule_kim_roush import RuleKimRoush
from svvamp.rules.rule_ranked_pairs import RuleRankedPairs
from svvamp.rules.rule_slater import RuleSlater
from svvamp.rules.rule_smith_irv import RuleSmithIRV
from svvamp.rules.rule_split_cycle import RuleSplitCycle
from svvamp.rules.rule_star import RuleSTAR
from svvamp.rules.rule_tideman import RuleTideman
from svvamp.rules.rule_veto import RuleVeto
from svvamp.rules.rule_woodall import RuleWoodall
from svvamp.rules.rule_young import RuleYoung

RuleInput = Profile | Sequence[Mapping[str, float]]


class RuleResult(Protocol):
    """Protocol for rule results that have been post-processed to include co-winners."""

    cowinners_: list[str]


RuleBuilder = Callable[[RuleInput, set[str] | None], RuleResult]
# Index
_RULE_BUILDERS: dict[str, RuleBuilder] = {}


def _label_for_candidate(rule: object, index: int) -> str:
    profile = getattr(rule, "profile_", None)
    if profile is not None:
        labels = getattr(profile, "labels_candidates", None)
        if labels is not None and 0 <= index < len(labels):
            return str(labels[index])
    return str(index)


def _winner_index(rule: object) -> int | None:
    winner = getattr(rule, "w_", None)
    if winner is None:
        return None
    try:
        winner_float = float(winner)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(winner_float):
        return None
    winner_index = int(winner_float)
    profile = getattr(rule, "profile_", None)
    n_candidates = getattr(profile, "n_c", None)
    if isinstance(n_candidates, int) and not (0 <= winner_index < n_candidates):
        return None
    return winner_index


def _compute_cowinners(rule: object) -> list[str]:
    scores = getattr(rule, "scores_", None)
    if scores is not None:
        try:
            scores_array = np.asarray(scores)
        except Exception:
            scores_array = None
        if scores_array is not None and scores_array.ndim == 1 and scores_array.size > 0:
            best_score = np.max(scores_array)
            winner_indices = np.flatnonzero(np.isclose(scores_array, best_score, equal_nan=False))
            if winner_indices.size > 0:
                return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in winner_indices]

    winner_index = _winner_index(rule)
    if winner_index is not None:
        return [_label_for_candidate(rule, winner_index)]

    profile = getattr(rule, "profile_", None)
    weak_winners = getattr(profile, "weak_condorcet_winners", None)
    if weak_winners is not None:
        weak_winner_indices = np.flatnonzero(np.asarray(weak_winners, dtype=bool))
        if weak_winner_indices.size > 0:
            return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in weak_winner_indices]
    return []


def _ensure_cowinners(rule: Any) -> RuleResult:
    if hasattr(rule, "cowinners_"):
        return rule
    rule.cowinners_ = _compute_cowinners(rule)
    return rule


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


def _build_with_rule(rule_factory: Callable[[Profile], Any]) -> RuleBuilder:
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return _ensure_cowinners(rule_factory(profile))

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


def _build_l4vd(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
    """L4VD rule : code L4VD"""
    raise NotImplementedError("L4VD rule is not implemented yet")


register_rule("L4VD", _build_l4vd)  # TODO: implement L4VD rule


def _build_rv(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
    """Range voting rule : code RV"""
    profile = _ensure_profile(profile_or_ballots, candidates)
    min_grade, max_grade = _grade_bounds(profile)
    return _ensure_cowinners(RuleRangeVoting(min_grade=min_grade, max_grade=max_grade, rescale_grades=False)(profile))


register_rule("RV", _build_rv)


register_rule("COPE", _build_with_rule(lambda profile: RuleCopeland(cm_option="exact")(profile)))


def _build_majority_judgment(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
    """Majority judgment rule : code MJ"""
    profile = _ensure_profile(profile_or_ballots, candidates)
    min_grade, max_grade = _grade_bounds(profile)
    return _ensure_cowinners(
        RuleMajorityJudgment(min_grade=min_grade, max_grade=max_grade, rescale_grades=False)(profile)
    )


register_rule("MJ", _build_majority_judgment)


def _build_star(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
    """STAR rule with grade bounds inferred from profile."""
    profile = _ensure_profile(profile_or_ballots, candidates)
    min_grade, max_grade = _grade_bounds(profile)
    return _ensure_cowinners(RuleSTAR(min_grade=min_grade, max_grade=max_grade, rescale_grades=False)(profile))


register_rule("BUCK_R", _build_with_rule(lambda profile: RuleBucklin()(profile)))
register_rule("DODG_S", _build_with_rule(lambda profile: RuleDodgson()(profile)))
register_rule("NANS", _build_with_rule(lambda profile: RuleNanson()(profile)))
register_rule("AP_T", _build_with_rule(lambda profile: RuleApproval(approval_threshold=0.7)(profile)))
register_rule("BALD", _build_with_rule(lambda profile: RuleBaldwin()(profile)))
register_rule("BUCK_I", _build_with_rule(lambda profile: RuleIteratedBucklin()(profile)))
register_rule("HARE", _build_with_rule(lambda profile: RuleIRV()(profile)))
register_rule("IRV", _build_with_rule(lambda profile: RuleIRV()(profile)))
register_rule("MMAX", _build_with_rule(lambda profile: RuleMaximin()(profile)))
register_rule("SCHU", _build_with_rule(lambda profile: RuleSchulze()(profile)))
register_rule("AP_K", _build_with_rule(lambda profile: RuleKApproval(k=2)(profile)))
register_rule("STAR", _build_star)
register_rule("DODG_C", _build_with_rule(lambda profile: RuleDodgson()(profile)))
register_rule("CSUM", _build_with_rule(lambda profile: RuleCondorcetSumDefeats()(profile)))
register_rule("IRVD", _build_with_rule(lambda profile: RuleIRVDuels()(profile)))
register_rule("KEME", _build_with_rule(lambda profile: RuleKemeny(winner_option="exact")(profile)))
register_rule("KIMR", _build_with_rule(lambda profile: RuleKimRoush()(profile)))
register_rule("RPAR", _build_with_rule(lambda profile: RuleRankedPairs()(profile)))
register_rule("SLAT", _build_with_rule(lambda profile: RuleSlater(winner_option="exact")(profile)))
register_rule("SPCY", _build_with_rule(lambda profile: RuleSplitCycle()(profile)))
register_rule("VETO", _build_with_rule(lambda profile: RuleVeto()(profile)))
register_rule("YOUN", _build_with_rule(lambda profile: RuleYoung()(profile)))
register_rule("EXHB", _build_with_rule(lambda profile: RuleExhaustiveBallot()(profile)))
register_rule("CAIR", _build_with_rule(lambda profile: RuleCondorcetAbsIRV()(profile)))
register_rule("CVIR", _build_with_rule(lambda profile: RuleCondorcetVtbIRV()(profile)))
register_rule("ICRV", _build_with_rule(lambda profile: RuleICRV()(profile)))
register_rule("IRVA", _build_with_rule(lambda profile: RuleIRVAverage()(profile)))
register_rule("SIRV", _build_with_rule(lambda profile: RuleSmithIRV()(profile)))
register_rule("TIDE", _build_with_rule(lambda profile: RuleTideman()(profile)))
register_rule("WOOD", _build_with_rule(lambda profile: RuleWoodall()(profile)))
