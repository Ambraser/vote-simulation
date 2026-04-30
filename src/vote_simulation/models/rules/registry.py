"""Rule index for mapping short codes to `svvamp` rule factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from svvamp import (
    Profile,
    RuleDodgson,
)
from svvamp.rules.rule_condorcet_abs_irv import RuleCondorcetAbsIRV
from svvamp.rules.rule_condorcet_sum_defeats import RuleCondorcetSumDefeats
from svvamp.rules.rule_condorcet_vtb_irv import RuleCondorcetVtbIRV
from svvamp.rules.rule_exhaustive_ballot import RuleExhaustiveBallot
from svvamp.rules.rule_irv_average import RuleIRVAverage
from svvamp.rules.rule_irv_duels import RuleIRVDuels
from svvamp.rules.rule_ranked_pairs import RuleRankedPairs
from svvamp.rules.rule_smith_irv import RuleSmithIRV

RuleInput = Profile | Sequence[Mapping[str, float]] | Sequence[Sequence[float]] | np.ndarray


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

    profile = getattr(rule, "profile_", None)
    weak_winners = getattr(profile, "weak_condorcet_winners", None)
    n_candidates = getattr(profile, "n_c", None)
    if weak_winners is not None:
        weak_winner_indices = np.flatnonzero(np.asarray(weak_winners, dtype=bool))
        if n_candidates == 2 and weak_winner_indices.size == 2:
            return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in weak_winner_indices]

        if weak_winner_indices.size > 1:
            matrix_victories = getattr(profile, "matrix_victories_ut_abs", None)
            if matrix_victories is not None:
                try:
                    matrix_victories_array = np.asarray(matrix_victories, dtype=float)
                except Exception:
                    matrix_victories_array = None
                if (
                    matrix_victories_array is not None
                    and matrix_victories_array.size > 0
                    and np.all(np.isclose(matrix_victories_array, 0.0, equal_nan=False))
                ):
                    return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in weak_winner_indices]

    scores = getattr(rule, "scores_", None)
    if scores is not None:
        try:
            scores_array = np.asarray(scores)
        except Exception:
            scores_array = None
        if scores_array is not None and scores_array.size > 0:
            if scores_array.ndim == 1:
                best_score = np.max(scores_array)
                winner_indices = np.flatnonzero(np.isclose(scores_array, best_score, equal_nan=False))
                if winner_indices.size > 0:
                    return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in winner_indices]

    winner_index = _winner_index(rule)
    if winner_index is not None:
        return [_label_for_candidate(rule, winner_index)]

    return []


def _ensure_cowinners(rule: Any) -> RuleResult:
    # print(rule)

    if hasattr(rule, "ws"):
        rule.cowinners_ = rule.ws
        return rule

    if hasattr(rule, "cowinners_"):
        return rule
    rule.cowinners_ = _compute_cowinners(rule)
    return rule


def _is_mapping_ballot(ballot: object) -> bool:
    return isinstance(ballot, Mapping)


def _default_labels(n_candidates: int) -> list[str]:
    return [f"Candidate {index + 1}" for index in range(n_candidates)]


def _infer_labels(ballots: RuleInput, candidates: set[str] | None) -> list[str]:
    if candidates:
        return sorted(str(candidate) for candidate in candidates)

    if isinstance(ballots, np.ndarray):
        if ballots.ndim != 2:
            raise ValueError("Ballot matrix must be a 2d array.")
        return _default_labels(ballots.shape[1])

    if len(ballots) == 0:
        raise ValueError("Unable to infer candidate labels from empty ballots.")

    first_ballot = ballots[0]
    if _is_mapping_ballot(first_ballot):
        if first_ballot:
            return [str(label) for label in first_ballot.keys()]
        raise ValueError("Unable to infer candidate labels from empty ballot mapping.")

    if isinstance(first_ballot, Sequence) and not isinstance(first_ballot, str | bytes):
        return _default_labels(len(first_ballot))

    raise TypeError("Ballots must be mappings of candidate labels to scores or 2d numeric sequences.")


def _ut_to_rk_stable(preferences_ut: np.ndarray) -> np.ndarray:
    """Convert utilities to ``preferences_rk`` as expected by svvamp.

    ``preferences_rk[v, k]`` is the candidate at rank ``k`` for voter ``v``
    (0 = most preferred). This is a permutation matrix (ranking), not a rank
    matrix. Equal utilities are broken by candidate index (stable sort), which
    is deterministic and consistent with svvamp's CTB convention.

    Args:
        preferences_ut: 2d array of shape ``(n_voters, n_candidates)``.

    Returns:
        2d int array of shape ``(n_voters, n_candidates)`` where
        ``result[v, k]`` is the candidate ranked ``k``-th by voter ``v``.
    """
    preferences_ut = np.asarray(preferences_ut, dtype=np.float64)
    if preferences_ut.ndim != 2:
        raise ValueError("preferences_ut must be a 2d array.")
    return np.argsort(-preferences_ut, axis=1, kind="stable")


def _ensure_profile(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> Profile:
    if isinstance(profile_or_ballots, Profile):
        return profile_or_ballots

    labels = _infer_labels(profile_or_ballots, candidates)
    if isinstance(profile_or_ballots, np.ndarray):
        matrix = np.asarray(profile_or_ballots, dtype=np.float64)
    elif len(profile_or_ballots) > 0 and _is_mapping_ballot(profile_or_ballots[0]):
        matrix = np.asarray(
            [[float(ballot[label]) for label in labels] for ballot in profile_or_ballots],
            dtype=np.float64,
        )
    else:
        matrix = np.asarray(profile_or_ballots, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError("preferences_ut must be a 2d array.")
    if matrix.shape[1] != len(labels):
        raise ValueError(
            f"Candidate label count ({len(labels)}) does not match ballot width ({matrix.shape[1]})."
        )

    preferences_rk = _ut_to_rk_stable(matrix)
    return Profile(preferences_ut=matrix, preferences_rk=preferences_rk, labels_candidates=labels)


def _grade_bounds(profile: Profile) -> tuple[float, float]:
    return float(np.min(profile.preferences_ut)), float(np.max(profile.preferences_ut))


def _build_with_rule(rule_factory: Callable[[Profile], Any]) -> RuleBuilder:
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return _ensure_cowinners(rule_factory(profile))

    return builder


def get_all_rules_codes() -> list[str]:
    """Return a list of all registered rule codes."""
    return sorted(_RULE_BUILDERS.keys())


def make_rule_builder(rule_factory: Callable[[Profile], Any]) -> RuleBuilder:
    """Create a public `RuleBuilder` from a `Profile -> rule result` factory.

    This helper is intended for external users who want to register custom rules
    while reusing the registry's profile conversion and co-winner post-processing.

    Args:
        rule_factory: Callable that takes a `svvamp.Profile` and returns a rule result.

    Returns:
        A `RuleBuilder` that can be registered in the registry.
    """
    return _build_with_rule(rule_factory)


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


# PLU1 is registered in rule_plurality.py with proper co-winner semantics.
# PLU2 is registered in rule_two_round.py with proper co-winner semantics.


# BLAC is registered in rule_black.py with proper co-winner semantics.
# BORD is registered in rule_borda.py with proper co-winner semantics.


# register_rule("COND", _build_with_rule(lambda profile: RuleCondorcet()(profile)))


# COOM is registered in rule_coombs.py with proper co-winner semantics.


# def _build_l4vd(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
#    """L4VD rule : code L4VD"""
#    raise NotImplementedError("L4VD rule is not implemented yet")


# register_rule("L4VD", _build_l4vd)  # TODO: implement L4VD rule


# RV is registered in rule_range_voting.py with proper co-winner semantics.


# COPE is registered in rule_copeland.py with proper co-winner semantics.


# MJ and MJ_RESCALE are registered in rule_majority_judgment.py with proper co-winner semantics.


# STAR is registered in rule_star.py with proper co-winner semantics.


# BUCK_R is registered in rule_bucklin.py with proper co-winner semantics.
register_rule("DODG_S", _build_with_rule(lambda profile: RuleDodgson()(profile)))
# NANS is registered in rule_nanson.py with proper co-winner semantics.
# AP_T* rules are registered in rule_approval.py with proper co-winner semantics.
# BALD is registered in rule_baldwin.py with proper co-winner semantics.
# BUCK_I is registered in rule_iterated_bucklin.py with proper co-winner semantics.
# HARE and IRV are registered in rule_irv.py with proper co-winner semantics.
# MMAX is registered in rule_maximin.py with proper co-winner semantics.
# SCHU is registered in rule_schulze.py with proper co-winner semantics.
# AP_K and AP_K2..AP_K12 are registered in rule_k_approval.py with proper co-winner semantics.
# STAR is registered in rule_star.py with proper co-winner semantics.
register_rule("DODG_C", _build_with_rule(lambda profile: RuleDodgson()(profile)))
register_rule("CSUM", _build_with_rule(lambda profile: RuleCondorcetSumDefeats()(profile)))
register_rule("IRVD", _build_with_rule(lambda profile: RuleIRVDuels()(profile)))
# KEME is registered in rule_kemeny.py with proper co-winner semantics.
# KIMR is registered in rule_kim_roush.py with proper co-winner semantics.
register_rule("RPAR", _build_with_rule(lambda profile: RuleRankedPairs()(profile)))
# SLAT is registered in rule_slater.py with proper co-winner semantics.
# SPCY is registered in rule_split_cycle.py with proper co-winner semantics.
# VETO is registered in rule_veto.py with proper co-winner semantics.
# YOUN is registered in rule_young.py with proper co-winner semantics.
register_rule("EXHB", _build_with_rule(lambda profile: RuleExhaustiveBallot()(profile)))
register_rule("CAIR", _build_with_rule(lambda profile: RuleCondorcetAbsIRV()(profile)))
register_rule("CVIR", _build_with_rule(lambda profile: RuleCondorcetVtbIRV()(profile)))
# ICRV is registered in rule_icrv.py with proper co-winner semantics.
register_rule("IRVA", _build_with_rule(lambda profile: RuleIRVAverage()(profile)))
register_rule("SIRV", _build_with_rule(lambda profile: RuleSmithIRV()(profile)))
# TIDE is registered in rule_tideman.py with proper co-winner semantics.
# WOOD is registered in rule_woodall.py with proper co-winner semantics.


# AP_K* and AP_T* rules are registered in their dedicated rule modules.
