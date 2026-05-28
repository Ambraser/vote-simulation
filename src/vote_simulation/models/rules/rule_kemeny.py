"""Kemeny method wrapper.

Kemeny finds the ranking of candidates that minimises the total Kendall-tau
distance to all voters.  The top candidate in the optimal order is the winner.

Co-winners are all candidates sharing the **maximum Kemeny score**.
``scores_`` is a 1-D integer array where ``scores_[c]`` is the rank position
of candidate ``c`` in the optimal Kemeny order (``n_c`` for the winner, 1 for
last).  All candidates with ``scores_[c] == max(scores_)`` are co-winners.

Note: computing the Kemeny winner is NP-hard, so this rule is expensive for
large profiles.
"""

from __future__ import annotations

from svvamp import Profile
from svvamp.rules.rule_kemeny import RuleKemeny

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class KemenyResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleKemeny` with proper co-winner semantics.

    Co-winners are all candidates sharing the top position in the optimal
    Kemeny order (i.e. those with the highest value in ``scores_``).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Kemeny.
    winner_option:
        ``'exact'`` (default) or ``'lazy'``. With ``'lazy'``, the winner is
        only resolved in the obvious Condorcet-winner case.
    """

    def __init__(self, profile: Profile, *, winner_option: str = "exact") -> None:
        self.profile_ = profile
        self._inner = RuleKemeny(winner_option=winner_option)(profile)
        self.cowinners_ = self._init_score_based()


def _build_kemeny(*, winner_option: str = "exact"):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Kemeny."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return KemenyResult(profile, winner_option=winner_option)

    return builder


register_rule("KEME", _build_kemeny(winner_option="exact"))
register_rule("KEME_LAZY", _build_kemeny(winner_option="lazy"))
