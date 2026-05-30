"""Nanson method wrapper.

Nanson eliminates all candidates with a Borda score strictly below the average
simultaneously each round, until all remaining candidates have equal Borda
scores.

``scores_[r, c]`` = Borda score of candidate ``c`` at round ``r``.
Eliminated candidates carry ``numpy.inf``.

Co-winners are all candidates that survive to the final round, i.e. all ``c``
with a finite score in ``scores_[-1, :]``.
"""

from __future__ import annotations

from svvamp import Profile, RuleNanson

from vote_simulation.models.rules.elimination_based import EliminationBasedRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class NansonResult(EliminationBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleNanson` with proper co-winner semantics.

    Co-winners are all candidates whose Borda score is finite in the last
    elimination round (i.e. they were not eliminated before the end).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Nanson.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleNanson()(profile)
        self.scores_ = self._inner.scores_
        self.cowinners_ = self._init_elimination_based()


def _build_nanson():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Nanson."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return NansonResult(profile)

    return builder


register_rule("NANS", _build_nanson())
