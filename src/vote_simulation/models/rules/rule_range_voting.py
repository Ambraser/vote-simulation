"""Range Voting (Average) wrapper

Each voter grades every candidate on the interval [min_grade, max_grade].
Co-winners are all candidates sharing the **maximum average grade**.

``scores_`` is a 1-D float array where ``scores_[c]`` is the average grade
received by candidate ``c``.

Grade bounds are derived automatically from the profile's utility matrix so
that the clip transformation (``rescale_grades=False``) is consistent with
the actual utility range.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleRangeVoting

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


def _grade_bounds(profile: Profile) -> tuple[float, float]:
    return float(np.min(profile.preferences_ut)), float(np.max(profile.preferences_ut))


class RangeVotingResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleRangeVoting` with proper co-winner semantics.

    Co-winners are all candidates sharing the maximum average grade.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Range Voting.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        lo, hi = _grade_bounds(profile)
        self._inner = RuleRangeVoting(min_grade=lo, max_grade=hi, rescale_grades=False)(profile)
        self.cowinners_ = self._init_score_based()


def _build_range_voting():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Range Voting."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return RangeVotingResult(profile)

    return builder

register_rule("RV", _build_range_voting())
