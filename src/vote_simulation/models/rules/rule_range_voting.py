"""Range Voting (Average) wrapper with semantically correct co-winner detection.

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_range_voting():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Range Voting."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return RangeVotingResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("RV", _build_range_voting())

if __name__ == "__main__":
    # Case 1 — clear winner
    result1 = RangeVotingResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 1 — clear winner:   scores_: {result1._inner.scores_}   cowinners_: {result1.cowinners_}")

    # Case 2 — 3-way tie (all voters indifferent between all)
    result2 = RangeVotingResult(
        _ensure_profile(
            [[1, 1, 1], [1, 1, 1]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 2 — 3-way tie:      scores_: {result2._inner.scores_}   cowinners_: {result2.cowinners_}")

    # Case 3 — 2-way tie at top
    result3 = RangeVotingResult(
        _ensure_profile(
            [[2, 2, 0], [2, 2, 0], [0, 0, 1]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 3 — 2-way tie:      scores_: {result3._inner.scores_}   cowinners_: {result3.cowinners_}")
