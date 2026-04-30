"""Maximin method wrapper with semantically correct co-winner detection.

Maximin (also called Simpson-Kramer) elects the candidate whose worst
pairwise duel result is the best.  ``scores_[c]`` is the minimum entry of
row ``c`` in the pairwise duel matrix (excluding the diagonal).

``scores_`` is a 1-D integer array — co-winners are all candidates sharing
the maximum score.
"""

from __future__ import annotations

from svvamp import Profile, RuleMaximin

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class MaximinResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleMaximin` with proper co-winner semantics.

    Co-winners are all candidates sharing the maximum ``min-duel`` score.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Maximin.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleMaximin()(profile)
        self.cowinners_ = self._init_score_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_maximin():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Maximin."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return MaximinResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("MMAX", _build_maximin())

if __name__ == "__main__":
    # Case 1 — clear winner
    result1 = MaximinResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 1 — clear winner:")
    print("  scores_:", result1._inner.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — 3-way tie
    result2 = MaximinResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            {"A", "B", "C"},
        )
    )
    print("Case 2 — 3-way tie:")
    print("  scores_:", result2._inner.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — 2-way tie
    result3 = MaximinResult(
        _ensure_profile(
            [[2, 1, 0], [2, 1, 0], [1, 2, 0], [1, 2, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 3 — 2-way tie:")
    print("  scores_:", result3._inner.scores_)
    print("  cowinners_:", result3.cowinners_)
