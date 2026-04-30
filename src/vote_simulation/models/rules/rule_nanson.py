"""Nanson method wrapper with semantically correct co-winner detection.

Nanson eliminates all candidates with a Borda score strictly below the average
simultaneously each round, until all remaining candidates have equal Borda
scores (a mutual tie in pairwise duels).

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_nanson():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Nanson."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return NansonResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("NANS", _build_nanson())

if __name__ == "__main__":
    # Case 1 — clear winner: A dominates in Borda every round
    result1 = NansonResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 1 — clear winner:")
    print("  scores_:", result1.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — 3-way tie: cyclic preferences, all below average simultaneously
    result2 = NansonResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            {"A", "B", "C"},
        )
    )
    print("Case 2 — 3-way tie:")
    print("  scores_:", result2.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — C eliminated, A and B survive tied
    result3 = NansonResult(
        _ensure_profile(
            [[2, 1, 0], [2, 1, 0], [1, 2, 0], [1, 2, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 3 — 2-way tie after elimination:")
    print("  scores_:", result3.scores_)
    print("  cowinners_:", result3.cowinners_)
