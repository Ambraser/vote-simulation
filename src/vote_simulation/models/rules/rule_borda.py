"""Borda rule wrapper with co-winner detection.

Co-winners are all candidates sharing the **maximum Borda score**
(total points received across all voters), regardless of the index-based
tie-break used internally by svvamp to resolve ``w_``.

``scores_`` is a 1-D integer array — handled by :class:`ScoreBasedRuleWrapper`.
"""

from __future__ import annotations

from svvamp import Profile, RuleBorda

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class BordaResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleBorda` with proper co-winner semantics.

    Co-winners are **all** candidates that share the maximum total Borda score.
    The lowest-index tie-break used internally by svvamp to produce a single
    winner ``w_`` does *not* define the set of co-winners.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Borda.
    cm_option:
        Coalition-manipulation option. ``'fast'`` or ``'exact'``.
        Defaults to ``'fast'``.

    Attributes
    ----------
    cowinners_:
        List of candidate labels that tied at the top Borda score.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        cm_option: str = "fast",
    ) -> None:
        self.profile_ = profile
        self._inner = RuleBorda(cm_option=cm_option)(profile)
        self.cowinners_ = self._init_score_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_borda(*, cm_option: str = "fast"):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Borda.

    Parameters
    ----------
    cm_option:
        ``'fast'`` or ``'exact'``. Defaults to ``'fast'``.
    """

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return BordaResult(profile, cm_option=cm_option)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("BORD", _build_borda(cm_option="fast"))
register_rule("BORD_EXACT", _build_borda(cm_option="exact"))



if __name__ == "__main__":
    import numpy as np
    from svvamp import Profile

    # Case 1: clear winner
    p1 = Profile(
        preferences_rk=np.array([[0, 1, 2], [0, 1, 2], [0, 2, 1]]),
        labels_candidates=["A", "B", "C"],
    )
    r1 = BordaResult(p1)
    print("Case 1 — clear winner:")
    print("  scores_   :", r1._inner.scores_)
    print("  cowinners_:", r1.cowinners_)  # expected: ['A']

    # Case 2: three-way Borda tie (Condorcet cycle → equal Borda scores)
    p2 = Profile(
        preferences_rk=np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
        labels_candidates=["A", "B", "C"],
    )
    r2 = BordaResult(p2)
    print("\nCase 2 — three-way Borda tie (Condorcet cycle):")
    print("  scores_   :", r2._inner.scores_)
    print("  cowinners_:", r2.cowinners_)  # expected: ['A', 'B', 'C']
