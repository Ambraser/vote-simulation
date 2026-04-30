"""Plurality method wrapper with semantically correct co-winner detection.

Each voter votes for their top-ranked candidate.  Co-winners are all
candidates sharing the **maximum number of first-place votes**.

``scores_`` is a 1-D integer array where ``scores_[c]`` is the number of
first-place votes for candidate ``c``.
"""

from __future__ import annotations

from svvamp import Profile, RulePlurality

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class PluralityResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RulePlurality` with proper co-winner semantics.

    Co-winners are all candidates sharing the maximum first-place vote count.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Plurality.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RulePlurality()(profile)
        self.cowinners_ = self._init_score_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_plurality():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Plurality."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return PluralityResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("PLU1", _build_plurality())

if __name__ == "__main__":
    # Case 1 — clear winner
    result1 = PluralityResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 1 — clear winner:")
    print("  scores_:", result1._inner.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — 3-way tie
    result2 = PluralityResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            {"A", "B", "C"},
        )
    )
    print("Case 2 — 3-way tie:")
    print("  scores_:", result2._inner.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — 2-way tie
    result3 = PluralityResult(
        _ensure_profile(
            [[2, 1, 0], [2, 1, 0], [1, 2, 0], [1, 2, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 3 — 2-way tie:")
    print("  scores_:", result3._inner.scores_)
    print("  cowinners_:", result3.cowinners_)
