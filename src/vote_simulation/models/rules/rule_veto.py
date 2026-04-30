"""Veto (Antiplurality) wrapper with semantically correct co-winner detection.

Each voter vetoes their least-liked candidate.  The candidate receiving the
fewest vetos wins.

``scores_[c]`` = minus the number of vetos against candidate ``c`` (1-D
integer array, higher = better).  Co-winners are all candidates sharing the
maximum score (i.e. the minimum veto count).
"""

from __future__ import annotations

from svvamp import Profile
from svvamp.rules.rule_veto import RuleVeto

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class VetoResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleVeto` with proper co-winner semantics.

    Co-winners are all candidates sharing the minimum veto count (maximum
    ``scores_`` value).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Veto.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleVeto()(profile)
        self.cowinners_ = self._init_score_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_veto():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Veto."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return VetoResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("VETO", _build_veto())
register_rule("APLUS", _build_veto())  # alias

if __name__ == "__main__":
    # Case 1 — clear winner (A vetoed fewest times)
    result1 = VetoResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 1 — clear winner:   scores_: {result1._inner.scores_}   cowinners_: {result1.cowinners_}")

    # Case 2 — 3-way tie (each voter vetoes a different candidate)
    result2 = VetoResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 2 — 3-way tie:      scores_: {result2._inner.scores_}   cowinners_: {result2.cowinners_}")

    # Case 3 — 2-way tie (A and B each vetoed once, C vetoed once)
    result3 = VetoResult(
        _ensure_profile(
            [[2, 1, 0], [1, 2, 0], [2, 1, 0], [1, 2, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 3 — 2-way tie:      scores_: {result3._inner.scores_}   cowinners_: {result3.cowinners_}")
