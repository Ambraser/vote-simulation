"""Split Cycle wrapper with semantically correct co-winner detection.

Split Cycle removes the weakest defeat in each cycle and declares winners
as all candidates with no remaining defeats.

``scores_[c]`` is the number of Split Cycle victories of candidate ``c``
(1-D integer array, higher = better).  Co-winners are all candidates sharing
the maximum score.
"""

from __future__ import annotations

from svvamp import Profile
from svvamp.rules.rule_split_cycle import RuleSplitCycle

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class SplitCycleResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleSplitCycle` with proper co-winner semantics.

    Co-winners are all candidates sharing the maximum Split Cycle score.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Split Cycle.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleSplitCycle()(profile)
        self.cowinners_ = self._init_score_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_split_cycle():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Split Cycle."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return SplitCycleResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("SPCY", _build_split_cycle())

if __name__ == "__main__":
    # Case 1 — clear Condorcet winner
    result1 = SplitCycleResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 1 — clear winner:   scores_: {result1._inner.scores_}   cowinners_: {result1.cowinners_}")

    # Case 2 — 3-way Condorcet cycle → all candidates tied
    result2 = SplitCycleResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 2 — 3-way cycle:    scores_: {result2._inner.scores_}   cowinners_: {result2.cowinners_}")

    # Case 3 — 2-way tie (A and B both beat C, draw between themselves)
    result3 = SplitCycleResult(
        _ensure_profile(
            [[2, 1, 0], [1, 2, 0], [2, 1, 0], [1, 2, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 3 — 2-way tie:      scores_: {result3._inner.scores_}   cowinners_: {result3.cowinners_}")
