"""Two-Round System wrapper with semantically correct co-winner detection.

Voters cast ballots in two rounds.  The two candidates with the most
first-round votes advance to the second round; the one preferred by more
voters in the second round wins.

``scores_[0, c]`` = votes for candidate ``c`` in round 1.
``scores_[1, c]`` = votes for candidate ``c`` in round 2 (0 if not a finalist).

Co-winner semantics
-------------------
*Finalists* are all candidates whose round-0 score is at least as large as
the value at the 2nd slot of the sorted round-0 scores (ties at the cutoff
are included).  Among finalists, co-winners share the maximum round-1 score.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleTwoRound

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class TwoRoundResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleTwoRound` with proper co-winner semantics.

    Co-winners are all candidates in the extended finalist set that share the
    maximum second-round vote count.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run the Two-Round System.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleTwoRound()(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = self._inner.scores_  # shape [2, n_c]
        round0 = scores[0, :]  # first-round vote counts
        round1 = scores[1, :]  # second-round vote counts (0 for non-finalists)

        # If no second round was held (e.g. one candidate got a majority),
        # co-winners are all candidates tied at the maximum first-round score.
        if np.max(round1) == 0:
            return self._resolve_cowinners(np.where(round0 == np.max(round0))[0])

        # Extended finalist set: all candidates tied at the top-2 score positions.
        sorted_scores = np.sort(round0)[::-1]
        cutoff = sorted_scores[1]  # value at 2nd slot (may equal 1st if tied)
        finalist_mask = round0 >= cutoff

        # Among finalists, co-winners share the maximum second-round score.
        finalist_runoff = np.where(finalist_mask, round1, -np.inf)
        max_runoff = np.max(finalist_runoff)
        cowin_mask = finalist_mask & (round1 == max_runoff)

        return self._resolve_cowinners(np.where(cowin_mask)[0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_two_round():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Two-Round."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return TwoRoundResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("PLU2", _build_two_round())

if __name__ == "__main__":
    # Case 1 — clear winner (A top first round, wins second round)
    result1 = TwoRoundResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print("Case 1 — clear winner:")
    print("  scores_:\n", result1._inner.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — second-round tie between top-2 finalists
    result2 = TwoRoundResult(
        _ensure_profile(
            [[2, 1, 0], [1, 2, 0], [2, 1, 0], [1, 2, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print("Case 2 — runoff tie:")
    print("  scores_:\n", result2._inner.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — first-round tie for 2nd place (A, B, C all tied → all are finalists)
    result3 = TwoRoundResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            candidates={"A", "B", "C"},
        )
    )
    print("Case 3 — 3-way first-round tie:")
    print("  scores_:\n", result3._inner.scores_)
    print("  cowinners_:", result3.cowinners_)
