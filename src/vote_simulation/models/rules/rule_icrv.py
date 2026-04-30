"""ICRV (Instant-Condorcet Runoff Voting / Benham rule) wrapper.

Co-winners are **all** candidates tied at the maximum victories score in the
final deciding even round, where the rule stops because a Condorcet winner
(or tied survivors) is detected.

``scores_[r, c]`` alternates:
  - even ``r``: number of victories for ``c`` in ``matrix_victories_rk``
    restricted to non-eliminated candidates (ties = 0.5).
  - odd ``r``: number of first-place votes for ``c`` (IRV elimination round).

``nan`` marks eliminated candidates.  The rule always terminates on an even
round, so the last row is the deciding Condorcet-check round.  All candidates
tied at the maximum score in that row are co-winners.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile
from svvamp.rules.rule_icrv import RuleICRV

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class ICRVResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleICRV` with proper co-winner semantics.

    Co-winners are all candidates tied at the maximum victories score in the
    last (deciding) even round.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run ICRV.
    cm_option:
        Coalition-manipulation option. ``'fast'``, ``'slow'``, ``'very_slow'``
        or ``'exact'``. Defaults to ``'fast'``.

    Attributes
    ----------
    cowinners_:
        List of candidate labels tied at the deciding-round maximum.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(self, profile: Profile, *, cm_option: str = "fast") -> None:
        self.profile_ = profile
        self._inner = RuleICRV(cm_option=cm_option)(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = np.asarray(self._inner.scores_, dtype=float)  # shape [n_rounds, n_c]
        # The rule always terminates on an even round (Condorcet check).
        # Find the last even-indexed row — that is the deciding round.
        n_rounds = scores.shape[0]
        deciding_row = None
        for r in range(n_rounds - 1, -1, -1):
            if r % 2 == 0:
                deciding_row = scores[r, :]
                break

        if deciding_row is None:
            deciding_row = scores[-1, :]

        row = np.where(np.isnan(deciding_row), -np.inf, deciding_row)
        best = np.max(row)
        return self._resolve_cowinners(np.flatnonzero(row == best))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_icrv(*, cm_option: str = "fast"):
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return ICRVResult(profile, cm_option=cm_option)

    return builder


# Rule registrations

register_rule("ICRV", _build_icrv(cm_option="fast"))
register_rule("ICRV_EXACT", _build_icrv(cm_option="exact"))



if __name__ == "__main__":
    from vote_simulation.models.rules.registry import _ensure_profile

    # Case 1: A is top for all voters → Condorcet winner at round 0
    # scores: A=2, B=1, C=0 / A=2, B=0, C=1 / A=2, B=1, C=0
    p1 = _ensure_profile(
        [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
        candidates={"A", "B", "C"},
    )
    r1 = ICRVResult(p1)
    print("Case 1 — Condorcet winner at round 0:")
    print("  scores_:\n", r1._inner.scores_)
    print("  cowinners_:", r1.cowinners_)  # expected: ['A']

    # Case 2: Condorcet cycle (A>B>C>A) → no Condorcet winner at round 0,
    # IRV eliminates one, then Condorcet check resolves.
    # Utilities: A=[2,1,0], B=[1,0,2], C=[0,2,1]
    p2 = _ensure_profile(
        [[2, 1, 0], [1, 0, 2], [0, 2, 1]],
        candidates={"A", "B", "C"},
    )
    r2 = ICRVResult(p2)
    print("\nCase 2 Condorcet cycle, IRV decides:")
    print("  scores_:\n", r2._inner.scores_)
    print("  cowinners_:", r2.cowinners_)

    # Case 3: 4-candidate doc example — A, B, C form Condorcet cycle, D ranked last by all.
    # Voter 1: A>B>C>D [3,2,1,0], Voter 2: B>C>A>D [2,1,3,0], Voter 3: C>A>B>D [1,3,2,0]
    p3 = _ensure_profile(
        [[3, 2, 1, 0], [2, 1, 3, 0], [1, 3, 2, 0]],
        candidates={"A", "B", "C", "D"},
    )
    r3 = ICRVResult(p3)
    print("\nCase 3 doc example (4 candidates, D eliminated first):")
    print("  scores_:\n", r3._inner.scores_)
    print("  cowinners_:", r3.cowinners_)
