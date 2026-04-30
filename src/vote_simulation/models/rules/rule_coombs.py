"""Coombs rule wrapper with correct co-winner detection.

Co-winners are **all** candidates that survive to the final elimination round
and share the maximum score (least negative = fewest last-place votes) in that
round. 

``scores_[r, c]`` = minus the number of voters who rank ``c`` last among
remaining candidates at round ``r``.  Eliminated candidates carry ``nan``.
Co-winners are the argmax of the last row after masking ``nan`` as ``-inf``.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleCoombs

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class CoombsResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleCoombs` with proper co-winner semantics.

    Co-winners are all candidates still active in the final round that share
    the maximum (least negative) score, i.e. the fewest last-place votes.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Coombs.
    cm_option:
        Coalition-manipulation option. ``'fast'`` or ``'exact'``.
        Defaults to ``'fast'``.

    Attributes
    ----------
    cowinners_:
        List of candidate labels that tied at the last-round maximum score.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(self, profile: Profile, *, cm_option: str = "fast") -> None:
        self.profile_ = profile
        self._inner = RuleCoombs(cm_option=cm_option)(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = np.asarray(self._inner.scores_, dtype=float)  # shape [n_rounds, n_c]
        for r in range(scores.shape[0]):
            row = scores[r, :]
            survivors = np.flatnonzero(~np.isnan(row))
            if survivors.size == 0:
                break
            survivor_scores = row[survivors]
            # If all survivors share the same score, they are tied no one can be
            # distinguished for elimination, so they are all co-winners.
            if np.all(survivor_scores == survivor_scores[0]):
                return self._resolve_cowinners(survivors)
        # Fallback: argmax in last row (should not happen in practice)
        last = scores[-1, :]
        last = np.where(np.isnan(last), -np.inf, last)
        return self._resolve_cowinners(np.flatnonzero(last == np.max(last)))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_coombs(*, cm_option: str = "fast"):
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return CoombsResult(profile, cm_option=cm_option)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("COOM", _build_coombs(cm_option="fast"))
register_rule("COOM_EXACT", _build_coombs(cm_option="exact"))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Case 1: clear winner
    p1 = Profile(
        preferences_rk=np.array([[0, 1, 2], [0, 1, 2], [0, 2, 1]]),
        labels_candidates=["A", "B", "C"],
    )
    r1 = CoombsResult(p1)
    print("Case 1 — clear winner:")
    print("  scores_:\n", r1._inner.scores_)
    print("  cowinners_:", r1.cowinners_)  # expected: ['A']

    # Case 2: two candidates tie in the final round
    # 4 voters: A and B each get 2 last-place votes in round 0 C eliminated by CTB (highest index)
    # Round 1: A and B remain, equal last-place votes  tie
    p2 = Profile(
        preferences_rk=np.array([[0, 1, 2], [1, 0, 2], [0, 2, 1], [1, 2, 0]]),
        labels_candidates=["A", "B", "C"],
    )
    r2 = CoombsResult(p2)
    print("\nCase 2 — tie in final round:")
    print("  scores_:\n", r2._inner.scores_)
    print("  cowinners_:", r2.cowinners_)  # expected: ['A', 'B']
