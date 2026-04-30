"""Woodall rule wrapper.

Co-winners are **all** candidates tied at the first counting round where no
unique elimination is possible — i.e. all surviving candidates share the same
first-place vote count.  The lowest-index tie-break used internally by svvamp
does *not* define the set of co-winners.

``scores_[r, c]`` = number of voters who rank ``c`` first among non-eliminated
candidates at round ``r``.  ``nan`` marks eliminated candidates.

Woodall restricts the IRV elimination to candidates in the Smith set, then
elects the one eliminated latest.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile
from svvamp.rules.rule_woodall import RuleWoodall

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class WoodallResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleWoodall` with proper co-winner semantics.

    Co-winners are all surviving candidates at the first round where they all
    share the same first-place vote count (no one can be unambiguously
    eliminated).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Woodall.

    Attributes
    ----------
    cowinners_:
        List of candidate labels that tied at the deciding-round maximum.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleWoodall()(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = np.asarray(self._inner.scores_, dtype=float)  # shape [n_rounds, n_c]
        for r in range(scores.shape[0]):
            row = scores[r, :]
            survivors = np.flatnonzero(~np.isnan(row))
            if survivors.size == 0:
                break
            survivor_scores = row[survivors]
            # If all survivors share the same first-place count, no one can be
            # eliminated unambiguously — they are all co-winners.
            if np.all(survivor_scores == survivor_scores[0]):
                return self._resolve_cowinners(survivors)
        # Fallback: last-round argmax (should not happen in practice)
        last = scores[-1, :]
        last = np.where(np.isnan(last), -np.inf, last)
        return self._resolve_cowinners(np.flatnonzero(last == np.max(last)))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_woodall():
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return WoodallResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("WOOD", _build_woodall())


if __name__ == "__main__":
    from vote_simulation.models.rules.registry import _ensure_profile

    # Case 1: clear winner — A gets majority first-place votes immediately
    p1 = _ensure_profile(
        [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
        candidates={"A", "B", "C"},
    )
    r1 = WoodallResult(p1)
    print("Case 1 — clear majority winner:")
    print("  scores_:\n", r1._inner.scores_)
    print("  cowinners_:", r1.cowinners_)  # expected: ['A']

    # Case 2: 3-way first-place tie at round 0 → all co-winners
    p2 = _ensure_profile(
        [[2, 1, 0], [1, 0, 2], [0, 2, 1]],
        candidates={"A", "B", "C"},
    )
    r2 = WoodallResult(p2)
    print("\nCase 2 — 3-way tie at round 0:")
    print("  scores_:\n", r2._inner.scores_)
    print("  cowinners_:", r2.cowinners_)  # expected: ['A', 'B', 'C']

    # Case 3: elimination then tie — A and B survive with equal counts
    p3 = _ensure_profile(
        [[2, 1, 0], [2, 0, 1], [1, 2, 0], [0, 2, 1]],
        candidates={"A", "B", "C"},
    )
    r3 = WoodallResult(p3)
    print("\nCase 3 — elimination then 2-way tie:")
    print("  scores_:\n", r3._inner.scores_)
    print("  cowinners_:", r3.cowinners_)  # expected: ['A', 'B']
