"""Tideman method wrapper with semantically correct co-winner detection.

Tideman's rule restricts the election to the Smith Set, then eliminates the
plurality loser, and iterates until a winner is found.

``scores_[r, c]`` is ``1`` if candidate ``c`` survived elimination round ``r``,
``0`` if already eliminated.

Co-winners are all candidates with a nonzero score in the **final** round, i.e.
all candidates that were never eliminated before the process stopped.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile
from svvamp.rules.rule_tideman import RuleTideman

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class TidemanResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleTideman` with proper co-winner semantics.

    Co-winners are all candidates surviving to the final elimination round,
    i.e. all ``c`` with ``scores_[-1, c] != 0``.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Tideman.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleTideman()(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = self._inner.scores_
        last_round = scores[-1, :]
        # Use > 0: NaN (eliminated) compares False, 0 (eliminated) compares False.
        survivors_last = np.where(last_round > 0)[0]

        # If only one candidate survived the last round but the penultimate round
        # had multiple survivors with identical scores, ctb broke a true tie →
        # return all penultimate survivors as co-winners.
        if len(survivors_last) == 1 and len(scores) >= 2:
            penult = scores[-2, :]
            survivors_penult = np.where(penult > 0)[0]
            if len(survivors_penult) > 1:
                vals = penult[survivors_penult]
                if np.all(vals == vals[0]):
                    return self._resolve_cowinners(survivors_penult)

        return self._resolve_cowinners(survivors_last)



def _build_tideman():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Tideman."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return TidemanResult(profile)

    return builder


register_rule("TIDE", _build_tideman())
