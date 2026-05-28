"""STAR (Score Then Automatic Runoff) wrapper.

STAR proceeds in two stages:

1. **Score stage**: each voter grades every candidate.  The two candidates
   with the highest total grades advance to the automatic runoff.
2. **Runoff stage**: the finalist preferred by more voters wins.

``scores_[0, c]`` is the total grade for candidate ``c``.
``scores_[1, c]`` is the number of voters who prefer ``c`` in the runoff
(``0`` if ``c`` was not selected as a finalist).

Grade bounds are derived automatically from the profile's utility matrix so
that the clip transformation (``rescale_grades=False``) is consistent with
the actual utility range.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile
from svvamp.rules.rule_star import RuleSTAR

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


def _grade_bounds(profile: Profile) -> tuple[float, float]:
    return float(np.min(profile.preferences_ut)), float(np.max(profile.preferences_ut))


class STARResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleSTAR` with proper co-winner semantics.

    Co-winners are all candidates in the extended finalist set that share the
    maximum runoff vote count.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run STAR.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        lo, hi = _grade_bounds(profile)
        self._inner = RuleSTAR(min_grade=lo, max_grade=hi, rescale_grades=False)(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = self._inner.scores_  # shape [2, n_c]
        round0 = scores[0, :]  # total grades (score stage)
        round1 = scores[1, :]  # runoff votes (0 for non-finalists)

        # Extended finalist set: all candidates tied at the top-2 score positions.
        # Using sorted_scores[1] (not unique) correctly handles ties at position 2
        # without pulling in candidates ranked 3rd or lower.
        sorted_scores = np.sort(round0)[::-1]
        cutoff = sorted_scores[1]  # value at the 2nd slot (may equal 1st if tied)
        finalist_mask = round0 >= cutoff

        # Among the extended finalist set, co-winners share the max runoff score.
        finalist_runoff = np.where(finalist_mask, round1, -np.inf)
        max_runoff = np.max(finalist_runoff)
        cowin_mask = finalist_mask & (round1 == max_runoff)

        return self._resolve_cowinners(np.where(cowin_mask)[0])



def _build_star():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for STAR."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return STARResult(profile)

    return builder

register_rule("STAR", _build_star())
