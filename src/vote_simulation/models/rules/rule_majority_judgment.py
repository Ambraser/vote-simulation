"""Majority Judgment wrapper with semantically correct co-winner detection.

Majority Judgment elects the candidate with the highest median grade.
Ties on the median are broken by the adjusted majority grade: if more voters
grade candidate ``c`` above the median than below, the tie-breaker is ``+p``
(number above); otherwise it is ``-q`` (number below).

``scores_`` is a 2-D array ``[2, n_c]``:

- ``scores_[0, c]`` — median grade of candidate ``c``.
- ``scores_[1, c]`` — ``p`` if ``p > q``, else ``-q``.

Co-winners are **all** candidates that share the **lexicographically maximum**
pair ``(scores_[0, c], scores_[1, c])``.

Note: grades are set to ``rescale_grades=False`` so that ``preferences_ut``
values are used directly (clipped to ``[min_grade, max_grade]``), ensuring
consistent results across profiles.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleMajorityJudgment

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


def _grade_bounds(profile: Profile) -> tuple[float, float]:
    """Return (min, max) of preferences_ut for the profile."""
    return float(np.min(profile.preferences_ut)), float(np.max(profile.preferences_ut))


class MajorityJudgmentResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleMajorityJudgment` with proper co-winner semantics.

    Co-winners are all candidates sharing the lexicographically maximum
    ``(median_grade, tie_breaker)`` pair.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Majority Judgment.
    min_grade:
        Lower bound for grades. Defaults to ``min(preferences_ut)``.
    max_grade:
        Upper bound for grades. Defaults to ``max(preferences_ut)``.
    rescale_grades:
        Whether sincere voters rescale utilities to fill the grade range.
        Defaults to ``False`` (clip instead of rescale).
    """

    def __init__(
        self,
        profile: Profile,
        *,
        min_grade: float | None = None,
        max_grade: float | None = None,
        rescale_grades: bool = False,
    ) -> None:
        self.profile_ = profile
        lo, hi = _grade_bounds(profile)
        self._inner = RuleMajorityJudgment(
            min_grade=lo if min_grade is None else min_grade,
            max_grade=hi if max_grade is None else max_grade,
            rescale_grades=rescale_grades,
        )(profile)
        self.scores_ = self._inner.scores_
        self.cowinners_ = self._compute_mj_cowinners()

    def _compute_mj_cowinners(self) -> list[str]:
        """Return labels of all candidates at the lexicographic top."""
        scores = self._inner.scores_          # shape [2, n_c]
        medians = scores[0, :]                # row 0: median grades
        tiebreaks = scores[1, :]              # row 1: +p or -q

        max_med = np.max(medians)
        med_mask = medians == max_med
        max_tb = np.max(tiebreaks[med_mask])
        winners_mask = med_mask & (tiebreaks == max_tb)

        return self._labels_for(np.flatnonzero(winners_mask))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_majority_judgment(
    *,
    min_grade: float | None = None,
    max_grade: float | None = None,
    rescale_grades: bool = False,
):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for MJ."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return MajorityJudgmentResult(
            profile,
            min_grade=min_grade,
            max_grade=max_grade,
            rescale_grades=rescale_grades,
        )

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("MJ", _build_majority_judgment())
register_rule("MJ_RESCALE", _build_majority_judgment(rescale_grades=True))

if __name__ == "__main__":
    # Case 1 — clear winner: A receives highest grades from everyone
    result1 = MajorityJudgmentResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 1 — clear winner:")
    print("  scores_:", result1.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — 3-way tie at median and tie-breaker
    result2 = MajorityJudgmentResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            {"A", "B", "C"},
        )
    )
    print("Case 2 — 3-way tie:")
    print("  scores_:", result2.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — 2-way tie: A and B have equal median and tie-breaker, C lower
    result3 = MajorityJudgmentResult(
        _ensure_profile(
            [[2, 1, 0], [2, 1, 0], [1, 2, 0], [1, 2, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 3 — 2-way tie:")
    print("  scores_:", result3.scores_)
    print("  cowinners_:", result3.cowinners_)
