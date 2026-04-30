"""Iterated Bucklin method wrapper with semantically correct co-winner detection.

Iterated Bucklin eliminates the candidate with the lowest adjusted median
Borda score each round (ties broken by highest index).  Co-winners are **all
candidates that survive to the final elimination round** — those whose
``scores_[-1, c]`` is finite.

``scores_`` is a 2-D array ``[round, candidate]``; eliminated candidates
carry ``numpy.inf`` from the round they were removed onward.
"""

from __future__ import annotations

from svvamp import Profile, RuleIteratedBucklin

from vote_simulation.models.rules.elimination_based import EliminationBasedRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class IteratedBucklinResult(EliminationBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleIteratedBucklin` with proper co-winner semantics.

    Co-winners are all candidates whose adjusted median Borda score is finite
    in the last elimination round (i.e. they were not eliminated before the
    end).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Iterated Bucklin.
    cm_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'lazy'``.
    im_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'lazy'``.
    um_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'lazy'``.
    tm_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'exact'``.

    Attributes
    ----------
    cowinners_:
        Labels of all candidates surviving to the final round.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        cm_option: str = "lazy",
        im_option: str = "lazy",
        um_option: str = "lazy",
        tm_option: str = "exact",
    ) -> None:
        self.profile_ = profile
        self._inner = RuleIteratedBucklin(
            cm_option=cm_option,
            im_option=im_option,
            um_option=um_option,
            tm_option=tm_option,
        )(profile)
        self.scores_ = self._inner.scores_
        self.candidates_worst_to_best_ = self._inner.candidates_by_scores_best_to_worst_
        self.cowinners_ = self._init_elimination_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_iterated_bucklin(
    *,
    cm_option: str = "lazy",
    im_option: str = "lazy",
    um_option: str = "lazy",
    tm_option: str = "exact",
):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Iterated Bucklin."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return IteratedBucklinResult(
            profile,
            cm_option=cm_option,
            im_option=im_option,
            um_option=um_option,
            tm_option=tm_option,
        )

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("BUCK_I", _build_iterated_bucklin())
register_rule("BUCK_I_EXACT", _build_iterated_bucklin(cm_option="exact"))

if __name__ == "__main__":
    import numpy as np

    # Case 1 — clear winner (candidate 0 has highest adjusted median every round)
    result1 = IteratedBucklinResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 1 — clear winner:")
    print("  scores_:", result1.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — 3-way tie (all candidates equal adjusted median every round)
    result2 = IteratedBucklinResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            {"A", "B", "C"},
        )
    )
    print("Case 2 — 3-way tie:")
    print("  scores_:", result2.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — two survive (third eliminated, remaining two tied)
    result3 = IteratedBucklinResult(
        _ensure_profile(
            [[2, 1, 0], [2, 1, 0], [1, 2, 0], [1, 2, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 3 — two survivors:")
    print("  scores_:", result3.scores_)
    print("  cowinners_:", result3.cowinners_)
