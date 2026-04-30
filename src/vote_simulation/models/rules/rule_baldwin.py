"""Baldwin method wrapper with semantically correct co-winner detection.

Baldwin eliminates the candidate with the lowest Borda score each round
(ties broken by highest index).  Co-winners are **all candidates that
survive to the final elimination round** — those whose ``scores_[-1, c]``
is finite.

``scores_`` is a 2-D array ``[round, candidate]``; eliminated candidates
carry ``numpy.inf`` from the round they were removed onward.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleBaldwin

from vote_simulation.models.rules.elimination_based import EliminationBasedRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class BaldwinResult(EliminationBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleBaldwin` with proper co-winner semantics.

    Co-winners are all candidates whose Borda score is finite in the last
    elimination round (i.e. they were not eliminated before the end).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Baldwin.
    cm_option:
        ``'fast'`` or ``'exact'``. Defaults to ``'exact'``.
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
        cm_option: str = "exact",
        im_option: str = "lazy",
        um_option: str = "lazy",
        tm_option: str = "exact",
    ) -> None:
        self.profile_ = profile
        self._inner = RuleBaldwin(
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


def _build_baldwin(
    *,
    cm_option: str = "exact",
    im_option: str = "lazy",
    um_option: str = "lazy",
    tm_option: str = "exact",
):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Baldwin."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return BaldwinResult(
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

register_rule("BALD", _build_baldwin(cm_option="exact"))
register_rule("BALD_FAST", _build_baldwin(cm_option="fast"))

if __name__ == "__main__":
    # Quick test: run Baldwin on the reference profile and check co-winners.
    ballot = [
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ]

    #if isinstance(ballot, list):
    #    ballot = np.array(ballot, dtype=np.float64)
    

    profile = _ensure_profile(ballot, {"A", "B", "C"})
    #profile.demo()
    result = BaldwinResult(_ensure_profile(ballot, {"A", "B", "C"}))
    print(result.scores_)
    print(result.candidates_worst_to_best_)
    print(result.cowinners_)
#    .demo()
#    assert sorted(result.cowinners_) == sorted(EXPECTED_REFERENCE_COWINNERS[2][1])