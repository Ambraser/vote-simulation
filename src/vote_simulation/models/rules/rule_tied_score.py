"""Copeland rule wrapper with semantically correct co-winner detection.

Co-winners are all candidates sharing the **maximum Copeland score**
(i.e. number of pairwise victories), regardless of the ``tie_break_rule``
used internally by svvamp to resolve ``w_``.
"""

from __future__ import annotations
from collections.abc import Callable

from svvamp import Profile, RuleCopeland

from vote_simulation.models.rules.base import SvvampRuleWrapper


class RulesResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleCopeland` with proper co-winner semantics.

    Co-winners are **all** candidates that share the maximum Copeland score
    (number of pairwise victories).  The ``tie_break_rule='lexico'`` is kept
    internally by svvamp only to produce a deterministic ``w_`` for
    manipulation computations — it does *not* define the set of co-winners.

    Attributes
    ----------
    cowinners_:
        List of candidate labels that tied at the top Copeland score.
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
        self._inner = RuleCopeland(
            tie_break_rule="lexico",  # stable, deterministic — does not affect co-winners
            cm_option=cm_option,
            im_option=im_option,
            um_option=um_option,
            tm_option=tm_option,
        )(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        """Return all candidates whose utility-based Copeland score equals the maximum.

        Svvamp's ``scores_`` is derived from ``matrix_victories_rk`` (rank-based),
        which randomises in case of utility ties.  Instead we build Copeland scores
        from ``matrix_victories_ut_abs``:

        * ``matrix_victories_ut_abs[c, d]`` = number of voters with
          ``ut[c] > ut[d]`` (strict utility preference).

        For each pair (c, d):
        * c gets **+1** if more voters prefer c over d than d over c (strict win).
        * c gets **+0.5** for an exact majority tie.
        * c gets **0** if c loses.

        This is the correct utility-consistent Copeland rule and guarantees that
        profiles with equal utilities produce equal scores — and thus correct
        co-winner sets — regardless of rank-breaking used internally by svvamp.
        """

        return self._max_score_cowinners(self._inner.scores_)
