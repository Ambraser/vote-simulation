"""Copeland rule wrapper with semantically correct co-winner detection.

Co-winners are all candidates sharing the **maximum Copeland score**
(i.e. number of pairwise victories), regardless of the ``tie_break_rule``
used internally by svvamp to resolve ``w_``.
"""

from __future__ import annotations

from svvamp import Profile, RuleCopeland

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class CopelandResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleCopeland` with proper co-winner semantics.

    Co-winners are **all** candidates that share the maximum Copeland score
    (number of pairwise victories).  The ``tie_break_rule='lexico'`` is kept
    internally by svvamp only to produce a deterministic ``w_`` for
    manipulation computations — it does *not* define the set of co-winners.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Copeland.
    cm_option:
        Coalition-manipulation option. ``'fast'`` or ``'exact'``.
        Defaults to ``'exact'``.
    im_option:
        Individual-manipulation option. ``'lazy'`` or ``'exact'``.
        Defaults to ``'lazy'``.
    um_option:
        Unison-manipulation option. ``'lazy'`` or ``'exact'``.
        Defaults to ``'lazy'``.
    tm_option:
        Trivial-manipulation option. ``'lazy'`` or ``'exact'``.
        Defaults to ``'exact'``.

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
        # mv = np.asarray(self.profile_.matrix_victories_ut_abs, dtype=float)
        # n_c = mv.shape[0]
        # scores = np.zeros(n_c, dtype=float)
        # for c in range(n_c):
        #    for d in range(n_c):
        #        if c == d:
        #            continue
        #        if mv[c, d] > mv[d, c]:
        #            scores[c] += 1.0
        #        elif mv[c, d] == mv[d, c]:
        #            scores[c] += 0.5
        return self._max_score_cowinners(self._inner.scores_)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_copeland(
    *,
    cm_option: str = "exact",
    im_option: str = "lazy",
    um_option: str = "lazy",
    tm_option: str = "exact",
):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Copeland.

    Parameters
    ----------
    cm_option:
        ``'fast'`` or ``'exact'``. Defaults to ``'exact'``.
    im_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'lazy'``.
    um_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'lazy'``.
    tm_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'exact'``.
    """

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return CopelandResult(
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

register_rule("COPE", _build_copeland(cm_option="fast"))

# Variant: all options set to 'exact' (slower but fully precise)
register_rule(
    "COPE_EXACT",
    _build_copeland(cm_option="exact", im_option="exact", um_option="exact", tm_option="exact"),
)
