"""Black rule wrapper with semantically correct co-winner detection.

Black's algorithm:
1. If a (strict) Condorcet winner exists → elect them (unique by definition).
2. Otherwise → elect the candidate(s) with the highest Borda score.

``scores_`` layout (2-D, shape ``[2, n_candidates]``):

* ``scores_[0, c]`` — ``1.0`` if ``c`` is the Condorcet winner, else ``0.0``.
* ``scores_[1, c]`` — Borda score of ``c``.

Co-winner semantics
--------------------
* **Condorcet path** — row 0 has exactly one ``1``: that candidate is the
  sole co-winner.
* **Borda fallback** — row 0 is all zeros (no Condorcet winner): co-winners
  are all candidates tied at ``max(scores_[1, :])``.

Note: a Condorcet winner is by definition unique, so the Condorcet path
never produces more than one co-winner.  A Borda tie in the fallback path
*can* produce multiple co-winners.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleBlack

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class BlackResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleBlack` with proper co-winner semantics.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Black.
    cm_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'exact'``.
    im_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'lazy'``.
    um_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'lazy'``.
    tm_option:
        ``'lazy'`` or ``'exact'``. Defaults to ``'exact'``.

    Attributes
    ----------
    cowinners_:
        * The Condorcet winner (singleton list) when one exists.
        * All Borda-score co-winners otherwise.
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
        self._inner = RuleBlack(
            cm_option=cm_option,
            im_option=im_option,
            um_option=um_option,
            tm_option=tm_option,
        )(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        """Return co-winners according to Black's two-stage logic.

        * Stage 1 — Condorcet row (``scores_[0, :]``): if any candidate has
          score ``1``, they are the unique Condorcet winner → return them.
        * Stage 2 — Borda row (``scores_[1, :]``): return all candidates
          tied at the maximum Borda score.
        """
        scores = np.asarray(self._inner.scores_, dtype=float)  # shape [2, n_c]

        # Stage 1: Condorcet winner check
        condorcet_row = scores[0, :]
        condorcet_indices = np.flatnonzero(condorcet_row == 1.0)
        if condorcet_indices.size > 0:
            return self._resolve_cowinners(condorcet_indices)

        # Stage 2: Borda fallback — all candidates tied at max Borda score
        borda_row = scores[1, :]
        return self._max_score_cowinners(borda_row)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_black(
    *,
    cm_option: str = "exact",
    im_option: str = "lazy",
    um_option: str = "lazy",
    tm_option: str = "exact",
):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Black."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return BlackResult(
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

register_rule("BLAC", _build_black(cm_option="exact"))
register_rule("BLAC_LAZY", _build_black(cm_option="lazy"))

if __name__ == "__main__":
    from vote_simulation.models.rules import get_rule_builder

    print('MRO:', [c.__name__ for c in BlackResult.__mro__])
    print()

    # Case 1: Condorcet winner (from svvamp doc example)
    p = Profile(
        preferences_rk=np.array([[0,1,2],[0,2,1],[1,0,2],[2,0,1],[2,1,0]]),
        labels_candidates=['A','B','C'],
    )
    r_inner = RuleBlack()(p)
    print('scores_:', r_inner.scores_)
    res = get_rule_builder('BLAC')(p)
    print('Condorcet case  cowinners_:', res.cowinners_, '  (expected: [A])')
    print()

    # Case 2: no Condorcet winner Borda fallback with tie
    # A and B have same Borda score, no Condorcet winner
    p2 = _ensure_profile(
        ([[1,0,0],[0,1,0],[0,0,1]]),
        candidates={'A','B','C'}
    )
    p2.demo()
    r2_inner = RuleBlack()(p2)
    print('scores_:', r2_inner.candidates_by_scores_best_to_worst_)
    print('scores_ (no Condorcet):', r2_inner.scores_)
    res2 = get_rule_builder('BLAC')(p2)
    print('Borda fallback  cowinners_:', res2.cowinners_)    
