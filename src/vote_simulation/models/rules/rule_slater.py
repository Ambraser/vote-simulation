"""Slater method wrapper with semantically correct co-winner detection.

The Slater method finds a linear order of candidates that disagrees with the
fewest pairwise majority comparisons (using the matrix of victories rather
than the matrix of duels, unlike Kemeny).

``scores_[c]`` is the position of candidate ``c`` in the optimal Slater order
(integers 1 to ``n_c``, with ``n_c`` for the winner).

.. note::
    The Slater method is NP-hard.  When several orderings are equally optimal,
    svvamp returns the **first lexicographic one**, so only a single winner is
    reported.  Detecting all tied co-winners would require enumerating all
    optimal Slater orders — not feasible in general.  The tie-detection test
    for this rule is therefore marked ``xfail``.
"""

from __future__ import annotations

from svvamp import Profile
from svvamp.rules.rule_slater import RuleSlater

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class SlaterResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleSlater` with proper co-winner semantics.

    Co-winners are all candidates sharing the maximum position score in the
    optimal Slater order.  In practice svvamp returns only the lexicographically
    first optimal order, so a unique winner is almost always reported.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Slater.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleSlater(winner_option="exact")(profile)
        self.cowinners_ = self._init_score_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_slater():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Slater."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return SlaterResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("SLAT", _build_slater())

if __name__ == "__main__":
    # Case 1 — clear Condorcet winner
    result1 = SlaterResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 1 — clear winner:   scores_: {result1._inner.scores_}   cowinners_: {result1.cowinners_}")

    # Case 2 — Condorcet cycle (A>B>C>A each 2-1)
    result2 = SlaterResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            candidates={"A", "B", "C"},
        )
    )
    print(f"Case 2 — 3-way cycle:    scores_: {result2._inner.scores_}   cowinners_: {result2.cowinners_}")
