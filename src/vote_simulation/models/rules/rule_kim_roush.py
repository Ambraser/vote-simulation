"""Kim-Roush method wrapper with semantically correct co-winner detection.

Kim-Roush eliminates all candidates with a Veto score strictly below average
simultaneously each round.  The process stops when all remaining candidates
have the same Veto score.

``scores_[r, c]`` = minus the Veto score of candidate ``c`` at round ``r``.
Eliminated candidates carry ``numpy.inf``.

Co-winners are all candidates that survive to the final round, i.e. all ``c``
with a finite score in ``scores_[-1, :]``.  This matches the rule's stated
tie-break: "when all remaining candidates have the same Veto score, the
candidate with lowest index is declared the winner" — with co-winners being
all such equally-scored survivors.
"""

from __future__ import annotations

from svvamp import Profile
from svvamp.rules.rule_kim_roush import RuleKimRoush

from vote_simulation.models.rules.elimination_based import EliminationBasedRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class KimRoushResult(EliminationBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleKimRoush` with proper co-winner semantics.

    Co-winners are all candidates whose Veto score is finite in the last
    elimination round (i.e. they were not eliminated before the end).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Kim-Roush.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleKimRoush()(profile)
        self.scores_ = self._inner.scores_
        self.cowinners_ = self._init_elimination_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_kim_roush():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Kim-Roush."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return KimRoushResult(profile)

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("KIMR", _build_kim_roush())

if __name__ == "__main__":
    # Case 1 — clear winner: A receives fewest vetos every round
    result1 = KimRoushResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 1 — clear winner:")
    print("  scores_:", result1.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — 3-way tie from round 0: each voter vetos a different candidate
    result2 = KimRoushResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            {"A", "B", "C"},
        )
    )
    print("Case 2 — 3-way tie:")
    print("  scores_:", result2.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — C eliminated first, A and B survive tied
    result3 = KimRoushResult(
        _ensure_profile(
            [[2, 1, 0], [2, 1, 0], [1, 2, 0], [1, 2, 0]],
            {"A", "B", "C"},
        )
    )
    print("Case 3 — 2-way tie after elimination:")
    print("  scores_:", result3.scores_)
    print("  cowinners_:", result3.cowinners_)
