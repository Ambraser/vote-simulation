"""For borda rule 

Ties in borda are compute via 
    """



from __future__ import annotations

from svvamp import Profile, RuleBorda

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule

class BordaResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleBorda` with proper co-winner semantics.

    Co-winners are **all** candidates that share the maximum Borda score
    (sum of points assigned by voters).  svvamp's internal tie-breaking (lowest index
    wins) resolves ``w_`` for manipulation computations only — it does *not*
    define the set of co-winners.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Borda.

    Attributes
    ----------
    cowinners_:
        List of candidate labels that tied at the top Borda score.
    """
    

    def __init__(
        self,
        profile: Profile,
    ) -> None:
        self.profile_ = profile
        self._inner = RuleBorda()(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        """Return all candidates whose Borda score equals the maximum."""
        return self._max_score_cowinners(self._inner.scores_)