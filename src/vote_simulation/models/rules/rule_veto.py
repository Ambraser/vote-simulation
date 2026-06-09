"""Veto (Antiplurality) wrapper with semantically correct co-winner detection.

Each voter vetoes their least-liked candidate.  The candidate receiving the
fewest vetos wins.

``scores_[c]`` = minus the number of vetos against candidate ``c`` (1-D
integer array, higher = better).  Co-winners are all candidates sharing the
maximum score (i.e. the minimum veto count).
"""

from __future__ import annotations

from svvamp import Profile
from svvamp.rules.rule_veto import RuleVeto

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class VetoResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleVeto` with proper co-winner semantics.

    Co-winners are all candidates sharing the minimum veto count (maximum
    ``scores_`` value).

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Veto.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleVeto()(profile)
        self.cowinners_ = self._init_score_based()


def _build_veto():
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Veto."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return VetoResult(profile)

    return builder


register_rule("VETO", _build_veto())
# register_rule("APLU", _build_veto())  # alias
