"""k-Approval voting wrapper.

k-Approval: each voter approves their top-k candidates.  Co-winners are all
candidates sharing the **maximum approval score**.

``scores_`` is a 1-D integer array where ``scores_[c]`` is the number of
voters who approved candidate ``c``.
"""

from __future__ import annotations

from svvamp import Profile, RuleKApproval

from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule
from vote_simulation.models.rules.score_based import ScoreBasedRuleWrapper


class KApprovalResult(ScoreBasedRuleWrapper):
    """Wrapper around :class:`svvamp.RuleKApproval` with proper co-winner semantics.

    Co-winners are **all** candidates that share the maximum approval count.
    svvamp's internal tie-breaking (lowest index wins) resolves ``w_`` for
    manipulation computations only.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run k-Approval.
    k:
        Number of candidates each voter approves (their top-k). Default ``2``.
    """

    def __init__(self, profile: Profile, *, k: int = 2) -> None:
        self.profile_ = profile
        self.k = k
        self._inner = RuleKApproval(k=k)(profile)
        self.cowinners_ = self._init_score_based()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_k_approval(*, k: int = 2):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for k-Approval."""

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return KApprovalResult(profile, k=k)

    return builder


register_rule("AP_K", _build_k_approval(k=2))  # legacy alias
register_rule("AP_K2", _build_k_approval(k=2))
register_rule("AP_K3", _build_k_approval(k=3))
register_rule("AP_K4", _build_k_approval(k=4))
register_rule("AP_K5", _build_k_approval(k=5))
register_rule("AP_K6", _build_k_approval(k=6))
register_rule("AP_K7", _build_k_approval(k=7))
register_rule("AP_K8", _build_k_approval(k=8))
register_rule("AP_K9", _build_k_approval(k=9))
register_rule("AP_K10", _build_k_approval(k=10))
register_rule("AP_K11", _build_k_approval(k=11))
register_rule("AP_K12", _build_k_approval(k=12))
