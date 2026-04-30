"""k-Approval voting wrapper with semantically correct co-winner detection.

k-Approval: each voter approves their top-k candidates.  Co-winners are all
candidates sharing the **maximum approval score**, regardless of svvamp's
internal tie-breaking.

``scores_`` is a 1-D integer array where ``scores_[c]`` is the number of
voters who approved candidate ``c``.

Registered rule codes
---------------------

=========  ===
Code         k
=========  ===
``AP_K``     2   (legacy alias, same as AP_K2)
``AP_K2``    2
``AP_K3``    3
``AP_K4``    4
``AP_K5``    5
``AP_K6``    6
``AP_K7``    7
``AP_K8``    8
``AP_K9``    9
``AP_K10``  10
``AP_K11``  11
``AP_K12``  12
=========  ===
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


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

register_rule("AP_K", _build_k_approval(k=2))   # legacy alias
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

if __name__ == "__main__":
    # Case 1 — clear winner: candidate A is everyone's top pick
    result1 = KApprovalResult(
        _ensure_profile(
            [[2, 1, 0], [2, 0, 1], [2, 1, 0]],
            {"A", "B", "C"},
        ),
        k=2,
    )
    print("Case 1 — clear winner (k=2):")
    print("  scores_:", result1._inner.scores_)
    print("  cowinners_:", result1.cowinners_)

    # Case 2 — 3-way tie: each voter approves a different single candidate
    result2 = KApprovalResult(
        _ensure_profile(
            [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
            {"A", "B", "C"},
        ),
        k=1,
    )
    print("Case 2 — 3-way tie (k=1, each approves one different):")
    print("  scores_:", result2._inner.scores_)
    print("  cowinners_:", result2.cowinners_)

    # Case 3 — 2-way tie: A and B both get approved by 2 voters, C by 1
    result3 = KApprovalResult(
        _ensure_profile(
            [[2, 1, 0], [2, 1, 0], [0, 1, 2], [0, 2, 1]],
            {"A", "B", "C"},
        ),
        k=1,
    )
    print("Case 3 — 2-way tie (k=1):")
    print("  scores_:", result3._inner.scores_)
    print("  cowinners_:", result3.cowinners_)
