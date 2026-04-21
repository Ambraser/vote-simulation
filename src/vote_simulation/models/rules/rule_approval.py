"""Approval voting wrapper with semantically correct co-winner detection.

Co-winners are all candidates sharing the **maximum approval score**
(i.e. maximum number of voters who approve them), regardless of the
``tie_break_rule`` used internally by svvamp to resolve ``w_``.

Parameters exposed per wrapper
-------------------------------
approval_threshold : number
    Utility above which a voter approves a candidate. Default ``0``.
approval_comparator : ``'>'`` | ``'>='``
    Strict (``'>'``, default) or non-strict (``'>='``) comparison.

Registered rule codes
---------------------
Threshold-based (``approval_comparator='>'`` unless noted):

================  ============  ==============
Code              threshold     comparator
================  ============  ==============
``AP_T0``         0             ``>``
``AP_T0GE``       0             ``>=``
``AP_T05``        0.5           ``>``
``AP_T05GE``      0.5           ``>=``
``AP_T06``        0.6           ``>``
``AP_T07``        0.7           ``>``
``AP_T08``        0.8           ``>``
``AP_T09``        0.9           ``>``
``AP_T``          0.7           ``>``   (legacy alias)
================  ============  ==============

Note: All manipulation options for :class:`svvamp.RuleApproval` are
``'exact'`` only — svvamp does not expose alternative options for this rule.
"""

from __future__ import annotations

from svvamp import Profile, RuleApproval

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class ApprovalResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleApproval` with proper co-winner semantics.

    Co-winners are **all** candidates that share the maximum approval score
    (number of approving voters).  svvamp's internal tie-breaking (lowest index
    wins) resolves ``w_`` for manipulation computations only — it does *not*
    define the set of co-winners.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Approval.
    approval_threshold:
        Utility above which a voter approves a candidate. Default ``0``.
    approval_comparator:
        ``'>'`` (default, strict) or ``'>='`` (non-strict).

    Attributes
    ----------
    cowinners_:
        List of candidate labels that tied at the top approval score.
    profile_:
        The svvamp profile used for the election.

    Examples
    --------
    >>> from svvamp import Profile
    >>> import numpy as np
    >>> profile = Profile(
    ...     preferences_ut=np.array([[1.0, -1.0], [-1.0, 1.0]]),
    ...     labels_candidates=["A", "B"],
    ... )
    >>> result = ApprovalResult(profile, approval_threshold=0.0)
    >>> result.cowinners_
    ['A', 'B']
    """

    def __init__(
        self,
        profile: Profile,
        *,
        approval_threshold: float = 0.0,
        approval_comparator: str = ">",
    ) -> None:
        self.profile_ = profile
        self.approval_threshold = approval_threshold
        self.approval_comparator = approval_comparator
        self._inner = RuleApproval(
            approval_threshold=approval_threshold,
            approval_comparator=approval_comparator,
            # All manipulation options are forced to 'exact' by svvamp for this rule —
            # no need to expose them.
        )(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        """Return all candidates whose approval score equals the maximum."""
        # scores_[c] = integer count of approving voters — use exact integer equality.
        return self._max_score_cowinners(self._inner.scores_)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_approval(
    *,
    approval_threshold: float = 0.0,
    approval_comparator: str = ">",
):
    """Return a :data:`~vote_simulation.models.rules.registry.RuleBuilder` for Approval.

    Parameters
    ----------
    approval_threshold:
        Utility above which a voter approves a candidate.
    approval_comparator:
        ``'>'`` (strict, default) or ``'>='`` (non-strict).
    """

    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return ApprovalResult(
            profile,
            approval_threshold=approval_threshold,
            approval_comparator=approval_comparator,
        )

    return builder


# ---------------------------------------------------------------------------
# Rule registrations
# ---------------------------------------------------------------------------

# ── Threshold = 0 (default svvamp behaviour) ────────────────────────────────
register_rule("AP_T0", _build_approval(approval_threshold=0.0, approval_comparator=">"))
register_rule("AP_T0GE", _build_approval(approval_threshold=0.0, approval_comparator=">="))

# ── Threshold = 0.5 ─────────────────────────────────────────────────────────
register_rule("AP_T05", _build_approval(approval_threshold=0.5, approval_comparator=">"))
register_rule("AP_T05GE", _build_approval(approval_threshold=0.5, approval_comparator=">="))

# ── Threshold = 0.6 ─────────────────────────────────────────────────────────
register_rule("AP_T06", _build_approval(approval_threshold=0.6, approval_comparator=">"))

# ── Threshold = 0.7 ─────────────────────────────────────────────────────────
register_rule("AP_T07", _build_approval(approval_threshold=0.7, approval_comparator=">"))
register_rule("AP_T", _build_approval(approval_threshold=0.7, approval_comparator=">"))  # legacy alias

# ── Threshold = 0.8 ─────────────────────────────────────────────────────────
register_rule("AP_T08", _build_approval(approval_threshold=0.8, approval_comparator=">"))

# ── Threshold = 0.9 ─────────────────────────────────────────────────────────
register_rule("AP_T09", _build_approval(approval_threshold=0.9, approval_comparator=">"))
