"""Bucklin rule wrapper.

Co-winners are **all** candidates that share the maximum accrued score at the
first counting round where at least one candidate strictly exceeds ``n_v / 2``
votes.  The lowest-index tie-break used internally by svvamp to resolve ``w_``
does *not* define the set of co-winners.

``scores_[r, c]`` is the accrued vote count of candidate ``c`` after counting
ranks 0 through ``r``.  The deciding round ``r*`` is the first row where
``max(scores[r, :]) > n_v / 2``.  All candidates tied at that maximum are
co-winners.
"""

from __future__ import annotations

import numpy as np
from svvamp import Profile, RuleBucklin

from vote_simulation.models.rules.base import SvvampRuleWrapper
from vote_simulation.models.rules.registry import RuleInput, RuleResult, _ensure_profile, register_rule


class BucklinResult(SvvampRuleWrapper):
    """Wrapper around :class:`svvamp.RuleBucklin` with proper co-winner semantics.

    Co-winners are all candidates tied at the highest accrued score at the
    first round where that score exceeds ``n_v / 2``.

    Parameters
    ----------
    profile:
        A :class:`svvamp.Profile` on which to run Bucklin.

    Attributes
    ----------
    cowinners_:
        List of candidate labels that tied at the deciding-round maximum.
    profile_:
        The svvamp profile used for the election.
    """

    def __init__(self, profile: Profile) -> None:
        self.profile_ = profile
        self._inner = RuleBucklin()(profile)
        self.cowinners_ = self._compute_cowinners()

    def _compute_cowinners(self) -> list[str]:
        scores = np.asarray(self._inner.scores_, dtype=float)  # shape [n_rounds, n_c]
        n_v = float(self.profile_.n_v)
        threshold = n_v / 2.0

        for r in range(scores.shape[0]):
            row = scores[r, :]
            if np.max(row) > threshold:
                return self._resolve_cowinners(np.flatnonzero(row == np.max(row)))

        # Fallback (should not happen): return candidate with highest score in last round
        last = scores[-1, :]
        return self._resolve_cowinners(np.flatnonzero(last == np.max(last)))


def _build_bucklin():
    def builder(profile_or_ballots: RuleInput, candidates: set[str] | None = None) -> RuleResult:
        profile = _ensure_profile(profile_or_ballots, candidates)
        return BucklinResult(profile)

    return builder


register_rule("BUCK_R", _build_bucklin())
